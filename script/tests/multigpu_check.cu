#include "multigpu/runtime.cuh"

namespace {
__global__ void mutate(RoutingDeviceView view, int round, int rank, int size) {
    const int index = (blockIdx.x * blockDim.x + threadIdx.x) * size + rank;
    if (index >= view.L * view.X * view.Y || (index + round) % 7) return;
    graph::add_demand(view, index, round & 1 ? -.5f : 1.f);
}

__global__ void compare(RoutingDeviceView actual, RoutingDeviceView reference, int* errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int X = actual.X, Y = actual.Y;
    if (i >= actual.L * X * Y) return;
    const int l = i / (X * Y), x = i / Y % X, y = i % Y;
    if (cudb::wcost_values(actual)[i] != reference.wcost[i]) atomicAdd(errors, 1);
    if (cudb::via_costs(actual)[i] != reference.vcost[i]) atomicAdd(errors + 1, 1);
    const bool along_y = (l & 1) ^ actual.DIR;
    const int start = along_y ? y : x, end = along_y ? Y : X;
    double expected = 0;
    for (int p = start; p < end; ++p) {
        const double observed = graph::wire_segment_cost(actual, l, x, along_y ? x : p, y, along_y ? p : y);
        if (fabs(observed - expected) > 1e-9 * fmax(1., fabs(expected))) atomicAdd(errors + 2, 1);
        expected += reference.wcost[i + (p - start) * (along_y ? 1 : Y)];
    }
}

void check(grotor::multi::Session& session, int X, int Y, int DIR) {
    cudb::RouterState state;
    auto& v = state.device;
    v.X = X; v.Y = Y; v.L = 4; v.XY = std::max(X, Y); v.DIR = DIR;
    v.unit_length_wire_cost = .00131579; v.unit_via_cost = 4;
    const size_t cells = size_t(v.L) * X * Y;
    auto allocate = [&](auto*& p, size_t n) { state.memory.allocate(&p, n * sizeof(*p)); };
    allocate(v.demand, cells); allocate(v.capacity, cells); allocate(v.wcost, cells);
    allocate(v.vcost, cells); allocate(v.presum, cells); allocate(v.timestamp, cells);
    allocate(v.cross_points, X * Y); allocate(v.x_edge_len, std::max(X, 2));
    allocate(v.y_edge_len, std::max(Y, 2)); allocate(v.unit_length_short_costs, v.L);
    std::vector<int> tracks;
    for (int l = 0; l < v.L; ++l)
        for (int t = 0; t < (((l & 1) ^ DIR) ? X : Y); ++t) tracks.push_back(l * v.XY + t);
    v.all_track_cnt = tracks.size(); allocate(v.idx2track, tracks.size());
    std::vector<float> demand(cells, 2.5f), capacity(cells);
    for (int i = 0; i < cells; ++i) capacity[i] = (i % 17) * .5f;
    std::vector<int> cross(X * Y), xlen(std::max(X, 2), 3), ylen(std::max(Y, 2), 7);
    for (int i = 0; i < X * Y; ++i) cross[i] = i % 2;
    grotor::multi::upload(v.demand, demand); grotor::multi::upload(v.capacity, capacity);
    grotor::multi::upload(v.cross_points, cross); grotor::multi::upload(v.x_edge_len, xlen);
    grotor::multi::upload(v.y_edge_len, ylen); grotor::multi::upload(v.idx2track, tracks);
    grotor::multi::upload(v.unit_length_short_costs, std::vector<double>{5, 7, 11, 13});
    grotor::cuda_check(cudaMemset(v.timestamp, 0, cells * sizeof(int)), "clear fixture timestamps");
    graph::initialize_incremental_cost(state);
    graph::refresh_cost(state);
    // Snapshot stale caches plus pending dirty flags, rather than refreshing
    // them first. This checks propagation of the serial executor's last batch.
    mutate<<<BLOCK_NUM(cells), THREAD_NUM>>>(v, 0, 0, 1);
    grotor::multi::Header header{}; header.dimensions = v;
    grotor::multi::SymmetricMemory prefix_memory;
    RoutingDeviceView distributed{};
    distributed.X = X; distributed.Y = Y; distributed.L = v.L; distributed.XY = v.XY; distributed.DIR = DIR;
    const int shard = (X + session.size - 1) / session.size;
    const size_t local_cells = size_t(v.L) * shard * Y;
    distributed.vcost = prefix_memory.allocate<float>(local_cells);
    distributed.presum = prefix_memory.allocate<double>(local_cells);
    for (int p = 0; p < session.size; ++p) {
        distributed.peer_vcost[p] = static_cast<float*>(nvshmem_ptr(distributed.vcost, p));
        distributed.peer_presum[p] = static_cast<double*>(nvshmem_ptr(distributed.presum, p));
    }
    grotor::multi::DistributedGrid grid(distributed, header, session.rank, session.size);
    grid.snapshot(state); session.sync(); grid.prepare_tables(); session.sync();
    int* errors = nullptr; allocate(errors, 3);
    for (int round = 0; round < 4; ++round) {
        if (round) {
            mutate<<<BLOCK_NUM(cells), THREAD_NUM>>>(distributed, round, session.rank, session.size);
            mutate<<<BLOCK_NUM(cells), THREAD_NUM>>>(v, round, 0, 1);
            session.sync();
        }
        grid.costs(); grid.prefix(); session.sync();
        graph::update_cost(state);
        grotor::cuda_check(cudaMemset(errors, 0, 3 * sizeof(int)), "clear comparison counters");
        compare<<<BLOCK_NUM(cells), THREAD_NUM>>>(distributed, v, errors);
        int counts[3]{};
        grotor::cuda_check(cudaMemcpy(counts, errors, sizeof(counts), cudaMemcpyDeviceToHost), "read distributed comparison");
        if (counts[0] || counts[1] || counts[2]) {
            fprintf(stderr, "rank=%d X=%d Y=%d DIR=%d round=%d mismatches wire=%d via=%d queries=%d\n",
                session.rank, X, Y, DIR, round, counts[0], counts[1], counts[2]);
            throw std::runtime_error("Distributed costs/prefix queries differ from full-grid reference");
        }
        // Clean refresh must retain both cached cost arrays and query results.
        grid.costs(); grid.prefix(); session.sync();
    }
    if (!session.rank) printf("PASS MULTIGPU_COST_PREFIX X=%d Y=%d DIR=%d ranks=%d\n", X, Y, DIR, session.size);
}
}

int main(int argc, char** argv) {
    try {
        grotor::multi::Session session(argc, argv);
        for (int dir : {0, 1}) {
            check(session, 1, 1, dir);
            check(session, 13, 7, dir); // Unaligned words and unequal last shard.
            check(session, 37, 65, dir);
        }
        if (session.rank) session.serve();
        else printf("MULTIGPU_COST_PREFIX_CHECK_PASSED\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
