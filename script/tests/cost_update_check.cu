#include "routing/demand.cuh"
#include <numeric>

namespace {
__global__ void change_demands(cudb::RoutingDeviceView view, int round) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.L * view.X * view.Y) return;
    const unsigned int hash = (i * 2654435761u) ^ (round * 2246822519u);
    if (round % 7 == 0 || hash % 31 == 0) {
        view.demand[i] = float((hash >> 8) % 128) * .5f;
        graph::mark_cost_dirty(view, i);
    }
}

__global__ void change_segments(cudb::RoutingDeviceView view, int round) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread >= 97) return;
    const int layer = thread % view.L;
    const bool along_y = (layer & 1) ^ view.DIR;
    const int length = along_y ? view.Y : view.X, tracks = along_y ? view.X : view.Y;
    const int track = (thread * 7 + round) % tracks;
    const int a = (thread * 13 + round) % length, b = (thread * 19 + 1) % length;
    const int base = layer * view.X * view.Y + (along_y ? track * view.Y : track);
    const int stride = along_y ? 1 : view.Y;
    for (int i = min(a, b); i < max(a, b); ++i) atomicAdd(view.demand + base + i * stride, .5f);
    graph::mark_wire_span(view, layer, along_y ? track : a, along_y ? track : b,
                          along_y ? a : track, along_y ? b : track);
}

__global__ void prepare_wire_counts(cudb::RoutingDeviceView view, int round) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < view.L * view.X * view.Y) view.pre_demand[i] = (i * 31u + round) % 17 == 0;
}

__global__ void compare_costs(cudb::RoutingDeviceView actual, cudb::RoutingDeviceView expected, int* errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= actual.L * actual.X * actual.Y) return;
    const int l = i / actual.X / actual.Y, x = i / actual.Y % actual.X, y = i % actual.Y;
    if (__float_as_int(actual.wcost[i]) != __float_as_int(expected.wcost[i])) atomicAdd(errors, 1);
    if (__float_as_int(actual.vcost[i]) != __float_as_int(expected.vcost[i])) atomicAdd(errors + 1, 1);
    const double prefix = graph::fenwick_prefix(actual, l, x, y);
    if (__double_as_longlong(prefix) != __double_as_longlong(expected.presum[i])) atomicAdd(errors + 2, 1);
    if (__double_as_longlong(actual.presum[i]) != __double_as_longlong(expected.presum[i])) atomicAdd(errors + 3, 1);
}

void check(int X, int Y, int DIR) {
    cudb::RouterState state;
    auto& v = state.device;
    v.L = 3; v.X = X; v.Y = Y; v.XY = std::max(X, Y); v.DIR = DIR;
    v.unit_length_wire_cost = .00131579; v.unit_via_cost = 4;
    const int cells = v.L * X * Y;
    const auto allocate = [&](auto*& p, size_t count) { state.memory.allocate(&p, count * sizeof(*p)); };
    allocate(v.wcost, cells); allocate(v.vcost, cells); allocate(v.presum, cells);
    allocate(v.capacity, cells); allocate(v.demand, cells); allocate(v.cross_points, X * Y);
    allocate(v.pre_demand, cells); allocate(v.timestamp, cells); allocate(v.statistics, 1);
    grotor::cuda_check(cudaMemset(v.statistics, 0, sizeof(*v.statistics)), "initialize cost fixture statistics");
    allocate(v.unit_length_short_costs, v.L); allocate(v.x_edge_len, X); allocate(v.y_edge_len, Y);
    std::vector<int> tracks;
    for (int l = 0; l < v.L; ++l)
        for (int t = 0; t < (((l & 1) ^ DIR) ? X : Y); ++t) tracks.push_back(l * v.XY + t);
    v.all_track_cnt = tracks.size(); allocate(v.idx2track, tracks.size());
    std::vector<float> capacities(cells), demands(cells, 1.5f);
    std::vector<int> crosses(X * Y), xlen(X), ylen(Y);
    for (int i = 0; i < cells; ++i) capacities[i] = i % 17 == 0 ? 0.f : float(i % 21) * .25f;
    for (int i = 0; i < X * Y; ++i) crosses[i] = i & 1;
    std::iota(xlen.begin(), xlen.end(), 1); std::iota(ylen.begin(), ylen.end(), 3);
    const std::vector<double> weights{5., 1e10, 1e-6};
    const auto upload = [](auto* p, const auto& data) {
        grotor::cuda_check(cudaMemcpy(p, data.data(), data.size() * sizeof(*p), cudaMemcpyHostToDevice), "upload cost fixture");
    };
    upload(v.capacity, capacities); upload(v.demand, demands); upload(v.cross_points, crosses);
    upload(v.unit_length_short_costs, weights); upload(v.x_edge_len, xlen); upload(v.y_edge_len, ylen);
    upload(v.idx2track, tracks);
    auto reference = v;
    reference.wcost = nullptr; reference.vcost = nullptr; reference.presum = nullptr;
    allocate(reference.wcost, cells); allocate(reference.vcost, cells); allocate(reference.presum, cells);
    graph::initialize_incremental_cost(state);
    grotor::cuda_check(cudaFuncSetAttribute(graph::compute_presum,
        cudaFuncAttributeMaxDynamicSharedMemorySize, v.XY * sizeof(double)), "reference prefix shared memory");
    int* error_counts = nullptr; allocate(error_counts, 4);
    for (int round = 0; round < 40; ++round) {
        if (round) {
            if (round % 3 == 0) change_segments<<<1, 128>>>(v, round);
            else if (round % 3 == 1) {
                prepare_wire_counts<<<BLOCK_NUM(cells), THREAD_NUM>>>(v, round);
                graph::commit_all_edge<<<BLOCK_NUM(cells), THREAD_NUM>>>(v, round);
            } else change_demands<<<BLOCK_NUM(cells), THREAD_NUM>>>(v, round);
        }
        if (round == 23) {
            v.of_cost_scale = reference.of_cost_scale = .7;
            const size_t bytes = ((size_t(cells) + 31) / 32) * sizeof(unsigned int);
            grotor::cuda_check(cudaMemset(v.dirty_wire, 0xff, 2 * bytes), "invalidate global cost weights");
        }
        graph::refresh_cost(state);
        // Refresh again without mutations: clean batches must retain all values.
        if (round % 5 == 0) graph::refresh_cost(state);
        graph::update_wcost_cuda_ispd24<<<BLOCK_NUM(cells), THREAD_NUM>>>(reference);
        graph::update_vcost_ispd24<<<BLOCK_NUM(cells), THREAD_NUM>>>(reference);
        graph::compute_presum<<<v.all_track_cnt, THREAD_NUM, v.XY * sizeof(double)>>>(reference);
        grotor::cuda_check(cudaMemset(error_counts, 0, 4 * sizeof(int)), "reset cost comparison");
        compare_costs<<<BLOCK_NUM(cells), THREAD_NUM>>>(v, reference, error_counts);
        int errors[4];
        grotor::cuda_check(cudaMemcpy(errors, error_counts, sizeof(errors), cudaMemcpyDeviceToHost), "read cost comparison");
        if (errors[0] || errors[1] || errors[2] || errors[3]) {
            fprintf(stderr, "Cost mismatch X=%d Y=%d DIR=%d round=%d wire=%d via=%d prefix=%d cache=%d\n",
                    X, Y, DIR, round, errors[0], errors[1], errors[2], errors[3]);
            throw std::runtime_error("Incremental cost/BIT must exactly match full updates/scans");
        }
    }
    printf("PASS COST_BIT X=%d Y=%d DIR=%d rounds=40 exact_values=1\n", X, Y, DIR);
}
}

int main() {
    try {
        for (int dir : {0, 1}) {
            check(1, 1, dir); check(37, 65, dir); check(64, 33, dir);
            check(3511, 9, dir); check(7, 4764, dir);
        }
        printf("COST_BIT_CHECK_PASSED\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
