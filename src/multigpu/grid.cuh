#pragma once
// Included inside grotor::multi after the symmetric allocation helpers.

__global__ void update_grid_costs(const GROTOR_GRID_CONSTANT RoutingDeviceView view, int rank, int shard_x) {
    const int width = min(shard_x, view.X - rank * shard_x);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= view.L * width * view.Y) return;
    const int y = i % view.Y, x = i / view.Y % width + rank * shard_x;
    const int l = i / view.Y / width, index = (l * view.X + x) * view.Y + y;
    const int local = (l * shard_x + x - rank * shard_x) * view.Y + y;
    if (graph::is_dirty(view.dirty_wire, local)) graph::update_single_wcost_ispd24(view, index);
    if (graph::is_dirty(view.dirty_via, local)) graph::update_single_vcost_ispd24(view, l, x, y);
}

__global__ void snapshot_dirty_costs(RoutingDeviceView source, unsigned int* wire, unsigned int* via,
                                    int rank, int shard_x, bool initialized) {
    const int word = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = min(shard_x, source.X - rank * shard_x);
    const int allocated = source.L * shard_x * source.Y;
    if (word >= (allocated + 31) / 32) return;
    unsigned int w = 0, v = 0;
    for (int b = 0; b < 32; ++b) {
        const int i = word * 32 + b;
        // Padded last shards use shard_x, not width, as their layer stride.
        const int x = i / source.Y % shard_x, layer = i / (shard_x * source.Y);
        if (i >= allocated || x >= width) continue;
        const int global = (layer * source.X + rank * shard_x + x) * source.Y + i % source.Y;
        if (!initialized || !source.dirty_wire || graph::is_dirty(source.dirty_wire, global)) w |= 1u << b;
        if (!initialized || !source.dirty_via || graph::is_dirty(source.dirty_via, global)) v |= 1u << b;
    }
    wire[word] = w; via[word] = v;
}

// Local inclusive prefix scans match the legacy NVSHMEM layout. Y tracks
// belong to one owner; X tracks are scanned independently on each X shard.
__global__ void compute_local_presum(const GROTOR_GRID_CONSTANT RoutingDeviceView view, int rank, int shard_x) {
    const int code = view.idx2track[blockIdx.x], layer = code / view.XY, track = code % view.XY;
    const bool along_y = (layer & 1) ^ view.DIR;
    const int width = min(shard_x, view.X - rank * shard_x);
    if (width <= 0 || (along_y && track / shard_x != rank)) return;
    const int length = along_y ? view.Y : width;
    const int stride = along_y ? 1 : view.Y;
    const int base = layer * shard_x * view.Y + (along_y ? track % shard_x * view.Y : track);
    extern __shared__ double sum[];
    for (int i = threadIdx.x; i < length; i += blockDim.x) sum[i] = view.wcost[base + i * stride];
    __syncthreads();
    for (int d = 0; (1 << d) < length; ++d) {
        for (int i = threadIdx.x; i < length; i += blockDim.x)
            if ((i >> d) & 1) sum[i] += sum[((i >> d) << d) - 1];
        __syncthreads();
    }
    for (int i = threadIdx.x; i < length; i += blockDim.x) view.presum[base + i * stride] = sum[i];
}
struct StatisticPeers { RoutingStatistics* values[8]{}; int count = 0; };
__global__ void collect_statistics(RoutingStatistics* target, StatisticPeers peers) {
    for (int p = 0; p < peers.count; ++p) {
        target->total_wirelength += peers.values[p]->total_wirelength;
        target->total_via_count += peers.values[p]->total_via_count;
    }
}

struct DistributedGrid {
    SymmetricMemory memory;
    RoutingDeviceView& view;
    int rank, size, shard_x;
    size_t shard_cells;
    size_t dirty_bytes;
    float *demand, *capacity, *wcost;
    int *timestamps, *cross;
    int *x_lengths, *y_lengths, *tracks;
    double* short_costs;
    RoutingStatistics* statistics;
    StatisticPeers statistic_peers;
    std::unique_ptr<flt::Stream> streams[8];
    std::unique_ptr<flt::Event> done[8];
    flt::Event ready;

    DistributedGrid(RoutingDeviceView& target, const Header& header, int my_rank, int ranks)
        : view(target), rank(my_rank), size(ranks), shard_x((view.X + ranks - 1) / ranks),
          shard_cells(size_t(view.L) * shard_x * view.Y) {
        demand = memory.allocate<float>(shard_cells);
        capacity = memory.allocate<float>(shard_cells);
        wcost = memory.allocate<float>(shard_cells);
        dirty_bytes = ((shard_cells + 31) / 32) * sizeof(unsigned int);
        view.dirty_wire = memory.allocate<unsigned int>(2 * dirty_bytes / sizeof(unsigned int));
        view.dirty_via = view.dirty_wire + dirty_bytes / sizeof(unsigned int);
        timestamps = memory.allocate<int>(shard_cells);
        cross = memory.allocate<int>(size_t(shard_x) * view.Y);
        x_lengths = memory.allocate<int>(view.X - 1);
        y_lengths = memory.allocate<int>(view.Y - 1);
        tracks = memory.allocate<int>(header.dimensions.all_track_cnt);
        short_costs = memory.allocate<double>(view.L);
        statistics = memory.allocate<RoutingStatistics>(1);
        view.cost_shard_x = size > 1 ? shard_x : 0;
        view.local_presum = true;
        view.demand = demand; view.capacity = capacity; view.wcost = wcost;
        view.timestamp = timestamps; view.cross_points = cross;
        view.pre_demand = nullptr; view.statistics = statistics;
        view.unit_length_wire_cost = header.dimensions.unit_length_wire_cost;
        view.unit_via_cost = header.dimensions.unit_via_cost;
        view.of_cost_scale = header.dimensions.of_cost_scale;
        view.all_track_cnt = header.dimensions.all_track_cnt;
        view.x_edge_len = x_lengths; view.y_edge_len = y_lengths;
        view.idx2track = tracks; view.unit_length_short_costs = short_costs;
        statistic_peers.count = size;
        for (int p = 0; p < size; ++p) {
            view.peer_dirty_wire[p] = mapped(view.dirty_wire, p);
            view.peer_dirty_via[p] = mapped(view.dirty_via, p);
            view.peer_demand[p] = mapped(demand, p);
            view.peer_capacity[p] = mapped(capacity, p);
            view.peer_wcost[p] = mapped(wcost, p);
            view.peer_timestamp[p] = mapped(timestamps, p);
            view.peer_cross_points[p] = mapped(cross, p);
            statistic_peers.values[p] = mapped(statistics, p);
            if (!rank) { streams[p] = std::make_unique<flt::Stream>(); done[p] = std::make_unique<flt::Event>(); }
        }
        cuda_check(cudaFuncSetAttribute(compute_local_presum,
            cudaFuncAttributeMaxDynamicSharedMemorySize, view.XY * sizeof(double)), "distributed prefix shared memory");
        cudaFuncAttributes attributes{};
        cuda_check(cudaFuncGetAttributes(&attributes, update_grid_costs), "load distributed grid costs");
        cuda_check(cudaFuncGetAttributes(&attributes, compute_local_presum), "load distributed prefix scan");
        cuda_check(cudaFuncGetAttributes(&attributes, graph::commit_wire_demand), "load distributed wire commit");
        cuda_check(cudaFuncGetAttributes(&attributes, graph::commit_via_demand), "load distributed via commit");
    }
    template<class T> T* mapped(T* pointer, int p) {
        auto* result = static_cast<T*>(nvshmem_ptr(pointer, p));
        if (!result) throw std::runtime_error("Grid shard lacks an NVLink peer mapping");
        return result;
    }
    template<class T> void move_shard(T* full, T* shard, int p, int layers, bool gather) {
        const int begin = p * shard_x, width = std::min(shard_x, view.X - begin);
        if (width <= 0) return;
        const size_t full_pitch = size_t(view.X) * view.Y * sizeof(T);
        const size_t shard_pitch = size_t(shard_x) * view.Y * sizeof(T);
        T* global = full + size_t(begin) * view.Y;
        cuda_check(cudaMemcpy2DAsync(gather ? global : shard, gather ? full_pitch : shard_pitch,
            gather ? shard : global, gather ? shard_pitch : full_pitch,
            size_t(width) * view.Y * sizeof(T), layers, cudaMemcpyDefault, streams[p]->value), "transfer grid shard over NVLink");
    }
    void wait_copies() {
        for (int p = 0; p < size; ++p) {
            cuda_check(cudaEventRecord(done[p]->value, streams[p]->value), "record grid copy completion");
            cuda_check(cudaStreamWaitEvent(nullptr, done[p]->value, 0), "join parallel grid copies");
        }
    }
    void snapshot(const RouterState& state) {
        if (rank) return;
        const auto& original = state.device;
        copy(x_lengths, original.x_edge_len, size_t(view.X - 1) * sizeof(int));
        copy(y_lengths, original.y_edge_len, size_t(view.Y - 1) * sizeof(int));
        copy(tracks, original.idx2track, size_t(view.all_track_cnt) * sizeof(int));
        copy(short_costs, original.unit_length_short_costs, size_t(view.L) * sizeof(double));
        cuda_check(cudaEventRecord(ready.value), "record pre-batch demand readiness");
        for (int p = 0; p < size; ++p) {
            cuda_check(cudaStreamWaitEvent(streams[p]->value, ready.value, 0), "wait for phase-one removal");
            move_shard(original.demand, view.peer_demand[p], p, view.L, false);
            move_shard(original.capacity, view.peer_capacity[p], p, view.L, false);
            move_shard(original.timestamp, view.peer_timestamp[p], p, view.L, false);
            move_shard(original.cross_points, view.peer_cross_points[p], p, 1, false);
            if (state.costs_initialized) {
                move_shard(original.wcost, view.peer_wcost[p], p, view.L, false);
                move_shard(original.vcost, view.peer_vcost[p], p, view.L, false);
            }
            snapshot_dirty_costs<<<BLOCK_NUM((shard_cells + 31) / 32), THREAD_NUM, 0, streams[p]->value>>>(
                original, view.peer_dirty_wire[p], view.peer_dirty_via[p], p, shard_x, state.costs_initialized);
        }
        wait_copies();
    }
    void prepare_tables() {
        // These tiny read-only tables are hot in every DP/cost/commit loop.
        // Repeated peer reads are much more expensive than one local copy.
        if (rank) {
            copy(x_lengths, root_pointer(x_lengths), size_t(view.X - 1) * sizeof(int));
            copy(y_lengths, root_pointer(y_lengths), size_t(view.Y - 1) * sizeof(int));
            copy(tracks, root_pointer(tracks), size_t(view.all_track_cnt) * sizeof(int));
            copy(short_costs, root_pointer(short_costs), size_t(view.L) * sizeof(double));
        }
        cuda_check(cudaMemsetAsync(statistics, 0, sizeof(RoutingStatistics)), "clear batch statistics");
    }
    void costs() {
        const int width = std::max(0, std::min(shard_x, view.X - rank * shard_x));
        if (width) update_grid_costs<<<BLOCK_NUM(view.L * width * view.Y), THREAD_NUM>>>(view, rank, shard_x);
        cuda_check(cudaMemsetAsync(view.dirty_wire, 0, 2 * dirty_bytes), "clear consumed distributed cost flags");
    }
    void prefix() {
        compute_local_presum<<<view.all_track_cnt, THREAD_NUM, view.XY * sizeof(double)>>>(view, rank, shard_x);
    }
    void distribute_routes(int* routes, const std::vector<int>& pins, const std::vector<int>& partitions) {
        if (rank) return;
        for (int p = 1; p < size; ++p) {
            const size_t begin = size_t(pins[partitions[p]]) * view.ROUTE_PER_PIN_PHASE2;
            const size_t count = size_t(pins[partitions[p + 1]] - pins[partitions[p]]) * view.ROUTE_PER_PIN_PHASE2;
            if (count) copy(mapped(routes, p) + begin, routes + begin, count * sizeof(int));
        }
    }
    void gather(RouterState& state, int* routes, const std::vector<int>& pins,
                const std::vector<int>& partitions, bool second) {
        if (rank) return;
        for (int p = 0; p < size; ++p) {
            move_shard(state.device.demand, view.peer_demand[p], p, view.L, true);
            move_shard(state.device.timestamp, view.peer_timestamp[p], p, view.L, true);
            if (p) {
                const size_t stride = second ? view.ROUTE_PER_PIN_PHASE2 : view.ROUTE_PER_PIN;
                const size_t begin = size_t(pins[partitions[p]]) * stride;
                const size_t count = size_t(pins[partitions[p + 1]] - pins[partitions[p]]) * stride;
                if (count) cuda_check(cudaMemcpyAsync(routes + begin, mapped(routes, p) + begin,
                    count * sizeof(int), cudaMemcpyDefault, streams[p]->value), "gather disjoint local routes");
            }
        }
        wait_copies();
        collect_statistics<<<1, 1>>>(state.device.statistics, statistic_peers);
        // The root's old cache was not updated by distributed commits. Rebuild
        // it on the next serial batch, using dense scans without a BIT.
        if (state.device.dirty_wire) {
            const size_t bytes = ((size_t(view.L) * view.X * view.Y + 31) / 32) * sizeof(unsigned int);
            cuda_check(cudaMemsetAsync(state.device.dirty_wire, 0xff, 2 * bytes), "invalidate root cost cache after distributed commit");
        }
    }
};
