#pragma once
#include "routing/congestion.cuh"
#include <array>
#include <cmath>

namespace grotor::maze {

__global__ void overflow_partials(cudb::RoutingDeviceView view, double* partials) {
    __shared__ double sums[256];
    const size_t cells = size_t(view.L) * view.X * view.Y;
    double value = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < cells; i += size_t(gridDim.x) * blockDim.x) {
        const int layer = i / (view.X * view.Y), x = i / view.Y % view.X, y = i % view.Y;
        const bool vertical = (layer & 1) ^ view.DIR;
        if (vertical ? y + 1 < view.Y : x + 1 < view.X)
            value += view.unit_length_short_costs[layer] * graph::of_cost(view, view.capacity[i], view.demand[i]);
    }
    sums[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) partials[blockIdx.x] = sums[0];
}

// A fixed reduction tree makes the acceptance decision independent of atomic
// floating-point addition order. Wirelength/via statistics are integer sums.
inline double routing_score(cudb::RouterState& state) {
    DeviceMemory memory;
    double* partials = nullptr;
    memory.allocate(&partials, 1024 * sizeof(double));
    overflow_partials<<<1024, 256>>>(state.device, partials);
    std::array<double, 1024> host{};
    cudb::RoutingStatistics statistics;
    cuda_check(cudaMemcpy(host.data(), partials, sizeof(host), cudaMemcpyDeviceToHost), "read maze score partials");
    cuda_check(cudaMemcpy(&statistics, state.device.statistics, sizeof(statistics), cudaMemcpyDeviceToHost), "read maze score statistics");
    double score = statistics.total_wirelength * state.device.unit_length_wire_cost
        + statistics.total_via_count * state.device.unit_via_cost;
    for (double value : host) score += value;
    return score;
}

__global__ void restore_reroute_routes(cudb::RoutingDeviceView view, const int* previous_routes,
    const int* previous_flags, const int* previous_offsets, unsigned long long previous_stride) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= view.NET_NUM || !previous_flags[id]) return;
    const int offset = previous_offsets[id];
    const int* source = previous_routes + offset * previous_stride;
    int* target = view.routes_phase2 + offset * view.ROUTE_PER_PIN_PHASE2;
    for (int i = 0; i < source[0]; ++i) target[i] = source[i];
}

// Only rerouted routes and demand need snapshots. Phase-one routes remain
// immutable. Keep one checkpoint, with a free-memory reserve, for this phase.
struct Checkpoint {
    DeviceMemory memory;
    float* demand = nullptr;
    int *routes = nullptr, *flags = nullptr, *offsets = nullptr;
    int pins;
    unsigned long long stride;
    size_t cells;
    std::vector<int> host_flags, host_offsets;
    cudb::RoutingStatistics statistics;

    explicit Checkpoint(cudb::RouterState& state)
        : pins(state.device.PIN_NUM_PHASE2), stride(state.device.ROUTE_PER_PIN_PHASE2),
          cells(size_t(state.device.L) * state.device.X * state.device.Y),
          host_flags(state.ripup_flag_cpu), host_offsets(state.pin_cnt_sum_phase2_cpu) {
        const size_t route_bytes = (stride * pins + 1) * sizeof(int);
        const size_t flags_bytes = (state.device.NET_NUM + 1) * sizeof(int);
        size_t available = 0, total = 0;
        cuda_check(cudaMemGetInfo(&available, &total), "maze checkpoint budget");
        if (route_bytes + cells * sizeof(float) + 2 * flags_bytes + (size_t(256) << 20) > available)
            throw std::runtime_error("Maze checkpoint exceeds the available device-memory budget");
        memory.allocate(&routes, route_bytes);
        memory.allocate(&demand, cells * sizeof(float));
        memory.allocate(&flags, flags_bytes);
        memory.allocate(&offsets, flags_bytes);
        cuda_check(cudaMemcpy(routes, state.device.routes_phase2, route_bytes, cudaMemcpyDeviceToDevice), "snapshot maze routes");
        cuda_check(cudaMemcpy(demand, state.device.demand, cells * sizeof(float), cudaMemcpyDeviceToDevice), "snapshot maze demand");
        cuda_check(cudaMemcpy(flags, state.device.ripup_flag, flags_bytes, cudaMemcpyDeviceToDevice), "snapshot maze flags");
        cuda_check(cudaMemcpy(offsets, state.device.pin_acc_num_phase2, flags_bytes, cudaMemcpyDeviceToDevice), "snapshot maze offsets");
        cuda_check(cudaMemcpy(&statistics, state.device.statistics, sizeof(statistics), cudaMemcpyDeviceToHost), "snapshot maze statistics");
        printf("MAZE_CHECKPOINT bytes=%zu\n", route_bytes + cells * sizeof(float) + 2 * flags_bytes);
    }

    void restore(cudb::RouterState& state) {
        auto& view = state.device;
        restore_reroute_routes<<<(view.NET_NUM + 127) / 128, 128>>>(view, routes, flags, offsets, stride);
        cuda_check(cudaMemcpy(view.demand, demand, cells * sizeof(float), cudaMemcpyDeviceToDevice), "restore maze demand");
        const size_t bytes = (view.NET_NUM + 1) * sizeof(int);
        cuda_check(cudaMemcpy(view.ripup_flag, flags, bytes, cudaMemcpyDeviceToDevice), "restore maze flags");
        cuda_check(cudaMemcpy(view.pin_acc_num_phase2, offsets, bytes, cudaMemcpyDeviceToDevice), "restore maze offsets");
        cuda_check(cudaMemcpy(view.statistics, &statistics, sizeof(statistics), cudaMemcpyHostToDevice), "restore maze statistics");
        state.ripup_flag_cpu = host_flags;
        state.pin_cnt_sum_phase2_cpu = host_offsets;
        view.PIN_NUM_PHASE2 = pins;
        state.costs_initialized = false;
        cuda_check(cudaGetLastError(), "restore maze checkpoint");
    }
};
} // namespace grotor::maze
