#pragma once

#include "router_state.hpp"

namespace graph {

constexpr double via_weight = 1;

__device__ double of_cost_scaled(const RoutingDeviceView& view, float capacity, float demand) {
    auto& of_cost_scale = view.of_cost_scale;
    if(capacity > 0.001) return __expf(min(0.5 * (demand - capacity), of_cost_scale * 0.5 * (demand - capacity)));
    if(demand > 0) return __expf(min(1.5 * demand, of_cost_scale * 1.5 * demand));
    return 0;
}

__device__ double of_cost(const RoutingDeviceView& view, float capacity, float demand) {

    if(capacity > 0.001) return __expf(0.5 * (demand - capacity));
    if(demand > 0) return __expf(1.5 * demand);
    return 0;
}

__device__ double incremental_of_cost(const RoutingDeviceView& view, int l, int x, int y, double incre_demand, double incre_base = 0) {
    auto& X = view.X;
    auto& Y = view.Y;
    auto& unit_length_short_costs = view.unit_length_short_costs;
    auto& capacity = view.capacity;
    auto& demand = view.demand;
    int idx = l * X * Y + x * Y + y;
    return min(1e12, (of_cost_scaled(view, capacity[idx], demand[idx] + incre_demand + incre_base) 
                    - of_cost_scaled(view, capacity[idx], demand[idx] + incre_base)) * unit_length_short_costs[l]);
}

__device__ void update_single_wcost_ispd24(const RoutingDeviceView& view, int idx) {
    auto& DIR = view.DIR;
    auto& L = view.L;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& x_edge_len = view.x_edge_len;
    auto& y_edge_len = view.y_edge_len;
    auto& unit_length_wire_cost = view.unit_length_wire_cost;
    auto& wcost = view.wcost;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y, dir = l & 1 ^ DIR;
    wcost[idx] = incremental_of_cost(view, l, x, y, 1);
    if(dir == 0 && x < X - 1) wcost[idx] += unit_length_wire_cost * x_edge_len[x];
    if(dir == 1 && y < Y - 1) wcost[idx] += unit_length_wire_cost * y_edge_len[y];
#ifdef GROTOR_GGR_ROUTING
    // Integer DP edges must remain nonnegative and below its finite sentinel.
    wcost[idx] = float(int64_t(fminf(fmaxf(wcost[idx], 0.f), 1.e7f)));
#endif
}

__global__ void update_wcost_cuda_ispd24(RoutingDeviceView view) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < view.L * view.X * view.Y) update_single_wcost_ispd24(view, idx);
}

__device__ void update_single_vcost_ispd24(const RoutingDeviceView& view, int l, int x, int y) {
    auto& DIR = view.DIR;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& unit_via_cost = view.unit_via_cost;
    auto& vcost = view.vcost;
    auto& cross_points = view.cross_points;
    int idx = IDX(l, x, y);
    int is_cross_point = cross_points[x * Y + y];
    if(l & 1 ^ DIR) {
        if(y == 0) 
            vcost[idx] = incremental_of_cost(view, l, x, y, 1, is_cross_point);
        else if(y == Y - 1)
            vcost[idx] = incremental_of_cost(view, l, x, y - 1, 1, is_cross_point);
        else
            vcost[idx] = incremental_of_cost(view, l, x, y, 0.5, is_cross_point) + incremental_of_cost(view, l, x, y - 1, 0.5, is_cross_point);
    } else {
        if(x == 0)
            vcost[idx] = incremental_of_cost(view, l, x, y, 1, is_cross_point);
        else if(x == X - 1)
            vcost[idx] = incremental_of_cost(view, l, x - 1, y, 1, is_cross_point);
        else
            vcost[idx] = incremental_of_cost(view, l, x, y, 0.5, is_cross_point) + incremental_of_cost(view, l, x - 1, y, 0.5, is_cross_point);
    }
    vcost[idx] += unit_via_cost;
}

__global__ void update_vcost_ispd24(RoutingDeviceView view) {
    auto& L = view.L;
    auto& X = view.X;
    auto& Y = view.Y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < L * X * Y) update_single_vcost_ispd24(view, idx / X / Y, idx / Y % X, idx % Y);
}

void update_cost_ispd24(RouterState& state) {
    auto& L = state.device.L;
    auto& X = state.device.X;
    auto& Y = state.device.Y;
    update_wcost_cuda_ispd24<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>>(state.device);
    update_vcost_ispd24<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>>(state.device);
}

void update_cost(RouterState& state) {

    update_cost_ispd24(state);
}

} // namespace graph
