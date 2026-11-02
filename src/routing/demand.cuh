#pragma once

#include "router_state.hpp"
#include "cost/incremental_update.cuh"

namespace graph {

__device__ __forceinline__ void add_demand(const RoutingDeviceView& view, int index, float value) {
    atomicAdd(view.demand + index, value);
    mark_cost_dirty(view, index);
}


__device__ void atomic_add_unit_demand_wire_segment(const RoutingDeviceView& view, int l, int minx, int maxx, int miny, int maxy, int stamp, int K = 1) {
    auto& X = view.X;
    auto& Y = view.Y;
    auto& timestamp = view.timestamp;
    auto& pre_demand = view.pre_demand;
    assert(minx == maxx || miny == maxy);

    atomicAdd(pre_demand + IDX(l, minx, miny), K);
    if(minx == maxx) {
        atomicAdd(pre_demand + IDX(l, minx, maxy), -K);
        timestamp[IDX(l, minx, maxy)] = stamp;
    }
    if(miny == maxy) {
        atomicAdd(pre_demand + IDX(l, maxx, miny), -K);
        timestamp[IDX(l, maxx, miny)] = stamp;
    }
}

__global__ void commit_all_edge(RoutingDeviceView view, int stamp) {
    auto& DIR = view.DIR;
    auto& L = view.L;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& x_edge_len = view.x_edge_len;
    auto& y_edge_len = view.y_edge_len;
    auto& demand = view.demand;
    auto& timestamp = view.timestamp;
    auto& pre_demand = view.pre_demand;
    auto& total_wirelength = view.statistics->total_wirelength;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    bool changed = false;
    if((l & 1 ^ DIR) == 0 && x + 1 < X && pre_demand[IDX(l, x + 1, y)] > 0) {
        demand[IDX(l, x, y)] += pre_demand[IDX(l, x + 1, y)];
        changed = true;
        timestamp[IDX(l, x, y)] = stamp;
        atomicAdd(&total_wirelength, pre_demand[IDX(l, x + 1, y)] * x_edge_len[x]);
    }
    if((l & 1 ^ DIR) == 1 && y + 1 < Y && pre_demand[IDX(l, x, y + 1)] > 0) {
        demand[IDX(l, x, y)] += pre_demand[IDX(l, x, y + 1)];
        changed = true;
        timestamp[IDX(l, x, y)] = stamp;
        atomicAdd(&total_wirelength, pre_demand[IDX(l, x, y + 1)] * y_edge_len[y]);
    }
    if (view.dirty_wire) {
        const unsigned int active = __activemask();
        const unsigned int mask = __ballot_sync(active, changed);
        const unsigned int vertical = __ballot_sync(active, changed && ((l & 1) ^ DIR));
        if (!(threadIdx.x & 31) && mask) {
            const int word = idx >> 5;
            atomicOr(view.dirty_wire + word, mask);
            atomicOr(view.dirty_via + word, mask);
            mark_shifted_word(view.dirty_via, word, vertical, 1);
            mark_shifted_word(view.dirty_via, word, mask ^ vertical, Y);
        }
    }

}

void batch_wire_update(RouterState& state, int stamp) {
    auto& L = state.device.L;
    auto& X = state.device.X;
    auto& Y = state.device.Y;
    auto& XY = state.device.XY;
    auto& pre_demand = state.device.pre_demand;
    auto& all_track_cnt = state.device.all_track_cnt;
    compute_presum_general<<<all_track_cnt, THREAD_NUM, sizeof(int) * XY>>>(state.device, pre_demand);
    commit_all_edge<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>>(state.device, stamp);
    grotor::cuda_check(cudaMemsetAsync(pre_demand, 0, sizeof(int) * L * X * Y), "cudaMemset");
}

__global__ void commit_wire_demand(RoutingDeviceView view, int net_cnt, int net_offset, int stamp, int K = 1, int need_work_dir = -1) {
    auto& ROUTE_PER_PIN = view.ROUTE_PER_PIN;
    auto& ROUTE_PER_PIN_PHASE2 = view.ROUTE_PER_PIN_PHASE2;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& x_edge_len = view.x_edge_len;
    auto& y_edge_len = view.y_edge_len;
    auto& pin_acc_num = view.pin_acc_num;
    auto& pin_acc_num_phase2 = view.pin_acc_num_phase2;
    auto& demand = view.demand;
    auto& routes = view.routes;
    auto& timestamp = view.timestamp;
    auto& routes_phase2 = view.routes_phase2;
    auto& ripup_flag = view.ripup_flag;
    auto& net_ids = view.net_ids;
    auto& total_wirelength = view.statistics->total_wirelength;
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= net_cnt) return;
    net_id = net_ids[net_id + net_offset];
    int *net_routes;
    if(ripup_flag[net_id]==0 || need_work_dir==0)
    {
        net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    }
    else{
        // assert(0);
        net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;
    }
    // int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    double wirelength = 0;
    //commit wires
    for(int i = 1; i < net_routes[0]; i += 2) if(net_routes[i] / X / Y == net_routes[i + 1] / X / Y) {
        int l = net_routes[i] / X / Y;
        int x0 = net_routes[i] / Y % X, y0 = net_routes[i] % Y;
        int x1 = net_routes[i + 1] / Y % X, y1 = net_routes[i + 1] % Y;
        mark_wire_span(view, l, x0, x1, y0, y1);
        if(x0 < x1) for(int x = x0; x < x1; x++) atomicAdd(demand + IDX(l, x, y0), K), wirelength += x_edge_len[x], timestamp[IDX(l, x, y0)] = stamp;
        if(x1 < x0) for(int x = x1; x < x0; x++) atomicAdd(demand + IDX(l, x, y0), K), wirelength += x_edge_len[x], timestamp[IDX(l, x, y0)] = stamp;
        if(y0 < y1) for(int y = y0; y < y1; y++) atomicAdd(demand + IDX(l, x0, y), K), wirelength += y_edge_len[y], timestamp[IDX(l, x0, y)] = stamp;
        if(y1 < y0) for(int y = y1; y < y0; y++) atomicAdd(demand + IDX(l, x0, y), K), wirelength += y_edge_len[y], timestamp[IDX(l, x0, y)] = stamp;
        timestamp[IDX(l, x0, y0)] = timestamp[IDX(l, x0, y1)] = timestamp[IDX(l, x1, y0)] = timestamp[IDX(l, x1, y1)] = stamp;
    }
    atomicAdd(&total_wirelength, wirelength * K);
}

__global__ void commit_via_demand(RoutingDeviceView view, int net_cnt, int net_offset, int stamp, int K = 1, int need_work_dir = -1) {
    auto& ROUTE_PER_PIN = view.ROUTE_PER_PIN;
    auto& ROUTE_PER_PIN_PHASE2 = view.ROUTE_PER_PIN_PHASE2;
    auto& DIR = view.DIR;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& pin_acc_num = view.pin_acc_num;
    auto& pin_acc_num_phase2 = view.pin_acc_num_phase2;
    auto& demand = view.demand;
    auto& routes = view.routes;
    auto& timestamp = view.timestamp;
    auto& routes_phase2 = view.routes_phase2;
    auto& ripup_flag = view.ripup_flag;
    auto& net_ids = view.net_ids;
    auto& total_via_count = view.statistics->total_via_count;
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= net_cnt) return;
    net_id = net_ids[net_id + net_offset];
    int *net_routes;
    if(ripup_flag[net_id]==0 || need_work_dir==0)
    {
        net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    }
    else{
        net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;
    }
    // int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    int via_count = 0;
    //commit vias
    for(int i = 1; i < net_routes[0]; i += 2) if(net_routes[i] / X / Y != net_routes[i + 1] / X / Y) {
        int x = net_routes[i] / Y % X, y = net_routes[i] % Y, l0 = net_routes[i] / X / Y, l1 = net_routes[i + 1] / X / Y;
        int minl = min(l0, l1), maxl = max(l0, l1);
        via_count += maxl - minl;
        for(int l = minl; l < maxl; l++) {
            if(timestamp[IDX(l, x, y)] < stamp) {
                
                if(l & 1 ^ DIR) {
                    if(y == 0) 
                        add_demand(view, IDX(l, x, y), K);
                    else if(y == Y - 1) 
                        add_demand(view, IDX(l, x, y - 1), K);
                    else {
                        add_demand(view, IDX(l, x, y - 1), 0.5 * K);
                        add_demand(view, IDX(l, x, y), 0.5 * K);
                    }
                } else {
                    if(x == 0) {
                        add_demand(view, IDX(l, x, y), K);
                    }
                    else if(x == X - 1)
                        add_demand(view, IDX(l, x - 1, y), K);
                    else {
                        add_demand(view, IDX(l, x - 1, y), 0.5 * K);
                        add_demand(view, IDX(l, x, y), 0.5 * K);
                    }
                }
                //timestamp[IDX(l, x, y)] = stamp;
            }
        }
    }
    atomicAdd(&total_via_count, via_count * K);
}

} // namespace graph
