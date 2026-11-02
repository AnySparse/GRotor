#pragma once

#include "router_state.hpp"
#include "cost/prefix_scan.cuh"

namespace graph {

__global__ void mark_overflow_edges(RoutingDeviceView view, int threshold) {
    auto& DIR = view.DIR;
    auto& L = view.L;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& capacity = view.capacity;
    auto& demand = view.demand;
    auto& of_edge_sum = view.of_edge_sum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if(((l & 1 ^ DIR) == 0 && x + 1 < X) || ((l & 1 ^ DIR) == 1 && y + 1 < Y))
        of_edge_sum[idx] = (capacity[idx] + threshold <= demand[idx] ? 1 : 0);
}

__global__ void mark_overflow_nets(RoutingDeviceView view) {
    auto& ROUTE_PER_PIN = view.ROUTE_PER_PIN;
    auto& ROUTE_PER_PIN_PHASE2 = view.ROUTE_PER_PIN_PHASE2;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& NET_NUM = view.NET_NUM;
    auto& pin_acc_num = view.pin_acc_num;
    auto& pin_acc_num_phase2 = view.pin_acc_num_phase2;
    auto& is_of_net = view.is_of_net;
    auto& of_edge_sum = view.of_edge_sum;
    auto& routes = view.routes;
    auto& routes_phase2 = view.routes_phase2;
    auto& ripup_flag = view.ripup_flag;
    int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(net_id >= NET_NUM) return;
    is_of_net[net_id] = false;
    int *net_routes;
    if(ripup_flag[net_id]==0)
    {
        net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    }
    else{
        assert(0);
        net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;
    }
    // int *net_routes = routes + pin_acc_num[net_id] * ROUTE_PER_PIN;
    int of_count = 0;
    for(int i = 1; i < net_routes[0]; i += 2) {
        int l = net_routes[i] / X / Y, x0 = net_routes[i] / Y % X, y0 = net_routes[i] % Y;
        int x1 = net_routes[i + 1] / Y % X, y1 = net_routes[i + 1] % Y;
        if(x0 != x1) of_count += of_edge_sum[IDX(l, max(x0, x1), y0)] - of_edge_sum[IDX(l, min(x0, x1), y0)];
        if(y0 != y1) of_count += of_edge_sum[IDX(l, x0, max(y0, y1))] - of_edge_sum[IDX(l, x0, min(y0, y1))];
        if(of_count > 10) {
            is_of_net[net_id] = true;
            break;
        }
    }
}

__global__ void extract_congestionView(RoutingDeviceView view) {
    auto& L = view.L;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& capacity = view.capacity;
    auto& demand = view.demand;
    auto& congestion = view.congestion;
    auto& congestion_xsum = view.congestion_xsum;
    auto& congestion_ysum = view.congestion_ysum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= X * Y) return;
    double worst = 0.0;
    for (int layer = 0; layer < L; ++layer) {
        int edge = layer * X * Y + idx;
        worst = max(worst, demand[edge] - capacity[edge]);
    }
    congestion[idx] = worst > 0.0;
    congestion_xsum[idx] = static_cast<float>(worst);
    congestion_ysum[idx] = static_cast<float>(worst);
}

__global__ void extract_congestionView_ysum(RoutingDeviceView view) {
    auto& Y = view.Y;
    auto& congestion_ysum = view.congestion_ysum;
    extern __shared__ float sum3[];
    if(threadIdx.x == 0) sum3[0] = 0;
    int x = blockIdx.x; // % XY;
    for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum3[y + 1] = congestion_ysum[x*Y+y];
    __syncthreads();
    for(int d = 0; (1 << d) < Y; d++) {
        for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
            if(idx >> d & 1) sum3[idx] += sum3[(idx >> d << d) - 1];
        __syncthreads();
    }
    for(int y = threadIdx.x; y < Y; y += blockDim.x) congestion_ysum[x*Y+y] = sum3[y];
}

__global__ void extract_congestionView_xsum(RoutingDeviceView view) {
    auto& X = view.X;
    auto& Y = view.Y;
    auto& congestion_xsum = view.congestion_xsum;
    extern __shared__ float sum3[];
    if(threadIdx.x == 0) sum3[0] = 0;
    int y = blockIdx.x;// % XY;
    for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum3[x + 1] = congestion_xsum[x*Y+y];
    __syncthreads();
    for(int d = 0; (1 << d) < X; d++) {
        for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
            if(idx >> d & 1) sum3[idx] += sum3[(idx >> d << d) - 1];
        __syncthreads();
    }
    for(int x = threadIdx.x; x < X; x += blockDim.x) congestion_xsum[x*Y+y] = sum3[x];
}

pair<vector<int>, vector<int>> ripup(RouterState& state, int of_threshold) {
    auto& L = state.device.L;
    auto& X = state.device.X;
    auto& Y = state.device.Y;
    auto& XY = state.device.XY;
    auto& NET_NUM = state.device.NET_NUM;
    auto& is_of_net = state.device.is_of_net;
    auto& of_edge_sum = state.device.of_edge_sum;
    auto& all_track_cnt = state.device.all_track_cnt;
    assert(NET_NUM <= db::MAX_NET_NUM);
    bool *is_of_net_cpu = state.host.allocate<bool>(NET_NUM); 
    
    grotor::cuda_check(cudaMemset(of_edge_sum, 0, sizeof(int) * L * X * Y), "cudaMemset");
    mark_overflow_edges<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>>(state.device, of_threshold);
    compute_presum_general<<<all_track_cnt, THREAD_NUM, sizeof(int) * XY>>>(state.device, of_edge_sum);
    mark_overflow_nets<<<BLOCK_NUM(NET_NUM), THREAD_NUM>>>(state.device);
    grotor::cuda_check(cudaMemcpy(is_of_net_cpu, is_of_net, sizeof(bool) * NET_NUM, cudaMemcpyDeviceToHost), "cudaMemcpy");
    vector<int> of_nets, no_of_nets;
    for(int i = 0; i < NET_NUM; i++) 
        (is_of_net_cpu[i] ? of_nets : no_of_nets).emplace_back(i);
    state.host.release(is_of_net_cpu);
    return make_pair(move(of_nets), move(no_of_nets));
}

} // namespace graph
