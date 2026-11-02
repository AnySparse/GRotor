#pragma once

#include "router_state.hpp"
#include "output_writer.hpp"
#include "cost/full_update.cuh"

namespace graph {

struct OutputNet {
    std::string name;
    std::vector<int> pins, extra_routes, subnets;
};

// Publication transfers completed input records to the output task. Routing
// no longer accesses those records, and their storage can be reclaimed there.
std::vector<OutputNet> collect_output_nets(std::queue<int> ready) {
    std::vector<OutputNet> records;
    records.reserve(ready.size());
    while (!ready.empty()) {
        auto& net = db::nets[ready.front()];
        ready.pop();
        records.push_back({std::move(net.name), std::move(net.pins),
                           std::move(net.extra_routes), std::move(net.subnets)});
    }
    return records;
}

__global__ void add_all_overflow_cost(RoutingDeviceView view) {
    auto& DIR = view.DIR;
    auto& L = view.L;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& unit_length_short_costs = view.unit_length_short_costs;
    auto& capacity = view.capacity;
    auto& demand = view.demand;
    auto& layer_overflow_cost = view.statistics->layer_overflow_cost;
    auto& total_overflow_cost = view.statistics->total_overflow_cost;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= L * X * Y) return;
    int l = idx / X / Y, x = idx / Y % X, y = idx % Y;
    if(((l & 1 ^ DIR) == 0 && x + 1 < X) || ((l & 1 ^ DIR) == 1 && y + 1 < Y)) {
        double c = unit_length_short_costs[l] * of_cost(view, capacity[idx], demand[idx]);
        atomicAdd(&total_overflow_cost, c);
        atomicAdd(&layer_overflow_cost[l], c);
    }
}

void report_score(RouterState& state) {
    auto& L = state.device.L;
    auto& X = state.device.X;
    auto& Y = state.device.Y;
    auto& layer_overflow_cost = state.statistics.layer_overflow_cost;
    auto& total_overflow_cost = state.statistics.total_overflow_cost;
    auto& total_via_count = state.statistics.total_via_count;
    auto& total_wirelength = state.statistics.total_wirelength;
    printf("\n----------------------------------------------------------------------\n");
    grotor::cuda_check(cudaMemcpy(&state.statistics, state.device.statistics, sizeof(RoutingStatistics), cudaMemcpyDeviceToHost), "read statistics");
    total_overflow_cost = 0;
    for(int i = 0; i < L; i++) layer_overflow_cost[i] = 0;
    grotor::cuda_check(cudaMemcpy(state.device.statistics, &state.statistics, sizeof(RoutingStatistics), cudaMemcpyHostToDevice), "reset overflow statistics");
    add_all_overflow_cost<<<BLOCK_NUM(L * X * Y), THREAD_NUM>>>(state.device);
    grotor::cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    grotor::cuda_check(cudaMemcpy(&state.statistics, state.device.statistics, sizeof(RoutingStatistics), cudaMemcpyDeviceToHost), "read statistics");
    printf("             WL            Via                  OF                                             Total\n%15.0f%15.0f%20.0f%50.0f\n", 
        total_wirelength * db::unit_length_wire_cost, total_via_count * db::unit_via_cost, total_overflow_cost, 
        total_wirelength * db::unit_length_wire_cost + total_via_count * db::unit_via_cost + total_overflow_cost);
    printf("----------------------------------------------------------------------\n\n");
}

void finish_nets(RouterState& state, vector<int> &finished_nets) {
    auto& nets = state.nets;
    for(auto _net_id : finished_nets) {
        int _ = nets[_net_id].original_net_id;
        if(--db::nets[_].unfinished_subnet_count == 0) nets2output.push(_);

    }
}

void output_nets(const RouterState& state, std::vector<OutputNet> ready) {
    const auto ROUTE_PER_PIN = state.device.ROUTE_PER_PIN;
    const auto ROUTE_PER_PIN_PHASE2 = state.device.ROUTE_PER_PIN_PHASE2;
    const auto X = state.device.X, Y = state.device.Y;
    const auto& pin_cnt_sum_cpu = state.pin_cnt_sum_cpu;
    const auto& ripup_flag_cpu = state.ripup_flag_cpu;
    const auto& pin_cnt_sum_phase2_cpu = state.pin_cnt_sum_phase2_cpu;
    grotor::RouteWriter writer(out_file);
    auto write_route = [&] (int pos0, int pos1) {
        if(pos0 > pos1) swap(pos0, pos1);
        writer.segment(pos0 / Y % X, pos0 % Y, pos0 / X / Y, pos1 / Y % X, pos1 % Y, pos1 / X / Y);
    };

    for (const auto& net : ready) {
        writer.text(net.name);
        writer.text("\n(\n");
        for(auto pin : net.pins) if(pin < X * Y) write_route(pin, pin + X * Y);
        for(int i = 0; i < net.extra_routes.size(); i += 2)
            write_route(net.extra_routes[i], net.extra_routes[i + 1]);
        for(auto net_id : net.subnets) {
            const int *net_routes;
            if(ripup_flag_cpu[net_id]==0)
            {
                net_routes = state.routes_cpu.get() + pin_cnt_sum_cpu[net_id] * ROUTE_PER_PIN;
            }
            else{
                net_routes = state.routes_phase2_cpu.get() + pin_cnt_sum_phase2_cpu[net_id] * ROUTE_PER_PIN_PHASE2;
            }
            for(int j = 1; j < net_routes[0]; j += 2) write_route(net_routes[j] + X * Y, net_routes[j + 1] + X * Y);
        }
        writer.text(")\n");
    }
    writer.flush();
}

} // namespace graph
