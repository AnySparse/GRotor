#include "pipeline_cugr2.hpp"
#include <iostream>

int main() {
    try {
        grotor::DeviceMemory memory;
        cudb::RoutingDeviceView view;
        view.L = 3; view.X = view.Y = 4; view.NET_NUM = 2;
        view.ROUTE_PER_PIN = 12; view.ROUTE_PER_PIN_PHASE2 = 128;
        auto upload = [&](auto*& target, const auto& values) {
            using T = typename std::decay_t<decltype(values)>::value_type;
            memory.allocate(&target, values.size() * sizeof(T));
            grotor::cuda_check(cudaMemcpy(target, values.data(), values.size() * sizeof(T),
                cudaMemcpyHostToDevice), "upload test data");
        };
        std::vector<int> first(73), second(769);
        const std::vector<int> route{7, 1, 9, 9, 25, 3, 35};
        std::copy(route.begin(), route.end(), first.begin());
        std::copy(route.begin(), route.end(), second.begin() + 384);
        upload(view.routes, first); upload(view.routes_phase2, second);
        upload(view.net_ids, std::vector<int>{0, 1});
        upload(view.pin_acc_num, std::vector<int>{0, 3, 6});
        upload(view.pin_acc_num_phase2, std::vector<int>{0, 3, 6});
        upload(view.ripup_flag, std::vector<int>{0, 1});
        upload(view.x_edge_len, std::vector<int>{1, 1, 1});
        upload(view.y_edge_len, std::vector<int>{1, 1, 1});
        upload(view.timestamp, std::vector<int>(48));
        upload(view.demand, std::vector<float>(48));
        upload(view.statistics, std::vector<cudb::RoutingStatistics>(1));
        // Serial commits provide an independent reference for overlapping nets,
        // endpoint wire/via overlap, interior half demand and boundary demand.
        for (int net = 0; net < 2; ++net) {
            graph::commit_wire_demand<<<1, 1>>>(view, 1, net, 10 + net);
            graph::commit_via_demand<<<1, 1>>>(view, 1, net, 10 + net);
        }
        grotor::maze::remove_routes<<<1, 128>>>(view, 2);
        grotor::cuda_check(cudaDeviceSynchronize(), "maze rip-up reference check");
        std::vector<float> demand(48);
        cudb::RoutingStatistics statistics;
        grotor::cuda_check(cudaMemcpy(demand.data(), view.demand, 48 * sizeof(float),
            cudaMemcpyDeviceToHost), "read remaining demand");
        grotor::cuda_check(cudaMemcpy(&statistics, view.statistics, sizeof(statistics),
            cudaMemcpyDeviceToHost), "read remaining statistics");
        for (float value : demand) if (value != 0) throw std::runtime_error("Maze rip-up demand differs from serial commits");
        if (statistics.total_wirelength != 0 || statistics.total_via_count != 0)
            throw std::runtime_error("Maze rip-up statistics differ from serial commits");
        cudb::net net;
        net.rsmt = {{1}, {0, 2}, {1}, {0, 4, 8}};
        const auto dag = grotor::maze::tree_dag(net);
        if (dag.node_index_cnt != 3 || dag.node_depth_cpu != std::vector<int>{0, 1, 2}
                || dag.par_nodes_cpu != std::vector<int>{0, 1})
            throw std::runtime_error("Maze tree DAG conversion failed");
        {
            cudb::RouterState state;
            state.device.NET_NUM = 3;
            state.nets.resize(3);
            for (auto& item : state.nets) item.pins = {0, 1};
            state.ripup_flag_cpu.assign(4, 0);
            state.pin_cnt_sum_phase2_cpu.assign(4, 0);
            state.memory.allocate(&state.device.ripup_flag, 4 * sizeof(int));
            state.memory.allocate(&state.device.pin_acc_num_phase2, 4 * sizeof(int));
            grotor::reserve_reroute_storage(state, {0, 1}, 128);
            const std::vector<int> first_route{3, 11, 12}, second_route{3, 21, 22};
            grotor::cuda_check(cudaMemcpy(state.device.routes_phase2, first_route.data(),
                3 * sizeof(int), cudaMemcpyHostToDevice), "seed first route");
            grotor::cuda_check(cudaMemcpy(state.device.routes_phase2 + 256, second_route.data(),
                3 * sizeof(int), cudaMemcpyHostToDevice), "seed second route");
            state.device.X = state.device.Y = state.device.L = 2;
            state.device.unit_length_wire_cost = state.device.unit_via_cost = 1;
            state.memory.allocate(&state.device.demand, 8 * sizeof(float));
            state.memory.allocate(&state.device.capacity, 8 * sizeof(float));
            state.memory.allocate(&state.device.unit_length_short_costs, 2 * sizeof(double));
            state.memory.allocate(&state.device.statistics, sizeof(cudb::RoutingStatistics));
            grotor::cuda_check(cudaMemset(state.device.demand, 0, 8 * sizeof(float)), "seed checkpoint demand");
            grotor::cuda_check(cudaMemset(state.device.statistics, 0, sizeof(cudb::RoutingStatistics)), "seed checkpoint statistics");
            const std::vector<float> capacity(8, 1);
            const std::vector<double> weights(2, 1);
            grotor::cuda_check(cudaMemcpy(state.device.capacity, capacity.data(), 8 * sizeof(float), cudaMemcpyHostToDevice), "seed checkpoint capacity");
            grotor::cuda_check(cudaMemcpy(state.device.unit_length_short_costs, weights.data(), 2 * sizeof(double), cudaMemcpyHostToDevice), "seed checkpoint weights");
            const double before_score = grotor::maze::routing_score(state);
            grotor::maze::Checkpoint checkpoint(state);
            grotor::reserve_reroute_storage(state, {0, 1, 2}, 216);
            state.download_routes(true);
            if (state.device.PIN_NUM_PHASE2 != 6 || state.pin_cnt_sum_phase2_cpu[2] != 4
                    || !std::equal(first_route.begin(), first_route.end(), state.routes_phase2_cpu.get())
                    || !std::equal(second_route.begin(), second_route.end(), state.routes_phase2_cpu.get() + 432))
                throw std::runtime_error("Resizing reroute storage changed existing routes");
            grotor::cuda_check(cudaMemset(state.device.routes_phase2, 0, 3 * sizeof(int)), "overwrite candidate route");
            checkpoint.restore(state);
            state.download_routes(true);
            if (state.device.PIN_NUM_PHASE2 != 4 || state.ripup_flag_cpu[2] != 0
                    || !std::equal(first_route.begin(), first_route.end(), state.routes_phase2_cpu.get())
                    || !std::equal(second_route.begin(), second_route.end(), state.routes_phase2_cpu.get() + 432)
                    || grotor::maze::routing_score(state) != before_score)
                throw std::runtime_error("Maze checkpoint did not restore routes, flags and score");
        }
        std::cout << "Maze rip-up and tree DAG checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
