#pragma once

#include "preprocess.hpp"
#include "routing/lshape.cuh"
#include "routing/flt.cuh"
#include "routing/congestion.cuh"
#include "output.hpp"
#include <future>

namespace grotor {

// Route with the selected batching backend and persistent cost/prefix state.
inline void route(cudb::RouterState& state) {
    using Clock = std::chrono::steady_clock;
    auto milliseconds = [](auto start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    };
    double output_ms = 0, output_wait_ms = 0;
    auto start_output = [&] {
        // A task owns its queue. No producer touches it while the writer runs.
        auto ready = std::move(nets2output);
        nets2output = {};
        auto records = graph::collect_output_nets(std::move(ready));
        return std::async(std::launch::async, [&state, records = std::move(records)]() mutable {
            const auto start = Clock::now();
            graph::output_nets(state, std::move(records));
            return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        });
    };
    auto wait_output = [&](auto& task) {
        const auto start = Clock::now();
        output_ms += task.get();
        output_wait_ms += milliseconds(start);
    };

    auto start = Clock::now();
    cudb::build_cuda_database(state);
    graph::initialize_incremental_cost(state);
    const double setup_ms = milliseconds(start);
    auto& view = state.device;
    for (int i = 0; i < db::nets.size(); ++i)
        if (db::nets[i].pins.size() == 1) nets2output.push(i);
    auto output_task = start_output();

    cuda_check(cudaFuncSetAttribute(graph::compute_presum,
               cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(double) * view.XY), "prefix shared memory");
    cuda_check(cudaFuncSetAttribute(graph::compute_presum_general,
               cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(int) * view.XY), "demand shared memory");
    std::vector<int> all_nets(state.nets.size());
    for (int i = 0; i < all_nets.size(); ++i) all_nets[i] = i;
    Lshape_route::Lshape_route(state, all_nets);
    if (LOG) graph::report_score(state);
    wait_output(output_task);
    start = Clock::now();
    state.download_routes(false);
    double download_ms = milliseconds(start);

    start = Clock::now();
    if (state.nets.size() > 5000000) mode = 1;
    if (state.nets.size() > 20000000) mode = 2;
    graph::extract_congestionView<<<BLOCK_NUM(view.X * view.Y), THREAD_NUM>>>(view);
    graph::extract_congestionView_xsum<<<view.Y, THREAD_NUM, sizeof(float) * view.X>>>(view);
    graph::extract_congestionView_ysum<<<view.X, THREAD_NUM, sizeof(float) * view.Y>>>(view);
    auto overflow_nets = graph::ripup(state, 0);
    graph::finish_nets(state, overflow_nets.second);

    int pin_count = 0;
    for (auto net_id : overflow_nets.first) {
        state.pin_cnt_sum_phase2_cpu[net_id] = pin_count;
        pin_count += state.nets[net_id].pins.size();
        state.ripup_flag_cpu[net_id] = 1;
    }
    view.PIN_NUM_PHASE2 = pin_count;
    state.memory.allocate(&view.routes_phase2,
                          (view.ROUTE_PER_PIN_PHASE2 * pin_count + 1) * sizeof(int));
    cuda_check(cudaMemcpy(view.ripup_flag, state.ripup_flag_cpu.data(),
                         (1 + view.NET_NUM) * sizeof(int), cudaMemcpyHostToDevice), "upload rip-up flags");
    cuda_check(cudaMemcpy(view.pin_acc_num_phase2, state.pin_cnt_sum_phase2_cpu.data(),
                         (1 + view.NET_NUM) * sizeof(int), cudaMemcpyHostToDevice), "upload FLT pin offsets");
    const double classification_ms = milliseconds(start);

    // Ready nets have no overflow subnets. Their flags and phase-one snapshot
    // remain immutable while FLT updates device routes for the other nets.
    output_task = start_output();
    auto topology_cleanup = std::async(std::launch::async,
        [&state, finished = std::move(overflow_nets.second)] {
            const auto begin = Clock::now();
            // FLT accesses only overflow nets. Output uses snapshots and the
            // input database, so completed solver objects can be reclaimed.
            for (const int id : finished) state.nets[id] = cudb::net{};
            vector<vector<cudb::net>>().swap(state.tmp_nets);
            return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
        });
    Lshape_route_detour::Lshape_route_detour_wrap(state, overflow_nets.first);
    if (LOG) graph::report_score(state);
    start = Clock::now();
    state.download_routes(true);
    download_ms += milliseconds(start);
    wait_output(output_task);
    graph::finish_nets(state, overflow_nets.first);
    cuda_check(cudaDeviceSynchronize(), "finish routing");
    cuda_check(cudaGetLastError(), "routing kernels");

    output_task = start_output();
    start = Clock::now();
    const double topology_cleanup_ms = topology_cleanup.get();
    state.release_workspaces();
    const double cleanup_ms = milliseconds(start);
    wait_output(output_task);
    if (state.profile_stages)
        printf("PIPELINE_TIMING setup_ms=%.3f lshape_ms=%.3f flt_ms=%.3f download_ms=%.3f classification_ms=%.3f output_ms=%.3f output_wait_ms=%.3f cleanup_ms=%.3f topology_cleanup_ms=%.3f\n",
               setup_ms, Lshape_time * 1000, DAG_time * 1000, download_ms,
               classification_ms, output_ms, output_wait_ms, cleanup_ms, topology_cleanup_ms);
}

} // namespace grotor
