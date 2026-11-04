#pragma once
#include "pipeline.hpp"
#include "routing/maze.cuh"
#include "routing/maze_quality.cuh"

namespace grotor {

inline void refresh_congestion(cudb::RouterState& state) {
    const auto view = state.device;
    graph::extract_congestionView<<<BLOCK_NUM(view.X * view.Y), THREAD_NUM>>>(view);
    graph::extract_congestionView_xsum<<<view.Y, THREAD_NUM, sizeof(float) * view.X>>>(view);
    graph::extract_congestionView_ysum<<<view.X, THREAD_NUM, sizeof(float) * view.Y>>>(view);
}

// Append phase-two slots without expanding routes for every net in the design.
inline void reserve_reroute_storage(cudb::RouterState& state, const std::vector<int>& ids,
                                    unsigned long long minimum_stride = 128) {
    auto& view = state.device;
    const auto old_pins = view.PIN_NUM_PHASE2;
    const auto old_stride = view.ROUTE_PER_PIN_PHASE2;
    for (int id : ids) if (!state.ripup_flag_cpu[id]) {
        state.pin_cnt_sum_phase2_cpu[id] = view.PIN_NUM_PHASE2;
        view.PIN_NUM_PHASE2 += state.nets[id].pins.size();
        state.ripup_flag_cpu[id] = 1;
    }
    const auto stride = std::max(old_stride, minimum_stride);
    if (!view.routes_phase2 || old_pins != view.PIN_NUM_PHASE2 || old_stride != stride) {
        // Reuse the eventual output snapshot as the migration buffer. Never
        // hold two complete phase-two device route allocations simultaneously.
        if (view.routes_phase2) {
            const int new_pins = view.PIN_NUM_PHASE2;
            view.PIN_NUM_PHASE2 = old_pins;
            state.download_routes(true);
            view.PIN_NUM_PHASE2 = new_pins;
        }
        state.memory.release(view.routes_phase2);
        view.ROUTE_PER_PIN_PHASE2 = stride;
        state.memory.allocate(&view.routes_phase2,
            (stride * view.PIN_NUM_PHASE2 + 1) * sizeof(int));
        if (old_pins && state.routes_phase2_cpu) {
            if (stride == old_stride) {
                cuda_check(cudaMemcpy(view.routes_phase2, state.routes_phase2_cpu.get(),
                    old_pins * stride * sizeof(int), cudaMemcpyHostToDevice), "restore reroute routes");
            } else {
                for (int id = 0; id < state.nets.size(); ++id) {
                    const int offset = state.pin_cnt_sum_phase2_cpu[id];
                    if (!state.ripup_flag_cpu[id] || offset >= old_pins) continue;
                    cuda_check(cudaMemcpy(view.routes_phase2 + offset * stride,
                        state.routes_phase2_cpu.get() + offset * old_stride,
                        state.nets[id].pins.size() * old_stride * sizeof(int),
                        cudaMemcpyHostToDevice), "restore resized net routes");
                }
            }
        }
        state.routes_phase2_cpu.reset();
    }
    cuda_check(cudaMemcpy(view.ripup_flag, state.ripup_flag_cpu.data(),
        (view.NET_NUM + 1) * sizeof(int), cudaMemcpyHostToDevice), "upload reroute flags");
    cuda_check(cudaMemcpy(view.pin_acc_num_phase2, state.pin_cnt_sum_phase2_cpu.data(),
        (view.NET_NUM + 1) * sizeof(int), cudaMemcpyHostToDevice), "upload reroute offsets");
    printf("REROUTE_STORAGE pins=%d stride=%llu bytes=%llu\n", view.PIN_NUM_PHASE2,
        view.ROUTE_PER_PIN_PHASE2, (view.ROUTE_PER_PIN_PHASE2 * view.PIN_NUM_PHASE2 + 1) * sizeof(int));
}

inline void route_cugr2(cudb::RouterState& state) {
    cudb::build_cuda_database(state);
    graph::initialize_incremental_cost(state);
    auto& view = state.device;
    cuda_check(cudaFuncSetAttribute(graph::compute_presum,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(double) * view.XY), "prefix shared memory");
    cuda_check(cudaFuncSetAttribute(graph::compute_presum_general,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(int) * view.XY), "demand shared memory");
    for (int i = 0; i < db::nets.size(); ++i)
        if (db::nets[i].pins.size() == 1) nets2output.push(i);
    std::vector<int> all_nets(state.nets.size());
    std::iota(all_nets.begin(), all_nets.end(), 0);

    printf("ROUTING_STAGE_BEGIN stage=lshape nets=%zu\n", all_nets.size());
    Lshape_route::Lshape_route(state, all_nets);
    printf("ROUTING_STAGE_END stage=lshape\n");
    if (state.nets.size() > 5000000) mode = 1;
    if (state.nets.size() > 20000000) mode = 2;
    refresh_congestion(state);
    auto detour = graph::ripup(state, 0).first;
    reserve_reroute_storage(state, detour);
    printf("ROUTING_STAGE_BEGIN stage=detour nets=%zu\n", detour.size());
    Lshape_route_detour::route_detour(state, detour);
    printf("ROUTING_STAGE_END stage=detour\n");

    refresh_congestion(state);
    auto maze_nets = graph::ripup(state, 0).first;
    printf("ROUTING_STAGE_BEGIN stage=maze nets=%zu\n", maze_nets.size());
    RoutingDagTimer maze_timer;
    if (!maze_nets.empty()) {
        maze::Checkpoint checkpoint(state);
        const double before_score = maze::routing_score(state);
        // Preserve the congestion context while choosing all sparse topologies.
        // Removing every candidate first would hide their competing demand.
        maze::prepare_topology(state, maze_nets);
        cuda_check(cudaMemcpy(view.net_ids, maze_nets.data(), maze_nets.size() * sizeof(int),
            cudaMemcpyHostToDevice), "upload maze rip-up nets");
        maze::remove_routes<<<BLOCK_NUM(maze_nets.size()), THREAD_NUM>>>(view, maze_nets.size());
        cuda_check(cudaDeviceSynchronize(), "remove maze routes");
        unsigned long long stride = 128;
        for (int id : maze_nets) {
            const auto words = 10ULL * state.nets[id].rsmt.back().size() + 1;
            const auto pins = state.nets[id].pins.size();
            stride = std::max(stride, (words + pins - 1) / pins);
        }
        reserve_reroute_storage(state, maze_nets, stride);
        Lshape_route_detour::route_prepared_topology(state, maze_nets, "maze", false);
        const double after_score = maze::routing_score(state);
        const bool accepted = std::isfinite(after_score) && after_score < before_score;
        if (!accepted) checkpoint.restore(state);
        printf("MAZE_QUALITY_GATE accepted=%d before_score=%.12g candidate_score=%.12g\n",
            int(accepted), before_score, after_score);
    }
    const auto maze_span = maze_timer.stop();
    printf("MAZE_PHASE_TIMING start_us=%lld end_us=%lld elapsed_ms=%.6f\n",
        maze_span.start_us, maze_span.end_us, (maze_span.end_us - maze_span.start_us) / 1000.0);
    printf("ROUTING_STAGE_END stage=maze\n");
    const auto remaining = graph::ripup(state, 0).first;
    printf("ROUTING_REMAINING_OVERFLOW nets=%zu\n", remaining.size());
    state.download_routes(false);
    state.download_routes(true);
    graph::finish_nets(state, all_nets);
    auto records = graph::collect_output_nets(std::move(nets2output));
    graph::output_nets(state, std::move(records));
    cuda_check(cudaGetLastError(), "CUGR2 routing kernels");
    state.release_workspaces();
}
} // namespace grotor
