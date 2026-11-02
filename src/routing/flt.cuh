#pragma once
#include "routing/flt_block.cuh"
#ifdef GROTOR_SERIAL_DAG
#include "routing/flt_depth.cuh"
#endif

namespace Lshape_route_detour {
void process_net(RouterState& state, int thread_idx, vector<int> &nets2route, int thread_num, std::atomic<int>& currentNetId) {
    auto& congestionView_cpu = state.packing.congestionView_cpu;
    auto& congestionView_xsum_cpu = state.packing.congestionView_xsum_cpu;
    auto& congestionView_ysum_cpu = state.packing.congestionView_ysum_cpu;
    auto& nets = state.nets;
    int is_show = false;
    grotor::flt::Builder builder;
    while (true) {
        int netId = currentNetId.fetch_add(1);
        if (netId >= nets2route.size()) {
            break;
        }
        nets[nets2route[netId]].generate_detours_reconstruct(state, builder, congestionView_cpu, congestionView_xsum_cpu, congestionView_ysum_cpu, false, is_show);
        if (state.rtcore_batching)
            state.rtcore_geometry.emit(netId, thread_idx, nets[nets2route[netId]], state.device.Y, true);
        state.flt_topology.emit(netId, thread_idx, builder);
    }
}

void multithreaded_processing(RouterState& state, vector<int> &nets2route) {

    std::vector<std::thread> threads;
    int max_threads = 8;
    threads.reserve(max_threads);
    std::atomic<int> currentNetId(0);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    for (int i = 0; i < max_threads; ++i) {
        threads.emplace_back([&, i] {
            try { process_net(state, i, nets2route, max_threads, currentNetId); }
            catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                currentNetId.store(static_cast<int>(nets2route.size()));
            }
        });
    }
    for (auto& t : threads) t.join();
    if (failure) std::rethrow_exception(failure);
}

void Lshape_route_detour_wrap(RouterState& state, vector<int>& nets2route) {
    const double phase_start = elapsed_time();
    if (nets2route.empty()) return;
    auto& nets = state.nets;
    auto& view = state.device;
    sort(nets2route.begin(), nets2route.end(), [&](int l, int r) { return nets[l].hpwl > nets[r].hpwl; });
    auto& packing = state.packing;
    const size_t cells = size_t(view.X) * view.Y;
    packing.congestionView_cpu = state.host.allocate<bool>(cells);
    packing.congestionView_xsum_cpu = state.host.allocate<float>(cells);
    packing.congestionView_ysum_cpu = state.host.allocate<float>(cells);
    grotor::cuda_check(cudaMemcpy(packing.congestionView_cpu, view.congestion, cells * sizeof(bool), cudaMemcpyDeviceToHost), "download congestion");
    grotor::cuda_check(cudaMemcpy(packing.congestionView_xsum_cpu, view.congestion_xsum, cells * sizeof(float), cudaMemcpyDeviceToHost), "download x congestion");
    grotor::cuda_check(cudaMemcpy(packing.congestionView_ysum_cpu, view.congestion_ysum, cells * sizeof(float), cudaMemcpyDeviceToHost), "download y congestion");
    const auto topology_start = std::chrono::steady_clock::now();
    for (int id : nets2route) if (nets[id].rsmt.empty()) nets[id].construct_rsmt(state);
    state.flt_topology.begin(nets2route, nets.size(), 8);
    if (state.rtcore_batching)
        state.rtcore_geometry.begin(nets2route, nets.size(), 8,
            {state.rtcore_z_gap, state.rtcore_margin, state.rtcore_coplanar, state.rtcore_compact_points});
    multithreaded_processing(state, nets2route);
    state.host.release(packing.congestionView_cpu);
    state.host.release(packing.congestionView_xsum_cpu);
    state.host.release(packing.congestionView_ysum_cpu);
    printf("TOPOLOGY_TIMING stage=flt elapsed_ms=%.6f geometry_fused=%d\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - topology_start).count(), int(state.rtcore_batching));
    vector<vector<int>> batches;
    if (state.rtcore_batching) {
#ifdef GROTOR_ENABLE_RTCORE
        grotor::batching::report_stage("flt", "cached", state, nets2route.size());
        auto batch_run = grotor::batching::generate_batches_rtcore_cached(state, nets2route, 300000);
        batches = grotor::batching::finish(state, nets2route, 300000, std::move(batch_run));
#else
        throw std::runtime_error("RT-Core build required");
#endif
    } else batches = generate_batches_cpu_timed(state, nets2route, 300000);

    const auto prepare_start = std::chrono::steady_clock::now();
    {
        grotor::flt::Plans plans(state.flt_topology, batches);
        std::vector<int>().swap(state.flt_topology.net_to_slot);
        // Batching and its optional validation have consumed all geometric input.
        for (int id : nets2route) {
            std::vector<std::vector<int>>().swap(nets[id].rsmt);
            std::vector<std::pair<int, int>>().swap(nets[id].rsmt_h_segments);
            std::vector<std::pair<int, int>>().swap(nets[id].rsmt_v_segments);
            std::vector<int>().swap(nets[id].points);
        }
#ifdef GROTOR_SERIAL_DAG
        const auto global_depths = prepare_global_depths(state.flt_topology, plans);
#endif
        grotor::flt::Workspace workspace(state.flt, plans);
        printf("FLT_PREPARATION elapsed_ms=%.6f batches=%zu\n",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prepare_start).count(), plans.batches.size());
        RoutingDagTimer timer;
        workspace.upload(state.flt_topology, plans);
        if (!workspace.streamed) {
            grotor::cuda_check(cudaEventSynchronize(workspace.ready.value), "release uploaded FLT host topology");
            state.flt_topology.clear();
        }
        for (const auto& batch : plans.batches) {
            workspace.assemble(state.flt_topology, plans, batch, view);
            graph::commit_wire_demand<<<BLOCK_NUM(batch.nets), THREAD_NUM>>>(view, batch.nets, 0, ++state.global_timestamp, -1, 0);
            graph::commit_via_demand<<<BLOCK_NUM(batch.nets), THREAD_NUM>>>(view, batch.nets, 0, state.global_timestamp, -1, 0);
            ++state.global_timestamp;
            graph::refresh_cost(state);
            precompute_cost_edge<<<BLOCK_NUM(batch.edges * 9), THREAD_NUM>>>(view, state.flt, batch.edges);
#ifdef GROTOR_SERIAL_DAG
            const size_t batch_index = &batch - plans.batches.data();
            route_flt_global_depths(view, state.flt, global_depths[batch_index], state.global_timestamp);
#else
            route_flt_block<<<batch.nets, 128>>>(view, state.flt, workspace.descriptors,
                batch.descriptor_begin, workspace.depth_offsets, state.global_timestamp);
#endif
            graph::batch_wire_update(state, state.global_timestamp);
            graph::commit_via_demand<<<BLOCK_NUM(batch.nets), THREAD_NUM>>>(view, batch.nets, 0, state.global_timestamp);
            grotor::cuda_check(cudaGetLastError(), "FLT batch launch");
        }
        const auto interval = timer.stop();
        printf("DAG_TIMING stage=detour batch_index=-1 start_us=%lld end_us=%lld elapsed_ms=%.6f nets=%zu batches=%zu\n",
            interval.start_us, interval.end_us, (interval.end_us - interval.start_us) / 1000.0, nets2route.size(), plans.batches.size());
        state.flt_topology.clear();
    } // Release all execution storage before reporting the complete FLT phase.
    DAG_time = elapsed_time() - phase_start;
}

} // namespace Lshape_route_detour
