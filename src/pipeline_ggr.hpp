#pragma once
#include "pipeline.hpp"
#include "routing/zshape.cuh"
#include "routing/zshape_topology.hpp"
#include <numeric>

namespace grotor {
inline void route_ggr(cudb::RouterState& state) {
    state.device.ROUTE_PER_PIN = 256;
    cudb::build_cuda_database(state);
    graph::initialize_incremental_cost(state);
    auto& view = state.device;
    const int n = std::max(view.X, view.Y), layers = view.L + 1;
    const size_t grid = size_t(layers) * n * n;
    std::vector<int> order(state.nets.size());
    std::iota(order.begin(), order.end(), 0);
    printf("ROUTING_STAGE_BEGIN stage=zshape nets=%zu\n", order.size());
    grotor::batching::prepare_geometry(state, order, 8);
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < order.size(); ++i) {
        state.nets[i].construct_rsmt(state, false);
        if (state.rtcore_batching) {
            int lane = 0;
#ifdef _OPENMP
            lane = omp_get_thread_num();
#endif
            grotor::batching::emit_geometry(state, i, lane, state.nets[i]);
        }
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return state.nets[a].hpwl == state.nets[b].hpwl ? a > b : state.nets[a].hpwl > state.nets[b].hpwl;
    });
    std::vector<std::vector<int>> groups, batches;
    long long batch_start = 0, batch_end = 0;
#ifdef GROTOR_ENABLE_RTCORE
    std::unique_ptr<rtcore::BatchWorkspace> workspace;
    if (state.rtcore_batching) {
        workspace = std::make_unique<rtcore::BatchWorkspace>();
        rtcore::prepare_batches_rtcore(*workspace, state, order, 1000000);
    }
#endif
    cudb::RoutingDagTimer batch_timer;
#ifdef GROTOR_ENABLE_RTCORE
    if (state.rtcore_batching) {
        rtcore::generate_batches_rtcore(*workspace);
        cuda_check(cudaDeviceSynchronize(), "Z-shape RSMT batching");
        groups = rtcore::collect_batches(workspace->params.batch_id, order, 1000000);
    } else
#endif
        groups = cudb::generate_batches_rsmt(state, order, 1000000, 1);
    for (const auto& group : groups) {
        auto refined = zshape::generate_bbox_batches(group, state.nets, view.X, view.Y);
        for (auto& batch : refined) batches.emplace_back(std::move(batch));
    }
    std::reverse(batches.begin(), batches.end());
    const auto span = batch_timer.stop();
    batch_start = span.start_us; batch_end = span.end_us;
    printf("BATCH_TIMING backend=%s start_us=%lld end_us=%lld elapsed_ms=%.6f nets=%zu batches=%zu used_for_routing=1\n",
        state.rtcore_batching ? "rt_core" : "cpu", batch_start, batch_end,
        (batch_end-batch_start)/1000., order.size(), batches.size());
    if (state.validate_rtcore_batches) {
        const auto cpu = cudb::generate_batches_rsmt(state, order, 1000000, 1);
        if (cpu != groups) throw std::runtime_error("Z-shape RT-Core/CPU RSMT batches differ");
        printf("BATCH_VALIDATION exact_lists_equal=1 nets=%zu\n", order.size());
    }
#ifdef GROTOR_ENABLE_RTCORE
    // Free the OptiX scene before allocating routing scratch.
    workspace.reset();
#endif
    grotor::batching::release_geometry(state);

    struct Batch { size_t points, gbpoints, offsets, ids; int count, point_count, gb_count, raw_words; };
    std::vector<Batch> spans;
    std::vector<int> points, gbpoints, offsets, ids, route_offsets(order.size());
    size_t max_points = 1, max_gb = 1, max_raw = 1;
    for (const auto& batch : batches) {
        int raw_words = 0;
        const size_t offset_begin = offsets.size();
        for (int id : batch) {
            route_offsets[id] = raw_words;
            offsets.push_back(raw_words);
            raw_words += int(state.nets[id].pins.size()) * 256;
        }
        auto encoded = zshape::encode_batch(batch, state.nets, route_offsets, view.X, view.Y, view.L);
        spans.push_back({points.size(), gbpoints.size(), offset_begin, ids.size(), int(batch.size()),
            int(encoded.points.size()), int(encoded.gbpoints.size()), raw_words});
        points.insert(points.end(), encoded.points.begin(), encoded.points.end());
        gbpoints.insert(gbpoints.end(), encoded.gbpoints.begin(), encoded.gbpoints.end());
        ids.insert(ids.end(), batch.begin(), batch.end());
        max_points = std::max(max_points, encoded.points.size());
        max_gb = std::max(max_gb, encoded.gbpoints.size());
        max_raw = std::max(max_raw, size_t(raw_words));
    }
    std::vector<std::vector<int>>().swap(groups);
    std::vector<std::vector<int>>().swap(batches);
    DeviceMemory memory;
    int *map=nullptr, *prev=nullptr, *raw=nullptr, *d_points=nullptr, *d_gb=nullptr, *d_offsets=nullptr, *error=nullptr;
    size_t free_bytes, total_bytes;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "Z-shape memory budget");
    const size_t scratch_bytes = (22*grid + max_raw + max_points + max_gb + 251) * sizeof(int);
    printf("ZSHAPE_STORAGE scratch_bytes=%zu free_bytes=%zu topology_host_bytes=%zu raw_route_scope=batch\n",
        scratch_bytes, free_bytes, (points.size()+gbpoints.size()+offsets.size()+ids.size())*sizeof(int));
    if (scratch_bytes > free_bytes * .9)
        throw std::runtime_error("Insufficient device memory for Z-shape scratch (explicit allocation; no managed spill)");
    memory.allocate(&map, 6*grid*sizeof(int));
    memory.allocate(&prev, 16*grid*sizeof(int));
    memory.allocate(&raw, max_raw*sizeof(int));
    memory.allocate(&d_points, max_points*sizeof(int));
    memory.allocate(&d_gb, max_gb*sizeof(int));
    memory.allocate(&d_offsets, 251*sizeof(int));
    memory.allocate(&error, sizeof(int));
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "allocated Z-shape storage");
    printf("ZSHAPE_MEMORY used_bytes=%zu free_bytes=%zu\n", total_bytes-free_bytes, free_bytes);
    cuda_check(cudaMemset(error, 0, sizeof(int)), "reset Z-shape status");
    const auto copy = [](int* dst, const int* src, size_t count) {
        cuda_check(cudaMemcpyAsync(dst, src, count*sizeof(int), cudaMemcpyHostToDevice), "upload prepared Z-shape batch");
    };
    cudb::RoutingDagTimer dag_timer;
    for (const auto& batch : spans) {
        graph::refresh_cost(state);
        copy(d_points, points.data()+batch.points, batch.point_count);
        copy(d_gb, gbpoints.data()+batch.gbpoints, batch.gb_count);
        copy(d_offsets, offsets.data()+batch.offsets, batch.count);
        copy(view.net_ids, ids.data()+batch.ids, batch.count);
        zshape::reset_nodes<<<batch.count, 128>>>(map, d_points, layers, n, view.DIR ^ 1, batch.count);
        zshape::solve<4><<<(batch.count+3)/4, 128>>>(map, d_points, {view,n}, {view,n}, prev,
            prev+10*grid, raw, d_gb, nullptr, layers, n, view.DIR ^ 1, batch.count);
        zshape::convert_routes<<<BLOCK_NUM(batch.count), THREAD_NUM>>>(batch.count, view.net_ids,
            d_offsets, raw, n, view.L, view.X, view.Y, view.DIR, view.ROUTE_PER_PIN,
            view.pin_acc_num, view.routes, error);
        const int stamp = ++state.global_timestamp;
        graph::commit_wire_demand<<<BLOCK_NUM(batch.count), THREAD_NUM>>>(view, batch.count, 0, stamp);
        graph::commit_via_demand<<<BLOCK_NUM(batch.count), THREAD_NUM>>>(view, batch.count, 0, stamp);
    }
    dag_timer.finish("zshape", -1, order.size(), spans.size());
    int status;
    cuda_check(cudaMemcpy(&status, error, sizeof(int), cudaMemcpyDeviceToHost), "read Z-shape status");
    if (status) throw std::runtime_error("Z-shape DP failure or route storage overflow: " + std::to_string(status));
    memory.clear();
    std::vector<int>().swap(points);
    std::vector<int>().swap(gbpoints);
    std::vector<int>().swap(offsets);
    std::vector<int>().swap(ids);
    state.download_routes(false);
    // Retain deterministic layer legalization from the Z-shape adapter.
    // Work in place on the single output snapshot.
    const auto post_begin = std::chrono::steady_clock::now();
    const int promoted = zshape::promote_congested_segments(view.NET_NUM, view.X, view.Y, view.L,
        view.ROUTE_PER_PIN, state.pin_cnt_sum_cpu, state.routes_cpu.get());
    zshape::canonicalize_routes(view.NET_NUM, view.X, view.Y, view.ROUTE_PER_PIN,
        state.pin_cnt_sum_cpu, state.routes_cpu.get());
    printf("ZSHAPE_POSTPROCESS promoted_segments=%d elapsed_ms=%.6f\n", promoted,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-post_begin).count());
    for (int i = 0; i < db::nets.size(); ++i)
        if (db::nets[i].pins.size() == 1) nets2output.push(i);
    graph::finish_nets(state, order);
    auto records = graph::collect_output_nets(std::move(nets2output));
    graph::output_nets(state, std::move(records));
    printf("ROUTING_STAGE_END stage=zshape\n");
    cuda_check(cudaGetLastError(), "Z-shape routing");
    state.release_workspaces();
}
} // namespace grotor
