#pragma once
#include "generate_batches_cpu.hpp"
#include "prepare_geometry.hpp"
#ifdef GROTOR_ENABLE_HIPRT
#include "hiprt/grid_chunk.cuh"
#endif
#ifdef GROTOR_ENABLE_RTCORE
#include "generate_batches_rtcore.cuh"
#include "generate_batches_rtcore_cached.cuh"
#include "generate_batches_rtcore_partitioned.cuh"
#endif

namespace cudb {
inline vector<vector<int>> generate_batches_cpu_timed(RouterState& state, vector<int>& order, int limit) {
    grotor::cuda_check(cudaDeviceSynchronize(), "before CPU batching");
    const auto start = std::chrono::steady_clock::now();
    auto batches = generate_batches_rsmt(state, order, limit, state.batch_sparsity_override);
    const auto end = std::chrono::steady_clock::now();
    const auto a = std::chrono::duration_cast<std::chrono::microseconds>(start-batch_clock_origin).count();
    const auto b = std::chrono::duration_cast<std::chrono::microseconds>(end-batch_clock_origin).count();
    printf("BATCH_BACKEND=cpu\n");
    printf("BATCH_TIMING backend=cpu start_us=%lld end_us=%lld elapsed_ms=%.6f nets=%zu batches=%zu used_for_routing=1\n",
           static_cast<long long>(a), static_cast<long long>(b), (b-a)/1000.0, order.size(), batches.size());
    return batches;
}
}

#ifdef GROTOR_ENABLE_RTCORE
namespace grotor::batching {
inline long long timestamp() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now()-cudb::batch_clock_origin).count();
}

template <class Workspace, class Generate>
inline rtcore::BatchResult measure_and_collect(Workspace& work, const std::vector<int>& order,
                                              int limit, Generate generate) {
    rtcore::BatchResult run;
    cuda_check(cudaDeviceSynchronize(), "before batch-generation interval");
    run.start_us = timestamp();
    generate(work);
    cuda_check(cudaDeviceSynchronize(), "complete batch-generation interval");
    run.end_us = timestamp();
    run.milliseconds = (run.end_us-run.start_us)/1000.0;
    run.batches = rtcore::collect_batches(work.params.batch_id, order, limit);
    return run;
}

inline rtcore::BatchResult generate_batches_partitioned(cudb::RouterState& state,
        const std::vector<int>& order, int limit, bool rounds, size_t scene_nets = 1000000) {
    const auto begin = timestamp();
    rtcore::BatchResult run;
    {
        rtcore::PreparedPipeline pipeline;
        const char* default_ptx = rounds ? GROTOR_RTCORE_PTX_PATH : GROTOR_RTCORE_CACHED_PTX_PATH;
        pipeline.prepare(state.rtcore_ptx.empty() ? default_ptx : state.rtcore_ptx, rounds);
        rtcore::partitioned::warm_pipeline(pipeline, rounds);
        cuda_check(cudaDeviceSynchronize(), "before partitioned batching");
        const auto start = timestamp();
        run = rtcore::partitioned::generate(state, order, limit, rounds, pipeline, scene_nets);
        cuda_check(cudaDeviceSynchronize(), "finish partitioned batching");
        const auto end = timestamp();
        run.start_us = start; run.end_us = end;
        run.milliseconds = (end-start)/1000.0;
    }
    const auto end = timestamp();
    run.preparation_ms = (run.start_us-begin)/1000.0;
    run.collection_ms = (end-run.end_us)/1000.0;
    run.call_ms = (end-begin)/1000.0;
    return run;
}

inline rtcore::BatchResult generate_batches_rtcore(cudb::RouterState& state, const std::vector<int>& order, int limit) {
    if (partitioned(state, order.size())) return generate_batches_partitioned(state, order, limit, true);
    const auto begin = timestamp();
    rtcore::BatchResult run;
    {
        rtcore::BatchWorkspace work;
        rtcore::prepare_batches_rtcore(work, state, order, limit);
        run = measure_and_collect(work, order, limit, rtcore::generate_batches_rtcore);
    }
    const auto end = timestamp();
    run.method = "rounds";
    run.timing_scope = "coloring_loop_wall";
    run.preparation_ms = (run.start_us-begin)/1000.0;
    run.collection_ms = (end-run.end_us)/1000.0;
    run.call_ms = (end-begin)/1000.0;
    return run;
}

inline rtcore::BatchResult generate_batches_rtcore_cached(cudb::RouterState& state, const std::vector<int>& order, int limit) {
    if (partitioned(state, order.size())) return generate_batches_partitioned(state, order, limit, false);
    const auto begin = timestamp();
    rtcore::BatchResult run;
    {
        rtcore::cached::CachedWorkspace work;
        rtcore::cached::prepare_batches_rtcore_cached(work, state, order, limit);
        run = measure_and_collect(work, order, limit, rtcore::cached::generate_batches_rtcore_cached);
    }
    const auto end = timestamp();
    run.method = "cached";
    run.timing_scope = "pipeline_to_coloring_wall";
    run.preparation_ms = (run.start_us-begin)/1000.0;
    run.collection_ms = (end-run.end_us)/1000.0;
    run.call_ms = (end-begin)/1000.0;
    return run;
}

inline void report_stage(const char* stage, const char* method, const cudb::RouterState& state, size_t count) {
    printf("RTCORE_STAGE stage=%s method=%s design_nets=%zu input_nets=%zu\n", stage, method, state.nets.size(), count);
}

inline std::vector<std::vector<int>> finish(cudb::RouterState& state, std::vector<int>& order,
                                           int limit, rtcore::BatchResult run) {
    state.rtcore_geometry.clear();
    // GPU workspaces are already released. Validation is outside every measured interval.
    if (state.validate_rtcore_batches) {
        const auto cpu = cudb::generate_batches_rsmt(state, order, limit, 1);
        const bool equal = cpu == run.batches;
        printf("BATCH_VALIDATION nets=%zu rtcore_batches=%zu cpu_batches=%zu exact_lists_equal=%d\n",
               order.size(), run.batches.size(), cpu.size(), int(equal));
        if (!equal) throw std::runtime_error("RT-Core batches differ from CPU");
    }
    printf("RTCORE_TIMING method=%s scope=%s elapsed_ms=%.6f\n", run.method, run.timing_scope, run.milliseconds);
    printf("BATCH_PREPARATION method=%s preparation_ms=%.6f collection_ms=%.6f call_ms=%.6f\n",
           run.method, run.preparation_ms, run.collection_ms, run.call_ms);
    printf("BATCH_BACKEND=rtcore\n");
    printf("BATCH_TIMING backend=rt_core start_us=%lld end_us=%lld elapsed_ms=%.6f nets=%zu batches=%zu used_for_routing=1\n",
           run.start_us, run.end_us, run.milliseconds, order.size(), run.batches.size());
    return std::move(run.batches);
}

} // namespace grotor::batching
#endif
