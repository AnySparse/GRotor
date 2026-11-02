#pragma once
#include "rtcore/runtime.cuh"
#include "rtcore/color_kernels.cuh"

namespace grotor::rtcore {
struct BatchWorkspace {
    Resources resources;
    Scene scene;
    OptixShaderBindingTable sbt{};
    LaunchParams params{};
    LaunchParams* device_params = nullptr;
    int* active_count = nullptr;
};

inline void prepare_batches_rtcore(BatchWorkspace& work, const cudb::RouterState& state,
                             const std::vector<int>& order, int max_batch_size) {
    if (order.empty()) return;
    auto& resources = work.resources;
    work.scene = build_scene(resources, state, order, max_batch_size);
    auto& scene = work.scene;
    const int total_nets = scene.total_nets;
    const size_t ray_count = scene.rays;
    work.sbt = build_pipeline(resources, state.rtcore_ptx.empty() ? GROTOR_RTCORE_PTX_PATH : state.rtcore_ptx,
                                    3, "__raygen__compute_mex", "__anyhit__collect_colors");
    auto& params = work.params;
    params.gas_handle = scene.handle;
    params.ray_origins = scene.origins;
    params.ray_to_net = scene.ray_to_net;
    params.tri_to_net_id = scene.triangle_to_net;
    params.total_nets = total_nets;
    params.num_rays = static_cast<int>(ray_count);
    params.ray_tmin = scene.ray_tmin;
    params.ray_tmax = scene.ray_tmax;
    params.batch_id = resources.allocate<int>(total_nets);
    params.is_ready = resources.allocate<int>(total_nets);
    params.used_mask = resources.allocate<unsigned int>(size_t(total_nets) * mask_words);
    params.blocker_id = resources.allocate<int>(total_nets);
    params.d_unassigned_count = resources.allocate<int>(1);
    cuda_check(cudaMemset(params.batch_id, 0xff, size_t(total_nets) * sizeof(int)), "initialize RT-Core batch IDs");
    cuda_check(cudaMemset(params.blocker_id, 0xff, size_t(total_nets) * sizeof(int)), "initialize RT-Core blockers");
    params.active_ray_indices = resources.allocate<unsigned int>(ray_count);
    thrust::sequence(thrust::device, params.active_ray_indices, params.active_ray_indices + ray_count);
    work.active_count = resources.allocate<int>(1);
    work.device_params = resources.allocate<LaunchParams>(1);
    auto* device_params = work.device_params;
    cuda_check(cudaMemcpy(device_params, &params, sizeof(params), cudaMemcpyHostToDevice), "upload initial RT-Core parameters");
    cuda_check(cudaDeviceSynchronize(), "finish RT-Core preparation");
    report_memory("traversal_ready");

    cuda_check(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking), "RT-Core stream");
    // Preload launch resources with an inactive query before the measured loop.
    cudaFuncAttributes attributes{};
    cuda_check(cudaFuncGetAttributes(&attributes, reset_status_kernel), "load RT-Core reset");
    cuda_check(cudaFuncGetAttributes(&attributes, assign_batch_kernel), "load RT-Core assignment");
    cuda_check(cudaFuncGetAttributes(&attributes, fast_compact_rays_kernel), "load RT-Core compaction");
    cuda_check(cudaMemset(params.is_ready, 0, size_t(total_nets) * sizeof(int)), "inactive RT-Core warm-up");
    cuda_check(cudaDeviceSynchronize(), "complete RT-Core preparation");
    optix_check(optixLaunch(resources.pipeline, resources.stream, address(device_params), sizeof(params), &work.sbt, 1, 1, 1), "RT-Core inactive launch");
    cuda_check(cudaStreamSynchronize(resources.stream), "RT-Core inactive launch completion");
}

// Only the batching algorithm belongs here; orchestration owns its timing.
inline void generate_batches_rtcore(BatchWorkspace& work) {
    const int total_nets = work.scene.total_nets;
    if (!total_nets) return;
    auto& resources = work.resources;
    auto& params = work.params;
    auto* device_params = work.device_params;
    auto* active_count = work.active_count;
    const auto& sbt = work.sbt;
    int remaining = total_nets, iteration = 0;
    constexpr int block = 256;
    cuda_check(cudaMemcpyAsync(params.d_unassigned_count, &remaining, sizeof(int), cudaMemcpyHostToDevice, resources.stream), "RT-Core remaining count");
    while (remaining > 0) {
        params.iteration = iteration;
        cuda_check(cudaMemcpyAsync(device_params, &params, sizeof(params), cudaMemcpyHostToDevice, resources.stream), "RT-Core round parameters");
        int width = params.num_rays;
        reset_status_kernel<<<(total_nets + block - 1) / block, block, 0, resources.stream>>>(params);
        if (iteration) {
            cuda_check(cudaMemsetAsync(active_count, 0, sizeof(int), resources.stream), "RT-Core active reset");
            fast_compact_rays_kernel<<<(params.num_rays + block - 1) / block, block, 0, resources.stream>>>(params.ray_to_net, params.batch_id, params.is_ready, params.active_ray_indices, active_count, params.num_rays);
            cuda_check(cudaMemcpyAsync(&width, active_count, sizeof(int), cudaMemcpyDeviceToHost, resources.stream), "RT-Core active rays");
            cuda_check(cudaStreamSynchronize(resources.stream), "read RT-Core active rays");
        }
        if (!width) throw std::runtime_error("RT-Core coloring stalled");
        optix_check(optixLaunch(resources.pipeline, resources.stream, address(device_params), sizeof(params), &sbt, width, 1, 1), "RT-Core tracing");
        assign_batch_kernel<<<(total_nets + block - 1) / block, block, 0, resources.stream>>>(params);
        const int previous = remaining;
        cuda_check(cudaMemcpyAsync(&remaining, params.d_unassigned_count, sizeof(int), cudaMemcpyDeviceToHost, resources.stream), "RT-Core unassigned nets");
        cuda_check(cudaStreamSynchronize(resources.stream), "read RT-Core unassigned nets");
        if (remaining >= previous || ++iteration > total_nets) throw std::runtime_error("RT-Core coloring stalled or exhausted its 1024-batch mask");
    }
}
} // namespace grotor::rtcore
