#pragma once
#include "rtcore/runtime.cuh"
#include "rtcore/cached_kernels.cuh"

namespace grotor::rtcore::cached {
struct CachedWorkspace {
    Resources resources;
    Scene scene;
    OptixShaderBindingTable sbt{};
    LaunchParams params{};
    LaunchParams* device_params = nullptr;
    int multiprocessors = 0, max_chunk_nets = 0, max_dependency_words = 0;
    std::string ptx;
    OptixPipeline borrowed_pipeline = nullptr;
};

inline void prepare_batches_rtcore_cached(CachedWorkspace& work, const cudb::RouterState& state,
                           const std::vector<int>& order, int max_batch_size, const PreparedPipeline* pipeline = nullptr) {
    if (order.empty()) return;
    if (pipeline) { pipeline->borrow(work.resources, work.sbt); work.borrowed_pipeline = pipeline->resources.pipeline; }
    work.scene = build_scene(work.resources, state, order, max_batch_size, true);
    work.max_chunk_nets = std::min(work.scene.total_nets, chunk_capacity);
    work.max_dependency_words = (work.max_chunk_nets+31)/32;
    work.ptx = state.rtcore_ptx.empty() ? GROTOR_RTCORE_CACHED_PTX_PATH : state.rtcore_ptx;
    int device = 0;
    cuda_check(cudaGetDevice(&device), "get batching device");
    cuda_check(cudaDeviceGetAttribute(&work.multiprocessors, cudaDevAttrMultiProcessorCount, device), "get batching multiprocessors");
    cuda_check(cudaDeviceSynchronize(), "finish cached scene preparation");
}

// Preserve the defined cached step: pipeline/graph initialization and all chunks.
// The caller measures initialization plus all chunks as one interval.
inline void initialize_coloring(CachedWorkspace& work) {
    const int total_nets = work.scene.total_nets;
    if (!total_nets) return;
    auto& resources = work.resources;
    const auto& scene = work.scene;
    const int max_chunk_nets = work.max_chunk_nets;
    const int max_dependency_words = work.max_dependency_words;
    if (!work.borrowed_pipeline) work.sbt = build_pipeline(resources, work.ptx,
                                    2, "__raygen__collect_dependencies", "__anyhit__dependency_bits");
    auto& params = work.params;
    params.gas_handle = scene.handle;
    params.ray_origins = scene.origins;
    params.ray_to_net = scene.ray_to_net;
    params.tri_to_net_id = scene.triangle_to_net;
    params.total_nets = total_nets;
    params.num_rays = static_cast<int>(scene.rays);
    params.ray_tmin = scene.ray_tmin;
    params.ray_tmax = scene.ray_tmax;
    params.batch_id = resources.allocate<int>(total_nets);
    params.dependencies = resources.allocate<unsigned int>(size_t(max_chunk_nets) * max_dependency_words);
    params.used_mask = resources.allocate<unsigned int>(size_t(max_chunk_nets) * mask_words);
    params.next_net = resources.allocate<int>(1);
    params.color_error = resources.allocate<int>(1);
    cuda_check(cudaMemset(params.batch_id, 0xff, size_t(total_nets) * sizeof(int)), "initialize RT-Core batch IDs");
    work.device_params = resources.allocate<LaunchParams>(1);
    auto* device_params = work.device_params;
    cuda_check(cudaMemcpy(device_params, &params, sizeof(params), cudaMemcpyHostToDevice), "upload initial RT-Core parameters");
    cuda_check(cudaDeviceSynchronize(), "finish RT-Core preparation");
    cuda_check(cudaMemsetAsync(params.color_error, 0, sizeof(int)), "initialize coloring status");
}

inline void color_chunk(CachedWorkspace& work, int begin, int count) {
    auto& params = work.params;
    params.chunk_begin = begin;
    params.chunk_nets = count;
    params.dependency_words = (count + 31) / 32;
    params.first_ray = work.scene.ray_offsets[begin];
    const int rays = work.scene.ray_offsets[begin + count] - params.first_ray;
    cuda_check(cudaMemcpy(work.device_params, &params, sizeof(params), cudaMemcpyHostToDevice), "upload chunk parameters");
    cuda_check(cudaMemsetAsync(params.dependencies, 0, size_t(count) * params.dependency_words * sizeof(unsigned int)), "initialize chunk dependencies");
    cuda_check(cudaMemsetAsync(params.used_mask, 0, size_t(count) * mask_words * sizeof(unsigned int)), "initialize preceding colors");
    cuda_check(cudaMemsetAsync(params.next_net, 0, sizeof(int)), "initialize coloring work queue");
    optix_check(optixLaunch(work.borrowed_pipeline ? work.borrowed_pipeline : work.resources.pipeline, 0, address(work.device_params), sizeof(params),
                &work.sbt, rays, 1, 1), "collect RT-Core chunk dependencies");
    const int blocks = std::min(work.multiprocessors * 2, (count + 7) / 8);
    color_dependency_graph<<<blocks, 256>>>(params);
}

inline void generate_batches_rtcore_cached(CachedWorkspace& work) {
    if (!work.scene.total_nets) return;
    initialize_coloring(work);
    for (int begin = 0; begin < work.scene.total_nets; begin += work.max_chunk_nets)
        color_chunk(work, begin, std::min(work.max_chunk_nets, work.scene.total_nets - begin));
    auto& params = work.params;
    int error = 0;
    cuda_check(cudaMemcpy(&error, params.color_error, sizeof(int), cudaMemcpyDeviceToHost), "check dependency coloring");
    if (error) throw std::runtime_error("RT-Core cached coloring failed or exhausted its 1024-batch mask");
}
} // namespace grotor::rtcore::cached
