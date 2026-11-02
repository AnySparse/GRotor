#pragma once

#include "router_state.hpp"
#include <cuda.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <fstream>
#include <cmath>
#include <limits>
#include <numeric>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

namespace grotor::rtcore {

inline void optix_check(OptixResult status, const char* operation) {
    if (status != OPTIX_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + optixGetErrorName(status));
}

inline CUdeviceptr address(const void* pointer) { return reinterpret_cast<CUdeviceptr>(pointer); }

struct Resources {
    DeviceMemory memory;
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup groups[3]{};
    OptixPipeline pipeline = nullptr;
    cudaStream_t stream = nullptr;

    ~Resources() {
        cudaDeviceSynchronize();
        if (stream) cudaStreamDestroy(stream);
        if (pipeline) optixPipelineDestroy(pipeline);
        for (auto group : groups) if (group) optixProgramGroupDestroy(group);
        if (module) optixModuleDestroy(module);
        memory.clear();
        if (context) optixDeviceContextDestroy(context);
    }

    template <class T> T* allocate(size_t count) {
        T* p = nullptr;
        memory.allocate(&p, count * sizeof(T));
        return p;
    }

    template <class T> T* upload(const std::vector<T>& values) {
        auto* p = allocate<T>(values.size());
        cuda_check(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
                   "upload RT-Core input");
        return p;
    }
};

inline void report_memory(const char* point) {
    size_t available = 0, total = 0;
    cuda_check(cudaMemGetInfo(&available, &total), "RT-Core memory usage");
    printf("RTCORE_MEMORY point=%s used_bytes=%zu free_bytes=%zu\n", point, total - available, available);
}

inline std::vector<std::vector<int>> collect_batches(const int* device_batches, const std::vector<int>& order, int max_batch_size) {
    if (order.empty()) return {};
    std::vector<int> ids(order.size());
    cuda_check(cudaMemcpy(ids.data(), device_batches, ids.size()*sizeof(int), cudaMemcpyDeviceToHost), "download RT-Core batch IDs");
    const int last = *std::max_element(ids.begin(), ids.end());
    if (last < 0 || last >= 1024) throw std::runtime_error("Invalid RT-Core color range");
    std::vector<std::vector<int>> batches(last+1);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= 1024) throw std::runtime_error("Unassigned RT-Core net");
        batches[ids[i]].push_back(order[i]);
    }
    for (const auto& batch : batches)
        if (batch.size() > size_t(max_batch_size)) throw std::runtime_error("RT-Core batch exceeds routing limit");
    return batches;
}

struct BatchResult {
    std::vector<std::vector<int>> batches;
    double milliseconds = 0, preparation_ms = 0, collection_ms = 0, call_ms = 0;
    const char* method = nullptr;
    const char* timing_scope = "batch_generation_wall";
    long long start_us = 0, end_us = 0;
};

struct Scene {
    OptixTraversableHandle handle = 0;
    float3* origins = nullptr;
    float ray_tmin = 0, ray_tmax = 1e16f;
    unsigned int* ray_to_net = nullptr;
    unsigned int* triangle_to_net = nullptr;
    std::vector<int> ray_offsets;
    int total_nets = 0;
    size_t rays = 0;
};

struct GeometryPlacement {
    unsigned int triangle_source, triangle_target, ray_source, ray_target, priority;
    float z;
};

__global__ void expand_triangles(const Triangle2D* source, size_t count, size_t source_begin,
                                 const GeometryPlacement* placements, float3* vertices,
                                 unsigned int* owners) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto triangle = source[i];
    const auto p = placements[triangle.slot];
    const size_t target = p.triangle_target + source_begin + i - p.triangle_source;
    vertices[target*3] = {triangle.a.x, triangle.a.y, p.z};
    vertices[target*3+1] = {triangle.b.x, triangle.b.y, p.z};
    vertices[target*3+2] = {triangle.c.x, triangle.c.y, p.z};
    owners[target] = p.priority;
}

__global__ void expand_rays(const Ray2D* source, size_t count, size_t source_begin,
                            const GeometryPlacement* placements, float3* origins,
                            unsigned int* owners, float offset, bool coplanar) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto ray = source[i];
    const auto p = placements[ray.slot];
    const size_t target = p.ray_target + source_begin + i - p.ray_source;
    const float z = coplanar ? -1.f : __fadd_rn(p.z, offset);
    origins[target] = {ray.origin.x, ray.origin.y, z};
    owners[target] = p.priority;
}

inline Scene build_scene(Resources& resources, const cudb::RouterState& state,
                         const std::vector<int>& order, int max_batch_size, bool need_offsets = false) {
    if (state.batch_sparsity_override != 1 || max_batch_size < 1)
        throw std::invalid_argument("RT-Core batching requires sparsity 1 and a positive batch limit");
    const auto& geometry = state.rtcore_geometry;
    if (geometry.spans.size() != order.size())
        throw std::invalid_argument("RT-Core geometry must be emitted during topology construction");
    cuda_check(cudaDeviceSynchronize(), "before RT-Core preparation");
    static const bool initialized = [] { optix_check(optixInit(), "optixInit"); return true; }();
    (void)initialized;
    optix_check(optixDeviceContextCreate(0, nullptr, &resources.context), "create OptiX context");
    const auto settings = geometry.config;
    const int total_nets = static_cast<int>(order.size());
    size_t triangles = 0, rays = 0, input_points = 0;
    std::vector<GeometryPlacement> placements(order.size());
    std::vector<int> offsets(need_offsets ? order.size()+1 : 0);
    for (int index = 0; index < total_nets; ++index) {
        const int id = order[index];
        if (id < 0 || size_t(id) >= geometry.net_to_slot.size())
            throw std::invalid_argument("Invalid prepared net ID");
        const auto slot = geometry.net_to_slot[id];
        if (slot >= geometry.spans.size()) throw std::invalid_argument("Missing prepared net");
        const auto span = geometry.spans[slot];
        if (span.lane >= geometry.arenas.size() || !span.rays)
            throw std::invalid_argument("Incomplete prepared geometry");
        if (need_offsets) offsets[index] = static_cast<int>(rays);
        placements[slot] = {span.triangle_begin, static_cast<unsigned int>(triangles), span.ray_begin,
                            static_cast<unsigned int>(rays), static_cast<unsigned int>(index),
                            settings.coplanar ? 0.f : (total_nets-1-index)*settings.z_gap};
        triangles += span.triangles;
        rays += span.rays;
    }
    for (const auto& arena : geometry.arenas) input_points += arena.input_points;
    if (triangles > 536870912 / 3 || rays > size_t(std::numeric_limits<int>::max()-255))
        throw std::runtime_error("RT-Core geometry exceeds supported vertex/ray count");
    if (need_offsets) offsets[total_nets] = static_cast<int>(rays);
    printf("RTCORE_SPACE model=%s point_model=%s z_gap=%.9g margin=%.9g\n",
           settings.coplanar ? "coplanar" : "layered", settings.compact_points ? "triangle" : "cross",
           double(settings.z_gap), double(settings.margin));
    printf("RTCORE_GEOMETRY nets=%d input_points=%zu unique_points=%zu triangles=%zu vertices=%zu fused=1\n",
           total_nets, input_points, rays, triangles, triangles*3);

    auto* device_placements = resources.upload(placements);
    std::vector<GeometryPlacement>().swap(placements);
    const size_t staging_bytes = std::min(size_t(32) << 20,
        std::max(triangles*sizeof(Triangle2D), rays*sizeof(Ray2D)));
    auto* staging = resources.allocate<unsigned char>(staging_bytes);
    auto* vertices = resources.allocate<float3>(triangles*3);
    auto* triangle_owners = resources.allocate<unsigned int>(triangles);
    const size_t triangles_per_chunk = staging_bytes / sizeof(Triangle2D);
    for (const auto& arena : geometry.arenas) {
        for (size_t begin = 0; begin < arena.triangles.size(); begin += triangles_per_chunk) {
            const size_t count = std::min(triangles_per_chunk, arena.triangles.size()-begin);
            cuda_check(cudaMemcpy(staging, arena.triangles.data()+begin, count*sizeof(Triangle2D), cudaMemcpyHostToDevice), "upload prepared triangles");
            expand_triangles<<<(count+255)/256,256>>>(reinterpret_cast<Triangle2D*>(staging), count, begin, device_placements, vertices, triangle_owners);
        }
    }
    cuda_check(cudaDeviceSynchronize(), "finish triangle expansion");
    CUdeviceptr vertex_buffer = address(vertices);
    uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput input{};
    input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    input.triangleArray.numVertices = static_cast<unsigned int>(triangles*3);
    input.triangleArray.vertexBuffers = &vertex_buffer;
    input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    input.triangleArray.flags = &flags;
    input.triangleArray.numSbtRecords = 1;
    OptixAccelBuildOptions options{};
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    optix_check(optixAccelComputeMemoryUsage(resources.context, &options, &input, 1, &sizes), "compute GAS memory requirements");
    size_t available = 0, total = 0;
    cuda_check(cudaMemGetInfo(&available, &total), "check BVH build budget");
    if (sizes.tempSizeInBytes + sizes.outputSizeInBytes + (size_t(64)<<20) > available)
        throw std::runtime_error("RT-Core BVH build exceeds free-memory budget");
    auto* scratch = resources.allocate<unsigned char>(sizes.tempSizeInBytes);
    auto* gas = resources.allocate<unsigned char>(sizes.outputSizeInBytes);
    report_memory("gas_build");
    OptixTraversableHandle handle = 0;
    optix_check(optixAccelBuild(resources.context, 0, &options, &input, 1, address(scratch), sizes.tempSizeInBytes,
                               address(gas), sizes.outputSizeInBytes, &handle, nullptr, 0), "build triangle GAS");
    cuda_check(cudaDeviceSynchronize(), "finish BVH build before releasing inputs");
    resources.memory.release(scratch);
    resources.memory.release(vertices);
    report_memory("build_inputs_released");

    Scene scene;
    scene.handle = handle;
    scene.ray_tmin = settings.coplanar ? 0.f : settings.z_gap*.1f;
    scene.ray_tmax = settings.coplanar ? 2.f : 1e16f;
    scene.origins = resources.allocate<float3>(rays);
    scene.ray_to_net = resources.allocate<unsigned int>(rays);
    const size_t rays_per_chunk = staging_bytes / sizeof(Ray2D);
    for (const auto& arena : geometry.arenas) {
        for (size_t begin = 0; begin < arena.rays.size(); begin += rays_per_chunk) {
            const size_t count = std::min(rays_per_chunk, arena.rays.size()-begin);
            cuda_check(cudaMemcpy(staging, arena.rays.data()+begin, count*sizeof(Ray2D), cudaMemcpyHostToDevice), "upload prepared rays");
            expand_rays<<<(count+255)/256,256>>>(reinterpret_cast<Ray2D*>(staging), count, begin, device_placements,
                                              scene.origins, scene.ray_to_net, settings.z_gap*.1f, settings.coplanar);
        }
    }
    cuda_check(cudaDeviceSynchronize(), "finish ray expansion");
    resources.memory.release(staging);
    resources.memory.release(device_placements);
    scene.triangle_to_net = triangle_owners;
    scene.ray_offsets = std::move(offsets);
    scene.total_nets = total_nets;
    scene.rays = rays;
    return scene;
}

inline OptixShaderBindingTable build_pipeline(Resources& resources, const std::string& ptx_path,
                                             unsigned int payload_values, const char* raygen_name, const char* hit_name) {
    std::ifstream file(ptx_path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open RT-Core PTX: " + ptx_path);
    const std::string ptx((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    OptixModuleCompileOptions module_options{};
    OptixPipelineCompileOptions pipeline_options{};
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = payload_values;
    pipeline_options.pipelineLaunchParamsVariableName = "optixLaunchParams_color";
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    char log[2048]{};
    size_t log_size = sizeof(log);
    optix_check(optixModuleCreate(resources.context, &module_options, &pipeline_options,
                ptx.data(), ptx.size(), log, &log_size, &resources.module), "compile OptiX module");
    OptixProgramGroupDesc descriptions[3]{};
    descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descriptions[0].raygen.module = resources.module;
    descriptions[0].raygen.entryFunctionName = raygen_name;
    descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptions[1].miss.module = resources.module;
    descriptions[1].miss.entryFunctionName = "__miss__aabb_miss";
    descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptions[2].hitgroup.moduleAH = resources.module;
    descriptions[2].hitgroup.entryFunctionNameAH = hit_name;
    OptixProgramGroupOptions group_options{};
    log_size = sizeof(log);
    optix_check(optixProgramGroupCreate(resources.context, descriptions, 3, &group_options,
                log, &log_size, resources.groups), "create OptiX program groups");
    OptixPipelineLinkOptions link_options{};
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    optix_check(optixPipelineCreate(resources.context, &pipeline_options, &link_options,
                resources.groups, 3, log, &log_size, &resources.pipeline), "create OptiX pipeline");
    OptixStackSizes stack{};
    for (auto group : resources.groups)
        optix_check(optixUtilAccumulateStackSizes(group, &stack, resources.pipeline), "accumulate OptiX stack sizes");
    unsigned int direct_traversal = 0, direct_state = 0, continuation = 0;
    optix_check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &direct_traversal, &direct_state, &continuation),
                "compute OptiX stack sizes");
    optix_check(optixPipelineSetStackSize(resources.pipeline, direct_traversal, direct_state, continuation, 1),
                "set OptiX stack size");
    struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
    std::vector<SbtRecord> records(3);
    for (int i = 0; i < 3; ++i)
        optix_check(optixSbtRecordPackHeader(resources.groups[i], &records[i]), "pack shader binding table");
    auto* device_records = resources.upload(records);
    OptixShaderBindingTable sbt{};
    sbt.raygenRecord = address(device_records);
    sbt.missRecordBase = address(device_records + 1);
    sbt.missRecordStrideInBytes = sizeof(SbtRecord); sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = address(device_records + 2);
    sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord); sbt.hitgroupRecordCount = 1;

    return sbt;
}

} // namespace grotor::rtcore
