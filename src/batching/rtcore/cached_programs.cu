#include <optix.h>
#include <vector_functions.h>
#include "cached_launch_params.h"

using namespace grotor::rtcore::cached;

extern "C" {
__constant__ LaunchParams optixLaunchParams_color;
}

namespace {
struct ColorPayload {
    unsigned int mask[mask_words];
    int net;
};
__forceinline__ __device__ ColorPayload* payload() {
    const auto address = (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
    return reinterpret_cast<ColorPayload*>(address);
}
} // namespace

extern "C" __global__ void __raygen__collect_dependencies() {
    if (optixLaunchParams_color.chunk_nets == 0) return;
    const int ray = optixLaunchParams_color.first_ray + optixGetLaunchIndex().x;
    ColorPayload local{};
    local.net = optixLaunchParams_color.ray_to_net[ray];
    if (optixLaunchParams_color.cell_colors) {
        const float3 origin = optixLaunchParams_color.ray_origins[ray];
        const size_t cell = size_t(__float2int_rn(origin.x)) * optixLaunchParams_color.grid_y
                            + __float2int_rn(origin.y);
        for (int word = 0; word < mask_words; ++word)
            local.mask[word] = optixLaunchParams_color.cell_colors[cell * mask_words + word];
    }
    const auto pointer = reinterpret_cast<unsigned long long>(&local);
    unsigned int high = pointer >> 32, low = pointer;
    optixTrace(optixLaunchParams_color.gas_handle,
               optixLaunchParams_color.ray_origins[ray], make_float3(0, 0, 1),
               optixLaunchParams_color.ray_tmin, optixLaunchParams_color.ray_tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, high, low);
    const int net = local.net - optixLaunchParams_color.chunk_begin;
    for (int word = 0; word < mask_words; ++word)
        if (local.mask[word])
            atomicOr(optixLaunchParams_color.used_mask + net * mask_words + word, local.mask[word]);
}

extern "C" __global__ void __anyhit__dependency_bits() {
    auto* local = payload();
    const unsigned int previous = optixLaunchParams_color.tri_to_net_id[optixGetPrimitiveIndex()];
    if (previous < static_cast<unsigned int>(local->net)) {
        const int begin = optixLaunchParams_color.chunk_begin;
        if (previous < static_cast<unsigned int>(begin)) {
            // Previous chunks have completed on this stream. Their colors
            // are immutable throughout this launch and can use read-only cache.
            const int color = __ldg(optixLaunchParams_color.batch_id + previous);
            if (color >= 0 && color < mask_words * 32)
                local->mask[color / 32] |= 1u << (color % 32);
            else atomicExch(optixLaunchParams_color.color_error, 1);
        } else {
            const int row = local->net - begin, column = previous - begin;
            atomicOr(optixLaunchParams_color.dependencies + size_t(row) * optixLaunchParams_color.dependency_words + column / 32,
                     1u << (column % 32));
        }
    }
    optixIgnoreIntersection();
}

extern "C" __global__ void __miss__aabb_miss() {}
