#include <optix.h>
#include <vector_functions.h>
#include "launch_params.h"

using namespace grotor::rtcore;

extern "C" {
__constant__ LaunchParams optixLaunchParams_color;
}

static __forceinline__ __device__ void pack_pointer(void* ptr, unsigned int& high, unsigned int& low) {
    const auto value = reinterpret_cast<unsigned long long>(ptr);
    high = static_cast<unsigned int>(value >> 32);
    low = static_cast<unsigned int>(value);
}

extern "C" __global__ void __raygen__compute_mex() {
    const unsigned int active = optixGetLaunchIndex().x;
    const unsigned int ray = optixLaunchParams_color.active_ray_indices[active];
    unsigned int net = optixLaunchParams_color.ray_to_net[ray];
    if (optixLaunchParams_color.batch_id[net] != -1) return;
    if (optixLaunchParams_color.is_ready[net] == 0) return;

    unsigned int local_mask[mask_words] = {};
    unsigned int high, low;
    pack_pointer(local_mask, high, low);
    optixTrace(optixLaunchParams_color.gas_handle,
               optixLaunchParams_color.ray_origins[ray], make_float3(0, 0, 1),
               optixLaunchParams_color.ray_tmin, optixLaunchParams_color.ray_tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, high, low, net);
    if (optixLaunchParams_color.is_ready[net] == 1) {
        for (int i = 0; i < mask_words; ++i)
            if (local_mask[i])
                atomicOr(&optixLaunchParams_color.used_mask[net * mask_words + i], local_mask[i]);
    }
}

extern "C" __global__ void __anyhit__collect_colors() {
    const unsigned int net = optixGetPayload_2();
    if (optixLaunchParams_color.iteration > 0 && optixLaunchParams_color.is_ready[net] == 0)
        optixTerminateRay();

    const unsigned int previous = optixLaunchParams_color.tri_to_net_id[optixGetPrimitiveIndex()];
    if (previous >= net) {
        optixIgnoreIntersection();
        return;
    }
    const int batch = optixLaunchParams_color.batch_id[previous];
    if (batch == -1) {
        optixLaunchParams_color.is_ready[net] = 0;
        if (optixLaunchParams_color.iteration == 0) {
            atomicMax(&optixLaunchParams_color.blocker_id[net], int(previous));
            optixIgnoreIntersection();
        } else {
            optixLaunchParams_color.blocker_id[net] = previous;
            optixTerminateRay();
        }
    } else {
        const auto address = (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
        auto* local_mask = reinterpret_cast<unsigned int*>(address);
        if (batch < mask_words * 32) local_mask[batch / 32] |= 1u << (batch % 32);
        optixIgnoreIntersection();
    }
}

extern "C" __global__ void __miss__aabb_miss() {}
