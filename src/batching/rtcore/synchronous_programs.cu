#include <optix.h>
#include <vector_functions.h>
#include "synchronous_params.h"

using grotor::rtcore::mask_words;
extern "C" {
__constant__ grotor::rtcore::synchronous::LaunchParams optixLaunchParams_color;
}

extern "C" __global__ void __raygen__rtcore_synchronous() {
    const auto& p = optixLaunchParams_color;
    const unsigned active = optixGetLaunchIndex().x;
    const unsigned ray = p.query.active_ray_indices[active];
    const unsigned net = p.query.ray_to_net[ray];
    // Eligibility is frozen before the launch; another ray cannot suppress this
    // ray by changing readiness while this launch is running.
    if (!p.eligible[net]) return;
    unsigned int local_mask[mask_words] = {};
    const auto pointer = reinterpret_cast<unsigned long long>(local_mask);
    unsigned high = pointer >> 32, low = pointer, payload_net = net;
    // Shard actual trace-call counters to avoid a single global atomic hotspot.
    atomicAdd(p.trace_counts + active / 256, 1u);
    optixTrace(p.query.gas_handle, p.query.ray_origins[ray], make_float3(0,0,1),
        p.query.ray_tmin, p.query.ray_tmax, 0.f, OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0, 1, 0, high, low, payload_net);
    for (int i=0; i<mask_words; ++i)
        if (local_mask[i]) atomicOr(p.query.used_mask + size_t(net)*mask_words+i, local_mask[i]);
}

extern "C" __global__ void __anyhit__rtcore_synchronous() {
    const auto& p = optixLaunchParams_color;
    const unsigned net = optixGetPayload_2();
    const unsigned previous = p.query.tri_to_net_id[optixGetPrimitiveIndex()];
    if (previous >= net) { optixIgnoreIntersection(); return; }
    const int color = p.query.batch_id[previous];
    if (color < 0) {
        atomicExch(p.query.is_ready + net, 0);
        // 0 means no blocker, priority+1 encodes a predecessor. Both arms
        // collect the same state; only the with-blocker reset consults it.
        atomicMax(reinterpret_cast<unsigned int*>(p.query.blocker_id + net), previous+1u);
        // Match standard traversal: round zero gathers the maximum encountered
        // unresolved predecessor; later rounds terminate this ray at a blocker.
        if (p.query.iteration == 0) optixIgnoreIntersection();
        else optixTerminateRay();
    } else {
        const auto pointer = (static_cast<unsigned long long>(optixGetPayload_0()) << 32) | optixGetPayload_1();
        auto* mask = reinterpret_cast<unsigned int*>(pointer);
        if (color < mask_words*32) mask[color/32] |= 1u << (color%32);
        optixIgnoreIntersection();
    }
}
extern "C" __global__ void __miss__aabb_miss() {}
