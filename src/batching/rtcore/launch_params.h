#pragma once

#include <optix.h>
#include <vector_types.h>

namespace grotor::rtcore {

constexpr int mask_words = 32;

struct LaunchParams {
    OptixTraversableHandle gas_handle;
    float3* ray_origins;
    unsigned int* tri_to_net_id;
    unsigned int* ray_to_net;
    int* batch_id;
    int* is_ready;
    unsigned int* used_mask;
    int* blocker_id;
    int* d_unassigned_count;
    int total_nets;
    int num_rays;
    int iteration;
    float ray_tmin, ray_tmax;
    unsigned int* active_ray_indices;
    const unsigned int* cell_colors;
    const unsigned int* full_colors;
    int grid_y;
};

} // namespace grotor::rtcore
