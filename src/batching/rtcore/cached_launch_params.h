#pragma once

#include <optix.h>
#include <vector_types.h>

namespace grotor::rtcore::cached {

constexpr int mask_words = 32;
constexpr int chunk_capacity = 16384;

struct LaunchParams {
    OptixTraversableHandle gas_handle;
    float3* ray_origins;
    unsigned int* tri_to_net_id;
    unsigned int* ray_to_net;
    int* batch_id;
    unsigned int* dependencies;
    unsigned int* used_mask;
    int* next_net;
    int* color_error;
    int total_nets;
    int num_rays;
    int dependency_words;
    int chunk_begin;
    int chunk_nets;
    int first_ray;
    float ray_tmin, ray_tmax;
};

} // namespace grotor::rtcore::cached
