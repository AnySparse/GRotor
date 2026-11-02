#pragma once

#include "launch_params.h"

namespace grotor::rtcore {

__global__ void reset_status_kernel(LaunchParams params) {
    const int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_id >= params.total_nets) return;
    if (params.batch_id[net_id] == -1) {
        const int blocker = params.blocker_id[net_id];
        if (blocker != -1) {
            if (params.batch_id[blocker] == -1) {
                params.is_ready[net_id] = 0;
                return;
            }
            params.blocker_id[net_id] = -1;
        }
        params.is_ready[net_id] = 1;
        // Each 128-byte mask row is aligned for four-word stores.
        auto* words = reinterpret_cast<uint4*>(params.used_mask + size_t(net_id) * mask_words);
        #pragma unroll
        for (int i = 0; i < mask_words / 4; ++i) words[i] = make_uint4(0, 0, 0, 0);
    }
}

__global__ void assign_batch_kernel(LaunchParams params) {
    const int net_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (net_id >= params.total_nets) return;
    if (params.batch_id[net_id] == -1 && params.is_ready[net_id] == 1) {
        int batch = -1;
        for (int i = 0; i < mask_words; ++i) {
            const unsigned int mask = params.used_mask[net_id * mask_words + i];
            if (mask != 0xffffffffu) {
                batch = i * 32 + __ffs(~mask) - 1;
                break;
            }
        }
        if (batch != -1) {
            params.batch_id[net_id] = batch;
            atomicSub(params.d_unassigned_count, 1);
        }
    }
}

__global__ void fast_compact_rays_kernel(unsigned int* ray_to_net, int* batch_id,
                                       int* is_ready, unsigned int* active_indices,
                                       int* active_count, int num_rays) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= num_rays) return;
    const int net = ray_to_net[ray];
    if (batch_id[net] == -1 && is_ready[net] == 1) {
        const int pos = atomicAdd(active_count, 1);
        active_indices[pos] = ray;
    }
}

} // namespace grotor::rtcore
