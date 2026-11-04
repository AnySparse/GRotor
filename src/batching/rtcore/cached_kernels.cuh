#pragma once

#include "cached_launch_params.h"
#include <cuda/atomic>

namespace grotor::rtcore::cached {

// Tasks are claimed by resident warps, rather than assigned by block index.
// Every unfinished predecessor is therefore already owned by a running warp.
__global__ void color_dependency_graph(LaunchParams params) {
    constexpr unsigned int full = 0xffffffffu;
    __shared__ unsigned int masks[8][mask_words];
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    while (true) {
        int net = 0;
        if (lane == 0) net = atomicAdd(params.next_net, 1);
        net = __shfl_sync(full, net, 0);
        if (net >= params.chunk_nets) return;
        masks[warp][lane] = params.used_mask[net * mask_words + lane];
        if (params.full_colors) masks[warp][lane] |= params.full_colors[lane];
        __syncwarp(full);
        bool invalid = false;
        for (int word = lane; word < params.dependency_words; word += 32) {
            unsigned int bits = params.dependencies[size_t(net) * params.dependency_words + word];
            while (bits) {
                const int previous = params.chunk_begin + word * 32 + __ffs(bits) - 1;
                int color;
                do {
                    color = cuda::atomic_ref<int, cuda::thread_scope_device>(params.batch_id[previous]).load(cuda::memory_order_acquire);
                    if (color == -1) __nanosleep(32);
                } while (color == -1);
                if (color < 0 || color >= mask_words * 32) invalid = true;
                else atomicOr(&masks[warp][color / 32], 1u << (color % 32));
                bits &= bits - 1;
            }
        }
        const bool bad = __any_sync(full, invalid);
        __syncwarp(full);
        const unsigned int used = masks[warp][lane];
        const unsigned int available = __ballot_sync(full, used != 0xffffffffu);
        const int word = available ? __ffs(available) - 1 : 0;
        const unsigned int first_mask = __shfl_sync(full, used, word);
        if (lane == 0) {
            const int color = bad || !available ? -2 : word * 32 + __ffs(~first_mask) - 1;
            cuda::atomic_ref<int, cuda::thread_scope_device>(params.batch_id[params.chunk_begin + net]).store(color, cuda::memory_order_release);
            if (color < 0) atomicExch(params.color_error, 1);
        }
        __syncwarp(full);
    }
}

} // namespace grotor::rtcore::cached
