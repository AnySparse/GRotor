#pragma once
#include "full_update.cuh"
#include "prefix_scan.cuh"

namespace graph {

inline void initialize_incremental_cost(RouterState& state) {
#ifndef GROTOR_FULL_COST_REFERENCE
    auto& view = state.device;
    const size_t cells = size_t(view.L) * view.X * view.Y;
    const size_t bytes = ((cells + 31) / 32) * sizeof(unsigned int);
#ifdef GROTOR_DENSE_PRESUM
    // Legacy multi-GPU policy: dirty cost cells, complete dense prefix scans.
    state.memory.allocate(&view.dirty_wire, 2 * bytes);
    view.dirty_via = view.dirty_wire + bytes / sizeof(unsigned int);
    grotor::cuda_check(cudaMemset(view.dirty_wire, 0xff, 2 * bytes), "initialize dirty costs without BIT");
    printf("COST_STORAGE cells=%zu bit_nodes=0 bit_bytes=0 dirty_bytes=%zu prefix_cache=dense\n", cells, 2 * bytes);
#else
    state.memory.allocate(&view.dirty_wire, 3 * bytes);
    view.dirty_via = view.dirty_wire + bytes / sizeof(unsigned int);
    view.dirty_tree = view.dirty_via + bytes / sizeof(unsigned int);
    state.memory.allocate(&view.dirty_tracks, view.all_track_cnt * sizeof(unsigned int));
    grotor::cuda_check(cudaMemset(view.dirty_wire, 0xff, bytes), "initialize dirty wire costs");
    grotor::cuda_check(cudaMemset(view.dirty_via, 0xff, bytes), "initialize dirty via costs");
    grotor::cuda_check(cudaMemset(view.dirty_tree, 0, bytes), "initialize dirty BIT nodes");
    grotor::cuda_check(cudaMemset(view.dirty_tracks, 0, view.all_track_cnt * sizeof(unsigned int)), "initialize dirty tracks");
    std::vector<int> offsets(1, 0);
    for (int layer = 0; layer < view.L; ++layer) {
        const bool along_y = (layer & 1) ^ view.DIR;
        const int length = along_y ? view.Y : view.X, tracks = along_y ? view.X : view.Y;
        for (int t = 0; t < tracks; ++t) offsets.push_back(offsets.back() + length / 2);
    }
    state.memory.allocate(&view.bit_track_offsets, offsets.size() * sizeof(int));
    state.memory.allocate(&view.cost_bit, size_t(offsets.back()) * sizeof(double));
    grotor::cuda_check(cudaMemcpy(view.bit_track_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice), "upload BIT track offsets");
    printf("COST_STORAGE cells=%zu bit_nodes=%d bit_bytes=%zu dirty_bytes=%zu prefix_cache=reused\n",
           cells, offsets.back(), size_t(offsets.back()) * sizeof(double), 3 * bytes);
    grotor::cuda_check(cudaFuncSetAttribute(refresh_fenwick_tracks,
        cudaFuncAttributeMaxDynamicSharedMemorySize, fenwick_shared_bytes(view.XY)), "BIT shared memory");
#endif
#endif
}

__device__ void update_dirty_wire_cost(const RoutingDeviceView& view, int index) {
    const float previous = view.wcost[index];
    update_single_wcost_ispd24(view, index);
    if (previous == view.wcost[index]) return;
    const int layer = index / (view.X * view.Y);
    const bool along_y = (layer & 1) ^ view.DIR;
    const int position = along_y ? index % view.Y : index / view.Y % view.X;
    const int length = along_y ? view.Y : view.X, stride = along_y ? 1 : view.Y;
    const int base = index - position * stride;
    for (int i = position + 1; i < length; i |= i + 1)
        set_dirty(view.dirty_tree, base + i * stride);
    if (position + 1 < length) {
        const int track = along_y ? index / view.Y % view.X : index % view.Y;
        atomicExch(view.dirty_tracks + cost_track_index(view, layer, track), 1u);
    }
}

// Adjacent lanes inspect adjacent cells and share the cached bitmap word.
__global__ void update_dirty_costs(RoutingDeviceView view) {
#ifdef GROTOR_ENABLE_NVSHMEM
    const int word = blockIdx.x * blockDim.x + threadIdx.x;
    const int cells = view.L * view.X * view.Y;
    if (word >= (cells + 31) / 32) return;
    const unsigned int wire = view.dirty_wire[word], via = view.dirty_via[word];
    unsigned int bits = wire | via;
    while (bits) {
        const int bit = __ffs(bits) - 1, index = word * 32 + bit;
        bits &= bits - 1;
        if (index >= cells) continue;
        if ((wire >> bit) & 1u) {
#ifdef GROTOR_DENSE_PRESUM
            update_single_wcost_ispd24(view, index);
#else
            update_dirty_wire_cost(view, index);
#endif
        }
        if ((via >> bit) & 1u)
            update_single_vcost_ispd24(view, index / view.X / view.Y, index / view.Y % view.X, index % view.Y);
    }
#else
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= view.L * view.X * view.Y) return;
    if (is_dirty(view.dirty_wire, index)) {
#ifdef GROTOR_DENSE_PRESUM
        update_single_wcost_ispd24(view, index);
#else
        update_dirty_wire_cost(view, index);
#endif
    }
    if (is_dirty(view.dirty_via, index))
        update_single_vcost_ispd24(view, index / view.X / view.Y, index / view.Y % view.X, index % view.Y);
#endif
}

inline void refresh_cost(RouterState& state) {
    const auto view = state.device;
    if (!view.dirty_wire) {
        update_cost(state);
        compute_presum<<<view.all_track_cnt, THREAD_NUM, view.XY * sizeof(double)>>>(view);
        return;
    }
    const int cells = view.L * view.X * view.Y;
#ifdef GROTOR_ENABLE_NVSHMEM
    const int update_cells = (cells + 31) / 32;
#else
    const int update_cells = cells;
#endif
#ifdef GROTOR_DENSE_PRESUM
    if (!state.costs_initialized) update_cost(state);
    else update_dirty_costs<<<BLOCK_NUM(update_cells), THREAD_NUM>>>(view);
    compute_presum<<<view.all_track_cnt, THREAD_NUM, view.XY * sizeof(double)>>>(view);
    state.costs_initialized = true;
    const size_t bytes = ((size_t(cells) + 31) / 32) * sizeof(unsigned int);
    grotor::cuda_check(cudaMemsetAsync(view.dirty_wire, 0, 2 * bytes), "clear consumed dense cost dirty bits");
#else
    if (!state.costs_initialized) {
        update_cost(state);
        refresh_fenwick_tracks<<<view.all_track_cnt, THREAD_NUM,
            fenwick_shared_bytes(view.XY)>>>(view, true);
        state.costs_initialized = true;
    } else {
        update_dirty_costs<<<BLOCK_NUM(update_cells), THREAD_NUM>>>(view);
        refresh_fenwick_tracks<<<view.all_track_cnt, THREAD_NUM,
            fenwick_shared_bytes(view.XY)>>>(view);
    }
    const size_t bytes = ((size_t(cells) + 31) / 32) * sizeof(unsigned int);
    grotor::cuda_check(cudaMemsetAsync(view.dirty_wire, 0, 3 * bytes), "clear consumed cost/BIT dirty bits");
#endif
}

} // namespace graph
