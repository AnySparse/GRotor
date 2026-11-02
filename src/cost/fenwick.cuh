#pragma once
#include "router_state.hpp"

namespace graph {

__device__ __forceinline__ void set_dirty(unsigned int* bits, int index) {
    atomicOr(bits + (index >> 5), 1u << (index & 31));
}

__device__ __forceinline__ bool is_dirty(const unsigned int* bits, int index) {
    return (bits[index >> 5] >> (index & 31)) & 1u;
}

// A changed edge affects its wire cost and both adjacent via landing costs.
__device__ __forceinline__ void mark_cost_dirty(const RoutingDeviceView& view, int index) {
    if (!view.dirty_wire) return;
    set_dirty(view.dirty_wire, index);
    set_dirty(view.dirty_via, index);
    const int layer = index / (view.X * view.Y);
    const bool along_y = (layer & 1) ^ view.DIR;
    const int position = along_y ? index % view.Y : index / view.Y % view.X;
    const int length = along_y ? view.Y : view.X;
    if (position + 1 < length)
        set_dirty(view.dirty_via, index + (along_y ? 1 : view.Y));
}

// Mark a contiguous interval with one atomic per bitmap word.
__device__ __forceinline__ void mark_dirty_range(unsigned int* bits, int first, int last) {
    const int a = first >> 5, b = last >> 5;
    for (int word = a; word <= b; ++word) {
        unsigned int mask = ~0u;
        if (word == a) mask &= ~0u << (first & 31);
        if (word == b) mask &= ~0u >> (31 - (last & 31));
        atomicOr(bits + word, mask);
    }
}

__device__ __forceinline__ void mark_wire_span(const RoutingDeviceView& view,
                                               int layer, int x0, int x1, int y0, int y1) {
    if (!view.dirty_wire || (x0 == x1 && y0 == y1)) return;
    const bool along_y = (layer & 1) ^ view.DIR;
    const int base = layer * view.X * view.Y;
    if (along_y && x0 == x1) {
        const int first = base + x0 * view.Y + min(y0, y1), last = base + x0 * view.Y + max(y0, y1);
        mark_dirty_range(view.dirty_wire, first, last - 1);
        mark_dirty_range(view.dirty_via, first, last);
    } else if (!along_y && y0 == y1) {
        const int lo = min(x0, x1), hi = max(x0, x1);
        for (int x = lo; x < hi; ++x) {
            set_dirty(view.dirty_wire, base + x * view.Y + y0);
            set_dirty(view.dirty_via, base + x * view.Y + y0);
        }
        set_dirty(view.dirty_via, base + hi * view.Y + y0);
    } else {
        for (int x = min(x0, x1); x < max(x0, x1); ++x) mark_cost_dirty(view, base + x * view.Y + y0);
        for (int y = min(y0, y1); y < max(y0, y1); ++y) mark_cost_dirty(view, base + x0 * view.Y + y);
    }
}

__device__ __forceinline__ void mark_shifted_word(unsigned int* bits, int word, unsigned int mask, int shift) {
    if (!mask) return;
    word += shift >> 5;
    shift &= 31;
    const unsigned int low = mask << shift;
    if (low) atomicOr(bits + word, low);
    if (shift) {
        const unsigned int high = mask >> (32 - shift);
        if (high) atomicOr(bits + word + 1, high);
    }
}

__device__ __forceinline__ int cost_track_index(const RoutingDeviceView& view, int layer, int track) {
    const int first = view.DIR ? view.X : view.Y, second = view.DIR ? view.Y : view.X;
    return (layer + 1) / 2 * first + layer / 2 * second + track;
}

// Zero-based BIT over the shifted input [0, w0, w1, ...]. Its prefix is exclusive
// in wire-edge coordinates, matching the routing segment endpoint convention.
__device__ __forceinline__ double fenwick_prefix(const RoutingDeviceView& view,
                                                int layer, int x, int y) {
    const bool along_y = (layer & 1) ^ view.DIR;
    int position = along_y ? y : x;
    const int stride = along_y ? 1 : view.Y;
    const int base = layer * view.X * view.Y + (along_y ? x * view.Y : y);
    const int track = cost_track_index(view, layer, along_y ? x : y);
    const double* nodes = view.cost_bit + view.bit_track_offsets[track];
    double sum = 0;
    for (; position >= 0; position = (position & (position + 1)) - 1) {
        // Even BIT indices are leaves; read their current cost directly.
        sum += (position & 1) ? nodes[position >> 1]
                            : (position ? double(view.wcost[base + (position - 1) * stride]) : 0.);
    }
    return sum;
}

__global__ void refresh_fenwick_tracks(RoutingDeviceView view, bool rebuild = false) {
    if (!rebuild && !view.dirty_tracks[blockIdx.x]) return;
    const int code = view.idx2track[blockIdx.x];
    const int layer = code / view.XY, track = code % view.XY;
    const bool along_y = (layer & 1) ^ view.DIR;
    const int length = along_y ? view.Y : view.X, stride = along_y ? 1 : view.Y;
    const int base = layer * view.X * view.Y + (along_y ? track * view.Y : track);
    double* nodes = view.cost_bit + view.bit_track_offsets[blockIdx.x];
    extern __shared__ double storage[];
    double* tree = storage;
    auto* dirty = reinterpret_cast<unsigned int*>(tree + view.XY);
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        const int index = base + i * stride;
        tree[i] = rebuild ? 0. : ((i & 1) ? nodes[i >> 1]
            : (i ? double(view.wcost[base + (i - 1) * stride]) : 0.));
        dirty[i] = rebuild || is_dirty(view.dirty_tree, index);
    }
    __syncthreads();
    // Children are complete before their parents. Rebuild affected nodes from
    // current costs, preserving the full scan's addition order without floating
    // atomic deltas or cancellation across successive route/rip-up updates.
    for (int width = 1; width <= length; width <<= 1) {
        for (int i = (2 * threadIdx.x + 1) * width - 1; i < length;
             i += 2 * blockDim.x * width) {
            if (!dirty[i]) continue;
            double value = i ? double(view.wcost[base + (i - 1) * stride]) : 0.;
            for (int offset = 1; offset < width; offset <<= 1)
                value += tree[i - offset];
            tree[i] = value;
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < length; i += blockDim.x) {
        if ((i & 1) && dirty[i]) nodes[i >> 1] = tree[i];
        double sum = 0.;
        for (int j = i; j >= 0; j = (j & (j + 1)) - 1) sum += tree[j];
        view.presum[base + i * stride] = sum;
    }
    if (!threadIdx.x) view.dirty_tracks[blockIdx.x] = 0;
}

} // namespace graph
