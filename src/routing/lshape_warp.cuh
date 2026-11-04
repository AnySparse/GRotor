#pragma once
#include "routing/demand.cuh"

namespace Lshape_route {

// One lane solves every layer state of one node. Nodes at the same tree
// depth are independent until their contributions are merged into parents.
__device__ __forceinline__ void node_diagonal(const RoutingDeviceView& view,
    const int* nodes, const int* parents, double* dist, int* from, int i, double* diagonal) {
    const int L = view.L, X = view.X, Y = view.Y, DIR = view.DIR;
    const auto vcost = cudb::via_costs(view);
    const int x = nodes[i] / Y % X, y = nodes[i] % Y;
    const int parent = parents[i], px = nodes[parent] / Y % X, py = nodes[parent] % Y;
    const int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
    const double* node_dist = dist + i * L * L;
    int* prev = from + i * L * L;
    for (int layer = 0; layer < L; ++layer) diagonal[layer] = INF;
    const bool straight = x == px || y == py;
    for (int curl = 0; curl < L; ++curl) {
        double current = INF;
        int choice = 0;
        for (int low = 0; low <= curl; ++low)
            for (int high = curl; high < L; ++high)
                if (node_dist[low * L + high] < current) {
                    current = node_dist[low * L + high];
                    choice = low * L + high;
                }
        if (straight) {
            if ((((curl & 1) ^ DIR) == 0 && y != py) ||
                (((curl & 1) ^ DIR) == 1 && x != px)) continue;
            diagonal[curl] = current + graph::wire_segment_cost(view, curl, minx, maxx, miny, maxy);
            prev[curl * L + curl] = choice * L * L + curl * L + curl;
        } else {
            if ((curl & 1) ^ DIR) current += graph::wire_segment_cost(view, curl, x, x, miny, maxy);
            else current += graph::wire_segment_cost(view, curl, minx, maxx, y, y);
            for (int parl = (curl & 1) ^ 1; parl < L; parl += 2) {
                double extra = 0;
                for (int layer = min(curl, parl) + 1; layer < max(curl, parl); ++layer)
                    extra += ((curl & 1) ^ DIR) ? vcost[IDX(layer, x, py)] : vcost[IDX(layer, px, y)];
                if ((parl & 1) ^ DIR) extra += graph::wire_segment_cost(view, parl, px, px, miny, maxy);
                else extra += graph::wire_segment_cost(view, parl, minx, maxx, py, py);
                if (current + extra < diagonal[parl]) {
                    diagonal[parl] = current + extra;
                    prev[parl * L + parl] = choice * L * L + curl * L + parl;
                }
            }
        }
    }
}

__device__ __forceinline__ int emit_lshape_node(const RoutingDeviceView& view,
    const int* nodes, const int* par_nodes, const int* ranges, const int* from,
    int i, int* output, int stamp) {
    const int L = view.L, X = view.X, Y = view.Y, DIR = view.DIR;
    const int* prev = from + i * L * L;
    int cursor = 0;
    int minl = ranges[i] / L, maxl = ranges[i] % L, x = nodes[i] / Y % X, y = nodes[i] % Y;
    if(minl < maxl) {
        output[cursor++] = IDX(minl, x, y);
        output[cursor++] = IDX(maxl, x, y);
    }

    if(i == 0) return cursor;
    int px = nodes[par_nodes[i]] / Y % X, py = nodes[par_nodes[i]] % Y;
    int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
    int curl = prev[ranges[par_nodes[i]]] / L % L, parl = prev[ranges[par_nodes[i]]] % L;
    if(px == x || py == y) {
        assert(curl == parl);
        output[cursor++] = IDX(curl, minx, miny);
        output[cursor++] = IDX(curl, maxx, maxy);
        graph::atomic_add_unit_demand_wire_segment(view, curl, minx, maxx, miny, maxy, stamp);
    } else {
        assert(curl % 2 != parl % 2);
        int minl = min(curl, parl), maxl = max(curl, parl);
        if(curl & 1 ^ DIR) {
            output[cursor++] = IDX(curl, x, y);
            output[cursor++] = IDX(curl, x, py);
            graph::atomic_add_unit_demand_wire_segment(view, curl, x, x, miny, maxy, stamp);

            output[cursor++] = IDX(curl, x, py);
            output[cursor++] = IDX(parl, x, py);

            output[cursor++] = IDX(parl, x, py);
            output[cursor++] = IDX(parl, px, py);
            graph::atomic_add_unit_demand_wire_segment(view, parl, minx, maxx, py, py, stamp);
        } else {
            output[cursor++] = IDX(curl, x, y);
            output[cursor++] = IDX(curl, px, y);
            graph::atomic_add_unit_demand_wire_segment(view, curl, minx, maxx, y, y, stamp);

            output[cursor++] = IDX(curl, px, y);
            output[cursor++] = IDX(parl, px, y);

            output[cursor++] = IDX(parl, px, y);
            output[cursor++] = IDX(parl, px, py);
            graph::atomic_add_unit_demand_wire_segment(view, parl, px, px, miny, maxy, stamp);
        }
    }
    return cursor;
}

// A warp owns a tree, with one lane per node of its current depth. Each parent
// is updated by one leader in descending original-node order, matching the
// scalar reverse-DFS accumulation exactly without floating-point atomics.
__global__ void Lshape_route_warp(const GROTOR_GRID_CONSTANT RoutingDeviceView view, int net_cnt, int net_offset,
    const int* node_cnt_sum, const int* all_nodes, const int* all_parents,
    double* all_dist, int* all_from, int* all_ranges, int stamp) {
    constexpr unsigned mask = 0xffffffffu;
    constexpr int range_bits = 8, range_mask = (1 << range_bits) - 1;
    const int lane = threadIdx.x & 31;
    int net_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (net_idx >= net_cnt) return;
    net_idx += net_offset;
    const int L = view.L, X = view.X, Y = view.Y;
    assert(L > 0 && L <= 10);
    const int node_cnt = node_cnt_sum[net_idx + 1] - node_cnt_sum[net_idx];
    const int* nodes = all_nodes + node_cnt_sum[net_idx];
    const int* parents = all_parents + node_cnt_sum[net_idx];
    double* dist = all_dist + size_t(node_cnt_sum[net_idx] - node_cnt_sum[net_offset]) * L * L;
    int* from = all_from + size_t(node_cnt_sum[net_idx] - node_cnt_sum[net_offset]) * L * L;
    int* ranges = all_ranges + node_cnt_sum[net_idx];
    const auto vcost = cudb::via_costs(view);
    const int net_id = view.net_ids[net_idx];
    int* routes = view.routes + view.pin_acc_num[net_id] * view.ROUTE_PER_PIN;
    int max_depth = 0;
    for (int i = lane; i < node_cnt; i += 32) {
        int depth = 0;
        for (int node = i; node > 0; node = parents[node]) {
            assert(parents[node] >= 0 && parents[node] < node);
            ++depth;
        }
        assert(depth <= (0x7fffffff >> range_bits));
        // Depth tags share the existing range buffer until traceback finishes.
        // No additional per-design topology or floating-point scratch is stored.
        ranges[i] = depth << range_bits;
        max_depth = max(max_depth, depth);
        const int fixed = nodes[i] / X / Y, x = nodes[i] / Y % X, y = nodes[i] % Y;
        double* values = dist + i * L * L;
        for (int low = 0; low < L; ++low) {
            values[low * L + low] = 0;
            for (int high = low + 1; high < L; ++high)
                values[low * L + high] = values[low * L + high - 1] + vcost[IDX(high - 1, x, y)];
            for (int high = low; high < L; ++high)
                if (fixed < L && (low > fixed || high < fixed)) values[low * L + high] = INF;
        }
    }
    for (int step = 16; step; step >>= 1)
        max_depth = max(max_depth, __shfl_xor_sync(mask, max_depth, step));
    __syncwarp(mask);
    for (int depth = max_depth; depth > 0; --depth) {
        for (int begin = node_cnt - 1; begin > 0; begin -= 32) {
            const int i = begin - lane;
            const bool active = i > 0 && (ranges[i] >> range_bits) == depth;
            if (!__ballot_sync(mask, active)) continue;
            const int parent = active ? parents[i] : -1;
            const unsigned peers = __match_any_sync(mask, parent);
            const unsigned group = active ? peers : 0;
            const bool leader = active && lane == __ffs(group) - 1;
            int max_group = __popc(group);
            for (int step = 16; step; step >>= 1)
                max_group = max(max_group, __shfl_xor_sync(mask, max_group, step));
            double diagonal[10];
            if (!active) for (int layer = 0; layer < L; ++layer) diagonal[layer] = INF;
            if (active) node_diagonal(view, nodes, parents, dist, from, i, diagonal);
            for (int low = 0; low < L; ++low) {
                double contribution = diagonal[low];
                int path = active ? from[i * L * L + low * L + low] : 0;
                for (int high = low; high < L; ++high) {
                    if (high > low && !(contribution <= diagonal[high])) {
                        contribution = diagonal[high];
                        if (active) path = from[i * L * L + high * L + high];
                    }
                    const int index = low * L + high;
                    if (active) from[i * L * L + index] = path;
                    if (max_group == 1) {
                        if (active) dist[parent * L * L + index] += contribution;
                    } else {
                        double value = leader ? dist[parent * L * L + index] : 0.;
                        unsigned remaining = group;
                        for (int child = 0; child < max_group; ++child) {
                            const int source = remaining ? __ffs(remaining) - 1 : 0;
                            const double message = __shfl_sync(mask, contribution, source);
                            if (leader && remaining) value += message;
                            remaining &= remaining - 1;
                        }
                        if (leader) dist[parent * L * L + index] = value;
                    }
                }
            }
            __syncwarp(mask);
        }
    }
    if (!lane) {
        int choice = 0;
        for (int low = 0; low < L; ++low)
            for (int high = low; high < L; ++high)
                if (dist[low * L + high] < dist[choice]) choice = low * L + high;
        ranges[0] = choice;
    }
    __syncwarp(mask);
    // Trace independent nodes of a depth together. Retain the high depth tag
    // until every level has completed, then restore ordinary range indices.
    for (int depth = 1; depth <= max_depth; ++depth) {
        for (int i = lane + 1; i < node_cnt; i += 32) {
            if ((ranges[i] >> range_bits) != depth) continue;
            const int parent_range = ranges[parents[i]] & range_mask;
            const int choice = from[i * L * L + parent_range] / L / L;
            assert(dist[i * L * L + choice] < INF);
            ranges[i] = (depth << range_bits) | choice;
        }
        __syncwarp(mask);
    }
    for (int i = lane; i < node_cnt; i += 32) ranges[i] &= range_mask;
    __syncwarp(mask);
    int next = 1;
    for (int begin = 0; begin < node_cnt; begin += 32) {
        const int i = begin + lane;
        int words = 0;
        if (i < node_cnt) {
            words = ranges[i] / L < ranges[i] % L ? 2 : 0;
            if (i) {
                const int a = nodes[i] % (X * Y), b = nodes[parents[i]] % (X * Y);
                words += (a / Y == b / Y || a % Y == b % Y) ? 2 : 6;
            }
        }
        int prefix = words;
        for (int step = 1; step < 32; step <<= 1) {
            const int other = __shfl_up_sync(mask, prefix, step);
            if (lane >= step) prefix += other;
        }
        const int count = __shfl_sync(mask, prefix, 31);
        assert(size_t(next + count) <= view.ROUTE_PER_PIN * (view.pin_acc_num[net_id + 1] - view.pin_acc_num[net_id]));
        if (i < node_cnt) {
            const int emitted = emit_lshape_node(view, nodes, parents, ranges, from, i,
                routes + next + prefix - words, stamp);
            assert(emitted == words);
        }
        next += count;
    }
    if (!lane) routes[0] = next;
#ifdef GROTOR_MULTI_GRID
    if (!view.pre_demand) {
        __syncwarp(mask);
        graph::commit_net_wires<32>(view, net_id, stamp, lane);
    }
#endif
}
} // namespace Lshape_route
