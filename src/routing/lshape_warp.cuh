#pragma once
#include "routing/demand.cuh"

namespace Lshape_route {

// A warp owns one tree. Layer states are parallel; child accumulation follows
// the original reverse DFS order so floating-point sums and ties stay fixed.
__global__ void Lshape_route_warp(RoutingDeviceView view, int net_cnt, int net_offset,
    const int* node_cnt_sum, const int* all_nodes, const int* all_parents,
    double* all_dist, int* all_from, int* all_ranges, int stamp) {
    constexpr unsigned mask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    int net_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (net_idx >= net_cnt) return;
    net_idx += net_offset;
    const int L = view.L, X = view.X, Y = view.Y, DIR = view.DIR;
    const int node_cnt = node_cnt_sum[net_idx + 1] - node_cnt_sum[net_idx];
    const int* nodes = all_nodes + node_cnt_sum[net_idx];
    const int* par_nodes = all_parents + node_cnt_sum[net_idx];
    double* dist = all_dist + size_t(node_cnt_sum[net_idx] - node_cnt_sum[net_offset]) * L * L;
    int* from = all_from + size_t(node_cnt_sum[net_idx] - node_cnt_sum[net_offset]) * L * L;
    int* layer_range = all_ranges + node_cnt_sum[net_idx];
    const auto vcost = view.vcost;
    int* net_routes = view.routes + view.pin_acc_num[view.net_ids[net_idx]] * view.ROUTE_PER_PIN;
    for (int row = lane; row < node_cnt * L; row += 32) {
        const int i = row / L, low = row % L;
        const int fixed = nodes[i] / X / Y, x = nodes[i] / Y % X, y = nodes[i] % Y;
        double* values = dist + i * L * L;
        values[low * L + low] = 0;
        for (int high = low + 1; high < L; ++high)
            values[low * L + high] = values[low * L + high - 1] + vcost[IDX(high - 1, x, y)];
        for (int high = low; high < L; ++high)
            if (fixed < L && (low > fixed || high < fixed)) values[low * L + high] = INF;
    }
    __syncwarp(mask);
    for (int i = node_cnt - 1; i >= 1; --i) {
        const int x = nodes[i] / Y % X, y = nodes[i] % Y;
        const int parent = par_nodes[i], px = nodes[parent] / Y % X, py = nodes[parent] % Y;
        const int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
        const double* values = dist + i * L * L;
        double* parent_values = dist + parent * L * L;
        int* prev = from + i * L * L;
        double current = INF;
        int current_from = 0;
        if (lane < L) {
            for (int low = 0; low <= lane; ++low)
                for (int high = lane; high < L; ++high)
                    if (values[low * L + high] < current) {
                        current = values[low * L + high]; current_from = low * L + high;
                    }
        }
        double diagonal = INF;
        int diagonal_from = 0;
        if (x == px || y == py) {
            if (lane < L && !(((lane & 1) ^ DIR) == 0 && y != py)
                         && !(((lane & 1) ^ DIR) == 1 && x != px)) {
                diagonal = current + graph::wire_segment_cost(view, lane, minx, maxx, miny, maxy);
                diagonal_from = current_from * L * L + lane * L + lane;
            }
        } else {
            for (int curl = 0; curl < L; ++curl) {
                double cost = __shfl_sync(mask, current, curl);
                const int choice = __shfl_sync(mask, current_from, curl);
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
                // Share this warp-uniform query. Each actual BIT traversal is
                // counted once; the returned value is broadcast to all lanes.
                double wire = 0.;
                if (lane == 0) {
                    if ((curl & 1) ^ DIR) wire = graph::wire_segment_cost(view, curl, x, x, miny, maxy);
                    else wire = graph::wire_segment_cost(view, curl, minx, maxx, y, y);
                }
                cost += __shfl_sync(mask, wire, 0);
#else
                if ((curl & 1) ^ DIR) cost += graph::wire_segment_cost(view, curl, x, x, miny, maxy);
                else cost += graph::wire_segment_cost(view, curl, minx, maxx, y, y);
#endif
                if (lane < L && ((lane & 1) != (curl & 1))) {
                    double cost2 = 0;
                    for (int l = min(curl, lane) + 1; l < max(curl, lane); ++l)
                        cost2 += ((curl & 1) ^ DIR) ? vcost[IDX(l, x, py)] : vcost[IDX(l, px, y)];
                    if ((lane & 1) ^ DIR) cost2 += graph::wire_segment_cost(view, lane, px, px, miny, maxy);
                    else cost2 += graph::wire_segment_cost(view, lane, minx, maxx, py, py);
                    if (cost + cost2 < diagonal) {
                        diagonal = cost + cost2;
                        diagonal_from = choice * L * L + curl * L + lane;
                    }
                }
            }
        }
        double best = diagonal;
        int path = diagonal_from;
        for (int high = 0; high < L; ++high) {
            const double candidate = __shfl_sync(mask, diagonal, high);
            const int candidate_from = __shfl_sync(mask, diagonal_from, high);
            if (lane < L && high >= lane) {
                if (high > lane && !(best <= candidate)) { best = candidate; path = candidate_from; }
                prev[lane * L + high] = path;
                parent_values[lane * L + high] += best;
            }
        }
        __syncwarp(mask);
    }
    if (lane == 0) {
    net_routes[0] = 1;
    layer_range[0] = 0;
    for(int minl = 0; minl < L; minl++)
        for(int maxl = minl; maxl < L; maxl++) 
            if(dist[minl * L + maxl] < dist[layer_range[0]]) {
                layer_range[0] = minl * L + maxl;
            }
    for(int i = 0; i < node_cnt; i++) {
        int *prev = from + i * L * L;
        if(i > 0) layer_range[i] = prev[layer_range[par_nodes[i]]] / L / L;
        assert(dist[i * L * L + layer_range[i]] < INF);
        int minl = layer_range[i] / L, maxl = layer_range[i] % L, x = nodes[i] / Y % X, y = nodes[i] % Y;
        if(minl < maxl) {
            net_routes[net_routes[0]++] = IDX(minl, x, y);
            net_routes[net_routes[0]++] = IDX(maxl, x, y);
        }

        if(i == 0) continue;
        int px = nodes[par_nodes[i]] / Y % X, py = nodes[par_nodes[i]] % Y;
        int minx = min(x, px), maxx = max(x, px), miny = min(y, py), maxy = max(y, py);
        int curl = prev[layer_range[par_nodes[i]]] / L % L, parl = prev[layer_range[par_nodes[i]]] % L;
        if(px == x || py == y) {
            assert(curl == parl);
            net_routes[net_routes[0]++] = IDX(curl, minx, miny);
            net_routes[net_routes[0]++] = IDX(curl, maxx, maxy);
            graph::atomic_add_unit_demand_wire_segment(view, curl, minx, maxx, miny, maxy, stamp);
        } else {        
            assert(curl % 2 != parl % 2);
            int minl = min(curl, parl), maxl = max(curl, parl);
            if(curl & 1 ^ DIR) {
                net_routes[net_routes[0]++] = IDX(curl, x, y);
                net_routes[net_routes[0]++] = IDX(curl, x, py);
                graph::atomic_add_unit_demand_wire_segment(view, curl, x, x, miny, maxy, stamp);

                net_routes[net_routes[0]++] = IDX(curl, x, py);
                net_routes[net_routes[0]++] = IDX(parl, x, py);

                net_routes[net_routes[0]++] = IDX(parl, x, py);
                net_routes[net_routes[0]++] = IDX(parl, px, py);
                graph::atomic_add_unit_demand_wire_segment(view, parl, minx, maxx, py, py, stamp);
            } else {
                net_routes[net_routes[0]++] = IDX(curl, x, y);
                net_routes[net_routes[0]++] = IDX(curl, px, y);
                graph::atomic_add_unit_demand_wire_segment(view, curl, minx, maxx, y, y, stamp);

                net_routes[net_routes[0]++] = IDX(curl, px, y);
                net_routes[net_routes[0]++] = IDX(parl, px, y);

                net_routes[net_routes[0]++] = IDX(parl, px, y);
                net_routes[net_routes[0]++] = IDX(parl, px, py);
                graph::atomic_add_unit_demand_wire_segment(view, parl, px, px, miny, maxy, stamp);
            }
        }
    }
    }
}

} // namespace Lshape_route
