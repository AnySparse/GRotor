#pragma once
#include "routing/maze_topology.hpp"
#include "routing/flt.cuh"
#include <atomic>
#include <thread>

namespace grotor::maze {

// Test membership in this net's wires, independently of other nets' stamps.
__device__ bool wire_touches(const int* route, int point, int plane, int y_size) {
    for (int i = 1; i < route[0]; i += 2) {
        const int a = route[i], b = route[i + 1];
        if (a / plane != point / plane || b / plane != point / plane) continue;
        const int x = point / y_size, y = point % y_size;
        if (a % y_size == y && b % y_size == y &&
            x >= min(a / y_size, b / y_size) && x <= max(a / y_size, b / y_size)) return true;
        if (a / y_size == x && b / y_size == x &&
            y >= min(a % y_size, b % y_size) && y <= max(a % y_size, b % y_size)) return true;
    }
    return false;
}

// Arbitrary overflow nets may overlap. Half-unit atomic demand changes and
// net-local wire membership avoid shared timestamp races during bulk rip-up.
__global__ void remove_routes(cudb::RoutingDeviceView view, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int id = view.net_ids[index], plane = view.X * view.Y;
    const int* route = view.ripup_flag[id]
        ? view.routes_phase2 + view.pin_acc_num_phase2[id] * view.ROUTE_PER_PIN_PHASE2
        : view.routes + view.pin_acc_num[id] * view.ROUTE_PER_PIN;
    double length = 0;
    int vias = 0;
    for (int i = 1; i < route[0]; i += 2) {
        const int a = route[i], b = route[i + 1];
        const int x = a / view.Y % view.X, y = a % view.Y;
        if (a / plane == b / plane) {
            const int bx = b / view.Y % view.X, by = b % view.Y;
            for (int xx = min(x, bx); xx < max(x, bx); ++xx) {
                graph::add_demand(view, a / plane * plane + xx * view.Y + y, -1);
                length += view.x_edge_len[xx];
            }
            for (int yy = min(y, by); yy < max(y, by); ++yy) {
                graph::add_demand(view, a / plane * plane + x * view.Y + yy, -1);
                length += view.y_edge_len[yy];
            }
        } else {
            const int first = min(a / plane, b / plane), last = max(a / plane, b / plane);
            vias += last - first;
            for (int layer = first; layer < last; ++layer) {
                const int point = layer * plane + x * view.Y + y;
                if (wire_touches(route, point, plane, view.Y)) continue;
                const bool vertical = (layer & 1) ^ view.DIR;
                const int position = vertical ? y : x, limit = vertical ? view.Y : view.X;
                const int step = vertical ? 1 : view.Y;
                if (!position) graph::add_demand(view, point, -1);
                else if (position == limit - 1) graph::add_demand(view, point - step, -1);
                else {
                    graph::add_demand(view, point - step, -.5f);
                    graph::add_demand(view, point, -.5f);
                }
            }
        }
    }
    atomicAdd(&view.statistics->total_wirelength, -length);
    atomicAdd(&view.statistics->total_via_count, -vias);
}

inline flt::Builder tree_dag(const cudb::net& net) {
    flt::Builder result;
    const int size = static_cast<int>(net.rsmt.size()) - 1;
    if (size < 2) throw std::runtime_error("Sparse maze produced an empty tree");
    std::vector<int> order{0}, parent(size, -1), depth(size, 0), slot(size, -1);
    parent[0] = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        const int node = order[i];
        for (int child : net.rsmt[node]) {
            if (child == parent[node]) continue;
            if (parent[child] != -1) throw std::runtime_error("Sparse maze tree contains a cycle");
            parent[child] = node;
            depth[child] = depth[node] + 1;
            order.push_back(child);
        }
    }
    if (order.size() != size) throw std::runtime_error("Sparse maze tree is disconnected");
    result.node_index_cnt = size;
    result.par_num_sum_cpu.push_back(0);
    for (int i = 0; i < size; ++i) slot[order[i]] = i;
    for (int i = 0; i < size; ++i) {
        const int node = order[i];
        const int children = net.rsmt[node].size() - (i != 0);
        if (children > 6) throw std::runtime_error("Sparse maze DAG fanout exceeds six");
        result.nodes_cpu.push_back(net.rsmt.back()[node]);
        result.node_depth_cpu.push_back(depth[node]);
        result.child_num_cpu.push_back(children);
        result.par_num_cpu.push_back(i != 0);
        if (i) {
            const int p = parent[node];
            int child_index = 0;
            for (int sibling : net.rsmt[p]) {
                if (sibling == parent[p] && p != 0) continue;
                if (sibling == node) break;
                ++child_index;
            }
            result.par_nodes_cpu.push_back(slot[p]);
            result.currentChildIDX_cpu.push_back(child_index);
        }
        result.par_num_sum_cpu.push_back(result.par_nodes_cpu.size());
    }
    return result;
}

inline double prepare_topology(cudb::RouterState& state, std::vector<int>& ids) {
    if (ids.empty()) return 0;
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        return state.nets[a].hpwl != state.nets[b].hpwl
            ? state.nets[a].hpwl > state.nets[b].hpwl : a < b;
    });
    const auto begin = std::chrono::steady_clock::now();
    const MazeCostView cost(state.device, state.device.demand);
    const SparseMazeRouter router(state.device.X, state.device.Y, state.device.L, 10);
    state.flt_topology.begin(ids, state.nets.size(), 8);
    if (state.rtcore_batching)
        state.rtcore_geometry.begin(ids, state.nets.size(), 8,
            {state.rtcore_z_gap, state.rtcore_margin, state.rtcore_coplanar, state.rtcore_compact_points});
    std::atomic<size_t> next{0};
    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::vector<std::thread> workers;
    for (int lane = 0; lane < 8; ++lane) workers.emplace_back([&, lane] {
        try {
            while (true) {
                const size_t slot = next.fetch_add(1);
                if (slot >= ids.size()) break;
                auto& net = state.nets[ids[slot]];
                if (!router.rebuild(net, cost, slot))
                    throw std::runtime_error("Sparse maze search failed for net " + std::to_string(ids[slot]));
                const auto dag = tree_dag(net);
                state.flt_topology.emit(slot, lane, dag);
                if (state.rtcore_batching)
                    state.rtcore_geometry.emit(slot, lane, net, state.device.Y, true);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
            next.store(ids.size());
        }
    });
    for (auto& worker : workers) worker.join();
    if (failure) std::rethrow_exception(failure);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    printf("MAZE_SEARCH elapsed_ms=%.6f nets=%zu threads=8 sparse_interval=10\n", ms, ids.size());
    return ms;
}
} // namespace grotor::maze
