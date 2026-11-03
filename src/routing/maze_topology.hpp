#pragma once

#include "router_state.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace grotor::maze {
inline void maze_copy(cudaError_t status) { cuda_check(status, "download maze cost view"); }

// A CPU copy of CUGR2's direction-collapsed 2-D maze cost view.  Prefix sums
// make every sparse-graph edge query O(1); the routing DP remains the existing
// GPU block-inc implementation.
class MazeCostView {
public:
    // The evaluator charges physical wirelength directly.  The original
    // direction-collapsed congestion term is otherwise strong enough to send
    // sparse paths on very long detours for a small overflow reduction.
    // Keep that topology search evaluator-aware with an explicit base-length
    // guard; the later layer-assignment DP still uses the normal GPU costs.
    MazeCostView(const cudb::RoutingDeviceView& router, const float* d_demand,
                 double wirelength_guard = 100.0)
        : x_(router.X), y_(router.Y),
          layers_(router.L), dir0_(router.DIR),
          wirelength_guard_(wirelength_guard) {
        const size_t cells = static_cast<size_t>(layers_) * x_ * y_;
        std::vector<float> capacity(cells);
        std::vector<float> demand(cells);
        std::vector<int> x_edge(std::max(0, x_ - 1));
        std::vector<int> y_edge(std::max(0, y_ - 1));
        std::vector<double> overflow_weight(layers_);

        maze_copy(cudaMemcpy(capacity.data(), router.capacity,
                              cells * sizeof(float), cudaMemcpyDeviceToHost));
        maze_copy(cudaMemcpy(demand.data(), d_demand,
                              cells * sizeof(float), cudaMemcpyDeviceToHost));
        if (x_ > 1) {
            maze_copy(cudaMemcpy(x_edge.data(), router.x_edge_len,
                                  (x_ - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        if (y_ > 1) {
            maze_copy(cudaMemcpy(y_edge.data(), router.y_edge_len,
                                  (y_ - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        }
        maze_copy(cudaMemcpy(overflow_weight.data(), router.unit_length_short_costs,
                              layers_ * sizeof(double), cudaMemcpyDeviceToHost));

        h_prefix_.assign(static_cast<size_t>(y_) * (x_ + 1), 0.0);
        v_prefix_.assign(static_cast<size_t>(x_) * (y_ + 1), 0.0);

        double direction_weight[2] = {
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()
        };
        for (int l = 0; l < layers_; ++l) {
            int direction = (l & 1) ^ dir0_;
            direction_weight[direction] =
                std::min(direction_weight[direction], overflow_weight[l]);
        }
        for (double& value : direction_weight) {
            if (!std::isfinite(value)) value = 1.0;
        }

        for (int yy = 0; yy < y_; ++yy) {
            for (int xx = 0; xx + 1 < x_; ++xx) {
                double cap = 0.0;
                double dem = 0.0;
                for (int l = 0; l < layers_; ++l) {
                    if (((l & 1) ^ dir0_) != 0) continue;
                    size_t idx = (static_cast<size_t>(l) * x_ + xx) * y_ + yy;
                    cap += capacity[idx];
                    dem += demand[idx];
                }
                const double congestion =
                    cap < 1.0 ? 1.0 : 1.0 / (1.0 + std::exp((cap - dem) * 0.5));
                const double edge_cost = x_edge[xx] *
                    (wirelength_guard_ * router.unit_length_wire_cost +
                     direction_weight[0] * congestion);
                h_prefix_[static_cast<size_t>(yy) * (x_ + 1) + xx + 1] =
                    h_prefix_[static_cast<size_t>(yy) * (x_ + 1) + xx] + edge_cost;
            }
        }

        for (int xx = 0; xx < x_; ++xx) {
            for (int yy = 0; yy + 1 < y_; ++yy) {
                double cap = 0.0;
                double dem = 0.0;
                for (int l = 0; l < layers_; ++l) {
                    if (((l & 1) ^ dir0_) != 1) continue;
                    size_t idx = (static_cast<size_t>(l) * x_ + xx) * y_ + yy;
                    cap += capacity[idx];
                    dem += demand[idx];
                }
                const double congestion =
                    cap < 1.0 ? 1.0 : 1.0 / (1.0 + std::exp((cap - dem) * 0.5));
                const double edge_cost = y_edge[yy] *
                    (wirelength_guard_ * router.unit_length_wire_cost +
                     direction_weight[1] * congestion);
                v_prefix_[static_cast<size_t>(xx) * (y_ + 1) + yy + 1] =
                    v_prefix_[static_cast<size_t>(xx) * (y_ + 1) + yy] + edge_cost;
            }
        }
    }

    double horizontal(int x0, int x1, int y) const {
        if (x0 > x1) std::swap(x0, x1);
        const size_t base = static_cast<size_t>(y) * (x_ + 1);
        return h_prefix_[base + x1] - h_prefix_[base + x0];
    }

    double vertical(int x, int y0, int y1) const {
        if (y0 > y1) std::swap(y0, y1);
        const size_t base = static_cast<size_t>(x) * (y_ + 1);
        return v_prefix_[base + y1] - v_prefix_[base + y0];
    }

private:
    int x_;
    int y_;
    int layers_;
    int dir0_;
    double wirelength_guard_;
    std::vector<double> h_prefix_;
    std::vector<double> v_prefix_;
};

struct MazeState {
    double cost;
    int vertex;
    int previous;
    bool operator>(const MazeState& rhs) const {
        return cost != rhs.cost ? cost > rhs.cost : vertex > rhs.vertex;
    }
};

class SparseMazeRouter {
public:
    SparseMazeRouter(int x_size, int y_size, int routing_layers,
                     int interval = 10)
        : x_size_(x_size), y_size_(y_size), routing_layers_(routing_layers),
          interval_(interval) {}

    bool rebuild(cudb::net& net, const MazeCostView& cost_view, int offset) const {
        if (net.pins.size() <= 1) return true;

        std::vector<int> xs;
        std::vector<int> ys;
        xs.reserve(x_size_ / interval_ + net.pins.size() + 2);
        ys.reserve(y_size_ / interval_ + net.pins.size() + 2);
        for (int x = offset % interval_; x < x_size_; x += interval_) xs.push_back(x);
        for (int y = offset % interval_; y < y_size_; y += interval_) ys.push_back(y);
        for (int pin : net.pins) {
            xs.push_back((pin % (x_size_ * y_size_)) / y_size_);
            ys.push_back(pin % y_size_);
        }
        sort_unique(xs);
        sort_unique(ys);
        if (xs.empty() || ys.empty()) return false;

        const int plane = static_cast<int>(xs.size() * ys.size());
        const int vertices = 2 * plane;
        auto vertex = [&](int direction, int xi, int yi) {
            return direction * plane + yi * static_cast<int>(xs.size()) + xi;
        };
        auto decode = [&](int v, int& direction, int& xi, int& yi) {
            direction = v / plane;
            int local = v % plane;
            yi = local / static_cast<int>(xs.size());
            xi = local % static_cast<int>(xs.size());
        };

        std::unordered_map<int, int> x_index;
        std::unordered_map<int, int> y_index;
        for (int i = 0; i < static_cast<int>(xs.size()); ++i) x_index[xs[i]] = i;
        for (int i = 0; i < static_cast<int>(ys.size()); ++i) y_index[ys[i]] = i;

        std::vector<int> pin_vertex(net.pins.size());
        std::unordered_map<int, int> vertex_pin;
        for (int i = 0; i < static_cast<int>(net.pins.size()); ++i) {
            const int pos = net.pins[i] % (x_size_ * y_size_);
            const int v = vertex(0, x_index[pos / y_size_], y_index[pos % y_size_]);
            pin_vertex[i] = v;
            vertex_pin.emplace(v, i);
        }

        std::vector<double> distance(vertices, std::numeric_limits<double>::max());
        std::vector<int> predecessor(vertices, -1);
        std::vector<char> attached(net.pins.size(), 0);
        std::priority_queue<MazeState, std::vector<MazeState>, std::greater<MazeState>> queue;
        std::vector<std::pair<int, int>> tree_edges;

        attached[0] = 1;
        distance[pin_vertex[0]] = 0.0;
        queue.push({0.0, pin_vertex[0], -1});
        int detached = static_cast<int>(net.pins.size()) - 1;

        while (detached > 0) {
            int found_vertex = -1;
            int found_pin = -1;
            while (!queue.empty()) {
                MazeState state = queue.top();
                queue.pop();
                if (state.cost != distance[state.vertex]) continue;
                auto pin_it = vertex_pin.find(state.vertex);
                if (pin_it != vertex_pin.end() && !attached[pin_it->second]) {
                    found_vertex = state.vertex;
                    found_pin = pin_it->second;
                    break;
                }

                int direction, xi, yi;
                decode(state.vertex, direction, xi, yi);
                auto relax = [&](int next, double edge_cost) {
                    const double candidate = state.cost + edge_cost;
                    if (candidate < distance[next]) {
                        distance[next] = candidate;
                        predecessor[next] = state.vertex;
                        queue.push({candidate, next, state.vertex});
                    }
                };
                if (direction == 0) {
                    if (xi > 0) {
                        relax(vertex(0, xi - 1, yi),
                              cost_view.horizontal(xs[xi - 1], xs[xi], ys[yi]));
                    }
                    if (xi + 1 < static_cast<int>(xs.size())) {
                        relax(vertex(0, xi + 1, yi),
                              cost_view.horizontal(xs[xi], xs[xi + 1], ys[yi]));
                    }
                } else {
                    if (yi > 0) {
                        relax(vertex(1, xi, yi - 1),
                              cost_view.vertical(xs[xi], ys[yi - 1], ys[yi]));
                    }
                    if (yi + 1 < static_cast<int>(ys.size())) {
                        relax(vertex(1, xi, yi + 1),
                              cost_view.vertical(xs[xi], ys[yi], ys[yi + 1]));
                    }
                }
                const int other = vertex(1 - direction, xi, yi);
                const double via_cost =
                    vertex_pin.count(vertex(0, xi, yi)) ? 0.0 : 1.0;
                relax(other, via_cost);
            }
            if (found_vertex < 0) return false;

            attached[found_pin] = 1;
            --detached;
            int current = found_vertex;
            while (current >= 0 && distance[current] != 0.0) {
                const int previous = predecessor[current];
                if (previous < 0) return false;
                tree_edges.emplace_back(current, previous);
                distance[current] = 0.0;
                queue.push({0.0, current, predecessor[current]});
                current = previous;
            }
        }

        return install_tree(net, xs, ys, tree_edges, plane);
    }

private:
    static void sort_unique(std::vector<int>& values) {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
    }

    bool install_tree(cudb::net& net, const std::vector<int>& xs,
                      const std::vector<int>& ys,
                      const std::vector<std::pair<int, int>>& state_edges,
                      int plane) const {
        std::unordered_map<int, int> point_to_node;
        std::vector<int> points;
        std::vector<std::pair<int, int>> edges;
        auto state_point = [&](int state) {
            int local = state % plane;
            int yi = local / static_cast<int>(xs.size());
            int xi = local % static_cast<int>(xs.size());
            return xs[xi] * y_size_ + ys[yi];
        };
        auto node_for = [&](int point) {
            auto it = point_to_node.find(point);
            if (it != point_to_node.end()) return it->second;
            int id = static_cast<int>(points.size());
            points.push_back(point);
            point_to_node.emplace(point, id);
            return id;
        };
        for (const auto& edge : state_edges) {
            int a = node_for(state_point(edge.first));
            int b = node_for(state_point(edge.second));
            if (a != b) edges.emplace_back(a, b);
        }
        if (points.empty()) return false;

        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        std::vector<std::vector<int>> graph(points.size());
        for (const auto& edge : edges) {
            graph[edge.first].push_back(edge.second);
            graph[edge.second].push_back(edge.first);
        }

        // Collapse non-pin degree-2 collinear vertices, matching CUGR2's
        // getSteinerTree cleanup and keeping ROUTE_PER_PIN bounded.
        std::unordered_map<int, int> pin_layer;
        for (int pin : net.pins) {
            int p2d = pin % (x_size_ * y_size_);
            pin_layer[p2d] = pin / (x_size_ * y_size_);
        }
        bool changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < static_cast<int>(graph.size()); ++i) {
                if (points[i] < 0 || pin_layer.count(points[i]) || graph[i].size() != 2) continue;
                int a = graph[i][0];
                int b = graph[i][1];
                int ix = points[i] / y_size_, iy = points[i] % y_size_;
                int ax = points[a] / y_size_, ay = points[a] % y_size_;
                int bx = points[b] / y_size_, by = points[b] % y_size_;
                if (!((ax == ix && bx == ix) || (ay == iy && by == iy))) continue;
                erase_neighbor(graph[a], i);
                erase_neighbor(graph[b], i);
                graph[a].push_back(b);
                graph[b].push_back(a);
                graph[i].clear();
                points[i] = -1;
                changed = true;
            }
        }

        std::vector<int> remap(points.size(), -1);
        std::vector<int> compact_points;
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            if (points[i] >= 0 && !graph[i].empty()) {
                remap[i] = static_cast<int>(compact_points.size());
                compact_points.push_back(points[i]);
            }
        }
        if (compact_points.size() < 2) return false;
        std::vector<std::vector<int>> compact_graph(compact_points.size());
        for (int i = 0; i < static_cast<int>(graph.size()); ++i) {
            if (remap[i] < 0) continue;
            for (int next : graph[i]) {
                if (remap[next] >= 0) compact_graph[remap[i]].push_back(remap[next]);
            }
            sort_unique(compact_graph[remap[i]]);
        }

        std::vector<int> encoded = compact_points;
        for (int& point : encoded) {
            auto it = pin_layer.find(point);
            point += (it == pin_layer.end() ? routing_layers_ : it->second)
                * x_size_ * y_size_;
        }
        compact_graph.push_back(std::move(encoded));
        net.rsmt = std::move(compact_graph);
        rebuild_collision_geometry(net);
        return true;
    }

    static void erase_neighbor(std::vector<int>& neighbors, int value) {
        neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), value), neighbors.end());
    }

    void rebuild_collision_geometry(cudb::net& net) const {
        net.rsmt_h_segments.clear();
        net.rsmt_v_segments.clear();
        net.points.clear();
        const int node_count = static_cast<int>(net.rsmt.size()) - 1;
        net.points.reserve(node_count);
        for (int i = 0; i < node_count; ++i) {
            int a = net.rsmt.back()[i] % (x_size_ * y_size_);
            net.points.push_back(a);
            for (int j : net.rsmt[i]) {
                if (j >= i) continue;
                int b = net.rsmt.back()[j] % (x_size_ * y_size_);
                int ax = a / y_size_, ay = a % y_size_;
                int bx = b / y_size_, by = b % y_size_;
                if (ay == by) {
                    net.rsmt_h_segments.emplace_back(
                        std::min(ax, bx) * y_size_ + ay,
                        std::max(ax, bx) * y_size_ + ay);
                } else if (ax == bx) {
                    net.rsmt_v_segments.emplace_back(
                        ax * y_size_ + std::min(ay, by),
                        ax * y_size_ + std::max(ay, by));
                }
            }
        }
    }

    int x_size_;
    int y_size_;
    int routing_layers_;
    int interval_;
};

} // namespace grotor::maze
