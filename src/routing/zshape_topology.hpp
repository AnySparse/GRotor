#pragma once
// BSD 3-Clause attribution: baseline/GGR/LICENSE and NOTICE.md.
#include "preprocess.hpp"
#include <map>
#include <set>
namespace grotor::zshape {
inline std::vector<std::vector<int>> generate_bbox_batches(
    const std::vector<int>& input_nets,
    const std::vector<cudb::net>& nets,
    int x_size,
    int y_size,
    int max_batch_size = 250) {
    std::vector<int> pending = input_nets;
    std::sort(pending.begin(), pending.end(), [&](int lhs, int rhs) {
        const long long lhs_area =
            static_cast<long long>(nets[lhs].maxx - nets[lhs].minx + 1) *
            (nets[lhs].maxy - nets[lhs].miny + 1);
        const long long rhs_area =
            static_cast<long long>(nets[rhs].maxx - nets[rhs].minx + 1) *
            (nets[rhs].maxy - nets[rhs].miny + 1);
        return lhs_area == rhs_area ? lhs > rhs : lhs_area > rhs_area;
    });
    std::vector<std::vector<int>> batches;
    std::vector<unsigned char> occupied(static_cast<size_t>(x_size) * y_size, 0);
    size_t routed = 0;
    while (routed < pending.size()) {
        std::fill(occupied.begin(), occupied.end(), 0);
        std::vector<int> batch;
        batch.reserve(max_batch_size);
        for (int& net_id : pending) {
            if (net_id < 0) continue;
            const cudb::net& net = nets[net_id];
            bool conflict = false;
            for (int x = net.minx; x <= net.maxx && !conflict; ++x) {
                for (int y = net.miny; y <= net.maxy; ++y) {
                    if (occupied[static_cast<size_t>(x) * y_size + y]) {
                        conflict = true;
                        break;
                    }
                }
            }
            if (conflict) continue;
            for (int x = net.minx; x <= net.maxx; ++x) {
                for (int y = net.miny; y <= net.maxy; ++y) {
                    occupied[static_cast<size_t>(x) * y_size + y] = 1;
                }
            }
            batch.push_back(net_id);
            net_id = -1;
            ++routed;
            if (static_cast<int>(batch.size()) == max_batch_size) break;
        }
        if (batch.empty()) break;
        batches.emplace_back(std::move(batch));
    }
    return batches;
}
struct EncodedBatch { std::vector<int> net_ids, points, gbpoints; };
inline EncodedBatch encode_batch(const std::vector<int>& batch,
                                 const std::vector<cudb::net>& nets,
                                 const std::vector<int>& route_offsets,
                                 int x_size,
                                 int y_size,
                                 int routing_layers) {
    EncodedBatch encoded;
    encoded.net_ids = batch;
    encoded.points.resize(batch.size());
    encoded.gbpoints.resize(batch.size());
    for (size_t bi = 0; bi < batch.size(); ++bi) {
        const cudb::net& net = nets[batch[bi]];
        const auto& graph = net.rsmt;
        const int node_count = static_cast<int>(graph.size()) - 1;
        const int point_base = static_cast<int>(encoded.points.size());
        const int gb_base = static_cast<int>(encoded.gbpoints.size());
        encoded.points[bi] = point_base;
        encoded.gbpoints[bi] = gb_base;
        encoded.points.push_back(route_offsets[batch[bi]]);
        encoded.points.push_back(node_count);
        int root = 0;
        for (int i = 0; i < node_count; ++i) {
            if (graph[i].size() == 1) { root = i; break; }
        }
        std::vector<int> order;
        std::vector<int> parent(node_count, -1);
        order.reserve(node_count);
        std::vector<int> stack{root};
        while (!stack.empty()) {
            int node = stack.back();
            stack.pop_back();
            order.push_back(node);
            for (auto it = graph[node].rbegin(); it != graph[node].rend(); ++it) {
                if (*it == parent[node]) continue;
                parent[*it] = node;
                stack.push_back(*it);
            }
        }
        std::vector<int> preorder_index(node_count, -1);
        for (int i = 0; i < node_count; ++i) preorder_index[order[i]] = i;
        std::vector<int> records(static_cast<size_t>(node_count) * 6, -1);
        for (int oi = 0; oi < node_count; ++oi) {
            int node = order[oi];
            int encoded_pos = graph.back()[node];
            int layer = encoded_pos / (x_size * y_size);
            int pos2d = encoded_pos % (x_size * y_size);
            records[oi * 6] = (pos2d / y_size) * std::max(x_size, y_size) + pos2d % y_size;
            if (layer < routing_layers) {
                records[oi * 6 + 1] = layer + 1;
                records[oi * 6 + 2] = layer + 1;
            }
            int child_slot = 3;
            for (int next : graph[node]) {
                if (parent[next] != node) continue;
                if (child_slot >= 6) throw std::runtime_error("GGR Z-shape node has more than three children");
                records[oi * 6 + child_slot++] = preorder_index[next] * 6;
            }
        }
        encoded.points.insert(encoded.points.end(), records.begin(), records.end());
        encoded.gbpoints.insert(encoded.gbpoints.end(), node_count, -1);
    }
    return encoded;
}
inline int promote_congested_segments(
    int net_count, int x_size, int y_size, int layers,
    unsigned long long route_per_pin, const std::vector<int>& pin_acc_num,
    int* routes) {
    const int xy = x_size * y_size;
    std::vector<float> usage(static_cast<size_t>(layers) * xy, 0.0f);
    auto visit_edges = [&](int p0, int p1, const auto& fn) {
        int l = p0 / xy;
        if (l != p1 / xy) return;
        int x0 = p0 / y_size % x_size, y0 = p0 % y_size;
        int x1 = p1 / y_size % x_size, y1 = p1 % y_size;
        if (x0 == x1) for (int y = std::min(y0, y1); y < std::max(y0, y1); ++y) fn(l, x0, y);
        if (y0 == y1) for (int x = std::min(x0, x1); x < std::max(x0, x1); ++x) fn(l, x, y0);
    };
    for (int net = 0; net < net_count; ++net) {
        int* r = routes + static_cast<size_t>(pin_acc_num[net]) * route_per_pin;
        for (int i = 1; i + 1 < r[0]; i += 2)
            visit_edges(r[i], r[i + 1], [&](int l, int x, int y) { usage[l * xy + x * y_size + y] += 1.0f; });
    }
    int promoted = 0;
    for (int net = 0; net < net_count; ++net) {
        int* r = routes + static_cast<size_t>(pin_acc_num[net]) * route_per_pin;
        const int original_end = r[0];
        const int limit = static_cast<int>((pin_acc_num[net + 1] - pin_acc_num[net]) * route_per_pin);
        for (int i = 1; i + 1 < original_end; i += 2) {
            int p0 = r[i], p1 = r[i + 1], old_l = p0 / xy;
            if (old_l != p1 / xy) continue;
            int x0 = p0 / y_size % x_size, y0 = p0 % y_size;
            int x1 = p1 / y_size % x_size, y1 = p1 % y_size;
            if (x0 == x1 && y0 == y1) continue;
            auto worst_excess = [&](int layer, float add) {
                double worst = -1e30;
                visit_edges((p0 % xy) + layer * xy, (p1 % xy) + layer * xy,
                    [&](int, int x, int y) {
                        double cap = db::capacity[layer + 1][x][y];
                        worst = std::max(worst, static_cast<double>(usage[layer * xy + x * y_size + y] + add) - cap);
                    });
                return worst;
            };
            double old_score = worst_excess(old_l, 0.0f);
            if (old_score <= 0.0) continue;
            int best_l = old_l;
            double best_score = old_score;
            for (int l = old_l + 2; l < layers; l += 2) {
                double score = worst_excess(l, 1.0f);
                if (score < best_score) best_score = score, best_l = l;
            }
            if (best_l == old_l || best_score >= old_score || r[0] + 4 >= limit) continue;
            visit_edges(p0, p1, [&](int l, int x, int y) { usage[l * xy + x * y_size + y] -= 1.0f; });
            int np0 = p0 % xy + best_l * xy, np1 = p1 % xy + best_l * xy;
            visit_edges(np0, np1, [&](int l, int x, int y) { usage[l * xy + x * y_size + y] += 1.0f; });
            r[i] = np0;
            r[i + 1] = np1;
            r[r[0]++] = p0;
            r[r[0]++] = np0;
            r[r[0]++] = p1;
            r[r[0]++] = np1;
            ++promoted;
        }
    }
    return promoted;
}

inline void canonicalize_routes(int net_count, int x_size, int y_size,
                                unsigned long long route_per_pin,
                                const std::vector<int>& pin_acc_num,
                                int* routes) {
    const int xy = x_size * y_size;
    for (int net = 0; net < net_count; ++net) {
        int* data = routes + static_cast<size_t>(pin_acc_num[net]) * route_per_pin;
        if (data[0] <= 1) continue;
        using Key = std::tuple<int,int,int>;
        std::map<Key, std::vector<std::pair<int,int>>> intervals;
        std::set<std::tuple<int,int,int>> unit_vias;
        for (int i = 1; i + 1 < data[0]; i += 2) {
            int a = data[i], b = data[i + 1];
            int la = a / xy, lb = b / xy;
            int ax = a / y_size % x_size, ay = a % y_size;
            int bx = b / y_size % x_size, by = b % y_size;
            if (la != lb) {
                for (int l = std::min(la, lb); l < std::max(la, lb); ++l)
                    unit_vias.emplace(ax, ay, l);
            } else if (ax != bx) {
                intervals[{la, 0, ay}].emplace_back(std::min(ax,bx), std::max(ax,bx));
            } else if (ay != by) {
                intervals[{la, 1, ax}].emplace_back(std::min(ay,by), std::max(ay,by));
            }
        }
        int out = 1;
        int limit = (pin_acc_num[net + 1] - pin_acc_num[net]) * route_per_pin;
        auto emit = [&](int a, int b) {
            if (out + 1 >= limit) throw std::runtime_error("Z-shape route storage exceeded");
            data[out++] = a; data[out++] = b;
        };
        for (auto& entry : intervals) {
            const auto key = entry.first;
            auto& values = entry.second;
            std::sort(values.begin(), values.end());
            int begin = values[0].first, end = values[0].second;
            auto flush = [&]() {
                auto [layer, dir, fixed] = key;
                if (dir == 0) emit(layer * xy + begin * y_size + fixed,
                                   layer * xy + end * y_size + fixed);
                else emit(layer * xy + fixed * y_size + begin,
                          layer * xy + fixed * y_size + end);
            };
            for (size_t j = 1; j < values.size(); ++j) {
                if (values[j].first <= end) end = std::max(end, values[j].second);
                else { flush(); begin = values[j].first; end = values[j].second; }
            }
            flush();
        }
        for (auto it = unit_vias.begin(); it != unit_vias.end();) {
            auto [x,y,l0] = *it;
            int l1 = l0 + 1;
            auto next = std::next(it);
            while (next != unit_vias.end()) {
                auto [nx,ny,nl] = *next;
                if (nx != x || ny != y || nl != l1) break;
                ++l1; ++next;
            }
            emit(l0 * xy + x * y_size + y, l1 * xy + x * y_size + y);
            it = next;
        }
        data[0] = out;
    }
}
__global__ inline void convert_routes(
    int count, const int* net_ids, const int* route_offsets,
    const int* ggr_routes, int n, int layers, int x_size, int y_size,
    int direction, unsigned long long route_per_pin, const int* pin_acc_num,
    int* routes, int* error) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int net = net_ids[i];
    const int* src = ggr_routes + route_offsets[i];
    int* dst = routes + static_cast<unsigned long long>(pin_acc_num[net]) * route_per_pin;
    int out = 1;
    for (int p = 1; p + 1 < src[0]; p += 2) {
        int start = src[p], len = src[p + 1];
        int ggr_layer = start / (n * n);
        int layer = ggr_layer - 1;
        int a = start / n % n, b = start % n;
        int x = ((layer & 1) ^ direction) ? a : b;
        int y = ((layer & 1) ^ direction) ? b : a;
        if (layer < 0 || layer >= layers || x >= x_size || y >= y_size) {
            atomicExch(error, 3); continue;
        }
        int pos0 = layer * x_size * y_size + x * y_size + y;
        int pos1 = pos0;
        if (len == -1) {
            if (layer + 1 >= layers) { atomicExch(error, 3); continue; }
            pos1 += x_size * y_size;
        } else {
            int end = start + len;
            int ea = end / n % n, eb = end % n;
            int ex = ((layer & 1) ^ direction) ? ea : eb;
            int ey = ((layer & 1) ^ direction) ? eb : ea;
            if (ex >= x_size || ey >= y_size) { atomicExch(error, 3); continue; }
            pos1 = layer * x_size * y_size + ex * y_size + ey;
        }
        if (out + 1 < static_cast<int>((pin_acc_num[net + 1] - pin_acc_num[net]) * route_per_pin)) {
            dst[out++] = pos0;
            dst[out++] = pos1;
        } else { atomicExch(error, 1); }
    }
    if (src[0] <= 0) atomicExch(error, 2);
    dst[0] = out;
}
}
