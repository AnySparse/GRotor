#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace grotor::flt {
struct Builder {
    std::vector<int> par_num_cpu, par_num_sum_cpu, currentChildIDX_cpu, par_nodes_cpu;
    std::vector<int> child_num_cpu, node_depth_cpu, nodes_cpu;
    int node_index_cnt = 0;
};
struct Node { int coordinate, depth, children, parent_end; };
struct Edge { int parent, source_slot; };

// Fixed-size slabs avoid temporarily duplicating a large arena on reallocation.
template<class T> struct Slabs {
    static constexpr size_t capacity = (4u << 20) / sizeof(T);
    std::vector<std::vector<T>> chunks;
    size_t count = 0;
    void push(T value) {
        if (chunks.empty() || chunks.back().size() == capacity) {
            chunks.emplace_back(); chunks.back().reserve(capacity);
        }
        chunks.back().push_back(value); ++count;
    }
    const T& operator[](size_t index) const { return chunks[index / capacity][index % capacity]; }
    void copy(size_t begin, size_t count, T* output) const {
        while (count) {
            const size_t n = std::min(count, capacity - begin % capacity);
            std::memcpy(output, &(*this)[begin], n * sizeof(T));
            begin += n; output += n; count -= n;
        }
    }
};
struct Arena { Slabs<Node> nodes; Slabs<Edge> edges; };
struct Span {
    size_t node_begin = 0, edge_begin = 0;
    int nodes = 0, edges = 0, children = 0, lane = -1, depths = 0;
};
struct Topology {
    std::vector<Arena> arenas;
    std::vector<Span> spans;
    std::vector<int> net_to_slot;
    void begin(const std::vector<int>& order, size_t total_nets, int threads) {
        arenas.clear(); arenas.resize(threads); spans.resize(order.size());
        net_to_slot.assign(total_nets, -1);
        for (size_t i = 0; i < order.size(); ++i) net_to_slot.at(order[i]) = i;
    }
    template<class Net> void emit(size_t slot, int lane, const Net& net) {
        auto& arena = arenas.at(lane);
        Span span{arena.nodes.count, arena.edges.count, net.node_index_cnt, 0, 0, lane};
        int edge = 0;
        for (int node = 0; node < span.nodes; ++node) {
            const int depth = net.node_depth_cpu[node];
            if (depth < 0 || depth >= 5000) throw std::runtime_error("FLT depth exceeds supported range");
            span.depths = std::max(span.depths, depth + 1);
            const int parents = net.par_num_cpu[node];
            for (int p = 0; p < parents; ++p, ++edge)
                arena.edges.push({net.par_nodes_cpu[edge], node * 10 + net.currentChildIDX_cpu[edge]});
            arena.nodes.push({net.nodes_cpu[node], net.node_depth_cpu[node], net.child_num_cpu[node], edge});
            span.children += net.child_num_cpu[node];
        }
        span.edges = edge;
        if (edge != net.par_num_sum_cpu[span.nodes] || !span.nodes)
            throw std::runtime_error("Invalid FLT topology counts");
        spans.at(slot) = span;
    }
    void clear() {
        std::vector<Arena>().swap(arenas); std::vector<Span>().swap(spans);
        std::vector<int>().swap(net_to_slot);
    }
};

// Source and destination offsets are counts of records, not byte addresses.
struct NetDescriptor {
    int net_id;
    unsigned int source_node, source_edge;
    int target_node, target_edge, nodes, edges, depth_begin, depths;
};
struct Batch {
    int descriptor_begin = 0, nets = 0, nodes = 0, edges = 0, children = 0, depth_slots = 0;
    std::vector<int> slots;
};
struct Plans {
    std::vector<NetDescriptor> nets;
    std::vector<Batch> batches;
    size_t total_nodes = 0, total_edges = 0;
    int max_nodes = 0, max_edges = 0, max_children = 0, max_depth_slots = 0;
    explicit Plans(const Topology& topology, const std::vector<std::vector<int>>& order) {
        nets.reserve(topology.spans.size()); batches.reserve(order.size());
        std::vector<size_t> node_base, edge_base;
        for (const auto& arena : topology.arenas) {
            node_base.push_back(total_nodes); edge_base.push_back(total_edges);
            total_nodes += arena.nodes.count; total_edges += arena.edges.count;
        }
        if (total_nodes > UINT32_MAX || total_edges > UINT32_MAX)
            throw std::runtime_error("FLT compact topology exceeds device index range");
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            if (it->empty()) continue;
            Batch batch; batch.descriptor_begin = nets.size(); batch.nets = it->size();
            batch.slots.reserve(it->size());
            for (int id : *it) {
                const int slot = topology.net_to_slot.at(id);
                const auto& span = topology.spans.at(slot);
                const auto& arena = topology.arenas.at(span.lane);
                nets.push_back({id, static_cast<unsigned int>(node_base[span.lane] + span.node_begin),
                    static_cast<unsigned int>(edge_base[span.lane] + span.edge_begin),
                    batch.nodes, batch.edges, span.nodes, span.edges, batch.depth_slots, span.depths});
                if (int64_t(batch.depth_slots) + span.depths + 1 > INT32_MAX)
                    throw std::runtime_error("FLT depth offsets exceed index range");
                batch.depth_slots += span.depths + 1;
                batch.slots.push_back(slot);
                const int64_t nodes = int64_t(batch.nodes) + span.nodes, edges = int64_t(batch.edges) + span.edges;
                const int64_t children = int64_t(batch.children) + span.children;
                if (nodes > INT32_MAX / 10 || edges > INT32_MAX / 81 || children > INT32_MAX / 10 - 1)
                    throw std::runtime_error("FLT batch exceeds solver index limits");
                batch.nodes = nodes; batch.edges = edges; batch.children = children;
            }
            max_nodes = std::max(max_nodes, batch.nodes); max_edges = std::max(max_edges, batch.edges);
            max_children = std::max(max_children, batch.children);
            max_depth_slots = std::max(max_depth_slots, batch.depth_slots);
            batches.push_back(std::move(batch));
        }
    }
};
} // namespace grotor::flt
