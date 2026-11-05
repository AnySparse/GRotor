#pragma once
#include "router_state.hpp"
#ifdef GROTOR_USE_HIP
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif
#include <type_traits>

namespace grotor::flt {
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create FLT upload stream"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { cuda_check(cudaEventCreateWithFlags(&value, cudaEventDisableTiming), "create FLT upload event"); }
    ~Event() { if (value) cudaEventDestroy(value); }
};
struct PinnedSlot {
    void* data = nullptr;
    Event done;
    bool used = false;
    ~PinnedSlot() { if (data) cudaFreeHost(data); }
};

__global__ void assemble_batch(const NetDescriptor* descriptors, int begin, const Node* source_nodes,
                               const Edge* source_edges, bool streamed, cudb::RoutingDeviceView routing,
                               cudb::FltDeviceView output, int* child_counts, unsigned long long* keys, int* indices,
                               int node_count) {
    const auto net = descriptors[begin + blockIdx.x];
    if (!threadIdx.x) {
        routing.net_ids[blockIdx.x] = net.net_id;
        if (!blockIdx.x) { output.par_num_sum[0] = 0; child_counts[node_count] = 0; }
    }
    const Node* nodes = source_nodes + (streamed ? net.target_node : net.source_node);
    const Edge* edges = source_edges + (streamed ? net.target_edge : net.source_edge);
    for (int i = threadIdx.x; i < net.nodes; i += blockDim.x) {
        const int target = net.target_node + i;
        const Node node = nodes[i];
        output.nodes[target] = node.coordinate;
        output.node_net_idx[target] = net.net_id;
        output.par_num_sum[target + 1] = net.target_edge + node.parent_end;
        child_counts[target] = node.children;
#ifdef GROTOR_SERIAL_DAG
        keys[target] = node.depth;
#else
        keys[target] = (static_cast<unsigned long long>(blockIdx.x) << 13) | node.depth;
#endif
        indices[target] = target;
    }
    for (int i = threadIdx.x; i < net.edges; i += blockDim.x) {
        output.par_nodes[net.target_edge + i] = edges[i].parent + net.target_node;
        output.currentChildIDX[net.target_edge + i] = edges[i].source_slot + net.target_node * 10;
    }
}

__global__ void initialize_batch(cudb::FltDeviceView output, int nodes, int groups) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nodes) output.layer_output[i] = -1;
    if (i < nodes * 10) output.costs[i] = 1e22;
    if (i < (groups + 1) * 10) {
        output.childCosts[i] = 1e22;
        output.childCosts_road[i] = 200000000;
        output.packed_information[i] = 7208259049189261824ULL;
    }
}

// Sorted keys group nodes by DAG and depth. Empty depths use the next boundary.
__global__ void assemble_depth_offsets(const NetDescriptor* descriptors, int begin,
    const unsigned long long* sorted_keys, int* offsets) {
    const auto net = descriptors[begin + blockIdx.x];
    for (int depth = threadIdx.x; depth <= net.depths; depth += blockDim.x) {
        const auto key = (static_cast<unsigned long long>(blockIdx.x) << 13) | depth;
        int low = net.target_node, high = net.target_node + net.nodes;
        while (low < high) {
            const int mid = low + (high - low) / 2;
            if (sorted_keys[mid] < key) low = mid + 1;
            else high = mid;
        }
        offsets[net.depth_begin + depth] = low;
    }
}

struct Workspace {
    cudb::FltDeviceView& output;
    DeviceMemory memory;
    Stream upload_stream;
    Event ready, packed;
    PinnedSlot slots[2];
    bool streamed = false, have_pack = false;
    size_t pinned_bytes = 0, scratch_bytes = 0, next_slot = 0;
    Node* input_nodes = nullptr;
    Edge* input_edges = nullptr;
    NetDescriptor* descriptors = nullptr;
    int *child_counts = nullptr, *indices = nullptr, *depth_offsets = nullptr;
    unsigned long long *keys = nullptr, *sorted_keys = nullptr;
    unsigned char* scratch = nullptr;

    static int sort_bits(int nets) {
        int bits = 13;
#ifdef GROTOR_SERIAL_DAG
        return bits;
#endif
        for (unsigned n = unsigned(nets - 1); n; n >>= 1) ++bits;
        return bits;
    }
    template<class T> void allocate(T*& p, size_t count) { memory.allocate(&p, count * sizeof(T)); }
    Workspace(cudb::FltDeviceView& target, const Plans& plans,
              size_t resident_limit = std::numeric_limits<size_t>::max(),
              size_t staging_limit = size_t(32) << 20) : output(target) {
        const size_t n = plans.max_nodes, e = plans.max_edges, c = plans.max_children + 1;
        allocate(output.nodes, n); allocate(output.par_nodes, e);
        allocate(output.par_num_sum, n + 1); allocate(output.child_num_sum, n + 1);
        allocate(output.currentChildIDX, e); allocate(output.depth_node, n);
        allocate(output.node_net_idx, n); allocate(output.layer_output, n);
        allocate(output.costs, n * 10); allocate(output.cost_edges, e * 81);
        allocate(output.best_change, e * 81); allocate(output.best_path, c * 10);
        allocate(output.childCosts, c * 10); allocate(output.childCosts_road, c * 10);
        allocate(output.packed_information, c * 10);
        allocate(child_counts, n + 1); allocate(keys, n); allocate(sorted_keys, n); allocate(indices, n);
        allocate(descriptors, plans.nets.size());
        allocate(depth_offsets, plans.max_depth_slots);
        for (const auto& batch : plans.batches) {
            size_t scan = 0, sort = 0;
            cuda_check(cub::DeviceScan::ExclusiveSum(nullptr, scan, child_counts, output.child_num_sum, batch.nodes + 1), "size FLT scan");
            cuda_check(cub::DeviceRadixSort::SortPairs(nullptr, sort, keys, sorted_keys, indices,
                output.depth_node, batch.nodes, 0, sort_bits(batch.nets)), "size FLT stable depth sort");
            scratch_bytes = std::max(scratch_bytes, std::max(scan, sort));
        }
        allocate(scratch, scratch_bytes);
        size_t available = 0, total = 0;
        cuda_check(cudaMemGetInfo(&available, &total), "FLT input memory budget");
        const size_t reserve = std::max(size_t(256) << 20, total / 16);
        const size_t resident_bytes = plans.total_nodes * sizeof(Node) + plans.total_edges * sizeof(Edge);
        streamed = resident_bytes > resident_limit || resident_bytes + reserve > available;
        const size_t input_bytes = streamed ? n * sizeof(Node) + e * sizeof(Edge) : resident_bytes;
        if (input_bytes + reserve > available) throw std::runtime_error("FLT batch inputs exceed free-memory budget");
        allocate(input_nodes, streamed ? n : plans.total_nodes);
        allocate(input_edges, streamed ? e : plans.total_edges);
        if (streamed) {
            pinned_bytes = std::max(sizeof(Node), std::min(staging_limit, std::max(n * sizeof(Node), e * sizeof(Edge))));
            for (auto& slot : slots) cuda_check(cudaMallocHost(&slot.data, pinned_bytes), "allocate bounded FLT staging");
        }
        printf("FLT_STORAGE mode=%s nodes=%zu edges=%zu input_bytes=%zu pinned_bytes=%zu max_batch_nodes=%d max_batch_edges=%d scratch_bytes=%zu\n",
            streamed ? "streamed" : "resident", plans.total_nodes, plans.total_edges, input_bytes, 2 * pinned_bytes,
            plans.max_nodes, plans.max_edges, scratch_bytes);
    }
    ~Workspace() {
        cudaDeviceSynchronize();
        output = {};
    }

    void upload(const Topology& topology, const Plans& plans) {
        cuda_check(cudaMemcpyAsync(descriptors, plans.nets.data(), plans.nets.size() * sizeof(NetDescriptor),
            cudaMemcpyHostToDevice, upload_stream.value), "upload FLT batch descriptors");
        if (!streamed) {
            size_t node = 0, edge = 0;
            for (const auto& arena : topology.arenas) {
                for (const auto& chunk : arena.nodes.chunks) {
                    cuda_check(cudaMemcpyAsync(input_nodes + node, chunk.data(), chunk.size() * sizeof(Node),
                        cudaMemcpyHostToDevice, upload_stream.value), "upload compact FLT nodes"); node += chunk.size();
                }
                for (const auto& chunk : arena.edges.chunks) {
                    cuda_check(cudaMemcpyAsync(input_edges + edge, chunk.data(), chunk.size() * sizeof(Edge),
                        cudaMemcpyHostToDevice, upload_stream.value), "upload compact FLT edges"); edge += chunk.size();
                }
            }
        }
        cuda_check(cudaEventRecord(ready.value, upload_stream.value), "record FLT input upload");
    }

    template<class T> void stage_batch(const Topology& topology, const Batch& batch, T* target) {
        int net_index = 0;
        size_t within_net = 0;
        while (net_index < batch.nets) {
            auto& slot = slots[next_slot++ % 2];
            if (slot.used) cuda_check(cudaEventSynchronize(slot.done.value), "reuse FLT pinned staging");
            const size_t capacity = pinned_bytes / sizeof(T);
            size_t filled = 0;
            while (filled < capacity && net_index < batch.nets) {
                const auto& span = topology.spans[batch.slots[net_index]];
                const auto& arena = topology.arenas[span.lane];
                const size_t count = std::is_same_v<T, Node> ? span.nodes : span.edges;
                const size_t n = std::min(capacity - filled, count - within_net);
                if constexpr (std::is_same_v<T, Node>)
                    arena.nodes.copy(span.node_begin + within_net, n, static_cast<T*>(slot.data) + filled);
                else arena.edges.copy(span.edge_begin + within_net, n, static_cast<T*>(slot.data) + filled);
                filled += n; within_net += n;
                if (within_net == count) { ++net_index; within_net = 0; }
            }
            if (!filled) continue;
            cuda_check(cudaMemcpyAsync(target, slot.data, filled * sizeof(T), cudaMemcpyHostToDevice,
                upload_stream.value), "stream FLT topology");
            cuda_check(cudaEventRecord(slot.done.value, upload_stream.value), "record FLT staging reuse");
            slot.used = true; target += filled;
        }
    }

    void assemble(const Topology& topology, const Plans& plans, const Batch& batch, cudb::RoutingDeviceView routing) {
        if (streamed) {
            if (have_pack) cuda_check(cudaStreamWaitEvent(upload_stream.value, packed.value, 0), "protect reused FLT input buffer");
            stage_batch(topology, batch, input_nodes);
            stage_batch(topology, batch, input_edges);
            cuda_check(cudaEventRecord(ready.value, upload_stream.value), "record streamed FLT inputs");
        }
        cuda_check(cudaStreamWaitEvent(nullptr, ready.value, 0), "wait for FLT input upload");
        assemble_batch<<<batch.nets, 128>>>(descriptors, batch.descriptor_begin, input_nodes, input_edges,
            streamed, routing, output, child_counts, keys, indices, batch.nodes);
        cuda_check(cudaEventRecord(packed.value), "release compact FLT input buffer");
        have_pack = true;
        size_t scan = scratch_bytes, sort = scratch_bytes;
        cuda_check(cub::DeviceScan::ExclusiveSum(scratch, scan, child_counts, output.child_num_sum, batch.nodes + 1), "assemble FLT child offsets");
        cuda_check(cub::DeviceRadixSort::SortPairs(scratch, sort, keys, sorted_keys, indices,
            output.depth_node, batch.nodes, 0, sort_bits(batch.nets)), "assemble stable FLT depth order");
#ifndef GROTOR_SERIAL_DAG
        assemble_depth_offsets<<<batch.nets, 128>>>(descriptors, batch.descriptor_begin, sorted_keys, depth_offsets);
#endif
        const int work = std::max(batch.nodes * 10, (batch.children + 1) * 10);
        initialize_batch<<<(work + 255) / 256, 256>>>(output, batch.nodes, batch.children);
        cuda_check(cudaGetLastError(), "assemble FLT batch");
    }
};
} // namespace grotor::flt
