#include "routing/flt_workspace.cuh"
#include <random>
#include <numeric>

namespace {
grotor::flt::Builder fixture(int n, unsigned int seed) {
    grotor::flt::Builder net;
    net.node_index_cnt = n;
    net.nodes_cpu.resize(n); net.node_depth_cpu.resize(n); net.child_num_cpu.assign(n, 0);
    std::vector<int> permutation(n); std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 random(seed); std::shuffle(permutation.begin(), permutation.end(), random);
    std::vector<std::vector<std::pair<int, int>>> parents(n);
    for (int i = 0; i < n; ++i) {
        const int id = permutation[i];
        net.nodes_cpu[id] = seed * 10000 + i;
        net.node_depth_cpu[id] = i;
        if (i) {
            const int parent = permutation[i - 1];
            parents[id].push_back({parent, net.child_num_cpu[parent]++});
        }
        if (i > 1 && i % 4 == 0) {
            const int parent = permutation[i - 2];
            parents[id].push_back({parent, net.child_num_cpu[parent]++});
        }
    }
    net.par_num_sum_cpu.push_back(0);
    for (const auto& list : parents) {
        net.par_num_cpu.push_back(list.size());
        for (const auto& edge : list) {
            net.par_nodes_cpu.push_back(edge.first); net.currentChildIDX_cpu.push_back(edge.second);
        }
        net.par_num_sum_cpu.push_back(net.par_nodes_cpu.size());
    }
    return net;
}

template<class T> void equal(const T* device, const std::vector<T>& expected, const char* name) {
    std::vector<T> actual(expected.size());
    grotor::cuda_check(cudaMemcpy(actual.data(), device, actual.size() * sizeof(T), cudaMemcpyDeviceToHost), "download FLT layout");
    if (actual != expected) throw std::runtime_error(std::string("FLT layout mismatch: ") + name);
}

void check(bool streamed) {
    cudb::RouterState state;
    std::vector<grotor::flt::Builder> fixtures;
    for (int n : {1, 17, 65, 129, 3, 43}) fixtures.push_back(fixture(n, fixtures.size() + 1));
    auto& topology = state.flt_topology;
    const std::vector<int> emission{4, 1, 5, 2, 0, 3};
    topology.begin(emission, fixtures.size(), 3);
    for (int i = 0; i < emission.size(); ++i) topology.emit(i, i % 3, fixtures[emission[i]]);
    const std::vector<std::vector<int>> batches{{2, 0, 4}, {1}, {5, 3}};
    grotor::flt::Plans plans(topology, batches);
    state.memory.allocate(&state.device.net_ids, fixtures.size() * sizeof(int));
    grotor::flt::Workspace workspace(state.flt, plans, streamed ? 0 : SIZE_MAX, 128);
    workspace.upload(topology, plans);
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (const auto& batch : plans.batches) {
            std::vector<int> nodes, parents, source_slots, child_prefix{0}, parent_prefix{0}, ids, depth, owners, net_ids;
            for (int i = 0; i < batch.nets; ++i) {
                const auto& descriptor = plans.nets[batch.descriptor_begin + i];
                const auto& net = fixtures[descriptor.net_id];
                net_ids.push_back(descriptor.net_id);
                for (int node = 0; node < net.node_index_cnt; ++node) {
                    nodes.push_back(net.nodes_cpu[node]); ids.push_back(descriptor.net_id);
                    depth.push_back(net.node_depth_cpu[node]); owners.push_back(i);
                    child_prefix.push_back(child_prefix.back() + net.child_num_cpu[node]);
                    parent_prefix.push_back(parent_prefix.back() + net.par_num_cpu[node]);
                    for (int p = net.par_num_sum_cpu[node]; p < net.par_num_sum_cpu[node + 1]; ++p) {
                        parents.push_back(net.par_nodes_cpu[p] + descriptor.target_node);
                        source_slots.push_back((node + descriptor.target_node) * 10 + net.currentChildIDX_cpu[p]);
                    }
                }
            }
            std::vector<int> order(nodes.size()); std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return std::make_pair(owners[a], depth[a]) < std::make_pair(owners[b], depth[b]); });
            workspace.assemble(topology, plans, batch, state.device);
            equal(state.flt.nodes, nodes, "coordinates"); equal(state.flt.par_nodes, parents, "parents");
            equal(state.flt.currentChildIDX, source_slots, "child slots");
            equal(state.flt.par_num_sum, parent_prefix, "parent offsets");
            equal(state.flt.child_num_sum, child_prefix, "child offsets");
            std::vector<int> boundaries(batch.depth_slots);
            for (int i = 0; i < batch.nets; ++i) {
                const auto& desc = plans.nets[batch.descriptor_begin + i];
                int pos = desc.target_node;
                for (int d = 0; d <= desc.depths; ++d) {
                    while (pos < desc.target_node + desc.nodes && depth[order[pos]] < d) ++pos;
                    boundaries[desc.depth_begin + d] = pos;
                }
            }
            equal(workspace.depth_offsets, boundaries, "per-DAG depth boundaries");
            equal(state.flt.depth_node, order, "stable depth order"); equal(state.flt.node_net_idx, ids, "node/net map");
            equal(state.device.net_ids, net_ids, "net order");
            equal(state.flt.layer_output, std::vector<int>(nodes.size(), -1), "layer initialization");
            equal(state.flt.costs, std::vector<double>(nodes.size() * 10, 1e22), "cost initialization");
            equal(state.flt.packed_information,
                std::vector<unsigned long long>((batch.children + 1) * 10, 7208259049189261824ULL), "packed cost initialization");
        }
    }
    struct Snapshot {
        int *nodes = nullptr, *parents = nullptr, *depth = nullptr;
        std::vector<int> expected_nodes, expected_parents, expected_depth;
    };
    grotor::DeviceMemory snapshots;
    std::vector<Snapshot> saved(plans.batches.size());
    for (int b = 0; b < plans.batches.size(); ++b) {
        const auto& batch = plans.batches[b]; auto& sample = saved[b];
        std::vector<int> depths, owners;
        for (int i = 0; i < batch.nets; ++i) {
            const auto& desc = plans.nets[batch.descriptor_begin + i];
            const auto& net = fixtures[desc.net_id];
            sample.expected_nodes.insert(sample.expected_nodes.end(), net.nodes_cpu.begin(), net.nodes_cpu.end());
            depths.insert(depths.end(), net.node_depth_cpu.begin(), net.node_depth_cpu.end());
            owners.insert(owners.end(), desc.nodes, i);
            for (int p : net.par_nodes_cpu) sample.expected_parents.push_back(p + desc.target_node);
        }
        sample.expected_depth.resize(batch.nodes); std::iota(sample.expected_depth.begin(), sample.expected_depth.end(), 0);
        std::stable_sort(sample.expected_depth.begin(), sample.expected_depth.end(), [&](int a, int b) { return std::make_pair(owners[a], depths[a]) < std::make_pair(owners[b], depths[b]); });
        snapshots.allocate(&sample.nodes, batch.nodes * sizeof(int));
        snapshots.allocate(&sample.parents, batch.edges * sizeof(int));
        snapshots.allocate(&sample.depth, batch.nodes * sizeof(int));
    }
    if (!streamed) {
        grotor::cuda_check(cudaEventSynchronize(workspace.ready.value), "release unit host topology");
        topology.clear();
    }
    for (int repeat = 0; repeat < 10; ++repeat) {
        for (int b = 0; b < plans.batches.size(); ++b) {
            const auto& batch = plans.batches[b]; auto& sample = saved[b];
            workspace.assemble(topology, plans, batch, state.device);
            grotor::cuda_check(cudaMemcpyAsync(sample.nodes, state.flt.nodes, batch.nodes * sizeof(int), cudaMemcpyDeviceToDevice), "snapshot pipelined nodes");
            grotor::cuda_check(cudaMemcpyAsync(sample.parents, state.flt.par_nodes, batch.edges * sizeof(int), cudaMemcpyDeviceToDevice), "snapshot pipelined edges");
            grotor::cuda_check(cudaMemcpyAsync(sample.depth, state.flt.depth_node, batch.nodes * sizeof(int), cudaMemcpyDeviceToDevice), "snapshot pipelined depths");
        }
    }
    for (const auto& sample : saved) {
        equal(sample.nodes, sample.expected_nodes, "pipelined nodes");
        equal(sample.parents, sample.expected_parents, "pipelined edges");
        equal(sample.depth, sample.expected_depth, "pipelined depths");
    }
    printf("PASS FLT_ASSEMBLY streamed=%d repeats=3 exact_arrays=1\n", int(streamed));
}
}

int main() {
    try {
        grotor::flt::Slabs<grotor::flt::Node> slabs;
        for (int i = 0; i < slabs.capacity + 19; ++i) slabs.push({i, i % 5, 0, 0});
        grotor::flt::Node crossing[24]; slabs.copy(slabs.capacity - 5, 24, crossing);
        for (int i = 0; i < 24; ++i)
            if (crossing[i].coordinate != slabs.capacity - 5 + i) throw std::runtime_error("FLT slab boundary copy failed");
        check(false); check(true);
        printf("FLT_WORKSPACE_CHECK_PASSED\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
