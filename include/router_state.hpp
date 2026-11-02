#pragma once
#include "input.hpp"
#include "cuda_memory.hpp"
#include "rtcore_geometry.hpp"
#include "flt_topology.hpp"
#include <mutex>
#include <stack>
#include <random>
#include <chrono>
#include <memory>
using namespace flute;
namespace cudb {
struct RouterState;
struct RoutingStatistics {
    double total_wirelength = 0, total_overflow_cost = 0, layer_overflow_cost[10]{};
    int total_via_count = 0;
};
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
struct PrefixAccessCounts {
    unsigned long long query_reads=0, prefix_writes=0, queries=0, maintenance_reads=0;
};
struct PrefixQueryCounts { unsigned long long reads=0, queries=0; };
#endif
struct RoutingDeviceView {
    unsigned long long ROUTE_PER_PIN = 12;
    unsigned long long ROUTE_PER_PIN_PHASE2 = 100;
    int DIR = 0, L = 0, X = 0, Y = 0, XY = 0, NET_NUM = 0, PIN_NUM = 0, PIN_NUM_PHASE2 = 0;
    int *x_edge_len = nullptr, *y_edge_len = nullptr, *pin_acc_num = nullptr, *pins = nullptr;
    int *pin_acc_num_phase2 = nullptr;
    double unit_length_wire_cost = 0, unit_via_cost = 0, of_cost_scale = 1;
    double *unit_length_short_costs = nullptr, *layer_min_len = nullptr;
    float *vcost = nullptr, *wcost = nullptr, *capacity = nullptr, *demand = nullptr;
    int *cross_points = nullptr;
    bool *is_of_net = nullptr;
    int *of_edge_sum = nullptr, *routes = nullptr, *timestamp = nullptr, *pre_demand = nullptr;
    int *routes_phase2 = nullptr, *ripup_flag = nullptr;
    int *last = nullptr, all_track_cnt = 0, *idx2track = nullptr;
    double *presum = nullptr, *cost_bit = nullptr;
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
    PrefixAccessCounts* prefix_access = nullptr;
    PrefixQueryCounts* prefix_queries = nullptr;
    int prefix_query_mode = 0; // Prefix access benchmark: dense, BIT, BIT with common-prefix pruning.
#endif
    int *bit_track_offsets = nullptr;
    unsigned int *dirty_wire = nullptr, *dirty_via = nullptr, *dirty_tree = nullptr;
    unsigned int *dirty_tracks = nullptr;
    int *net_ids = nullptr;
    bool *congestion = nullptr;
    float *congestion_xsum = nullptr, *congestion_ysum = nullptr;
    RoutingStatistics *statistics = nullptr;
};
struct FltDeviceView {
    double *cost_edges = nullptr;
    int *best_change = nullptr;
    int edge_cnt = 0;
    int *macroBorder = nullptr;
    int *node_cnt_sum = nullptr;
    int *nodes = nullptr;
    int *par_nodes = nullptr;
    int *from = nullptr;
    int *layer_range = nullptr;
    int *child_num = nullptr;
    int *child_num_sum = nullptr;
    int *in_degree = nullptr;
    int *currentChildIDX = nullptr;
    int *par_num = nullptr;
    int *par_num_sum = nullptr;
    int *locks = nullptr;
    double *childCosts = nullptr;
    int *childCosts_road = nullptr;
    unsigned long long *packed_information = nullptr;
    unsigned long long *best_path = nullptr;
    int *layer_output = nullptr;
    double *costs = nullptr;
    int *fixed_layers = nullptr;
    int *node_net_idx = nullptr;
    int *node_net_idx2 = nullptr;
    int *lock_gpu = nullptr;
    int *node_depth = nullptr;
    int *net_depth = nullptr;
    int *batch_depth = nullptr;
    int *depth_node = nullptr;
    int *depth_node_cnt = nullptr;
    int log = 0;
};
struct net {
    void construct_rsmt(RouterState& state, bool, bool);
    void generate_detours(RouterState& state, grotor::flt::Builder& topology, bool* const & congestionView_cpu, 
                          float* const & congestionView_xsum_cpu, 
                          float* const & congestionView_ysum_cpu,
                          bool construct_segments = true, bool display = false);
    void generate_detours_reconstruct(RouterState& state, grotor::flt::Builder& topology, bool* const & congestionView_cpu, 
                          float* const & congestionView_xsum_cpu, 
                          float* const & congestionView_ysum_cpu,
                          bool construct_segments = true, bool display = false, int net_id = 0);
    void calc_hpwl(RouterState& state);
    
    int minx, maxx, miny, maxy, hpwl, original_net_id;
    vector<int> pins;
    vector<vector<int>> rsmt;
    vector<pair<int, int>> rsmt_h_segments, rsmt_v_segments;
    vector<int> points;
    int MAX_LAYER=10;
    int select_root = 0;

    Tree tree;
    unordered_map<int, int> layer;
};

// A routing run owns its persistent storage and stage workspaces. Views are
// copied into launch parameters, so kernels never depend on mutable globals.
struct RouterState {
    RoutingDeviceView device;
    FltDeviceView flt;
    grotor::flt::Topology flt_topology;
    grotor::DeviceMemory memory;
    grotor::HostMemory host;
    RoutingStatistics statistics;
    bool costs_initialized = false;
    int batch_sparsity_override = 1;
    bool rtcore_batching = false;
    grotor::rtcore::PreparedGeometry rtcore_geometry;
    float rtcore_z_gap = 0x1p-16f;
    float rtcore_margin = .1f;
    bool rtcore_coplanar = false;
    bool rtcore_compact_points = true;
    bool validate_rtcore_batches = false;
    std::string rtcore_ptx;
    int global_timestamp = 0;
    vector<net> nets;
    vector<vector<net>> tmp_nets;
    vector<int> pin_cnt_sum_cpu, ripup_flag_cpu, pin_cnt_sum_phase2_cpu;
    std::unique_ptr<int[]> routes_cpu, routes_phase2_cpu;
    bool profile_stages = false;
    struct FltHostWorkspace {
        bool *congestionView_cpu = nullptr;
        float *congestionView_xsum_cpu = nullptr;
        float *congestionView_ysum_cpu = nullptr;
    } packing;

    void download_routes(bool second_phase) {
        auto count = second_phase ? device.ROUTE_PER_PIN_PHASE2 * device.PIN_NUM_PHASE2
                                  : device.ROUTE_PER_PIN * device.PIN_NUM;
        auto& destination = second_phase ? routes_phase2_cpu : routes_cpu;
        destination.reset(new int[count + 1]);
        grotor::cuda_check(cudaMemcpy(destination.get(), second_phase ? device.routes_phase2 : device.routes,
                                     (count + 1) * sizeof(int), cudaMemcpyDeviceToHost), "download routes");
    }

    // Output uses only immutable route snapshots, offsets and input metadata.
    // Release solver storage while the final output task is still writing.
    void release_workspaces() {
        memory.clear();
        host.clear();
        vector<net>().swap(nets);
        vector<vector<net>>().swap(tmp_nets);
    }
};
const auto batch_clock_origin = std::chrono::steady_clock::now();
struct RoutingDagInterval {
    long long start_us, end_us;
};

struct RoutingDagTimer {
    std::chrono::steady_clock::time_point start;

    static void synchronize() {
        auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }

    RoutingDagTimer() {
        synchronize(); // Exclude GPU work preceding this region.
        start = std::chrono::steady_clock::now();
    }

    RoutingDagInterval stop() {
        synchronize(); // Include completion of the last demand-commit kernel.
        const auto end = std::chrono::steady_clock::now();
        const auto start_us = std::chrono::duration_cast<std::chrono::microseconds>(start - batch_clock_origin).count();
        const auto end_us = std::chrono::duration_cast<std::chrono::microseconds>(end - batch_clock_origin).count();
        return {static_cast<long long>(start_us), static_cast<long long>(end_us)};
    }

    void finish(const char *stage, int batch_index, size_t net_count, size_t batch_count) {
        const auto interval = stop();
        printf("DAG_TIMING stage=%s batch_index=%d start_us=%lld end_us=%lld elapsed_ms=%.6f nets=%zu batches=%zu\n",
               stage, batch_index, interval.start_us, interval.end_us,
               (interval.end_us - interval.start_us) / 1000.0, net_count, batch_count);
    }
};

#define IDX(l, x, y) ((l) * X * Y + (x) * Y + (y))
#define THREAD_NUM 512
#define BLOCK_NUM(n) ((n) / THREAD_NUM + 1)
#define BLOCK_CNT(tot, thread_cnt) ((tot) / (thread_cnt) + ((tot) % (thread_cnt) > 0))
#define INF 1e22
} // namespace cudb
using namespace cudb;
