#pragma once

#include "preprocess.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "batching/generate_batches.hpp"
#include "cost/full_update.cuh"
#include "cost/prefix_scan.cuh"
#include "routing/demand.cuh"

#include "routing/lshape_warp.cuh"
#ifdef GROTOR_ENABLE_NVSHMEM
#include "multigpu/runtime.cuh"
#endif
#ifdef GROTOR_SERIAL_DAG
#include "routing/lshape_thread.cuh"
#endif

namespace Lshape_route {
void Lshape_route(RouterState& state, vector<int> &nets2route) {
    auto& L = state.device.L;
    auto& XY = state.device.XY;
    auto& all_track_cnt = state.device.all_track_cnt;
    auto& net_ids = state.device.net_ids;
    auto& global_timestamp = state.global_timestamp;
    auto& nets = state.nets;
    double Lshape_start_time = elapsed_time();

    const auto topology_start = std::chrono::steady_clock::now();
    grotor::batching::prepare_geometry(state, nets2route, 8);
    if(LOG) printf("[%5.1f] FLUTE", elapsed_time()); cerr << endl;
    #pragma omp parallel for num_threads(8)
    for(int i = 0; i < nets2route.size(); i++) {
        bool log_flag = false;
        nets[nets2route[i]].construct_rsmt(state, log_flag);
        if (state.rtcore_batching) {
            int lane = 0;
#ifdef _OPENMP
            lane = omp_get_thread_num();
#endif
            grotor::batching::emit_geometry(state, i, lane, nets[nets2route[i]]);
        }
    }
    if(LOG) printf("[%5.1f] FLUTE END", elapsed_time()); cerr << endl;
    printf("TOPOLOGY_TIMING stage=lshape elapsed_ms=%.6f geometry_fused=%d\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-topology_start).count(), int(grotor::batching::geometry_fused(state)));


    //generate batches
    
    if(LOG) printf("[%5.1f] SORT\n", elapsed_time());
    sort(nets2route.begin(), nets2route.end(), [&] (int l, int r) {
        return nets[l].hpwl > nets[r].hpwl;
    });
    if(LOG) printf("[%5.1f] SORT END\n", elapsed_time());

    vector<vector<int>> batches;
    if (state.rtcore_batching) {
#ifdef GROTOR_ENABLE_RTCORE
        grotor::batching::report_stage("lshape", grotor::batching::partitioned(state, nets2route.size()) ? "partitioned" : "rounds", state, nets2route.size());
        auto batch_run = grotor::batching::generate_batches_rtcore(state, nets2route, 1000000);
        batches = grotor::batching::finish(state, nets2route, 1000000, std::move(batch_run));
#elif defined(GROTOR_ENABLE_HIPRT)
        batches = grotor::hiprt_batching::generate(state, nets2route, 1000000, "lshape");
#else
        throw std::runtime_error("RT-Core build required");
#endif
    } else batches = generate_batches_cpu_timed(state, nets2route, 1000000);
    reverse(batches.begin(), batches.end());

    vector<int> batch_cnt_sum(batches.size() + 1, 0);
    for(int i = 0; i < batches.size(); i++) {
        batch_cnt_sum[i + 1] = batch_cnt_sum[i] + batches[i].size();
        for(int j = 0; j < batches[i].size(); j++)
            nets2route[batch_cnt_sum[i] + j] = batches[i][j];
    }
    int pin_cnt = 0, net_cnt = nets2route.size();
    for(auto net_id : nets2route) pin_cnt += nets[net_id].pins.size();
    int *node_cnt_sum_cpu = state.host.allocate<int>(net_cnt + 1, true);
    int *nodes_cpu = state.host.allocate<int>(pin_cnt * 2);
    int *par_nodes_cpu = state.host.allocate<int>(pin_cnt * 2);


    if(LOG) printf("[%5.1f] DFS\n", elapsed_time());
    for(int id = 0; id < net_cnt; id++) {
        auto &graph = nets[nets2route[id]].rsmt;
        node_cnt_sum_cpu[id + 1] = node_cnt_sum_cpu[id] + graph.back().size();
    }
    #pragma omp parallel for num_threads(8)
    for(int id = 0; id < net_cnt; id++) {
        auto &graph = nets[nets2route[id]].rsmt;
        int node_idx_cnt = 0;
        function<void(int, int, int)> dfs = [&] (int x, int par, int par_node_idx) {
            int node_idx = node_idx_cnt++;
            nodes_cpu[node_cnt_sum_cpu[id] + node_idx] = graph.back()[x];
            par_nodes_cpu[node_cnt_sum_cpu[id] + node_idx] = par_node_idx;
            for(auto e : graph[x]) if(e != par) dfs(e, x, node_idx);
        };
        dfs(0, -1, -1);
        assert((node_cnt_sum_cpu[id + 1] - node_cnt_sum_cpu[id]) == graph.back().size());
    }
    if(LOG) printf("[%5.1f] DFS END\n", elapsed_time());
#ifdef GROTOR_MULTI_LEGACY_LSHAPE
    // Do not allocate/use the optimized warp-per-layer executor on N=1.
    // All L-shape batches run through the same original distributed path.
    grotor::multi::session->lshape_all(state,batches,nets2route,node_cnt_sum_cpu,nodes_cpu,par_nodes_cpu);
    state.host.release(node_cnt_sum_cpu);
    state.host.release(nodes_cpu);
    state.host.release(par_nodes_cpu);
    Lshape_time=elapsed_time()-Lshape_start_time;
    printf("Lshape END.   "); print_GPU_memory_usage();
    return;
#endif
    int max_num_nodes = 0;
    for(int i = 0; i < batches.size(); i++)
#ifdef GROTOR_ENABLE_NVSHMEM
        if (i + 1 < batches.size())
#endif
        max_num_nodes = max(max_num_nodes, node_cnt_sum_cpu[batch_cnt_sum[i + 1]] - node_cnt_sum_cpu[batch_cnt_sum[i]]);

    double *dist = nullptr;
    int *from = nullptr, *layer_range = nullptr, *node_cnt_sum = nullptr, *nodes = nullptr, *par_nodes = nullptr;

    print_GPU_memory_usage();
    grotor::cuda_check(cudaMemcpy(net_ids, nets2route.data(), net_cnt * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");

    state.memory.allocate(&node_cnt_sum, (net_cnt + 1) * sizeof(int));
    grotor::cuda_check(cudaMemcpy(node_cnt_sum, node_cnt_sum_cpu, (net_cnt + 1) * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    state.memory.allocate(&nodes, node_cnt_sum_cpu[net_cnt] * sizeof(int));
    grotor::cuda_check(cudaMemcpy(nodes, nodes_cpu, node_cnt_sum_cpu[net_cnt] * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");
    state.memory.allocate(&par_nodes, node_cnt_sum_cpu[net_cnt] * sizeof(int));
    grotor::cuda_check(cudaMemcpy(par_nodes, par_nodes_cpu, node_cnt_sum_cpu[net_cnt] * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy");

    print_GPU_memory_usage();
    state.memory.allocate(&dist, max_num_nodes * L * L * sizeof(double));
    state.memory.allocate(&from, max_num_nodes * L * L * sizeof(int));
    state.memory.allocate(&layer_range, node_cnt_sum_cpu[net_cnt] * sizeof(int));
    
    print_GPU_memory_usage();
    
    if(LOG) printf("[%5.1f] Lshape_cuda\n", elapsed_time());
#ifdef GROTOR_DENSE_PRESUM
    // Legacy L-shape starts each batch with resident, current costs and
    // charges its post-commit refresh to that batch (FLT refreshes before DP).
    graph::refresh_cost(state);
#endif
    RoutingDagTimer dag_timer;
    auto routing_view = state.device;
#ifndef GROTOR_ENABLE_NVSHMEM
    routing_view.pre_demand = nullptr;
#endif
     for(int i = 0; i < batches.size(); i++) {
#ifdef GROTOR_ENABLE_NVSHMEM
        if (i + 1 == batches.size()) {
            grotor::multi::session->lshape(state, batches[i], node_cnt_sum_cpu + batch_cnt_sum[i],
                nodes_cpu, par_nodes_cpu, i, batches.size());
            continue;
        }
#endif
        global_timestamp++;
#ifndef GROTOR_DENSE_PRESUM
        graph::refresh_cost(state);
#endif
#ifdef GROTOR_SERIAL_DAG
        Lshape_route_cuda<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>>(routing_view, batches[i].size(), batch_cnt_sum[i], node_cnt_sum, nodes, par_nodes, dist, from, layer_range, global_timestamp);
#else
        Lshape_route_warp<<<(batches[i].size() + 3) / 4, 128>>>(routing_view, batches[i].size(), batch_cnt_sum[i], node_cnt_sum, nodes, par_nodes, dist, from, layer_range, global_timestamp);
#endif
#ifdef GROTOR_ENABLE_NVSHMEM
        graph::batch_wire_update(state, global_timestamp);
        graph::commit_via_demand<<<BLOCK_NUM(batches[i].size()), THREAD_NUM>>>(state.device, batches[i].size(), batch_cnt_sum[i], global_timestamp);
#else
        graph::commit_batch_demand(state.device, batches[i].size(), batch_cnt_sum[i], global_timestamp);
#endif
#ifdef GROTOR_DENSE_PRESUM
        graph::refresh_cost(state);
#endif
    }
    dag_timer.finish("lshape", -1, nets2route.size(), batches.size());
    if(LOG) printf("[%5.1f] Lshape_cuda END\n", elapsed_time());

    state.memory.release(node_cnt_sum);
    state.memory.release(nodes);
    state.memory.release(par_nodes);
    state.memory.release(dist);
    state.memory.release(from);
    state.memory.release(layer_range);

    state.host.release(node_cnt_sum_cpu);
    state.host.release(nodes_cpu);
    state.host.release(par_nodes_cpu);
    printf("Lshape END.   ");
    print_GPU_memory_usage();

    Lshape_time = elapsed_time() - Lshape_start_time;
}

}
