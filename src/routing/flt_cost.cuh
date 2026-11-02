#pragma once

#include "preprocess.hpp"
#include "batching/generate_batches.hpp"
#include "cost/full_update.cuh"
#include "cost/prefix_scan.cuh"
#include "routing/demand.cuh"
#include "routing/flt_workspace.cuh"
#include <omp.h>
#include <atomic>
#include <unistd.h>
#include <queue>
#define INF_LAYER 20
#define MAX_LAYER 10
#define MIN_ROUTE_LAYER 1
#define MAX_DEPTH 5000
// node_cnt*81




namespace Lshape_route_detour {
#define COST_BITS 32
#define COST_SHIFT (64 - COST_BITS)
__device__ int32_t float_to_ordered_int32(float f)
{
    int32_t i = __float_as_int(f);
    return (i >= 0) ? i : i ^ 0x7FFFFFFF;
}

__device__ float ordered_int32_to_float(int32_t i)
{
    return __int_as_float((i >= 0) ? i : i ^ 0x7FFFFFFF);
}

__device__ unsigned long long packCostAndNodeId(float cost, int node_id) {
    int32_t cost_int = float_to_ordered_int32(cost);
    unsigned long long packed = ((unsigned long long)cost_int << COST_SHIFT) | (unsigned int)node_id;
    return packed;
}

__device__ void unpackCostAndNodeId(unsigned long long packed, double* cost, int* node_id) {
    int32_t cost_int = (int32_t)(packed >> COST_SHIFT);
    *cost = (double)ordered_int32_to_float(cost_int);
    *node_id = (int)(packed & ((1ULL << COST_SHIFT) - 1));
}

constexpr unsigned long long init_cost_with_road = 7208259049189261824;

__global__ void precompute_cost_edge(RoutingDeviceView view, FltDeviceView flt, int par_sum) {
    auto& DIR = view.DIR;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& vcost = view.vcost;
    auto& cost_edges = flt.cost_edges;
    auto& best_change = flt.best_change;
    auto& nodes = flt.nodes;
    auto& par_nodes = flt.par_nodes;
    auto& currentChildIDX = flt.currentChildIDX;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int edge_id = index / 9;
    if(edge_id>=par_sum) return;
    int target_node_idx = par_nodes[edge_id];
    int from_layer = index%9;
    int from_node_idx = currentChildIDX[edge_id] / 10;
    int tox = nodes[target_node_idx] / Y % X, toy = nodes[target_node_idx] % Y;
    int fromx = nodes[from_node_idx] / Y % X, fromy = nodes[from_node_idx] % Y;
    if(((from_layer & 1 ^ DIR) == 0 && fromy != toy)||((from_layer & 1 ^ DIR) == 1 && fromx != tox))
    {
        for(int to_layer = 0; to_layer < 9; to_layer++)
        cost_edges[edge_id * 81 + from_layer * 9 + to_layer] = INF;
        return;
    }
    assert(fromx == tox || fromy == toy);
    int distance = abs(tox - fromx) + abs(toy - fromy);
    if(distance==0) return;
    int dd=distance / min(distance, 20);
    bool isVertical = (fromx == tox);
    int start_search_pos = isVertical?min(fromy, toy):min(fromx, tox);
    int end_search_pos = isVertical?max(fromy, toy):max(fromx, tox);
    for(int to_layer = 0; to_layer < 9; to_layer++)
    {
        if((to_layer%2)!=(from_layer%2)) continue;
        double min_costt = INF;
        int decide_pos = -1;
        assert(start_search_pos >= 0);
        assert(end_search_pos >= 0);
        assert(dd>0);
        for(int search_pos = start_search_pos+dd; ; search_pos+=dd)
        {
            if(search_pos >= end_search_pos)
            {
                break;
            }
            int search_x, search_y;
            if(isVertical)
            {
                search_x = fromx;
                search_y = search_pos;
            }
            else{
                search_x = search_pos;
                search_y = fromy;
            }
            assert(search_x>=0);
            assert(search_y>=0);
            double via_cost_tmp = 0;
            for(int ll = min(from_layer, to_layer); ll < max(from_layer, to_layer); ll++)
            {
                via_cost_tmp += vcost[IDX(ll, search_x, search_y)];
            }
            double cost_tmp = via_cost_tmp;
            if(isVertical)
            {
                cost_tmp += (graph::wire_segment_cost(view, from_layer, fromx, fromx, min(fromy, search_y), max(fromy, search_y))
                            + graph::wire_segment_cost(view, to_layer, tox, tox, min(toy, search_y), max(toy, search_y)));
            }else{
                cost_tmp += (graph::wire_segment_cost(view, from_layer, min(fromx, search_x), max(fromx, search_x), fromy, fromy)
                            + graph::wire_segment_cost(view, to_layer, min(tox, search_x), max(tox, search_x), toy, toy));
            }
            if(cost_tmp < min_costt)
            {
                min_costt = cost_tmp;
                decide_pos = search_pos;
            }
        }
        cost_edges[edge_id * 81 + from_layer * 9 + to_layer] = min_costt;
        best_change[edge_id * 81 + from_layer * 9 + to_layer] = decide_pos;
    }
}


} // namespace Lshape_route_detour
