#pragma once
#include "routing/flt_cost.cuh"

namespace Lshape_route_detour {
// Preserve the scalar node arithmetic and packed integer candidate ordering.
__device__ __forceinline__ void solve_flt_node(RoutingDeviceView view, FltDeviceView flt, int node_idx) {

    auto& DIR = view.DIR;
    auto& L = view.L;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& vcost = view.vcost;
    auto& cost_edges = flt.cost_edges;
    auto& nodes = flt.nodes;
    auto& par_nodes = flt.par_nodes;
    auto& child_num_sum = flt.child_num_sum;
    auto& currentChildIDX = flt.currentChildIDX;
    auto& par_num_sum = flt.par_num_sum;
    auto& packed_information = flt.packed_information;
    auto& best_path = flt.best_path;
    auto& costs = flt.costs;
    auto& depth_node = flt.depth_node;
    int parent_num_cur = par_num_sum[node_idx + 1]-par_num_sum[node_idx];
    int fixed_layer_low = 1 + nodes[node_idx] / X / Y;
    int x = nodes[node_idx] / Y % X, y = nodes[node_idx] % Y;
    int fixed_layer_high = fixed_layer_low==10?0:fixed_layer_low;
    int cur_child_num = child_num_sum[node_idx + 1]-child_num_sum[node_idx];

    unsigned long long *cur_best_path = best_path + child_num_sum[node_idx] * MAX_LAYER;
    unsigned long long *cur_packed_information = packed_information + child_num_sum[node_idx] * MAX_LAYER;
    double minChildCosts[6];
    assert(cur_child_num <= 6);
    unsigned long long bestPaths[6];
    for (int lowLayerIndex = MIN_ROUTE_LAYER; lowLayerIndex <= fixed_layer_low; lowLayerIndex++) {
        for(int cid = 0; cid<cur_child_num; cid++)
        {
            minChildCosts[cid] = INF;
        }
        double via_cost = 0;
        for (int layerIndex = lowLayerIndex; layerIndex < (L + 1); layerIndex++) {
            if(layerIndex>lowLayerIndex)
            {   
                via_cost += vcost[IDX(layerIndex - 2, x, y)];
            }
            for (int childIndex = 0; childIndex < cur_child_num; childIndex++) {
                double cost;
                int nodeddd;
                unpackCostAndNodeId(cur_packed_information[childIndex * MAX_LAYER + layerIndex], &cost, &nodeddd);
                double cur_child_cost = cost;
                assert(nodeddd >= 0);
                if (cur_child_cost < minChildCosts[childIndex]) {
                    minChildCosts[childIndex] = cur_child_cost;
                    bestPaths[childIndex] = (unsigned long long)nodeddd * MAX_LAYER + layerIndex;
                }
            }
            if (layerIndex >= fixed_layer_high) {
                double cost = via_cost;
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cost += minChildCosts[childIndex];
                }
                if (cost<INF && cost < costs[node_idx * MAX_LAYER+layerIndex]) {
                    costs[node_idx * MAX_LAYER + layerIndex] = cost;
                    for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                    {
                        cur_best_path[childIndex * MAX_LAYER + layerIndex] = bestPaths[childIndex];
                    }
                }
            }
        }
        for (int layerIndex = (L + 1) - 2; layerIndex >= lowLayerIndex; layerIndex--) {//
            if (costs[node_idx * MAX_LAYER + layerIndex + 1] < costs[node_idx * MAX_LAYER + layerIndex]) {
                costs[node_idx * MAX_LAYER + layerIndex] = costs[node_idx * MAX_LAYER + layerIndex + 1];
                for (int childIndex = 0; childIndex < cur_child_num; childIndex++)
                {
                    cur_best_path[childIndex * MAX_LAYER + layerIndex] = cur_best_path[childIndex * MAX_LAYER + layerIndex + 1];
                }
            }
        }
    }
    int node_x = nodes[node_idx] / Y % X, node_y = nodes[node_idx] % Y; 
    unsigned long long packed_information_cache[11];
    for(int par_id = 0; par_id < parent_num_cur; par_id++)
    {
        int parent_IDX = par_nodes[par_num_sum[node_idx] + par_id];
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_idx] + par_id] % 10;
        int parent_shift_base = child_num_sum[parent_IDX] * MAX_LAYER;
        unsigned long long *parent_packed_information = packed_information + parent_shift_base;
        int px = nodes[parent_IDX] / Y % X, py = nodes[parent_IDX] % Y;
        int minx = min(node_x, px), maxx = max(node_x, px), miny = min(node_y, py), maxy = max(node_y, py);
        assert(px == node_x || py == node_y);
        
        for(int target_layer=0; target_layer<MAX_LAYER; target_layer++)
        {
            if((target_layer<MIN_ROUTE_LAYER)||(target_layer>=MAX_LAYER)) continue;
            int index_ = child_index_of_current_node * MAX_LAYER + target_layer;
            packed_information_cache[target_layer] = init_cost_with_road;
        }
        for(int layer = MIN_ROUTE_LAYER; layer < MAX_LAYER; layer++) // output=1-9
        {
            if((layer & 1 ^ DIR) == 1 && node_y != py) continue;
            if((layer & 1 ^ DIR) == 0 && node_x != px) continue;
            double cost_base = costs[node_idx * MAX_LAYER + layer];
            assert(cost_base<INF);
            for(int target_layer = MIN_ROUTE_LAYER + (1 - layer%2); target_layer< MAX_LAYER; target_layer+=2)
            {
                if((target_layer < MIN_ROUTE_LAYER) || (target_layer >= MAX_LAYER)) continue;
                double cost2 = INF;
                int possibility_choosed = 82;
                if((target_layer == layer))
                {
                    cost2 = cost_base + graph::wire_segment_cost(view, layer-1, minx, maxx, miny, maxy);
                }
                else if(abs(node_x - px) + abs(node_y - py)>4){
                    int possibility_sequence = (layer-1) * 9 + (target_layer-1);
                    double c_e = cost_edges[(par_num_sum[node_idx] + par_id) * 81 + possibility_sequence];
                    if(c_e < INF){
                        cost2 = cost_base + c_e;
                        possibility_choosed = possibility_sequence;
                    } else{
                        continue;
                    }
                }
                else{
                    continue;
                }
                volatile unsigned long long new_value = packCostAndNodeId(cost2, node_idx * 100 + possibility_choosed);
                if(new_value<packed_information_cache[target_layer])
                {
                    packed_information_cache[target_layer] = new_value;
                }
            }
        }
        
        for(int target_layer = MIN_ROUTE_LAYER; target_layer< MAX_LAYER; target_layer++)
        {
            if((target_layer<MIN_ROUTE_LAYER)||(target_layer>=MAX_LAYER)) continue;
            int index_ = child_index_of_current_node * MAX_LAYER + target_layer;
            atomicMin(parent_packed_information + index_, packed_information_cache[target_layer]);
        }   
    }
}

__device__ __forceinline__ void trace_flt_node(RoutingDeviceView view, FltDeviceView flt, int node_id, int stamp) {

    auto& ROUTE_PER_PIN_PHASE2 = view.ROUTE_PER_PIN_PHASE2;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& NET_NUM = view.NET_NUM;
    auto& pin_acc_num = view.pin_acc_num;
    auto& pin_acc_num_phase2 = view.pin_acc_num_phase2;
    auto& routes_phase2 = view.routes_phase2;
    auto& best_change = flt.best_change;
    auto& nodes = flt.nodes;
    auto& par_nodes = flt.par_nodes;
    auto& child_num_sum = flt.child_num_sum;
    auto& currentChildIDX = flt.currentChildIDX;
    auto& par_num_sum = flt.par_num_sum;
    auto& best_path = flt.best_path;
    auto& layer_output = flt.layer_output;
    auto& costs = flt.costs;
    auto& node_net_idx = flt.node_net_idx;
    auto& depth_node = flt.depth_node;
    int net_id = node_net_idx[node_id];
    assert(net_id<NET_NUM);
    assert(pin_acc_num_phase2[net_id]>=0);
    int *net_routes = routes_phase2 + pin_acc_num_phase2[net_id] * ROUTE_PER_PIN_PHASE2;

    unsigned long long *cur_best_path = best_path + child_num_sum[node_id] * MAX_LAYER;

    int l = nodes[node_id] / Y / X;
    int cur_x = nodes[node_id] / Y % X, cur_y = nodes[node_id] % Y;
    if(par_num_sum[node_id + 1]-par_num_sum[node_id]==0)
    {
        int min_layer = 0;
        double min_cost = costs[node_id * MAX_LAYER];
        for(int layer = 1; layer < MAX_LAYER; layer++)
        {
            if(costs[node_id * MAX_LAYER + layer] < min_cost)
            {
                min_cost = costs[ node_id * MAX_LAYER + layer];
                min_layer = layer;
            }
        }
        layer_output[node_id] = min_layer;
        net_routes[0] = 1;
    } else{
        int par_layer = -1;
        int par_idx = -1;
        int par_sequence = -1;
        for(int par_id = 0; par_id< par_num_sum[node_id + 1]-par_num_sum[node_id]; par_id++)
        {
            int edge_id = par_num_sum[node_id]+par_id;
            int par_node = par_nodes[par_num_sum[node_id]+par_id];
            if(layer_output[par_node]>0)
            {
                par_idx = par_node;
                par_sequence = par_id;
                par_layer = layer_output[par_node];
                int child_index_of_current_node = currentChildIDX[par_num_sum[node_id]+par_sequence]%10;
                unsigned long long *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
                unsigned long long path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
                unsigned long long child_idx = path / MAX_LAYER / 100;
                unsigned long long possibility_choosed = (path / MAX_LAYER) % 100;
                if(child_idx == node_id)
                {
                    if(possibility_choosed==82)
                    {
                        layer_output[node_id] = path % MAX_LAYER;
                    }
                    else{
                        layer_output[node_id] = possibility_choosed / 9 + 1;
                    }
                    int px = nodes[par_idx] / Y % X, py = nodes[par_idx] % Y;
                    assert(px==cur_x||py==cur_y); 
                    if(px==cur_x && cur_y!=py)
                    {
                        if(possibility_choosed==82)
                        {
                            graph::atomic_add_unit_demand_wire_segment(view, layer_output[node_id] - 1, px, px, min(py,cur_y), max(py,cur_y), stamp);
                            int idd1 = atomicAdd(net_routes,2);
                            assert(idd1 + 1 < ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
                            net_routes[idd1] = IDX(layer_output[node_id] - 1, px, min(py,cur_y));
                            net_routes[idd1 + 1] = IDX(layer_output[node_id] - 1, px, max(py,cur_y));
                            assert(IDX(layer_output[node_id] - 1, px, min(py,cur_y))>=0);
                            assert(idd1 + 1<ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
                        }
                        else{
                            assert(possibility_choosed<81);
                            int from_layer = possibility_choosed / 9;
                            int to_layer = possibility_choosed % 9;
                            assert((path % MAX_LAYER)==(to_layer + 1));
                            int change_pos = best_change[edge_id * 81 + from_layer * 9 + to_layer];

                            assert(change_pos>min(cur_y, py));
                            assert(change_pos<max(cur_y, py));
                            assert((from_layer%2)==(to_layer%2));
                            assert((from_layer%2)==0);// vertical
                            graph::atomic_add_unit_demand_wire_segment(view, from_layer, cur_x, cur_x, min(cur_y,change_pos), max(cur_y,change_pos), stamp);
                            graph::atomic_add_unit_demand_wire_segment(view, to_layer, cur_x, cur_x, min(py,change_pos), max(py,change_pos), stamp);
                            int idd1 = atomicAdd(net_routes,6);
                            net_routes[idd1] = IDX(from_layer, cur_x, min(cur_y,change_pos));
                            net_routes[idd1 + 1] = IDX(from_layer, cur_x,  max(cur_y,change_pos));
                            assert(from_layer!=to_layer);
                            net_routes[idd1 + 2] = IDX(min(from_layer, to_layer), cur_x, change_pos);
                            net_routes[idd1 + 3] = IDX(max(from_layer, to_layer), cur_x, change_pos);
                            net_routes[idd1 + 4] = IDX(to_layer, cur_x, min(py,change_pos));
                            net_routes[idd1 + 5] = IDX(to_layer, cur_x,  max(py,change_pos));
                            assert(idd1 + 5<ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1] - pin_acc_num[net_id]));
                        }
                    }
                    else if(py==cur_y && cur_x != px)
                    {
                        if(possibility_choosed==82)
                        {
                            graph::atomic_add_unit_demand_wire_segment(view, layer_output[node_id] - 1, min(px,cur_x), max(px,cur_x), py, py, stamp);
                            int idd1 = atomicAdd(net_routes,2);
                            net_routes[idd1] = IDX(layer_output[node_id] - 1, min(px, cur_x), py);
                            net_routes[idd1 + 1] = IDX(layer_output[node_id] - 1, max(px, cur_x), py);
                            assert(IDX(layer_output[node_id] - 1, min(px,cur_x), py) >= 0);
                            assert(idd1 + 1 < ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
                        }
                        else{
                            assert(possibility_choosed<81);
                            int from_layer = possibility_choosed / 9;
                            int to_layer = possibility_choosed % 9;
                            assert((path % MAX_LAYER)==(to_layer + 1));
                            int change_pos = best_change[edge_id * 81 + from_layer * 9 + to_layer];
                            assert(change_pos > min(cur_x, px));
                            assert(change_pos < max(cur_x, px));
                            assert((from_layer % 2)==(to_layer % 2));
                            assert((from_layer % 2)==1);
                            graph::atomic_add_unit_demand_wire_segment(view, from_layer, min(cur_x, change_pos), max(cur_x, change_pos), cur_y, cur_y, stamp);
                            graph::atomic_add_unit_demand_wire_segment(view, to_layer, min(px, change_pos), max(px, change_pos), cur_y, cur_y, stamp);
                            int idd1 = atomicAdd(net_routes, 6);
                            net_routes[idd1] = IDX(from_layer, min(cur_x, change_pos), cur_y);
                            net_routes[idd1 + 1] = IDX(from_layer, max(cur_x, change_pos), cur_y);
                            assert(from_layer!=to_layer);
                            net_routes[idd1 + 2] = IDX(min(from_layer, to_layer), change_pos, cur_y);
                            net_routes[idd1 + 3] = IDX(max(from_layer, to_layer), change_pos, cur_y);
                            net_routes[idd1 + 4] = IDX(to_layer, min(px,change_pos), cur_y);
                            net_routes[idd1 + 5] = IDX(to_layer, max(px,change_pos), cur_y);
                            assert(idd1 + 5 < ROUTE_PER_PIN_PHASE2 * (pin_acc_num[net_id + 1] - pin_acc_num[net_id]));
                        }
                    }
                    break;
                }else{
                    layer_output[node_id] = -1;
                }
            }
        }
        if(par_layer==-1)
        {
            layer_output[node_id] = -1;
            return;
        }
        int child_index_of_current_node = currentChildIDX[par_num_sum[node_id] + par_sequence]%10;
        unsigned long long *par_best_path = best_path + child_num_sum[par_idx] * MAX_LAYER;
        unsigned long long path = par_best_path[child_index_of_current_node * MAX_LAYER + par_layer];
        unsigned long long child_idx = path / MAX_LAYER / 100;
        if( child_idx != node_id)
        {
            layer_output[node_id] = -1;
            return;
        }
    }
    int num_child = child_num_sum[node_id + 1] - child_num_sum[node_id];
    int minl = l + 1; //l:0-8-->2nd - 10th layers
    int maxl = (l + 1)==MAX_LAYER?1:minl;
    minl = min(minl,layer_output[node_id]);
    maxl = max(maxl,layer_output[node_id]);
    assert(num_child>=0);
    if(num_child>0)
    {
        for(int child_id=0; child_id<num_child; child_id++)
        {
            int layer_of_child = cur_best_path[child_id * MAX_LAYER + layer_output[node_id]] % MAX_LAYER;
            assert(layer_of_child!=0);
            minl = min(layer_of_child, minl);
            maxl = max(layer_of_child, maxl);
        }
    }
    if(minl<maxl)
    {
        int idd1 = atomicAdd(net_routes,2);
        net_routes[idd1] = IDX(minl - 1, cur_x, cur_y);
        net_routes[idd1 + 1] = IDX(maxl - 1, cur_x, cur_y);
        assert(IDX(minl - 1, cur_x, cur_y)>=0);
        assert(idd1 + 1<ROUTE_PER_PIN_PHASE2*(pin_acc_num[net_id + 1]-pin_acc_num[net_id]));
    }
}

// One CTA owns a complete DAG; only CTA barriers separate its depths.
__global__ __launch_bounds__(128) void route_flt_block(RoutingDeviceView view,
    FltDeviceView flt, const grotor::flt::NetDescriptor* descriptors, int begin,
    const int* depth_offsets, int stamp) {
    const auto net = descriptors[begin + blockIdx.x];
    for (int depth = net.depths - 1; depth >= 0; --depth) {
        const int first = depth_offsets[net.depth_begin + depth];
        const int end = depth_offsets[net.depth_begin + depth + 1];
        for (int i = first + threadIdx.x; i < end; i += blockDim.x)
            solve_flt_node(view, flt, flt.depth_node[i]);
        __syncthreads();
    }
    for (int depth = 0; depth < net.depths; ++depth) {
        const int first = depth_offsets[net.depth_begin + depth];
        const int end = depth_offsets[net.depth_begin + depth + 1];
        for (int i = first + threadIdx.x; i < end; i += blockDim.x)
            trace_flt_node(view, flt, flt.depth_node[i], stamp);
        __syncthreads();
    }
}
} // namespace Lshape_route_detour
