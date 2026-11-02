#pragma once
#include "routing/flt_block.cuh"

namespace Lshape_route_detour {

// Per-batch depth boundaries are tiny histograms, not packed copies of every
// DAG. Build them before the common execution timer; GPU sorting supplies the
// corresponding stable global depth order during batch assembly.
inline std::vector<std::vector<int>> prepare_global_depths(
    const grotor::flt::Topology& topology,const grotor::flt::Plans& plans) {
    std::vector<std::vector<int>> result;
    result.reserve(plans.batches.size());
    for(const auto& batch:plans.batches){
        std::vector<int> counts;
        for(int slot:batch.slots){
            const auto& span=topology.spans[slot];
            if(counts.size()<size_t(span.depths))counts.resize(span.depths);
            const auto& nodes=topology.arenas[span.lane].nodes;
            for(int i=0;i<span.nodes;++i)++counts[nodes[span.node_begin+i].depth];
        }
        std::vector<int> offsets(counts.size()+1);
        for(size_t i=0;i<counts.size();++i)offsets[i+1]=offsets[i]+counts[i];
        if(offsets.back()!=batch.nodes)throw std::runtime_error("FLT depth plan does not cover its batch");
        result.push_back(std::move(offsets));
    }
    return result;
}

__global__ __launch_bounds__(512) void solve_flt_global_depth(RoutingDeviceView view,FltDeviceView flt,int begin,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)solve_flt_node(view,flt,flt.depth_node[begin+i]);
}
__global__ __launch_bounds__(512) void trace_flt_global_depth(RoutingDeviceView view,FltDeviceView flt,int begin,int count,int stamp) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)trace_flt_node(view,flt,flt.depth_node[begin+i],stamp);
}

inline void route_flt_global_depths(RoutingDeviceView view,FltDeviceView flt,
                                    const std::vector<int>& offsets,int stamp) {
    for(int d=int(offsets.size())-2;d>=0;--d){
        const int count=offsets[d+1]-offsets[d];
        if(count){
            solve_flt_global_depth<<<BLOCK_NUM(count),THREAD_NUM>>>(view,flt,offsets[d],count);
            grotor::cuda_check(cudaGetLastError(),"FLT global-depth solve launch");
        }
    }
    for(size_t d=0;d+1<offsets.size();++d){
        const int count=offsets[d+1]-offsets[d];
        if(count){
            trace_flt_global_depth<<<BLOCK_NUM(count),THREAD_NUM>>>(view,flt,offsets[d],count,stamp);
            grotor::cuda_check(cudaGetLastError(),"FLT global-depth trace launch");
        }
    }
}
} // namespace Lshape_route_detour
