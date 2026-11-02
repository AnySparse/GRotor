#pragma once
#include "router_state.hpp"

namespace grotor::experiments {

// Both endpoints are exclusive edge prefixes in a zero-based shifted BIT.
// Clear the differing suffix to find the shared ancestor of their query paths.
__host__ __device__ inline int common_prefix_node(unsigned first, unsigned last) {
    unsigned difference=(first+1)^(last+1), mask=0;
    while(difference){mask=(mask<<1)|1u;difference>>=1;}
    return int((first+1)&~mask)-1;
}

template<bool Prune>
__device__ double bit_range_sum(const double* tree,int stride,int first,int last,unsigned& reads) {
    const int stop=Prune ? common_prefix_node(first,last) : -1;
    double low=0,high=0;
    for(int p=last;p>stop;p=(p&(p+1))-1){high+=tree[p*stride];++reads;}
    for(int p=first;p>stop;p=(p&(p+1))-1){low+=tree[p*stride];++reads;}
    return high-low;
}

__device__ inline double bit_range_unpruned(const double* tree,int stride,int first,int last,unsigned& reads) {
    return bit_range_sum<false>(tree,stride,first,last,reads);
}
__device__ inline double bit_range_pruned(const double* tree,int stride,int first,int last,unsigned& reads) {
    return bit_range_sum<true>(tree,stride,first,last,reads);
}

#ifdef GROTOR_PREFIX_ACCESS_COUNTS
__device__ inline double wire_cost_counted(const cudb::RoutingDeviceView& view,int layer,
                                          int xmin,int xmax,int ymin,int ymax) {
    const int X=view.X,Y=view.Y;
    unsigned reads=0;
    double result;
    if(view.prefix_query_mode==0){
        result=view.presum[(layer*X+xmax)*Y+ymax]-view.presum[(layer*X+xmin)*Y+ymin];
        reads=2;
    }else{
        const bool along_y=(layer&1)^view.DIR;
        const int base=layer*X*Y+(along_y?xmin*Y:ymin),stride=along_y?1:Y;
        const int first=along_y?ymin:xmin,last=along_y?ymax:xmax;
        if(view.prefix_query_mode==1)result=bit_range_unpruned(view.presum+base,stride,first,last,reads);
        else result=bit_range_pruned(view.presum+base,stride,first,last,reads);
    }
    if(view.prefix_queries){
        auto& counter=view.prefix_queries[blockIdx.x*blockDim.x+threadIdx.x];
        counter.reads+=reads;++counter.queries;
    }
    return result;
}
#endif
} // namespace grotor::experiments
