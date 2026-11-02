#pragma once
#include "routing/demand.cuh"

namespace grotor::experiments {

// Full BIT storage keeps every logical node, including the zero sentinel.
// Rebuild changed nodes once from current costs, with the deterministic scan
// addition order. No dense prefix cache is materialized by either BIT arm.
__global__ void refresh_direct_bit(cudb::RoutingDeviceView view,bool rebuild) {
    if(!rebuild && !view.dirty_tracks[blockIdx.x])return;
    const int code=view.idx2track[blockIdx.x],layer=code/view.XY,track=code%view.XY;
    const bool along_y=(layer&1)^view.DIR;
    const int length=along_y?view.Y:view.X,stride=along_y?1:view.Y;
    const int base=layer*view.X*view.Y+(along_y?track*view.Y:track);
    extern __shared__ double storage[];
    double* tree=storage;
    auto* dirty=reinterpret_cast<unsigned*>(tree+view.XY);
    for(int i=threadIdx.x;i<length;i+=blockDim.x){
        tree[i]=rebuild?0.:view.presum[base+i*stride];
        dirty[i]=rebuild || graph::is_dirty(view.dirty_tree,base+i*stride);
    }
    __syncthreads();
    for(int width=1;width<=length;width<<=1){
        for(int i=(2*threadIdx.x+1)*width-1;i<length;i+=2*blockDim.x*width){
            if(!dirty[i])continue;
            double value=i?double(view.wcost[base+(i-1)*stride]):0.;
            for(int offset=1;offset<width;offset<<=1)value+=tree[i-offset];
            tree[i]=value;
        }
        __syncthreads();
    }
    unsigned writes=0;
    for(int i=threadIdx.x;i<length;i+=blockDim.x)if(dirty[i]){
        view.presum[base+i*stride]=tree[i];++writes;
    }
    for(int offset=16;offset;offset>>=1)writes+=__shfl_down_sync(0xffffffffu,writes,offset);
    __shared__ unsigned warp_writes[32];
    if(!(threadIdx.x&31))warp_writes[threadIdx.x>>5]=writes;
    __syncthreads();
    if(!threadIdx.x && view.prefix_access){
        unsigned total=0;for(int warp=0;warp<blockDim.x/32;++warp)total+=warp_writes[warp];
        atomicAdd(&view.prefix_access->prefix_writes,(unsigned long long)total);
        if(!rebuild)atomicAdd(&view.prefix_access->maintenance_reads,(unsigned long long)length);
    }
    if(!threadIdx.x)view.dirty_tracks[blockIdx.x]=0;
}

inline void refresh_prefix(cudb::RouterState& state) {
    auto& v=state.device;
    if(!v.prefix_query_mode){graph::refresh_cost(state);return;}
    const bool rebuild=!state.costs_initialized;
    if(rebuild)graph::update_cost(state);
    else graph::update_dirty_costs<<<BLOCK_NUM((v.L*v.X*v.Y+31)/32),THREAD_NUM>>>(v);
    refresh_direct_bit<<<v.all_track_cnt,THREAD_NUM,v.XY*(sizeof(double)+sizeof(unsigned))>>>(v,rebuild);
    state.costs_initialized=true;
    const size_t bytes=((size_t(v.L)*v.X*v.Y+31)/32)*sizeof(unsigned);
    grotor::cuda_check(cudaMemsetAsync(v.dirty_wire,0,3*bytes),"consume prefix dirty masks");
}

inline void initialize_direct_bit(cudb::RouterState& state) {
    auto& v=state.device;
    if(!v.dirty_wire){
        graph::initialize_incremental_cost(state);
        // The experiment reuses the existing prefix allocation for a full BIT.
        state.memory.release(v.cost_bit);state.memory.release(v.bit_track_offsets);
    }
    const size_t bytes=((size_t(v.L)*v.X*v.Y+31)/32)*sizeof(unsigned);
    grotor::cuda_check(cudaMemset(v.dirty_wire,0xff,2*bytes),"reset prefix cost dirty masks");
    grotor::cuda_check(cudaMemset(v.dirty_tree,0,bytes),"reset prefix tree dirty mask");
    grotor::cuda_check(cudaMemset(v.dirty_tracks,0,v.all_track_cnt*sizeof(unsigned)),"reset prefix dirty tracks");
    grotor::cuda_check(cudaFuncSetAttribute(refresh_direct_bit,cudaFuncAttributeMaxDynamicSharedMemorySize,
        v.XY*(sizeof(double)+sizeof(unsigned))),"direct BIT shared memory");
}

__global__ void reduce_query_counts(cudb::RoutingDeviceView view,int threads) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long reads=i<threads?view.prefix_queries[i].reads:0,queries=i<threads?view.prefix_queries[i].queries:0;
    for(int offset=16;offset;offset>>=1){
        reads+=__shfl_down_sync(0xffffffffu,reads,offset);queries+=__shfl_down_sync(0xffffffffu,queries,offset);
    }
    if(!(threadIdx.x&31)){
        if(reads)atomicAdd(&view.prefix_access->query_reads,reads);
        if(queries)atomicAdd(&view.prefix_access->queries,queries);
    }
}
} // namespace grotor::experiments
