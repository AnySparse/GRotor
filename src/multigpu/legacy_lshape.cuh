#pragma once
// Included inside grotor::multi, after the MPI/NVSHMEM ownership helpers.
namespace legacy {

struct Payload {
    std::vector<int> ids, batches{0}, offsets{0}, nodes, parents;
    std::vector<int> depth_counts{0}, depth_offsets{0};
    int max_nodes = 1;
    void exchange(int peer, bool send) {
        transfer(ids,peer,send); transfer(batches,peer,send); transfer(offsets,peer,send);
        transfer(nodes,peer,send); transfer(parents,peer,send);
        transfer(depth_counts,peer,send); transfer(depth_offsets,peer,send);
        if (!send) for (size_t i=1;i<offsets.size();++i) max_nodes=std::max(max_nodes,offsets[i]-offsets[i-1]);
    }
    void append(int id, const int* source, const int* parent, int count) {
        ids.push_back(id); max_nodes=std::max(max_nodes,count);
        // Stable BFS, as in the original executor. Keep each parent's child
        // order from the InstantGR2 RSMT; fixed-layer labels are unchanged.
        std::vector<int> depth(count), order(count), remap(count);
        std::iota(order.begin(),order.end(),0);
        for (int i=1;i<count;++i) depth[i]=depth[parent[i]]+1;
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return depth[a]<depth[b];});
        for (int i=0;i<count;++i) remap[order[i]]=i;
        int level=0;
        for (int i=0;i<count;++i) {
            const int old=order[i];
            if (depth[old]!=level) { depth_offsets.push_back(nodes.size()); level=depth[old]; }
            nodes.push_back(source[old]); parents.push_back(old ? remap[parent[old]] : -1);
        }
        offsets.push_back(nodes.size()); depth_offsets.push_back(nodes.size());
        depth_counts.push_back(depth_counts.back()+level+1);
    }
};

struct Header {
    RoutingDeviceView dimensions;
    int batches=0, nets=0, last_nets=0, initial_stamp=0;
};

struct Costs {
    RoutingDeviceView formula;
    P2PContext peers;
    int rank, ranks;
};

struct Stream {
    cudaStream_t value=nullptr;
    Stream(){cuda_check(cudaStreamCreate(&value),"create original blocking CUDA stream");}
    ~Stream(){if(value)cudaStreamDestroy(value);}
};

struct WirePeers { float* values[8]{}; };
__global__ void check_costs(RoutingDeviceView reference,P2PContext peers,WirePeers wires,int ranks,int* errors) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x,X=reference.X,Y=reference.Y;
    if(i>=reference.L*X*Y)return;
    const int l=i/(X*Y),x=i/Y%X,y=i%Y,owner=x/((X+ranks-1)/ranks);
    if(__float_as_int(wires.values[owner][i])!=__float_as_int(reference.wcost[i])){atomicAdd(errors,1);atomicMin(errors+3,i);}
    if(__float_as_int(peers.vcost[owner][i])!=__float_as_int(reference.vcost[i])){atomicAdd(errors+1,1);atomicMin(errors+3,i);}
    const double observed=((l&1)^reference.DIR)?wire_segment_cost(l,x,x,0,y,X,Y,ranks,peers):wire_segment_cost(l,0,x,y,y,X,Y,ranks,peers);
    if(__double_as_longlong(observed)!=__double_as_longlong(reference.presum[i]))atomicAdd(errors+2,1);
    const bool along_y=(l&1)^reference.DIR;
    const int start=along_y?y*11%(y+1):x*17%(x+1);
    const int first=along_y?(l*X+x)*Y+start:(l*X+start)*Y+y;
    const double segment=along_y?wire_segment_cost(l,x,x,start,y,X,Y,ranks,peers):wire_segment_cost(l,start,x,y,y,X,Y,ranks,peers);
    if(__double_as_longlong(segment)!=__double_as_longlong(reference.presum[i]-reference.presum[first]))atomicAdd(errors+4,1);
}
__global__ void demand_hash(const float* demand,int cells,unsigned long long* hash){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned long long value=i<cells?(0x9e3779b97f4a7c15ull*(i+1))*(unsigned int)(demand[i]*2):0;
    for(int step=16;step;step>>=1)value+=__shfl_down_sync(0xffffffffu,value,step);
    if(!(threadIdx.x&31))atomicAdd(hash,value);
}

__device__ double incremental_cost(const Costs& ctx,int l,int x,int y,double increase,double base=0.) {
    const int X=ctx.formula.X,Y=ctx.formula.Y,index=(l*X+x)*Y+y;
    const int chunk=(X+ctx.ranks-1)/ctx.ranks;
    // Only the preceding X row can cross the local cost domain. Unlike the
    // old replicated read, this landing cost observes the owner's current
    // demand at a partition boundary (required by InstantGR2).
    const int owner=x/chunk;
    const float demand=.5f*ctx.peers.demand[owner][index];
    const float capacity=ctx.formula.capacity[index];
    return min(1e12,(graph::of_cost_scaled(ctx.formula,capacity,demand+increase+base)
        -graph::of_cost_scaled(ctx.formula,capacity,demand+base))*ctx.formula.unit_length_short_costs[l]);
}

__global__ void update_wire_cost(const GROTOR_GRID_CONSTANT Costs ctx) {
    const int X=ctx.formula.X,Y=ctx.formula.Y,L=ctx.formula.L;
    const int chunk=(X+ctx.ranks-1)/ctx.ranks;
    const int width=min(chunk,X-ctx.rank*chunk),i=blockIdx.x*blockDim.x+threadIdx.x;
    if (width<=0 || i>=L*width*Y) return;
    const int l=i/(width*Y),x=i/Y%width+ctx.rank*chunk,y=i%Y,index=(l*X+x)*Y+y;
    if (!ctx.peers.wcost_inc[ctx.rank][index]) return;
    float cost=incremental_cost(ctx,l,x,y,1.);
    const bool along_y=(l&1)^ctx.formula.DIR;
    if (!along_y && x<X-1) cost+=ctx.formula.unit_length_wire_cost*ctx.formula.x_edge_len[x];
    if (along_y && y<Y-1) cost+=ctx.formula.unit_length_wire_cost*ctx.formula.y_edge_len[y];
    ctx.formula.wcost[index]=cost;
    ctx.peers.wcost_inc[ctx.rank][index]=false;
}

__global__ void update_via_cost(const GROTOR_GRID_CONSTANT Costs ctx) {
    const int X=ctx.formula.X,Y=ctx.formula.Y,L=ctx.formula.L;
    const int chunk=(X+ctx.ranks-1)/ctx.ranks;
    const int width=min(chunk,X-ctx.rank*chunk),i=blockIdx.x*blockDim.x+threadIdx.x;
    if (width<=0 || i>=L*width*Y) return;
    const int l=i/(width*Y),x=i/Y%width+ctx.rank*chunk,y=i%Y,index=(l*X+x)*Y+y;
    if (!ctx.peers.vcost_inc[ctx.rank][index]) return;
    const double base=ctx.formula.cross_points[x*Y+y];
    double cost;
    if ((l&1)^ctx.formula.DIR) {
        if (!y) cost=incremental_cost(ctx,l,x,y,1.,base);
        else if (y==Y-1) cost=incremental_cost(ctx,l,x,y-1,1.,base);
        else cost=incremental_cost(ctx,l,x,y,.5,base)+incremental_cost(ctx,l,x,y-1,.5,base);
    } else {
        if (!x) cost=incremental_cost(ctx,l,x,y,1.,base);
        else if (x==X-1) cost=incremental_cost(ctx,l,x-1,y,1.,base);
        else cost=incremental_cost(ctx,l,x,y,.5,base)+incremental_cost(ctx,l,x-1,y,.5,base);
    }
    ctx.formula.vcost[index]=cost;
    ctx.formula.vcost[index]+=ctx.formula.unit_via_cost;
    ctx.peers.vcost_inc[ctx.rank][index]=false;
}

__global__ void initialize_demand(int* target,const float* source,int cells) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<cells) { assert(source[i]*2==roundf(source[i]*2)); target[i]=int(source[i]*2); }
}
__global__ void gather_demand(float* target,const int* source,int L,int X,int Y,int begin,int width) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<L*width*Y) {
        const int index=(i/(width*Y)*X+i/Y%width+begin)*Y+i%Y;
        target[index]=.5f*source[index];
    }
}
__global__ void route_statistics(RoutingDeviceView view,const int* ids,int nets) {
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if (slot>=nets) return;
    const int id=ids[slot],X=view.X,Y=view.Y;
    const int* routes=view.routes+size_t(view.pin_acc_num[id])*view.ROUTE_PER_PIN;
    double length=0.;int vias=0;
    for (int i=1;i<routes[0];i+=2) {
        const int a=routes[i],b=routes[i+1];
        if (a/(X*Y)!=b/(X*Y)) vias+=abs(a/(X*Y)-b/(X*Y));
        else {
            const int x0=a/Y%X,x1=b/Y%X,y0=a%Y,y1=b%Y;
            for (int x=min(x0,x1);x<max(x0,x1);++x) length+=view.x_edge_len[x];
            for (int y=min(y0,y1);y<max(y0,y1);++y) length+=view.y_edge_len[y];
        }
    }
    atomicAdd(&view.statistics->total_wirelength,length);
    atomicAdd(&view.statistics->total_via_count,vias);
}

inline void synchronize() {
    cuda_check(cudaDeviceSynchronize(),"complete legacy GPU work");
    nvshmem_quiet(); nvshmem_barrier_all();
}

// One initialization and collection around the entire L-shape batch loop.
// Every rank, including N=1, allocates the original full matrices and uses the
// same P2P context, fixed launch geometry and node-parallel kernel.
inline void run(RouterState* state,int rank,int ranks,const Header* root_header=nullptr,
                std::vector<Payload>* root_payloads=nullptr,Clock::time_point preparation_start=Clock::now()) {
    Header h{}; if (!rank) h=*root_header; broadcast(h);
    Payload payload;
    if (!rank) {
        payload=std::move((*root_payloads)[0]);
        for (int p=1;p<ranks;++p) (*root_payloads)[p].exchange(p,true);
    } else payload.exchange(rank,false);
    const auto d=h.dimensions;
    const size_t cells=size_t(d.L)*d.X*d.Y;
    const size_t route_ints=size_t(d.PIN_NUM)*d.ROUTE_PER_PIN+1;
    SymmetricMemory symmetric; DeviceMemory local;
    auto allocate=[&](auto*& pointer,size_t n){local.allocate(&pointer,std::max(size_t(1),n)*sizeof(*pointer));};
    int* demand=symmetric.allocate<int>(cells);
    int* timestamp=symmetric.allocate<int>(cells);
    int* pre_demand=symmetric.allocate<int>(cells);
    bool* wire_dirty=symmetric.allocate<bool>(cells);
    bool* via_dirty=symmetric.allocate<bool>(cells);
    float* wire_cost=symmetric.allocate<float>(cells);
    float* via_cost=symmetric.allocate<float>(cells);
    double* prefix=symmetric.allocate<double>(cells);
    P2PContext peers{};
    auto mapped=[&](auto* pointer,int pe){
        auto* result=static_cast<std::remove_pointer_t<decltype(pointer)>*>(nvshmem_ptr(pointer,pe));
        if (!result) throw std::runtime_error("Legacy matrix lacks NVLink mapping");
        return result;
    };
    for (int p=0;p<ranks;++p) {
        peers.demand[p]=mapped(demand,p); peers.timestamp[p]=mapped(timestamp,p);
        peers.wcost_inc[p]=mapped(wire_dirty,p); peers.vcost_inc[p]=mapped(via_dirty,p);
        peers.vcost[p]=mapped(via_cost,p); peers.presum[p]=mapped(prefix,p);
    }
    RoutingDeviceView formula=d;
    formula.wcost=wire_cost; formula.vcost=via_cost; formula.presum=prefix;
    auto replicate=[&](auto*& target,auto* source,size_t n){
        using T=std::remove_pointer_t<decltype(source)>;
        std::vector<T> host(n);
        if (!rank && n) cuda_check(cudaMemcpy(host.data(),source,n*sizeof(T),cudaMemcpyDeviceToHost),"read legacy immutable input");
        broadcast(host.data(),n*sizeof(T)); target=nullptr; allocate(target,n); upload(target,host);
    };
    replicate(formula.capacity,d.capacity,cells);
    replicate(formula.cross_points,d.cross_points,size_t(d.X)*d.Y);
    replicate(formula.x_edge_len,d.x_edge_len,d.X-1);
    replicate(formula.y_edge_len,d.y_edge_len,d.Y-1);
    replicate(formula.unit_length_short_costs,d.unit_length_short_costs,d.L);
    replicate(formula.idx2track,d.idx2track,d.all_track_cnt);
    int* pins=nullptr; replicate(pins,d.pin_acc_num,size_t(d.NET_NUM)+1);
    int *ids=nullptr,*offsets=nullptr,*nodes=nullptr,*parents=nullptr,*depths=nullptr,*depth_offsets=nullptr;
    allocate(ids,payload.ids.size()); upload(ids,payload.ids);
    allocate(offsets,payload.offsets.size()); upload(offsets,payload.offsets);
    allocate(nodes,payload.nodes.size()); upload(nodes,payload.nodes);
    allocate(parents,payload.parents.size()); upload(parents,payload.parents);
    allocate(depths,payload.depth_counts.size()); upload(depths,payload.depth_counts);
    allocate(depth_offsets,payload.depth_offsets.size()); upload(depth_offsets,payload.depth_offsets);
    int *routes=nullptr,*ranges=nullptr,*from=nullptr,*unused_counter=nullptr;
    double* dist=nullptr;
    allocate(routes,route_ints); allocate(ranges,payload.nodes.size()); allocate(unused_counter,1);
    const size_t scratch=size_t(blocks)*(block_size/warp_size)*payload.max_nodes*d.L*d.L;
    allocate(dist,scratch); allocate(from,scratch);
    cuda_check(cudaMemset(routes,0,route_ints*sizeof(int)),"initialize full legacy route matrix");
    cuda_check(cudaMemset(demand,0,cells*sizeof(int)),"initialize integer demand");
    cuda_check(cudaMemset(timestamp,0,cells*sizeof(int)),"initialize legacy timestamps");
    cuda_check(cudaMemset(pre_demand,0,cells*sizeof(int)),"initialize legacy difference array");
    cuda_check(cudaMemset(wire_dirty,1,cells),"initialize legacy wire flags");
    cuda_check(cudaMemset(via_dirty,1,cells),"initialize legacy via flags");
    if (!rank) for (int p=0;p<ranks;++p)
        initialize_demand<<<(cells+255)/256,256>>>(peers.demand[p],d.demand,cells);
    cuda_check(cudaFuncSetAttribute(compute_presum_nvshmem,cudaFuncAttributeMaxDynamicSharedMemorySize,120*1024),"legacy prefix shared memory");
    ncclUniqueId uid{}; if (!rank && ncclGetUniqueId(&uid)!=ncclSuccess) throw std::runtime_error("NCCL unique ID failed");
    broadcast(uid); ncclComm_t comm;
    if (ncclCommInitRank(&comm,ranks,uid,rank)!=ncclSuccess) throw std::runtime_error("Legacy NCCL initialization failed");
    Stream stream;
    const int chunk=(d.X+ranks-1)/ranks,width=std::max(0,std::min(chunk,d.X-rank*chunk));
    Costs cost{formula,peers,rank,ranks};
    const bool validate=std::getenv("GROTOR_LEGACY_VALIDATE_COSTS")!=nullptr;
    RoutingDeviceView reference=formula; WirePeers wire_peers{};int* errors=nullptr;unsigned long long* signature=nullptr;
    if(validate && !rank){
        reference.demand=state->device.demand;reference.wcost=nullptr;reference.vcost=nullptr;reference.presum=nullptr;
        reference.cost_shard_x=0;reference.local_presum=false;
        allocate(reference.wcost,cells);allocate(reference.vcost,cells);allocate(reference.presum,cells);allocate(errors,5);allocate(signature,1);
        for(int p=0;p<ranks;++p)wire_peers.values[p]=mapped(wire_cost,p);
    }
    auto hash_state=[&](int batch,const char* phase){
        if(!validate)return;
        synchronize();
        if(!rank){
            for(int p=0;p<ranks;++p){const int w=std::max(0,std::min(chunk,d.X-p*chunk));if(w)
                gather_demand<<<(size_t(d.L)*w*d.Y+255)/256,256>>>(reference.demand,peers.demand[p],d.L,d.X,d.Y,p*chunk,w);}
            cuda_check(cudaMemset(signature,0,sizeof(*signature)),"clear demand hash");
            demand_hash<<<cells/256+1,256>>>(reference.demand,cells,signature);
            unsigned long long value;cuda_check(cudaMemcpy(&value,signature,sizeof(value),cudaMemcpyDeviceToHost),"read demand hash");
            printf("LEGACY_DEMAND_SIGNATURE batch=%d phase=%s hash=%llu\n",batch,phase,value);fflush(stdout);
        }
        synchronize();
    };
    auto check=[&](int batch){
        if(!validate)return;
        synchronize();
        if(!rank){
            for(int p=0;p<ranks;++p){const int w=std::max(0,std::min(chunk,d.X-p*chunk));if(w)
                gather_demand<<<(size_t(d.L)*w*d.Y+255)/256,256>>>(reference.demand,peers.demand[p],d.L,d.X,d.Y,p*chunk,w);}
            graph::update_wcost_cuda_ispd24<<<cells/256+1,256>>>(reference);
            graph::update_vcost_ispd24<<<cells/256+1,256>>>(reference);
            graph::compute_presum<<<d.all_track_cnt,256,d.XY*sizeof(double)>>>(reference);
            const int initial[5]={0,0,0,2147483647,0};
            cuda_check(cudaMemcpy(errors,initial,sizeof(initial),cudaMemcpyHostToDevice),"initialize legacy diagnostic");
            check_costs<<<cells/256+1,256>>>(reference,peers,wire_peers,ranks,errors);
            int counters[5];cuda_check(cudaMemcpy(counters,errors,sizeof(counters),cudaMemcpyDeviceToHost),"read legacy diagnostic");
            printf("LEGACY_COST_CHECK batch=%d wire=%d via=%d prefix=%d first=%d segment=%d\n",batch,counters[0],counters[1],counters[2],counters[3],counters[4]);fflush(stdout);
            if(counters[0]||counters[1])throw std::runtime_error("Legacy cost cache differs from full InstantGR2 recompute");
        }
        synchronize();
    };
    auto refresh=[&]{
        if (width) update_wire_cost<<<d.L*width*d.Y/256+1,256>>>(cost);
        compute_presum_nvshmem<<<d.all_track_cnt,256,d.XY*sizeof(double)>>>(d.L,d.X,d.Y,d.XY,d.DIR,formula.idx2track,wire_cost,prefix,rank,ranks);
        if (width) update_via_cost<<<d.L*width*d.Y/256+1,256>>>(cost);
    };
    synchronize(); refresh(); synchronize();
    check(-1);
    TimingEvent begin,solved,committed,end;
    const double preparation_ms=ms(preparation_start);
    double final_times[4]{};
    for (int b=0;b<h.batches;++b) {
        const int offset=payload.batches[b],count=payload.batches[b+1]-offset,stamp=h.initial_stamp+b+1;
        begin.record();
        L_shape_route_inc_warp_centric_nvshmem_p2p<<<blocks,block_size>>>(count,offset,offsets,nodes,depth_offsets,depths,parents,
            dist,from,ranges,stamp,d.ROUTE_PER_PIN,routes,pins,ids,d.L,d.X,d.Y,d.DIR,via_cost,wire_cost,prefix,pre_demand,timestamp,payload.max_nodes,rank,ranks,peers);
        solved.record();
        commit_edge_inc_set_update_nvshmem_p2p<<<blocks,block_size>>>(count,offset,stamp,d.X,d.Y,d.DIR,ids,routes,d.ROUTE_PER_PIN,unused_counter,pins,peers,ranks,2);
        // Timestamp PGAS writes and the default-stream wire commits must be
        // complete before any PE's via kernel reads them. Synchronizing the
        // separate NCCL stream alone does not establish that dependency.
        synchronize();
        hash_state(b,"wire");
        commit_via_demand_set_update_nvshmem_p2p<<<blocks,block_size>>>(count,offset,stamp,d.X,d.Y,d.DIR,ids,routes,d.ROUTE_PER_PIN,unused_counter,pins,peers,ranks,2);
        synchronize(); committed.record();
        hash_state(b,"via");
        refresh();
        // A following batch may read any owner's costs. The original entry
        // only waited for its local stop event here; make the immutable
        // next-batch snapshot complete on every PE, inside the measured span.
        synchronize(); end.record();
        cuda_check(cudaEventSynchronize(end.value),"finish legacy batch event");
        final_times[0]=end.since(begin); final_times[1]=end.since(committed);
        final_times[2]=solved.since(begin); final_times[3]=committed.since(solved);
        printf("LEGACY_LSHAPE_BATCH rank=%d batch_index=%d nets=%d elapsed_ms=%.6f\n",rank,b,count,final_times[0]);
        check(b);
    }
    const auto finalization_start=Clock::now();
    double times[8*4]{};
    mpi_check(MPI_Allgather(final_times,4,MPI_DOUBLE,times,4,MPI_DOUBLE,MPI_COMM_WORLD));
    int critical=0;for(int p=1;p<ranks;++p) if(times[p*4]>times[critical*4])critical=p;
    // The original L-shape entry collects complete routes after the batch loop.
    if (ncclReduce(routes,!rank?state->device.routes:routes,route_ints,ncclInt,ncclSum,0,comm,stream.value)!=ncclSuccess)
        throw std::runtime_error("Legacy final route reduction failed");
    cuda_check(cudaStreamSynchronize(stream.value),"collect full legacy routes");
    synchronize();
    if (!rank) {
        for (int p=0;p<ranks;++p) {
            const int begin_x=p*chunk,local_width=std::max(0,std::min(chunk,d.X-begin_x));
            if (!local_width) continue;
            gather_demand<<<(size_t(d.L)*local_width*d.Y+255)/256,256>>>(state->device.demand,peers.demand[p],d.L,d.X,d.Y,begin_x,local_width);
            cuda_check(cudaMemcpy2DAsync(state->device.timestamp+size_t(begin_x)*d.Y,size_t(d.X)*d.Y*sizeof(int),
                peers.timestamp[p]+size_t(begin_x)*d.Y,size_t(d.X)*d.Y*sizeof(int),size_t(local_width)*d.Y*sizeof(int),d.L,cudaMemcpyDefault),"gather legacy timestamp domain");
        }
        const size_t bytes=((cells+31)/32)*sizeof(unsigned int);
        cuda_check(cudaMemsetAsync(state->device.dirty_wire,0xff,2*bytes),"invalidate root dense cost cache");
        // Gather local statistics after timing, without changing either the
        // original DP/commit kernel or the original measured interval.
    }
    RoutingDeviceView stat_view=formula; stat_view.routes=routes;stat_view.pin_acc_num=pins;
    RoutingStatistics* statistics=nullptr;allocate(statistics,1);stat_view.statistics=statistics;
    cuda_check(cudaMemset(statistics,0,sizeof(RoutingStatistics)),"clear local route statistics");
    route_statistics<<<payload.ids.size()/256+1,256>>>(stat_view,ids,payload.ids.size());
    RoutingStatistics local_stats{};cuda_check(cudaMemcpy(&local_stats,statistics,sizeof(local_stats),cudaMemcpyDeviceToHost),"download local route statistics");
    double wire=0;int vias=0;
    mpi_check(MPI_Reduce(&local_stats.total_wirelength,&wire,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));
    mpi_check(MPI_Reduce(&local_stats.total_via_count,&vias,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD));
    if (!rank) {
        RoutingStatistics totals=state->statistics;totals.total_wirelength+=wire;totals.total_via_count+=vias;
        cuda_check(cudaMemcpy(state->device.statistics,&totals,sizeof(totals),cudaMemcpyHostToDevice),"publish legacy statistics");
        state->global_timestamp=h.initial_stamp+h.batches;
    }
    synchronize(); const double finalization_ms=ms(finalization_start);
    printf("MULTIGPU_PARTITION stage=lshape rank=%d nets=%d nodes=%d\n",rank,
        payload.batches.back()-payload.batches[h.batches-1],payload.offsets.back()-payload.offsets[payload.batches[h.batches-1]]);
    printf("MULTIGPU_RANK_TIMING stage=lshape rank=%d elapsed_ms=%.6f\n",rank,final_times[0]);
    if (!rank) {
        const double* t=times+critical*4;
        printf("MULTIGPU_TIMING_POLICY stage=lshape topology=preparation maintenance=after-commit\n");
        printf("MULTIGPU_BATCH stage=lshape gpus=%d batch_index=%d batches=%d nets=%d elapsed_ms=%.6f preparation_ms=%.6f snapshot_ms=%.6f solve_ms=%.6f commit_ms=%.6f finalization_ms=%.6f end_to_end_ms=%.6f critical_rank=%d\n",
            ranks,h.batches-1,h.batches,h.last_nets,t[0],preparation_ms,t[1],t[2],t[3],finalization_ms,ms(preparation_start),critical);
    }
    if (ncclCommDestroy(comm)!=ncclSuccess) throw std::runtime_error("Destroy legacy NCCL communicator failed");
}
} // namespace legacy
