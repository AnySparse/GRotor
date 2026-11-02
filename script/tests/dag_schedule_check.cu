#include "lshape_thread_oracle.cuh"
#include "routing/lshape_warp.cuh"
#include "routing/flt_block.cuh"
#ifdef GROTOR_SERIAL_DAG
#include "routing/flt_depth.cuh"
#endif
#include "flt_thread_oracle.cuh"
#include <numeric>

namespace {
template<class T> std::vector<T> download(const T* p, size_t n) {
    std::vector<T> result(n);
    grotor::cuda_check(cudaMemcpy(result.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download DAG fixture");
    return result;
}
template<class T> void upload(T* p, const std::vector<T>& data) {
    grotor::cuda_check(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice), "upload DAG fixture");
}
struct Fixture {
    cudb::RouterState state;
    explicit Fixture(int direction, bool ties) {
        auto& v = state.device;
        v.L = 9; v.X = 24; v.Y = 24; v.XY = 24; v.DIR = direction;
        v.NET_NUM = 5; v.ROUTE_PER_PIN = v.ROUTE_PER_PIN_PHASE2 = 100;
        const int cells = v.L * v.X * v.Y;
        alloc(v.vcost, cells); alloc(v.presum, cells); alloc(v.pre_demand, cells); alloc(v.timestamp, cells);
        alloc(v.net_ids, 5); alloc(v.pin_acc_num, 6); alloc(v.pin_acc_num_phase2, 5);
        alloc(v.routes, 5 * 256 * 100); alloc(v.routes_phase2, 5 * 256 * 100);
        std::vector<float> via(cells);
        std::vector<double> prefix(cells);
        for (int i = 0; i < cells; ++i) via[i] = ties ? .25f : float((i * 271u) % 997u + 1) * .000173f;
        for (int l = 0; l < v.L; ++l) {
            const bool y_direction = (l & 1) ^ direction;
            for (int track = 0; track < 24; ++track) {
                double sum = 0;
                for (int pos = 0; pos < 24; ++pos) {
                    const int i = l * 24 * 24 + (y_direction ? track * 24 + pos : pos * 24 + track);
                    prefix[i] = sum;
                    sum += ties ? .125f : float((i * 37u) % 991u + 1) * .000319f;
                }
            }
        }
        upload(v.vcost, via); upload(v.presum, prefix);
        upload(v.net_ids, std::vector<int>{0,1,2,3,4});
        upload(v.pin_acc_num, std::vector<int>{0,256,512,768,1024,1280});
        upload(v.pin_acc_num_phase2, std::vector<int>{0,256,512,768,1024});
        reset();
    }
    template<class T> void alloc(T*& p, size_t n) { state.memory.allocate(&p, n * sizeof(T)); }
    void reset() {
        auto& v = state.device;
        grotor::cuda_check(cudaMemset(v.pre_demand, 0, 9*24*24*sizeof(int)), "reset fixture demand");
        grotor::cuda_check(cudaMemset(v.timestamp, 0, 9*24*24*sizeof(int)), "reset fixture timestamps");
        grotor::cuda_check(cudaMemset(v.routes, 0, 5*256*100*sizeof(int)), "reset fixture routes");
        grotor::cuda_check(cudaMemset(v.routes_phase2, 0, 5*256*100*sizeof(int)), "reset fixture FLT routes");
    }
};
void check_lshape(int direction, bool ties) {
    Fixture f(direction, ties); auto& v = f.state.device;
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
    f.alloc(v.prefix_queries, 256);
#endif
    std::vector<int> offsets{0}, nodes, parents;
    for (int n : {1, 7, 19, 43, 121}) {
        for (int i = 0; i < n; ++i) {
            const int fixed = (i * 7 + n) % 10;
            nodes.push_back(fixed * 24*24 + ((i * 7 + n) % 24) * 24 + ((i * 11 + n) % 24));
            parents.push_back(i ? (i - 1) / 2 : -1);
        }
        offsets.push_back(nodes.size());
    }
    int *d_offsets=nullptr,*d_nodes=nullptr,*d_parents=nullptr,*paths=nullptr,*ranges=nullptr;
    double* costs=nullptr;
    f.alloc(d_offsets, offsets.size());f.alloc(d_nodes,nodes.size());f.alloc(d_parents,parents.size());
    f.alloc(paths,nodes.size()*81);f.alloc(costs,nodes.size()*81);f.alloc(ranges,nodes.size());
    upload(d_offsets,offsets);upload(d_nodes,nodes);upload(d_parents,parents);
    for (const auto span : std::vector<std::pair<int,int>>{{0,5},{1,3},{4,1}}) {
        std::vector<double> expected_cost;
        std::vector<int> expected_path,expected_range,expected_routes,expected_demand;
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
        unsigned long long expected_reads=0,expected_queries=0;
#endif
        for (int repeat=0;repeat<4;++repeat) {
            f.reset();
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
            grotor::cuda_check(cudaMemset(v.prefix_queries,0,256*sizeof(cudb::PrefixQueryCounts)),"reset DAG query fixture");
#endif
            grotor::cuda_check(cudaMemset(costs,0,nodes.size()*81*sizeof(double)),"reset L-shape costs");
            grotor::cuda_check(cudaMemset(paths,0,nodes.size()*81*sizeof(int)),"reset L-shape paths");
            grotor::cuda_check(cudaMemset(ranges,0,nodes.size()*sizeof(int)),"reset L-shape ranges");
            if (!repeat) Lshape_route::Lshape_route_cuda<<<1,128>>>(v,span.second,span.first,d_offsets,d_nodes,d_parents,costs,paths,ranges,7);
            else Lshape_route::Lshape_route_warp<<<(span.second+3)/4,128>>>(v,span.second,span.first,d_offsets,d_nodes,d_parents,costs,paths,ranges,7);
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
            const auto counters=download(v.prefix_queries,256);
            unsigned long long reads=0,queries=0;
            for(const auto& counter:counters){reads+=counter.reads;queries+=counter.queries;}
            if(!repeat){expected_reads=reads;expected_queries=queries;}
            else if(reads!=expected_reads || queries!=expected_queries)
                throw std::runtime_error("Warp broadcast query counts differ from thread-per-DAG oracle");
#endif
            auto c=download(costs,nodes.size()*81);
            auto p=download(paths,nodes.size()*81),r=download(ranges,nodes.size());
            auto routes=download(v.routes,5*256*100),demand=download(v.pre_demand,9*24*24);
            if (!repeat) {expected_cost=c;expected_path=p;expected_range=r;expected_routes=routes;expected_demand=demand;}
            else if (c!=expected_cost || p!=expected_path || r!=expected_range || routes!=expected_routes || demand!=expected_demand)
                throw std::runtime_error("Warp L-shape differs from serial-node oracle");
        }
    }
    printf("PASS LSHAPE_WARP direction=%d ties=%d exact_costs_paths_demands=1\n",direction,int(ties));
}
grotor::flt::Builder diamond(int diamonds, int fixed) {
    grotor::flt::Builder b;
    b.node_index_cnt=1+3*diamonds;
    b.par_num_sum_cpu.push_back(0);
    for (int i=0;i<b.node_index_cnt;++i) {
        int x=5,y=5,depth=0;
        if (i) {
            const int level=(i-1)/3,role=(i-1)%3;
            const int a=level%2 ? 12:5,other=level%2 ? 5:12;
            if (role==0) {x=a;y=other;depth=2*level+1;b.par_nodes_cpu.push_back(3*level);}
            if (role==1) {x=other;y=a;depth=2*level+1;b.par_nodes_cpu.push_back(3*level);}
            if (role==2) {x=other;y=other;depth=2*level+2;b.par_nodes_cpu.push_back(i-2);b.par_nodes_cpu.push_back(i-1);}
        }
        b.nodes_cpu.push_back((i==0 ? fixed : (i+1==b.node_index_cnt ? 2:9))*24*24+x*24+y);
        b.node_depth_cpu.push_back(depth);
        b.child_num_cpu.push_back(i+1==b.node_index_cnt ? 0:1);
        b.par_num_cpu.push_back(b.par_nodes_cpu.size()-b.par_num_sum_cpu.back());
        b.par_num_sum_cpu.push_back(b.par_nodes_cpu.size());
    }
    b.currentChildIDX_cpu.assign(b.par_nodes_cpu.size(),0);
    return b;
}
grotor::flt::Builder alternatives(int count, bool branches) {
    grotor::flt::Builder b;
    b.node_index_cnt = count + 1;
    b.par_num_sum_cpu = {0, 0};
    b.par_num_cpu.push_back(0); b.node_depth_cpu.push_back(0);
    b.nodes_cpu.push_back(5*24+5); b.child_num_cpu.push_back(branches ? count : 1);
    for (int i=0; i<count; ++i) {
        const int x = branches && (i&1) ? 5 : 12;
        const int y = branches && (i&1) ? 12 + i : 5;
        b.nodes_cpu.push_back((2+i%3)*24*24+x*24+y);
        b.node_depth_cpu.push_back(1); b.child_num_cpu.push_back(0);
        b.par_num_cpu.push_back(1); b.par_nodes_cpu.push_back(0);
        b.currentChildIDX_cpu.push_back(branches ? i : 0);
        b.par_num_sum_cpu.push_back(i+1);
    }
    return b;
}
std::vector<std::vector<std::pair<int,int>>> normalized_routes(Fixture& f, const std::vector<int>& net_ids) {
    std::vector<std::vector<std::pair<int,int>>> result;
    for (int id:net_ids) {
        auto buffer=download(f.state.device.routes_phase2+id*256*100,256*100);
        std::vector<std::pair<int,int>> segments;
        for (int i=1;i<buffer[0];i+=2)segments.emplace_back(buffer[i],buffer[i+1]);
        std::sort(segments.begin(),segments.end());result.push_back(std::move(segments));
    }
    return result;
}
void check_flt(int direction, bool ties, bool streamed) {
    Fixture f(direction,ties);auto& v=f.state.device;
    std::vector<grotor::flt::Builder> builders{diamond(1,0),diamond(5,3),diamond(70,7),alternatives(4,true),alternatives(257,false)};
    std::vector<int> ids{2,0,1,3,4};
    auto& topology=f.state.flt_topology;topology.begin(ids,5,3);
    for (int i=0;i<builders.size();++i)topology.emit(i,i%3,builders[i]);
    grotor::flt::Plans plans(topology,{ids});
    grotor::flt::Workspace workspace(f.state.flt,plans,streamed ? 0:SIZE_MAX,128);
    workspace.upload(topology,plans);
    const auto& batch=plans.batches[0];
#ifdef GROTOR_SERIAL_DAG
    const auto global_depths=Lshape_route_detour::prepare_global_depths(topology,plans);
#endif
    std::vector<int> depths;
    for (const auto& b:builders)depths.insert(depths.end(),b.node_depth_cpu.begin(),b.node_depth_cpu.end());
    std::vector<int> order(batch.nodes);std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return depths[a]<depths[b];});
    int maxdepth=*std::max_element(depths.begin(),depths.end());
    std::vector<double> expected_cost;
    std::vector<unsigned long long> expected_packed,expected_path;
    std::vector<int> expected_layers,expected_demand;
    std::vector<std::vector<std::pair<int,int>>> expected_routes;
    for (int repeat=0;repeat<4;++repeat) {
        f.reset();workspace.assemble(topology,plans,batch,v);
        auto& flt=f.state.flt;
        grotor::cuda_check(cudaMemset(flt.best_path,0,(batch.children+1)*10*sizeof(unsigned long long)),"reset FLT paths");
        Lshape_route_detour::precompute_cost_edge<<<(batch.edges*9+127)/128,128>>>(v,flt,batch.edges);
        if (!repeat) {
            upload(flt.depth_node,order);
            for (int d=maxdepth;d>=0;--d) {
                auto first=std::lower_bound(order.begin(),order.end(),d,[&](int n,int depth){return depths[n]<depth;});
                auto end=std::upper_bound(order.begin(),order.end(),d,[&](int depth,int n){return depth<depths[n];});
                if (first!=end)Lshape_route_detour::Lshape_route_node_layer_change_cuda<<<(end-first+511)/512,512>>>(v,flt,first-order.begin(),end-order.begin());
            }
            for (int d=0;d<=maxdepth;++d) {
                auto first=std::lower_bound(order.begin(),order.end(),d,[&](int n,int depth){return depths[n]<depth;});
                auto end=std::upper_bound(order.begin(),order.end(),d,[&](int depth,int n){return depth<depths[n];});
                if (first!=end)Lshape_route_detour::get_routing_tree_layer_change_cuda<<<(end-first+511)/512,512>>>(v,flt,first-order.begin(),end-order.begin(),d,7);
            }
        } else {
#ifdef GROTOR_SERIAL_DAG
            Lshape_route_detour::route_flt_global_depths(v,flt,global_depths[0],7);
#else
            Lshape_route_detour::route_flt_block<<<batch.nets,128>>>(v,flt,workspace.descriptors,0,workspace.depth_offsets,7);
#endif
        }
        auto costs=download(flt.costs,batch.nodes*10);
        auto packed=download(flt.packed_information,(batch.children+1)*10),path=download(flt.best_path,(batch.children+1)*10);
        auto layers=download(flt.layer_output,batch.nodes),demand=download(v.pre_demand,9*24*24);
        auto routes=normalized_routes(f,ids);
        if (!repeat) {expected_cost=costs;expected_packed=packed;expected_path=path;expected_layers=layers;expected_demand=demand;expected_routes=routes;}
        else if (costs!=expected_cost || packed!=expected_packed || path!=expected_path || layers!=expected_layers || demand!=expected_demand || routes!=expected_routes)
            throw std::runtime_error("Block FLT differs from global-depth oracle");
    }
    printf("PASS FLT_BLOCK direction=%d ties=%d streamed=%d exact_costs_paths_demands=1\n",direction,int(ties),int(streamed));
}
}
int main() {
    try {
        for (int direction:{0,1})for (bool ties:{false,true}) {
            check_lshape(direction,ties);
            // FLT path encodings use the benchmark's vertical-even layout (DIR=1).
            if (direction) {check_flt(direction,ties,false);check_flt(direction,ties,true);}
        }
        puts("DAG_SCHEDULE_CHECK_PASSED");return 0;
    } catch (const std::exception& e) {fprintf(stderr,"%s\n",e.what());return 1;}
}
