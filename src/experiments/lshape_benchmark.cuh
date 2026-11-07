#pragma once
#include "preprocess.hpp"
#include "batching/generate_batches_cpu.hpp"
#include "routing/lshape_thread.cuh"
#include "routing/lshape_warp.cuh"
#include "output_writer.hpp"
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <spawn.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <cerrno>
#ifdef _OPENMP
#include <omp.h>
#endif

extern char** environ;
namespace grotor::experiments {
struct Batch {
    int begin=0,nets=0,nodes=0,color=0;
    unsigned long long plan_hash=14695981039346656037ULL;
};
struct Workspace {
    grotor::DeviceMemory memory;
    std::vector<Batch> batches;
    int *node_offsets=nullptr,*nodes=nullptr,*parents=nullptr,*from=nullptr,*ranges=nullptr;
    double* dist=nullptr;
    size_t matrix_entries=0,total_nodes=0;
    template<class T> void allocate(T*& p,size_t count){memory.allocate(&p,count*sizeof(T));}
    explicit Workspace(cudb::RouterState& state,const std::vector<std::vector<int>>& colors) {
        std::vector<int> order;
        order.reserve(state.device.NET_NUM);
        for (int c=int(colors.size())-1;c>=0;--c) {
            if (colors[c].empty()) throw std::runtime_error("Empty logical batch");
            Batch batch;batch.begin=order.size();batch.nets=colors[c].size();batch.color=c;
            for (int id:colors[c]) {
                order.push_back(id);
                for (int byte=0;byte<4;++byte) {
                    batch.plan_hash^=(static_cast<unsigned>(id)>>(byte*8))&255u;
                    batch.plan_hash*=1099511628211ULL;
                }
            }
            batches.push_back(batch);
        }
        if (order.size()!=state.nets.size())throw std::runtime_error("Batch plan does not cover every routing net");
        std::vector<int> offsets(order.size()+1);
        for (int i=0;i<order.size();++i) {
            const size_t count=state.nets[order[i]].rsmt.back().size();
            if (size_t(offsets[i])+count>size_t(std::numeric_limits<int>::max()))throw std::runtime_error("Node index overflow");
            offsets[i+1]=offsets[i]+count;
        }
        total_nodes=offsets.back();
        int max_nodes=0;
        for (auto& b:batches){b.nodes=offsets[b.begin+b.nets]-offsets[b.begin];max_nodes=std::max(max_nodes,b.nodes);}
        const int L=state.device.L;
        if (max_nodes>std::numeric_limits<int>::max()/(L*L))throw std::runtime_error("DAG matrix index overflow");
        matrix_entries=size_t(max_nodes)*L*L;
        size_t available=0,total=0;
        grotor::cuda_check(cudaMemGetInfo(&available,&total),"L-shape benchmark memory budget");
        const size_t required=(order.size()+1+3*total_nodes)*sizeof(int)+matrix_entries*(sizeof(double)+sizeof(int));
        const size_t reserve=size_t(state.device.L)*state.device.X*state.device.Y*5+(size_t(256)<<20);
        if (required+reserve>available)throw std::runtime_error("L-shape benchmark workspace and BIT reserve exceed free GPU memory");
        std::unique_ptr<int[]> host_nodes(new int[total_nodes]),host_parents(new int[total_nodes]);
        #pragma omp parallel for num_threads(8)
        for (int i=0;i<order.size();++i) {
            const auto& graph=state.nets[order[i]].rsmt;
            int count=0;
            std::function<void(int,int,int)> dfs=[&](int current,int previous,int parent){
                const int index=count++;
                host_nodes[offsets[i]+index]=graph.back()[current];host_parents[offsets[i]+index]=parent;
                for (int next:graph[current])if(next!=previous)dfs(next,current,index);
            };
            dfs(0,-1,-1);
            if (count!=offsets[i+1]-offsets[i])throw std::runtime_error("Incomplete L-shape DFS");
        }
        allocate(node_offsets,offsets.size());allocate(nodes,total_nodes);allocate(parents,total_nodes);allocate(ranges,total_nodes);
        allocate(dist,matrix_entries);allocate(from,matrix_entries);
        grotor::cuda_check(cudaMemcpy(node_offsets,offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice),"upload DAG throughput benchmark node offsets");
        grotor::cuda_check(cudaMemcpy(nodes,host_nodes.get(),total_nodes*sizeof(int),cudaMemcpyHostToDevice),"upload DAG throughput benchmark nodes");
        grotor::cuda_check(cudaMemcpy(parents,host_parents.get(),total_nodes*sizeof(int),cudaMemcpyHostToDevice),"upload DAG throughput benchmark parents");
        grotor::cuda_check(cudaMemcpy(state.device.net_ids,order.data(),order.size()*sizeof(int),cudaMemcpyHostToDevice),"upload DAG throughput benchmark batch plan");
        printf("LSHAPE_PLAN nets=%zu batches=%zu nodes=%zu max_batch_nodes=%d shared_plan=1\n",order.size(),batches.size(),total_nodes,max_nodes);
    }
    void reset(cudb::RouterState& state,const cudb::RoutingStatistics& initial) {
        auto& v=state.device;
        const size_t cells=size_t(v.L)*v.X*v.Y;
        grotor::cuda_check(cudaMemset(v.demand,0,cells*sizeof(float)),"reset DAG throughput benchmark demand");
        grotor::cuda_check(cudaMemset(v.pre_demand,0,cells*sizeof(int)),"reset DAG throughput benchmark pre-demand");
        grotor::cuda_check(cudaMemset(v.timestamp,0,cells*sizeof(int)),"reset DAG throughput benchmark timestamps");
        grotor::cuda_check(cudaMemset(v.cross_points,0,size_t(v.X)*v.Y*sizeof(int)),"reset DAG throughput benchmark cross points");
        grotor::cuda_check(cudaMemset(v.routes,0,(v.ROUTE_PER_PIN*v.PIN_NUM+1)*sizeof(int)),"reset DAG throughput benchmark routes");
        grotor::cuda_check(cudaMemcpy(v.statistics,&initial,sizeof(initial),cudaMemcpyHostToDevice),"reset DAG throughput benchmark statistics");
        grotor::cuda_check(cudaMemset(from,0,matrix_entries*sizeof(int)),"reset DAG throughput benchmark traceback scratch");
        grotor::cuda_check(cudaMemset(dist,0,matrix_entries*sizeof(double)),"reset DAG throughput benchmark DP scratch");
        grotor::cuda_check(cudaMemset(ranges,0,total_nodes*sizeof(int)),"reset DAG throughput benchmark layer decisions");
        state.global_timestamp=0;state.costs_initialized=false;
        grotor::cuda_check(cudaDeviceSynchronize(),"complete DAG throughput benchmark reset");
    }
    void execute(cudb::RouterState& state,bool optimized,const std::filesystem::path& path) {
        std::ofstream out(path);
        if(!out)throw std::runtime_error("Cannot write DAG throughput benchmark timing table");
        out<<"batch,color_id,nets,nodes,plan_hash,start_us,end_us,elapsed_ms,nets_per_second\n"<<std::setprecision(17);
        for (int i=0;i<batches.size();++i) {
            const auto& b=batches[i];
            cudb::RoutingDagTimer timer;
            const int stamp=++state.global_timestamp;
            graph::refresh_cost(state);
            auto routing_view=state.device;
            routing_view.pre_demand=nullptr;
            if (optimized)
                Lshape_route::Lshape_route_warp<<<(b.nets+3)/4,128>>>(routing_view,b.nets,b.begin,node_offsets,nodes,parents,dist,from,ranges,stamp);
            else
                Lshape_route::Lshape_route_cuda<<<BLOCK_NUM(b.nets),THREAD_NUM>>>(routing_view,b.nets,b.begin,node_offsets,nodes,parents,dist,from,ranges,stamp);
            graph::commit_batch_demand(state.device,b.nets,b.begin,stamp);
            const auto interval=timer.stop();
            const auto us=interval.end_us-interval.start_us;
            if(us<=0)throw std::runtime_error("Nonpositive DAG throughput benchmark batch duration");
            const double ms=us/1000.0,throughput=double(b.nets)*1e6/double(us);
            out<<i<<','<<b.color<<','<<b.nets<<','<<b.nodes<<','<<b.plan_hash<<','<<interval.start_us<<','<<interval.end_us<<','<<ms<<','<<throughput<<'\n';
            printf("DAG_THROUGHPUT_BATCH optimized=%d batch=%d nets=%d elapsed_ms=%.6f throughput=%.9g\n",int(optimized),i,b.nets,ms,throughput);
        }
    }
};

void write_output(cudb::RouterState& state,const std::filesystem::path& path) {
    // Stream existing input metadata instead of retaining another full copy.
    state.routes_cpu.reset();state.download_routes(false);
    out_file=fopen(path.c_str(),"w");
    if(!out_file)throw std::runtime_error("Cannot create DAG throughput benchmark routing output");
    setvbuf(out_file,nullptr,_IONBF,0);
    {
        grotor::RouteWriter writer(out_file);
        const int X=state.device.X,Y=state.device.Y;
        const auto segment=[&](int a,int b){
            if(a>b)std::swap(a,b);
            writer.segment(a/Y%X,a%Y,a/X/Y,b/Y%X,b%Y,b/X/Y);
        };
        for(const auto& n:db::nets) {
            writer.text(n.name);writer.text("\n(\n");
            for(int p:n.pins)if(p<X*Y)segment(p,p+X*Y);
            for(int i=0;i<n.extra_routes.size();i+=2)segment(n.extra_routes[i],n.extra_routes[i+1]);
            for(int id:n.subnets) {
                const int* routes=state.routes_cpu.get()+size_t(state.pin_cnt_sum_cpu[id])*state.device.ROUTE_PER_PIN;
                for(int i=1;i<routes[0];i+=2)segment(routes[i]+X*Y,routes[i+1]+X*Y);
            }
            writer.text(")\n");
        }
        writer.flush();
    }
    const int status=fclose(out_file);out_file=nullptr;
    state.routes_cpu.reset();
    if(status)throw std::runtime_error("Failed to close DAG throughput benchmark routing output");
}
int evaluate(const std::string& evaluator,const std::string& cap,const std::string& net,
             const std::filesystem::path& output,const std::filesystem::path& log) {
    std::vector<std::string> args{evaluator,cap,net,output.string()};
    std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());argv.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    int status=posix_spawn_file_actions_init(&actions);
    if(status)throw std::runtime_error("Cannot initialize evaluator process");
    status=posix_spawn_file_actions_addopen(&actions,STDOUT_FILENO,log.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0644);
    if(!status)status=posix_spawn_file_actions_adddup2(&actions,STDOUT_FILENO,STDERR_FILENO);
    pid_t pid=-1;
    if(!status)status=posix_spawn(&pid,evaluator.c_str(),&actions,nullptr,argv.data(),environ);
    posix_spawn_file_actions_destroy(&actions);
    if(status)throw std::runtime_error("Cannot launch external evaluator");
    while(waitpid(pid,&status,0)<0)if(errno!=EINTR)throw std::runtime_error("Cannot wait for external evaluator");
    if(!WIFEXITED(status))throw std::runtime_error("Evaluator did not exit normally");
    return WEXITSTATUS(status);
}
}

