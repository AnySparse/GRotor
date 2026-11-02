#include "experiments/lshape_benchmark.cuh"

using namespace grotor::experiments;

int main(int argc,char** argv) {
    std::filesystem::path temporary;
    try {
        std::string cap,net,output,evaluator,lut=G_ROTOR_RESOURCE_DIR;
        if(argc<11 || argc%2!=1)throw std::invalid_argument("Usage: grotor_dag_throughput --cap FILE --net FILE --output-dir DIR --route-output FILE --evaluator FILE [--lut-dir DIR]");
        for(int i=1;i<argc;i+=2) {
            const std::string key=argv[i],value=argv[i+1];
            if(key=="--cap")cap=value;
            else if(key=="--net")net=value;
            else if(key=="--output-dir")output=value;
            else if(key=="--route-output")temporary=value;
            else if(key=="--evaluator")evaluator=value;
            else if(key=="--lut-dir")lut=value;
            else throw std::invalid_argument("Unknown DAG throughput benchmark argument: "+key);
        }
        if(cap.empty()||net.empty()||output.empty()||temporary.empty()||evaluator.empty())throw std::invalid_argument("Missing DAG throughput benchmark arguments");
        std::filesystem::create_directories(output);
        program_start=std::chrono::high_resolution_clock::now();
        grotor::cuda_check(cudaFree(nullptr),"initialize DAG throughput benchmark CUDA");
        db::read(cap.data(),net.data());
        const auto cwd=std::filesystem::current_path();std::filesystem::current_path(lut);flute::readLUT();std::filesystem::current_path(cwd);
        cudb::RouterState state;cudb::build_cuda_database(state);
        grotor::cuda_check(cudaDeviceSynchronize(),"finish DAG throughput benchmark database initialization");
        decltype(db::capacity)().swap(db::capacity);
        std::vector<std::vector<cudb::net>>().swap(state.tmp_nets);
        std::vector<int> order(state.nets.size());std::iota(order.begin(),order.end(),0);
        if(order.empty())throw std::runtime_error("No L-shape routing nets");
        #pragma omp parallel for num_threads(8)
        for(int i=0;i<order.size();++i)state.nets[order[i]].construct_rsmt(state);
        std::sort(order.begin(),order.end(),[&](int a,int b){return state.nets[a].hpwl>state.nets[b].hpwl;});
        auto colors=cudb::generate_batches_rsmt(state,order,1000000,1);
        Workspace work(state,colors);
        std::vector<std::vector<int>>().swap(colors);std::vector<int>().swap(order);
        std::vector<cudb::net>().swap(state.nets);
        auto& v=state.device;
        grotor::cuda_check(cudaFuncSetAttribute(graph::compute_presum,cudaFuncAttributeMaxDynamicSharedMemorySize,v.XY*sizeof(double)),"DAG throughput benchmark prefix shared memory");
        grotor::cuda_check(cudaFuncSetAttribute(graph::compute_presum_general,cudaFuncAttributeMaxDynamicSharedMemorySize,v.XY*sizeof(int)),"DAG throughput benchmark demand shared memory");
        // Load both solver entry points outside all measured batch intervals.
        Lshape_route::Lshape_route_cuda<<<1,THREAD_NUM>>>(v,0,0,work.node_offsets,work.nodes,work.parents,work.dist,work.from,work.ranges,0);
        Lshape_route::Lshape_route_warp<<<1,128>>>(v,0,0,work.node_offsets,work.nodes,work.parents,work.dist,work.from,work.ranges,0);
        grotor::cuda_check(cudaDeviceSynchronize(),"warm DAG throughput benchmark solver entries");
        const auto initial=state.statistics;
        std::vector<int> statuses;
        const std::filesystem::path folder(output);
        for(bool optimized:{false,true}) {
            const std::string label=optimized?"with-optimizations":"without-optimizations";
            work.reset(state,initial);
            if(optimized)graph::initialize_incremental_cost(state);
            if((v.dirty_wire!=nullptr)!=optimized)throw std::runtime_error("Unexpected DAG throughput benchmark cost-update mode");
            printf("DAG_THROUGHPUT_CONFIG optimized=%d solver=%s cost=%s prefix=%s\n",int(optimized),optimized?"warp":"thread",optimized?"incremental":"full",optimized?"bit":"full");
            work.execute(state,optimized,folder/(label+".csv"));
            write_output(state,temporary);
            const int code=evaluate(evaluator,cap,net,temporary,folder/(label+".evaluator.log"));
            std::filesystem::remove(temporary);statuses.push_back(code);
            printf("DAG_THROUGHPUT_EVALUATED optimized=%d exit_status=%d output_removed=1\n",int(optimized),code);
            if(code!=0 && code!=1)throw std::runtime_error("External evaluator failed");
        }
        std::ofstream report(folder/"measurement.json");
        if(!report)throw std::runtime_error("Cannot write DAG throughput benchmark measurement metadata");
        report<<"{\n  \"status\": \"completed\",\n  \"stage\": \"lshape\",\n  \"shared_batch_plan\": true,\n"
            <<"  \"same_initial_demand\": true,\n  \"sparsity\": 1,\n  \"nets\": "<<v.NET_NUM<<",\n  \"batches\": "<<work.batches.size()<<",\n"
            <<"  \"timer\": \"synchronized_steady_clock\",\n  \"timing_scope\": \"cost_prefix_solve_and_demand_commit\",\n"
            <<"  \"without_solver\": \"thread\",\n  \"without_cost\": \"full\",\n  \"without_prefix\": \"full\",\n"
            <<"  \"with_solver\": \"warp\",\n  \"with_cost\": \"incremental\",\n  \"with_prefix\": \"bit\",\n"
            <<"  \"evaluator_exit_status\": ["<<statuses[0]<<","<<statuses[1]<<"],\n  \"routing_outputs_retained\": 0\n}\n";
        return 0;
    }catch(const std::exception& e){
        if(out_file){fclose(out_file);out_file=nullptr;}
        if(!temporary.empty()){std::error_code ignored;std::filesystem::remove(temporary,ignored);}
        fprintf(stderr,"DAG throughput benchmark error: %s\n",e.what());return 1;
    }
}
