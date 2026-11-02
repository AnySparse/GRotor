#include "experiments/lshape_benchmark.cuh"
#include "experiments/prefix_update.cuh"

using namespace grotor::experiments;

#ifndef GROTOR_PREFIX_ACCESS_COUNTS
#error "Prefix access benchmark requires the isolated prefix-access profiling target"
#endif

int main(int argc,char** argv) {
    std::filesystem::path temporary;
    try {
        std::string cap,net,output,evaluator,lut=G_ROTOR_RESOURCE_DIR;
        if(argc<11 || argc%2!=1)throw std::invalid_argument("Usage: grotor_prefix_access_benchmark --cap FILE --net FILE --output-dir DIR --route-output FILE --evaluator FILE [--lut-dir DIR]");
        for(int i=1;i<argc;i+=2) {
            const std::string key=argv[i],value=argv[i+1];
            if(key=="--cap")cap=value;
            else if(key=="--net")net=value;
            else if(key=="--output-dir")output=value;
            else if(key=="--route-output")temporary=value;
            else if(key=="--evaluator")evaluator=value;
            else if(key=="--lut-dir")lut=value;
            else throw std::invalid_argument("Unknown Prefix access benchmark argument: "+key);
        }
        if(cap.empty()||net.empty()||output.empty()||temporary.empty()||evaluator.empty())throw std::invalid_argument("Missing Prefix access benchmark arguments");
        std::filesystem::create_directories(output);
        program_start=std::chrono::high_resolution_clock::now();
        grotor::cuda_check(cudaFree(nullptr),"initialize Prefix access benchmark CUDA");
        db::read(cap.data(),net.data());
        const auto cwd=std::filesystem::current_path();std::filesystem::current_path(lut);flute::readLUT();std::filesystem::current_path(cwd);
        cudb::RouterState state;cudb::build_cuda_database(state);
        grotor::cuda_check(cudaDeviceSynchronize(),"finish Prefix access benchmark database initialization");
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
        grotor::cuda_check(cudaFuncSetAttribute(graph::compute_presum,cudaFuncAttributeMaxDynamicSharedMemorySize,v.XY*sizeof(double)),"Prefix access benchmark prefix shared memory");
        grotor::cuda_check(cudaFuncSetAttribute(graph::compute_presum_general,cudaFuncAttributeMaxDynamicSharedMemorySize,v.XY*sizeof(int)),"Prefix access benchmark demand shared memory");
        // Load both solver entry points outside all measured batch intervals.
        Lshape_route::Lshape_route_cuda<<<1,THREAD_NUM>>>(v,0,0,work.node_offsets,work.nodes,work.parents,work.dist,work.from,work.ranges,0);
        Lshape_route::Lshape_route_warp<<<1,128>>>(v,0,0,work.node_offsets,work.nodes,work.parents,work.dist,work.from,work.ranges,0);
        grotor::cuda_check(cudaDeviceSynchronize(),"warm Prefix access benchmark solver entries");
        const auto initial=state.statistics;
        cudb::PrefixAccessCounts* counts=nullptr;
        state.memory.allocate(&counts,sizeof(*counts));v.prefix_access=counts;
        size_t max_threads=0;
        for(const auto& batch:work.batches)max_threads=std::max(max_threads,size_t((batch.nets+3)/4)*128);
        state.memory.allocate(&v.prefix_queries,max_threads*sizeof(cudb::PrefixQueryCounts));
        std::vector<int> statuses;
        const std::filesystem::path folder(output);
        for(int mode=0;mode<3;++mode) {
            const std::string label=mode==0?"original":mode==1?"bit":"bit-pruning";
            work.reset(state,initial);v.prefix_query_mode=mode;
            if(mode)initialize_direct_bit(state);
            printf("PREFIX_ACCESS_CONFIG mode=%s solver=warp prefix=%s cost=%s\n",label.c_str(),mode?"direct_bit":"dense",mode?"incremental":"full");
            std::ofstream table(folder/(label+".csv"));
            if(!table)throw std::runtime_error("Cannot write Prefix access benchmark counts");
            table<<"batch,color_id,nets,nodes,plan_hash,query_calls,query_reads,prefix_writes,total_accesses,maintenance_reads\n";
            for(int i=0;i<work.batches.size();++i){
                const auto& b=work.batches[i];
                const int threads=(b.nets+3)/4*128;
                grotor::cuda_check(cudaMemset(counts,0,sizeof(*counts)),"reset prefix counters");
                grotor::cuda_check(cudaMemset(v.prefix_queries,0,threads*sizeof(cudb::PrefixQueryCounts)),"reset thread query counters");
                const int stamp=++state.global_timestamp;
                refresh_prefix(state);
                Lshape_route::Lshape_route_warp<<<threads/128,128>>>(v,b.nets,b.begin,work.node_offsets,work.nodes,work.parents,work.dist,work.from,work.ranges,stamp);
                graph::batch_wire_update(state,stamp);
                graph::commit_via_demand<<<BLOCK_NUM(b.nets),THREAD_NUM>>>(v,b.nets,b.begin,stamp);
                reduce_query_counts<<<(threads+255)/256,256>>>(v,threads);
                cudb::PrefixAccessCounts measured;
                grotor::cuda_check(cudaMemcpy(&measured,counts,sizeof(measured),cudaMemcpyDeviceToHost),"download prefix access counts");
                const auto total=measured.query_reads+measured.prefix_writes;
                table<<i<<','<<b.color<<','<<b.nets<<','<<b.nodes<<','<<b.plan_hash<<','<<measured.queries<<','
                     <<measured.query_reads<<','<<measured.prefix_writes<<','<<total<<','<<measured.maintenance_reads<<'\n';
                printf("PREFIX_ACCESS_BATCH mode=%s batch=%d nets=%d reads=%llu writes=%llu total=%llu\n",label.c_str(),i,b.nets,measured.query_reads,measured.prefix_writes,total);
            }
            table.close();
            write_output(state,temporary);
            const int code=evaluate(evaluator,cap,net,temporary,folder/(label+".evaluator.log"));
            std::filesystem::remove(temporary);statuses.push_back(code);
            printf("PREFIX_ACCESS_EVALUATED mode=%s exit_status=%d output_removed=1\n",label.c_str(),code);
            if(code!=0 && code!=1)throw std::runtime_error("External evaluator failed");
        }
        std::ofstream report(folder/"measurement.json");
        if(!report)throw std::runtime_error("Cannot write Prefix access benchmark measurement metadata");
        report<<"{\n  \"status\": \"completed\",\n  \"stage\": \"lshape\",\n"
            <<"  \"shared_batch_plan\": true,\n  \"same_initial_demand\": true,\n  \"sparsity\": 1,\n"
            <<"  \"nets\": "<<v.NET_NUM<<",\n  \"batches\": "<<work.batches.size()<<",\n"
            <<"  \"cells\": "<<size_t(v.L)*v.X*v.Y<<",\n  \"solver\": \"warp\",\n  \"query_schedule\": \"warp_broadcast\",\n"
            <<"  \"counter\": \"logical_prefix_elements\",\n  \"total_scope\": \"query_reads_plus_update_writes\",\n"
            <<"  \"bit_storage\": \"full_shifted_bit_without_dense_cache\",\n  \"first_batch_initialization_included\": true,\n"
            <<"  \"evaluator_exit_status\": ["<<statuses[0]<<","<<statuses[1]<<","<<statuses[2]<<"],\n  \"routing_outputs_retained\": 0\n}\n";
        return 0;
    }catch(const std::exception& e){
        if(out_file){fclose(out_file);out_file=nullptr;}
        if(!temporary.empty()){std::error_code ignored;std::filesystem::remove(temporary,ignored);}
        fprintf(stderr,"Prefix access benchmark error: %s\n",e.what());return 1;
    }
}
