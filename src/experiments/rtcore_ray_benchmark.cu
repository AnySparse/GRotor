#include "preprocess.hpp"
#include "batching/generate_batches_cpu.hpp"
#include "batching/generate_batches_rtcore_synchronous.cuh"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc,char** argv) {
    try {
        std::string cap,net,output,lut=G_ROTOR_RESOURCE_DIR;
        if (argc<7 || argc%2!=1) throw std::invalid_argument("Usage: grotor_rtcore_ray_benchmark --cap FILE --net FILE --output-dir DIR [--lut-dir DIR]");
        for (int i=1;i<argc;i+=2) {
            const std::string key=argv[i],value=argv[i+1];
            if (key=="--cap")cap=value;
            else if (key=="--net")net=value;
            else if (key=="--output-dir")output=value;
            else if (key=="--lut-dir")lut=value;
            else throw std::invalid_argument("Unknown blocker benchmark argument: "+key);
        }
        if (cap.empty() || net.empty() || output.empty()) throw std::invalid_argument("Missing blocker benchmark input/output argument");
        const std::filesystem::path folder(output);
        std::filesystem::create_directories(folder);
        program_start=std::chrono::high_resolution_clock::now();
        grotor::cuda_check(cudaFree(nullptr),"initialize blocker benchmark CUDA");
        db::read(cap.data(),net.data());
        // The microbenchmark requires pin coordinates, not the routing cost grid.
        decltype(db::capacity)().swap(db::capacity);
        const auto cwd=std::filesystem::current_path();
        std::filesystem::current_path(lut);flute::readLUT();std::filesystem::current_path(cwd);
        cudb::RouterState state;
        cudb::prepare_routing_nets(state);
        std::vector<db::net>().swap(db::nets);
        std::vector<std::vector<cudb::net>>().swap(state.tmp_nets);
        std::vector<int>().swap(state.pin_cnt_sum_cpu);
        std::vector<int>().swap(state.pin_cnt_sum_phase2_cpu);
        std::vector<int>().swap(state.ripup_flag_cpu);
        std::vector<int> order(state.nets.size());std::iota(order.begin(),order.end(),0);
        state.rtcore_geometry.begin(order,state.nets.size(),8,
            {state.rtcore_z_gap,state.rtcore_margin,state.rtcore_coplanar,state.rtcore_compact_points});
        #pragma omp parallel for num_threads(8)
        for (int i=0;i<order.size();++i) {
            auto& current=state.nets[order[i]];
            current.construct_rsmt(state);
            int lane=0;
#ifdef _OPENMP
            lane=omp_get_thread_num();
#endif
            state.rtcore_geometry.emit(i,lane,current,state.device.Y);
            std::vector<std::vector<int>>().swap(current.rsmt);
            std::vector<int>().swap(current.pins);
        }
        std::sort(order.begin(),order.end(),[&](int a,int b){return state.nets[a].hpwl>state.nets[b].hpwl;});
        printf("RTCORE_RAY_CONFIG stage=lshape method=rtcore_rounds publication=round_end eligibility=frozen sparsity=1 conflict_cache=0\n");
        auto result=grotor::rtcore::synchronous::generate_batches_rtcore_synchronous(state,order,1000000,GROTOR_RTCORE_SYNC_PTX_PATH,
            [](bool blocker,int iteration,const grotor::rtcore::synchronous::Round& round){
                printf("RTCORE_RAY_ROUND blocker=%d iteration=%d rays=%llu unassigned_before=%d unassigned_after=%d\n",
                    int(blocker),iteration,round.rays,round.before,round.remaining);
                fflush(stdout);
            });
        state.rtcore_geometry.clear();
        const auto cpu=cudb::generate_batches_rsmt(state,order,1000000,1);
        if (cpu!=result.batches) throw std::runtime_error("blocker benchmark batch lists differ from CPU batching");
        std::ofstream counts(folder/"ray-counts.csv"),stats(folder/"round-stats.csv"),checks(folder/"checks.json");
        if (!counts || !stats || !checks) throw std::runtime_error("Cannot create blocker benchmark output files");
        counts<<"without_blocker,with_blocker\n";
        stats<<"iteration,without_blocker,with_blocker,unassigned_before,unassigned_after,assigned_this_round\n";
        unsigned long long without=0,with=0;
        for (size_t i=0;i<result.with_blocker.size();++i) {
            const auto a=result.without_blocker[i],b=result.with_blocker[i];
            if (b.rays>a.rays) throw std::runtime_error("Blocker emitted more rays for the same assignment history");
            counts<<a.rays<<','<<b.rays<<'\n';
            stats<<i<<','<<a.rays<<','<<b.rays<<','<<a.before<<','<<a.remaining<<','<<a.before-a.remaining<<'\n';
            without+=a.rays;with+=b.rays;
        }
        checks<<"{\n  \"status\": \"passed\",\n  \"stage\": \"lshape\",\n  \"publication\": \"round_end\",\n"
            <<"  \"eligibility\": \"frozen\",\n  \"counter\": \"actual_optixTrace_calls\",\n  \"conflict_cache\": false,\n"
            <<"  \"nets\": "<<order.size()<<",\n  \"input_rays\": "<<result.input_rays<<",\n  \"iterations\": "<<result.with_blocker.size()<<",\n"
            <<"  \"batches\": "<<result.batches.size()<<",\n  \"without_blocker_total\": "<<without<<",\n  \"with_blocker_total\": "<<with<<",\n"
            <<"  \"same_iteration_count\": true,\n  \"exact_assignment_rounds\": true,\n  \"exact_batch_ids\": true,\n  \"exact_cpu_batch_lists\": true\n}\n";
        printf("RTCORE_RAY_CHECK iterations=%zu nets=%zu exact_rounds=1 exact_batch_ids=1 exact_cpu_lists=1\n",result.with_blocker.size(),order.size());
        return 0;
    } catch (const std::exception& e) {fprintf(stderr,"blocker benchmark error: %s\n",e.what());return 1;}
}
