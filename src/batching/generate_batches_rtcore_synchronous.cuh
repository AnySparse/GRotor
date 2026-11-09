#pragma once
#include "rtcore/runtime.cuh"
#include "rtcore/color_kernels.cuh"
#include "rtcore/synchronous_params.h"
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>

namespace grotor::rtcore::synchronous {
struct Round {
    unsigned long long rays = 0;
    int before = 0, remaining = 0;
};
struct Result {
    std::vector<Round> without_blocker, with_blocker;
    std::vector<std::vector<int>> batches;
    size_t input_rays = 0;
    bool exact_assignment_rounds = false, exact_batch_ids = false;
};

__global__ void reset_round(LaunchParams p) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=p.query.total_nets) return;
    const int blocker = p.query.blocker_id[i];
    const bool eligible = p.query.batch_id[i]<0 &&
        (!p.use_blocker || blocker==0 || p.query.batch_id[blocker-1]>=0);
    p.eligible[i] = p.query.is_ready[i] = int(eligible);
    p.pending[i] = -1;
    if (!eligible) return;
    p.query.blocker_id[i] = 0;
    auto* words = reinterpret_cast<uint4*>(p.query.used_mask + size_t(i)*mask_words);
    for (int word=0; word<mask_words/4; ++word) words[word] = make_uint4(0,0,0,0);
}

__global__ void assign_pending(LaunchParams p) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=p.query.total_nets || !p.eligible[i] || !p.query.is_ready[i]) return;
    for (int word=0; word<mask_words; ++word) {
        const unsigned mask = p.query.used_mask[size_t(i)*mask_words+word];
        if (mask != 0xffffffffu) {
            p.pending[i] = word*32+__ffs(~mask)-1;
            return;
        }
    }
}

__global__ void commit_round(LaunchParams p) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=p.query.total_nets || p.pending[i]<0) return;
    p.query.batch_id[i] = p.pending[i];
    p.assigned_round[i] = p.query.iteration;
    atomicSub(p.query.d_unassigned_count,1);
}

__global__ void compare_assignments(LaunchParams p, const int* colors, const int* rounds, int* errors) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=p.query.total_nets) return;
    if (p.query.batch_id[i]!=colors[i]) atomicAdd(errors,1);
    if (p.assigned_round[i]!=rounds[i]) atomicAdd(errors+1,1);
}

// Shares one immutable scene and one workspace between the two ablations.
// No conflict graph is materialized. Counts are taken at actual optixTrace calls.
inline Result generate_batches_rtcore_synchronous(cudb::RouterState& state,
    const std::vector<int>& order, int limit, const std::string& ptx,
    const std::function<void(bool,int,const Round&)>& progress = {}) {
    Result result;
    if (order.empty()) { result.exact_assignment_rounds=result.exact_batch_ids=true; return result; }
    Resources resources;
    auto scene = build_scene(resources,state,order,limit);
    const auto sbt = build_pipeline(resources,ptx,3,"__raygen__rtcore_synchronous","__anyhit__rtcore_synchronous");
    cuda_check(cudaStreamCreateWithFlags(&resources.stream,cudaStreamNonBlocking),"create synchronous batching stream");
    LaunchParams p{};
    auto& q = p.query;
    q.gas_handle=scene.handle;q.ray_origins=scene.origins;q.ray_to_net=scene.ray_to_net;
    q.tri_to_net_id=scene.triangle_to_net;q.num_rays=scene.rays;q.total_nets=scene.total_nets;
    q.ray_tmin=scene.ray_tmin;q.ray_tmax=scene.ray_tmax;
    q.batch_id=resources.allocate<int>(q.total_nets);
    q.is_ready=resources.allocate<int>(q.total_nets);
    q.blocker_id=resources.allocate<int>(q.total_nets);
    q.used_mask=resources.allocate<unsigned>(size_t(q.total_nets)*mask_words);
    q.active_ray_indices=resources.allocate<unsigned>(scene.rays);
    q.d_unassigned_count=resources.allocate<int>(1);
    p.eligible=resources.allocate<int>(q.total_nets);
    p.pending=resources.allocate<int>(q.total_nets);
    p.assigned_round=resources.allocate<int>(q.total_nets);
    const size_t bins=(scene.rays+255)/256;
    p.trace_counts=resources.allocate<unsigned>(bins);
    auto* active_count=resources.allocate<int>(1);
    auto* params=resources.allocate<LaunchParams>(1);
    auto* expected_colors=resources.allocate<int>(q.total_nets);
    auto* expected_rounds=resources.allocate<int>(q.total_nets);
    auto* errors=resources.allocate<int>(2);
    result.input_rays=scene.rays;
    report_memory("synchronous_batch_workspace_ready");
    constexpr int threads=256;
    const int blocks=(q.total_nets+threads-1)/threads;
    // With-blocker first establishes the reference history; the control is an
    // independent reset, not a padded trace or a replay of precomputed colors.
    for (bool blocker : {true,false}) {
        p.use_blocker=blocker;
        cuda_check(cudaMemset(q.batch_id,0xff,size_t(q.total_nets)*sizeof(int)),"reset synchronous batching colors");
        cuda_check(cudaMemset(q.blocker_id,0,size_t(q.total_nets)*sizeof(int)),"reset synchronous batching blockers");
        cuda_check(cudaMemset(p.assigned_round,0xff,size_t(q.total_nets)*sizeof(int)),"reset synchronous batching assignment rounds");
        thrust::sequence(thrust::device,q.active_ray_indices,q.active_ray_indices+scene.rays);
        int remaining=q.total_nets;
        cuda_check(cudaMemcpy(q.d_unassigned_count,&remaining,sizeof(int),cudaMemcpyHostToDevice),"reset synchronous batching remaining count");
        auto& rows=blocker ? result.with_blocker:result.without_blocker;
        while (remaining) {
            q.iteration=rows.size();
            reset_round<<<blocks,threads,0,resources.stream>>>(p);
            int width=q.num_rays;
            if (q.iteration) {
                cuda_check(cudaMemsetAsync(active_count,0,sizeof(int),resources.stream),"reset synchronous batching active count");
                fast_compact_rays_kernel<<<(q.num_rays+255)/256,256,0,resources.stream>>>(
                    q.ray_to_net,q.batch_id,p.eligible,q.active_ray_indices,active_count,q.num_rays);
                cuda_check(cudaMemcpyAsync(&width,active_count,sizeof(int),cudaMemcpyDeviceToHost,resources.stream),"download synchronous batching active count");
                cuda_check(cudaStreamSynchronize(resources.stream),"complete synchronous batching compaction");
            }
            if (!width) throw std::runtime_error("synchronous batching coloring stalled");
            cuda_check(cudaMemsetAsync(p.trace_counts,0,bins*sizeof(unsigned),resources.stream),"reset synchronous batching trace counters");
            cuda_check(cudaMemcpyAsync(params,&p,sizeof(p),cudaMemcpyHostToDevice,resources.stream),"upload synchronous batching launch parameters");
            optix_check(optixLaunch(resources.pipeline,resources.stream,address(params),sizeof(p),&sbt,width,1,1),"synchronous batching ray tracing");
            assign_pending<<<blocks,threads,0,resources.stream>>>(p);
            commit_round<<<blocks,threads,0,resources.stream>>>(p);
            const auto rays=thrust::reduce(thrust::cuda::par.on(resources.stream),p.trace_counts,p.trace_counts+bins,0ULL,thrust::plus<unsigned long long>());
            const int before=remaining;
            cuda_check(cudaMemcpyAsync(&remaining,q.d_unassigned_count,sizeof(int),cudaMemcpyDeviceToHost,resources.stream),"download synchronous batching progress");
            cuda_check(cudaStreamSynchronize(resources.stream),"complete synchronous batching round");
            if (rays!=static_cast<unsigned long long>(width)) throw std::runtime_error("Actual synchronous batching trace count differs from frozen launch eligibility");
            if (remaining>=before || rows.size()>=size_t(q.total_nets))
                throw std::runtime_error("synchronous batching stalled or exhausted its 1024-batch mask");
            rows.push_back({rays,before,remaining});
            if (progress) progress(blocker,q.iteration,rows.back());
            if (!blocker && (rows.size()>result.with_blocker.size() ||
                remaining!=result.with_blocker[rows.size()-1].remaining ||
                before!=result.with_blocker[rows.size()-1].before))
                throw std::runtime_error("Blocker ablation changed the synchronous assignment history");
        }
        if (blocker) {
            cuda_check(cudaMemcpy(expected_colors,q.batch_id,size_t(q.total_nets)*sizeof(int),cudaMemcpyDeviceToDevice),"save synchronous batching colors");
            cuda_check(cudaMemcpy(expected_rounds,p.assigned_round,size_t(q.total_nets)*sizeof(int),cudaMemcpyDeviceToDevice),"save synchronous batching assignment history");
        } else {
            cuda_check(cudaMemset(errors,0,2*sizeof(int)),"reset synchronous batching comparison");
            compare_assignments<<<blocks,threads>>>(p,expected_colors,expected_rounds,errors);
            int counts[2];
            cuda_check(cudaMemcpy(counts,errors,sizeof(counts),cudaMemcpyDeviceToHost),"compare synchronous batching per-net history");
            result.exact_batch_ids=counts[0]==0;
            result.exact_assignment_rounds=counts[1]==0;
        }
    }
    if (!result.exact_batch_ids || !result.exact_assignment_rounds || result.with_blocker.size()!=result.without_blocker.size())
        throw std::runtime_error("synchronous batching variants differ in colors or assignment rounds");
    result.batches=collect_batches(q.batch_id,order,limit);
    return result;
}
} // namespace grotor::rtcore::synchronous
