#pragma once

#include "fenwick.cuh"
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
#include "experiments/prefix_queries.cuh"
#endif

namespace graph {

__device__ double wire_segment_cost(const RoutingDeviceView& view, int layer, int xmin, int xmax, int ymin, int ymax) {
    auto& X = view.X;
    auto& Y = view.Y;
    const auto presum = cudb::prefix_costs(view);
#ifdef GROTOR_MULTI_GRID
    if (view.local_presum) {
        if (xmin == xmax && ymin == ymax) return 0;
        // Each shard stores an inclusive local scan, as in my-gr. A query
        // combines only its intersected shards; no global prefix propagation.
        if ((layer & 1) ^ view.DIR) {
            return presum[IDX(layer, xmin, ymax - 1)]
                - (ymin ? presum[IDX(layer, xmin, ymin - 1)] : 0.);
        }
        double result = 0;
        const int chunk = view.cost_shard_x ? view.cost_shard_x : X;
        for (int begin = xmin; begin < xmax;) {
            const int base = begin / chunk * chunk;
            const int end = min(xmax, base + chunk);
            result += presum[IDX(layer, end - 1, ymin)]
                - (begin > base ? presum[IDX(layer, begin - 1, ymin)] : 0.);
            begin = end;
        }
        return result;
    }
#endif
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
    return grotor::experiments::wire_cost_counted(view, layer, xmin, xmax, ymin, ymax);
#else
    return presum[IDX(layer, xmax, ymax)] - presum[IDX(layer, xmin, ymin)];
#endif
}

__global__ void compute_presum_general(RoutingDeviceView view, int *to_sum) {
    auto& DIR = view.DIR;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& XY = view.XY;
    auto& idx2track = view.idx2track;
    extern __shared__ int sum2[];
    if(threadIdx.x == 0) sum2[0] = 0;
    int l = idx2track[blockIdx.x] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[blockIdx.x] % XY;
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum2[y + 1] = to_sum[IDX(l, x, y)];
        __syncthreads();
        for(int d = 0; (1 << d) < Y; d++) {
            for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
                if(idx >> d & 1) sum2[idx] += sum2[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y; y += blockDim.x) to_sum[IDX(l, x, y)] = sum2[y];
    } else {
        int y = idx2track[blockIdx.x] % XY;
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum2[x + 1] = to_sum[IDX(l, x, y)];
        __syncthreads();
        for(int d = 0; (1 << d) < X; d++) {
            for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
                if(idx >> d & 1) sum2[idx] += sum2[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int x = threadIdx.x; x < X; x += blockDim.x) to_sum[IDX(l, x, y)] = sum2[x];
    }
}

__global__ void compute_presum(const GROTOR_GRID_CONSTANT RoutingDeviceView view, int rank = 0, int ranks = 1) {
    auto& DIR = view.DIR;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& XY = view.XY;
    const auto wcost = cudb::wcost_values(view);
    auto& idx2track = view.idx2track;
    const auto presum = cudb::prefix_costs(view);
    extern __shared__ double sum[];
    const int track = blockIdx.x * ranks + rank;
    if (track >= view.all_track_cnt) return;
    if(threadIdx.x == 0) sum[0] = 0;
    int l = idx2track[track] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[track] % XY;
        for(int y = threadIdx.x; y < Y - 1; y += blockDim.x) sum[y + 1] = wcost[l * X * Y + x * Y + y];
        __syncthreads();
        for(int d = 0; (1 << d) < Y; d++) {
            for(int idx = threadIdx.x; idx < Y; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int y = threadIdx.x; y < Y; y += blockDim.x) presum[l * X * Y + x * Y + y] = sum[y];
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
        if (!threadIdx.x && view.prefix_access) atomicAdd(&view.prefix_access->prefix_writes, (unsigned long long)Y);
#endif
    } else {
        int y = idx2track[track] % XY;
        for(int x = threadIdx.x; x < X - 1; x += blockDim.x) sum[x + 1] = wcost[l * X * Y + x * Y + y];
        __syncthreads();
        for(int d = 0; (1 << d) < X; d++) {
            for(int idx = threadIdx.x; idx < X; idx += blockDim.x) 
                if(idx >> d & 1) sum[idx] += sum[(idx >> d << d) - 1];
            __syncthreads();
        }
        for(int x = threadIdx.x; x < X; x += blockDim.x) presum[l * X * Y + x * Y + y] = sum[x];
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
        if (!threadIdx.x && view.prefix_access) atomicAdd(&view.prefix_access->prefix_writes, (unsigned long long)X);
#endif
    }
}

} // namespace graph
