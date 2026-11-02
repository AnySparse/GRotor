#pragma once

#include "fenwick.cuh"
#ifdef GROTOR_PREFIX_ACCESS_COUNTS
#include "experiments/prefix_queries.cuh"
#endif

namespace graph {

__device__ double wire_segment_cost(const RoutingDeviceView& view, int layer, int xmin, int xmax, int ymin, int ymax) {
    auto& X = view.X;
    auto& Y = view.Y;
    auto& presum = view.presum;
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

__global__ void compute_presum(RoutingDeviceView view) {
    auto& DIR = view.DIR;
    auto& X = view.X;
    auto& Y = view.Y;
    auto& XY = view.XY;
    auto& wcost = view.wcost;
    auto& idx2track = view.idx2track;
    auto& presum = view.presum;
    extern __shared__ double sum[];
    if(threadIdx.x == 0) sum[0] = 0;
    int l = idx2track[blockIdx.x] / XY;
    if(l & 1 ^ DIR) {
        int x = idx2track[blockIdx.x] % XY;
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
        int y = idx2track[blockIdx.x] % XY;
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
