#pragma once

// Keep one routing source tree; map only the runtime API on HIP.
#ifdef GROTOR_USE_HIP
#include <hip/hip_runtime.h>
#define cudaDevAttrMultiProcessorCount hipDevAttrMultiProcessorCount
#define cudaDevP2PAttrNativeAtomicSupported hipDevP2PAttrNativeAtomicSupported
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetP2PAttribute hipDeviceGetP2PAttribute
#define cudaDeviceProp hipDeviceProp
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributes hipFuncAttributes
#define cudaFuncGetAttributes(attr, kernel) hipFuncGetAttributes(attr, reinterpret_cast<const void*>(kernel))
#define cudaFuncSetAttribute(kernel, attr, value) hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attr, value)
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipMallocHost
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

// Logical 32-lane warp operations. Runtime rejects wave64 devices.
__device__ inline void gr_syncwarp(unsigned mask = 0xffffffffu) { __syncwarp(static_cast<unsigned long long>(mask)); }
__device__ inline unsigned gr_ballot_sync(unsigned mask, int value) { return static_cast<unsigned>(__ballot_sync(static_cast<unsigned long long>(mask), value)); }
template<class T> __device__ inline unsigned gr_match_any_sync(unsigned mask, T value) { return static_cast<unsigned>(__match_any_sync(static_cast<unsigned long long>(mask), value)); }
template<class T> __device__ inline T gr_shfl_sync(unsigned mask, T value, int lane, int width = 32) { return __shfl_sync(static_cast<unsigned long long>(mask), value, lane, width); }
template<class T> __device__ inline T gr_shfl_up_sync(unsigned mask, T value, int lane, int width = 32) { return __shfl_up_sync(static_cast<unsigned long long>(mask), value, lane, width); }
template<class T> __device__ inline T gr_shfl_down_sync(unsigned mask, T value, int lane, int width = 32) { return __shfl_down_sync(static_cast<unsigned long long>(mask), value, lane, width); }
template<class T> __device__ inline T gr_shfl_xor_sync(unsigned mask, T value, int lane, int width = 32) { return __shfl_xor_sync(static_cast<unsigned long long>(mask), value, lane, width); }
#else
#include <cuda_runtime.h>
#define gr_syncwarp __syncwarp
#define gr_ballot_sync __ballot_sync
#define gr_match_any_sync __match_any_sync
#define gr_shfl_sync __shfl_sync
#define gr_shfl_up_sync __shfl_up_sync
#define gr_shfl_down_sync __shfl_down_sync
#define gr_shfl_xor_sync __shfl_xor_sync
#endif
