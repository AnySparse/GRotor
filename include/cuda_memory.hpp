#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace grotor {

inline void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

// Owns device allocations. Kernel views borrow pointers; they never own memory.
// A workspace can release individual buffers or all buffers at a stage boundary.
class DeviceMemory {
public:
    DeviceMemory() = default;
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;
    ~DeviceMemory() { clear(); }

    template <class T>
    void allocate(T** destination, std::size_t bytes) {
        if (*destination) throw std::logic_error("Device buffer already allocated");
        void* pointer = nullptr;
        cuda_check(cudaMalloc(&pointer, std::max(bytes, sizeof(T))), "cudaMalloc");
        allocations_.emplace(pointer, bytes);
        *destination = static_cast<T*>(pointer);
    }

    template <class T>
    void release(T*& pointer) {
        if (!pointer) return;
        auto allocation = allocations_.find(pointer);
        if (allocation == allocations_.end()) throw std::logic_error("Unowned device buffer");
        cuda_check(cudaFree(pointer), "cudaFree");
        allocations_.erase(allocation);
        pointer = nullptr;
    }

    void clear() noexcept {
        for (const auto& allocation : allocations_) cudaFree(allocation.first);
        allocations_.clear();
    }

private:
    std::unordered_map<void*, std::size_t> allocations_;
};

// Owns CPU packing arrays, including error paths.
class HostMemory {
public:
    HostMemory() = default;
    HostMemory(const HostMemory&) = delete;
    HostMemory& operator=(const HostMemory&) = delete;
    ~HostMemory() { clear(); }

    void clear() noexcept {
        for (const auto& allocation : allocations_) allocation.second(allocation.first);
        allocations_.clear();
    }

    template <class T>
    T* allocate(std::size_t count, bool zero = false) {
        T* pointer = zero ? new T[count]() : new T[count];
        allocations_.emplace(pointer, [](void* p) { delete[] static_cast<T*>(p); });
        return pointer;
    }

    template <class T>
    void release(T*& pointer) {
        if (!pointer) return;
        auto allocation = allocations_.find(pointer);
        if (allocation == allocations_.end()) throw std::logic_error("Unowned host buffer");
        allocation->second(pointer);
        allocations_.erase(allocation);
        pointer = nullptr;
    }

private:
    std::unordered_map<void*, void (*)(void*)> allocations_;
};

} // namespace grotor
