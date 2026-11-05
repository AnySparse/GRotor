#pragma once
// HIPRT batching adapted from amd-gr-workspace-4; routing kernels remain shared.
#include "router_state.hpp"
#include <hiprt/hiprt.h>
#include <hiprt/hiprt_device.h>

// HIPRT's direct hipcc path expects these dispatch hooks even when the geometry
// consists only of built-in triangles and no user intersection/filter function
// table is supplied.
HIPRT_DEVICE bool intersectFunc(
    uint32_t, uint32_t, const hiprtFuncTableHeader&, const hiprtRay&, void*, hiprtHit&) {
    return false;
}
HIPRT_DEVICE bool filterFunc(
    uint32_t, uint32_t, const hiprtFuncTableHeader&, const hiprtRay&, void*, const hiprtHit&) {
    return false;
}

#include <hiprt/impl/hiprt_device_impl.h>

static constexpr int HIPRT_BATCH_WORDS = 32;
static constexpr int HIPRT_GRID_CHUNK_NETS = 512;
static constexpr int HIPRT_GRID_PRED_WORDS = HIPRT_GRID_CHUNK_NETS / 32;

struct HiprtInt2 {
    int x, y;
};


namespace grotor::hiprt_batching {
struct HiprtGridColorParams {
    unsigned int *external_mask;
    unsigned int *cell_batch_mask;
    int *batch_id;
    const int *points;
    const int *point_offsets;
    const HiprtInt2 *h_segments;
    const int *h_offsets;
    const HiprtInt2 *v_segments;
    const int *v_offsets;
    int x_size;
    int y_size;
    int *batch_counts;
    int batch_limit;
    bool capacity_order;
};

// Query the batch occupancy of a routing-grid cell through HIPRT.  The scene
// contains one triangle per cell, so closest-hit returns the cell index while
// the actual BVH traversal is executed by Navi3x's RT hardware.
__global__ void hiprt_grid_query_kernel(
    hiprtGeometry geometry,
    const hiprtFloat3 *ray_origins,
    const unsigned int *ray_to_net,
    const unsigned int *cell_batch_mask,
    unsigned int *net_used_mask,
    int ray_offset,
    int ray_count,
    const int *max_batch_id) {
    int local_ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_ray >= ray_count) return;
    int ray_idx = ray_offset + local_ray;

    hiprtRay ray;
    ray.origin = ray_origins[ray_idx];
    ray.direction = {0.0f, 0.0f, -1.0f};
    ray.minT = 0.0f;
    ray.maxT = 2.0f;
    hiprtGeomTraversalClosest traversal(geometry, ray);
    hiprtHit hit = traversal.getNextHit();
    if (!hit.hasHit()) return;

    unsigned int net = ray_to_net[ray_idx];
    size_t src = static_cast<size_t>(hit.primID) * HIPRT_BATCH_WORDS;
    size_t dst = static_cast<size_t>(net) * HIPRT_BATCH_WORDS;
    int max_batch = *max_batch_id;
    int active_batch_words = max_batch < 0 ? 0 : min(HIPRT_BATCH_WORDS, max_batch / 32 + 1);
    for (int word = 0; word < active_batch_words; ++word) {
        unsigned int bits = cell_batch_mask[src + word];
        if (bits) atomicOr(&net_used_mask[dst + word], bits);
    }
}

__device__ __forceinline__ void hiprt_grid_mark_local_net(
    unsigned int *cell_bits, int *cell_epoch, int pos, int local_net, int epoch) {
    volatile int *stamp = &cell_epoch[pos];
    int state = *stamp;
    while (state != epoch) {
        if (state != -epoch) {
            int previous = atomicCAS(&cell_epoch[pos], state, -epoch);
            if (previous == state) {
                size_t base = static_cast<size_t>(pos) * HIPRT_GRID_PRED_WORDS;
                for (int word = 0; word < HIPRT_GRID_PRED_WORDS; ++word)
                    cell_bits[base + word] = 0;
                __threadfence();
                atomicExch(&cell_epoch[pos], epoch);
                break;
            }
            state = previous;
        } else {
            state = *stamp;
        }
    }
    atomicOr(&cell_bits[static_cast<size_t>(pos) * HIPRT_GRID_PRED_WORDS + local_net / 32],
             1u << (local_net & 31));
}

__global__ void hiprt_grid_raster_chunk_kernel(
    HiprtGridColorParams p, unsigned int *cell_bits, int *cell_epoch,
    int begin, int end, int epoch) {
    int net = begin + blockIdx.x;
    if (net >= end) return;
    int local = net - begin;

    for (int i = p.h_offsets[net] + threadIdx.x; i < p.h_offsets[net + 1]; i += blockDim.x) {
        HiprtInt2 segment = p.h_segments[i];
        int lo = min(segment.x, segment.y), hi = max(segment.x, segment.y);
        for (int pos = lo; pos <= hi; pos += p.y_size)
            hiprt_grid_mark_local_net(cell_bits, cell_epoch, pos, local, epoch);
    }
    for (int i = p.v_offsets[net] + threadIdx.x; i < p.v_offsets[net + 1]; i += blockDim.x) {
        HiprtInt2 segment = p.v_segments[i];
        int lo = min(segment.x, segment.y), hi = max(segment.x, segment.y);
        for (int pos = lo; pos <= hi; ++pos)
            hiprt_grid_mark_local_net(cell_bits, cell_epoch, pos, local, epoch);
    }
    for (int i = p.point_offsets[net] + threadIdx.x; i < p.point_offsets[net + 1]; i += blockDim.x) {
        int pos = p.points[i], x = pos / p.y_size, y = pos % p.y_size;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy < 1; ++dy) {
                if (dx != 0 && dy != 0) continue;
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < p.x_size && ny >= 0 && ny < p.y_size)
                    hiprt_grid_mark_local_net(
                        cell_bits, cell_epoch, nx * p.y_size + ny, local, epoch);
            }
        }
    }
}

__global__ void hiprt_grid_collect_predecessors_kernel(
    HiprtGridColorParams p,
    const unsigned int *cell_bits,
    const int *cell_epoch,
    unsigned int *predecessors,
    int begin,
    int end,
    int epoch) {
    int net = begin + blockIdx.x;
    if (net >= end) return;
    __shared__ unsigned int pred[HIPRT_GRID_PRED_WORDS];
    if (threadIdx.x < HIPRT_GRID_PRED_WORDS) pred[threadIdx.x] = 0;
    __syncthreads();

    for (int i = p.point_offsets[net] + threadIdx.x; i < p.point_offsets[net + 1]; i += blockDim.x) {
        int pos = p.points[i];
        if (cell_epoch[pos] == epoch) {
            size_t base = static_cast<size_t>(pos) * HIPRT_GRID_PRED_WORDS;
            for (int word = 0; word < HIPRT_GRID_PRED_WORDS; ++word)
                atomicOr(&pred[word], cell_bits[base + word]);
        }
    }
    __syncthreads();

    if (threadIdx.x < HIPRT_GRID_PRED_WORDS) {
        int local = net - begin;
        int word = threadIdx.x;
        unsigned int lower = word < local / 32
            ? 0xffffffffu
            : (word == local / 32 ? ((1u << (local & 31)) - 1u) : 0u);
        predecessors[static_cast<size_t>(net) * HIPRT_GRID_PRED_WORDS + word] = pred[word] & lower;
    }
}

// Reproduce the CPU greedy order inside a fixed-size chunk.  A net becomes ready
// only after all lower-index conflicting nets have a color.  Scanning the
// occupancy a word at a time avoids a 128-byte private array per thread.
__global__ void hiprt_grid_color_wave_kernel(
    HiprtGridColorParams p,
    const unsigned int *predecessors,
    int *max_batch_id,
    int begin,
    int end) {
    int local = threadIdx.x;
    int net = begin + local;
    __shared__ int ready[HIPRT_GRID_CHUNK_NETS];
    __shared__ int remaining;
    __shared__ int chunk_max;
    __shared__ bool serial_capacity;
    if (local == 0) {
        remaining = end - begin;
        chunk_max = -1;
        // Capacity cannot affect greedy order if this entire chunk fits in
        // every existing batch. Serialize only chunks close to a full batch.
        serial_capacity = false;
        if (p.capacity_order) {
            for (int i = 0; i < HIPRT_BATCH_WORDS * 32; ++i)
                if (p.batch_counts[i] > p.batch_limit - (end - begin)) serial_capacity = true;
        }
    }
    if (net < end) p.batch_id[net] = -1;
    __syncthreads();

    while (remaining > 0) {
        bool can_assign = net < end && p.batch_id[net] == -1;
        if (can_assign && serial_capacity && local > 0 && p.batch_id[net - 1] == -1) can_assign = false;
        if (can_assign) {
            for (int word = 0; word < HIPRT_GRID_PRED_WORDS && can_assign; ++word) {
                unsigned int bits = predecessors[static_cast<size_t>(net) * HIPRT_GRID_PRED_WORDS + word];
                while (bits) {
                    int bit = __ffs(bits) - 1;
                    if (p.batch_id[begin + word * 32 + bit] == -1) can_assign = false;
                    bits &= bits - 1;
                }
            }
        }
        ready[local] = can_assign ? 1 : 0;
        __syncthreads();

        if (can_assign) {
            int selected_batch = -1;
            for (int batch_word = 0; batch_word < HIPRT_BATCH_WORDS; ++batch_word) {
                unsigned int used = p.external_mask[
                    static_cast<size_t>(net) * HIPRT_BATCH_WORDS + batch_word];
                for (int pred_word = 0; pred_word < HIPRT_GRID_PRED_WORDS; ++pred_word) {
                    unsigned int bits = predecessors[
                        static_cast<size_t>(net) * HIPRT_GRID_PRED_WORDS + pred_word];
                    while (bits) {
                        int bit = __ffs(bits) - 1;
                        int pred_batch = p.batch_id[begin + pred_word * 32 + bit];
                        if (pred_batch >= 0 && (pred_batch >> 5) == batch_word)
                            used |= 1u << (pred_batch & 31);
                        bits &= bits - 1;
                    }
                }
                if (used != 0xffffffffu) {
                    while (used != 0xffffffffu) {
                        int candidate = batch_word * 32 + (__ffs(~used) - 1);
                        if (!p.capacity_order || p.batch_counts[candidate] < p.batch_limit) {
                            selected_batch = candidate;
                            break;
                        }
                        used |= 1u << (candidate & 31);
                    }
                    if (selected_batch >= 0) break;
                }
            }
            if (selected_batch >= 0 && p.capacity_order) atomicAdd(p.batch_counts + selected_batch, 1);
            p.batch_id[net] = selected_batch >= 0 ? selected_batch : -2;
            atomicMax(&chunk_max, selected_batch);
        }
        __syncthreads();
        if (local == 0) {
            int assigned = 0;
            for (int i = 0; i < end - begin; ++i) assigned += ready[i];
            remaining -= assigned;
        }
        __syncthreads();
    }
    if (local == 0) atomicMax(max_batch_id, chunk_max);
}

__global__ void hiprt_grid_commit_chunk_kernel(HiprtGridColorParams p, int begin, int end) {
    int net = begin + blockIdx.x;
    if (net >= end) return;
    int batch = p.batch_id[net];
    if (batch < 0) return;
    int word = batch / 32;
    unsigned int bit = 1u << (batch & 31);

    for (int i = p.h_offsets[net] + threadIdx.x; i < p.h_offsets[net + 1]; i += blockDim.x) {
        HiprtInt2 segment = p.h_segments[i];
        int lo = min(segment.x, segment.y), hi = max(segment.x, segment.y);
        for (int pos = lo; pos <= hi; pos += p.y_size)
            atomicOr(&p.cell_batch_mask[static_cast<size_t>(pos) * HIPRT_BATCH_WORDS + word], bit);
    }
    for (int i = p.v_offsets[net] + threadIdx.x; i < p.v_offsets[net + 1]; i += blockDim.x) {
        HiprtInt2 segment = p.v_segments[i];
        int lo = min(segment.x, segment.y), hi = max(segment.x, segment.y);
        for (int pos = lo; pos <= hi; ++pos)
            atomicOr(&p.cell_batch_mask[static_cast<size_t>(pos) * HIPRT_BATCH_WORDS + word], bit);
    }
    for (int i = p.point_offsets[net] + threadIdx.x; i < p.point_offsets[net + 1]; i += blockDim.x) {
        int pos = p.points[i], x = pos / p.y_size, y = pos % p.y_size;
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy < 1; ++dy) {
                if (dx != 0 && dy != 0) continue;
                int nx = x + dx, ny = y + dy;
                if (nx >= 0 && nx < p.x_size && ny >= 0 && ny < p.y_size)
                    atomicOr(&p.cell_batch_mask[
                        static_cast<size_t>(nx * p.y_size + ny) * HIPRT_BATCH_WORDS + word], bit);
            }
        }
    }
}

struct Scene {
    hiprtContext context = nullptr;
    hiprtGeometry geometry = nullptr;
    void clear() noexcept {
        if (geometry) { (void)hiprtDestroyGeometry(context, geometry); geometry = nullptr; }
        if (context) { (void)hiprtDestroyContext(context); context = nullptr; }
    }
    ~Scene() { clear(); }
};
// Execute only batching on the already prepared scene/workspace. Scene setup,
// allocation, uploads and result collection belong to the caller, outside timing.
inline void generate_batch_ids(
        HiprtGridColorParams color, hiprtGeometry geometry,
        const hiprtFloat3* d_ray_origins, const unsigned int* d_ray_to_net,
        unsigned int* d_cell_net_bits, int* d_cell_epoch,
        unsigned int* d_predecessors, int* d_max_batch_id,
        const vector<int>& ray_offsets, int total_nets, size_t cell_batch_bytes) {
    auto check_hip = grotor::cuda_check;
    auto* d_cell_batch_mask = color.cell_batch_mask;
    auto* d_net_used_mask = color.external_mask;
    check_hip(hipMemsetAsync(d_cell_batch_mask, 0, cell_batch_bytes),
              "cell batch mask initialization");

    constexpr int chunk_nets = HIPRT_GRID_CHUNK_NETS;
    constexpr int raster_threads = 128;
    constexpr int query_threads = 256;
    for (int begin = 0; begin < total_nets; begin += chunk_nets) {
        int end = std::min(total_nets, begin + chunk_nets);
        int epoch = begin / chunk_nets + 1;
        check_hip(hipMemsetAsync(
                      d_net_used_mask + static_cast<size_t>(begin) * HIPRT_BATCH_WORDS,
                      0,
                      static_cast<size_t>(end - begin) * HIPRT_BATCH_WORDS * sizeof(unsigned int)),
                  "chunk net mask reset");

        int ray_offset = ray_offsets[begin];
        int ray_count = ray_offsets[end] - ray_offset;
        if (ray_count > 0) {
            int query_blocks = (ray_count + query_threads - 1) / query_threads;
            hiprt_grid_query_kernel<<<query_blocks, query_threads>>>(
                geometry, d_ray_origins, d_ray_to_net, d_cell_batch_mask,
                d_net_used_mask, ray_offset, ray_count, d_max_batch_id);
        }
        hiprt_grid_raster_chunk_kernel<<<end - begin, raster_threads>>>(
            color, d_cell_net_bits, d_cell_epoch, begin, end, epoch);
        hiprt_grid_collect_predecessors_kernel<<<end - begin, raster_threads>>>(
            color, d_cell_net_bits, d_cell_epoch, d_predecessors, begin, end, epoch);
        hiprt_grid_color_wave_kernel<<<1, chunk_nets>>>(
            color, d_predecessors, d_max_batch_id, begin, end);
        hiprt_grid_commit_chunk_kernel<<<end - begin, raster_threads>>>(color, begin, end);
    }

    check_hip(hipGetLastError(), "HIPRT grid batch kernel launch");
    check_hip(hipDeviceSynchronize(), "complete HIPRT batching function");
}

inline vector<vector<int>> generate(cudb::RouterState& state, vector<int>& nets2route, int limit, const char* stage) {
    if (limit < 1) throw std::invalid_argument("HIPRT batch capacity must be positive");
    const int total_nets = static_cast<int>(nets2route.size());
    const int total_cells = state.device.X * state.device.Y;
    if (total_nets == 0) {

        return {};
    }
    if (std::getenv("HIPRT_PATH") == nullptr) setenv("HIPRT_PATH", GROTOR_HIPRT_ROOT, 0);

    auto check_hip = grotor::cuda_check;
    auto check_hiprt = [](hiprtError error, const char *what) {
        if (error != hiprtSuccess) throw std::runtime_error(std::string(what) + ": HIPRT error " + std::to_string(int(error)));
    };
    grotor::DeviceMemory memory;
    Scene scene;
    auto stamp = [] { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - cudb::batch_clock_origin).count(); };
    check_hip(hipDeviceSynchronize(), "before HIPRT preparation");
    // One non-indexed triangle per routing-grid cell.  Primitive ID is then
    // exactly the flattened cell ID returned by closest-hit traversal.
    vector<hiprtFloat3> cell_vertices(static_cast<size_t>(total_cells) * 3);
    for (int cell = 0; cell < total_cells; ++cell) {
        float x = static_cast<float>(cell / state.device.Y);
        float y = static_cast<float>(cell % state.device.Y);
        cell_vertices[static_cast<size_t>(cell) * 3 + 0] = {x - 0.45f, y - 0.45f, 0.0f};
        cell_vertices[static_cast<size_t>(cell) * 3 + 1] = {x + 0.45f, y - 0.45f, 0.0f};
        cell_vertices[static_cast<size_t>(cell) * 3 + 2] = {x, y + 0.45f, 0.0f};
    }

    vector<hiprtFloat3> ray_origins;
    vector<unsigned int> ray_to_net;
    vector<int> ray_offsets(total_nets + 1, 0);
    vector<int> points;
    vector<int> point_offsets(total_nets + 1, 0);
    vector<HiprtInt2> h_segments, v_segments;
    vector<int> h_offsets(total_nets + 1, 0), v_offsets(total_nets + 1, 0);
    for (int i = 0; i < total_nets; ++i) {
        const auto &net = state.nets[nets2route[i]];
        for (int pos : net.points) {
            ray_origins.push_back({static_cast<float>(pos / state.device.Y),
                                   static_cast<float>(pos % state.device.Y), 1.0f});
            ray_to_net.push_back(static_cast<unsigned int>(i));
            points.push_back(pos);
        }
        ray_offsets[i + 1] = static_cast<int>(ray_origins.size());
        point_offsets[i + 1] = static_cast<int>(points.size());
        for (const auto &segment : net.rsmt_h_segments)
            h_segments.push_back({segment.first, segment.second});
        for (const auto &segment : net.rsmt_v_segments)
            v_segments.push_back({segment.first, segment.second});
        h_offsets[i + 1] = static_cast<int>(h_segments.size());
        v_offsets[i + 1] = static_cast<int>(v_segments.size());
    }

    hiprtFloat3 *d_cell_vertices = nullptr, *d_ray_origins = nullptr;
    unsigned int *d_ray_to_net = nullptr;
    int *d_points = nullptr, *d_point_offsets = nullptr;
    HiprtInt2 *d_h_segments = nullptr, *d_v_segments = nullptr;
    int *d_h_offsets = nullptr, *d_v_offsets = nullptr;

    auto alloc_copy = [&](auto **device, const auto &host, const char *what) {
        using Element = typename std::decay_t<decltype(host)>::value_type;
        size_t bytes = host.size() * sizeof(Element);
        if (bytes == 0) return;
        memory.allocate(device, bytes);
        check_hip(hipMemcpy(*device, host.data(), bytes, hipMemcpyHostToDevice), what);
    };
    alloc_copy(&d_cell_vertices, cell_vertices, "grid vertex upload");
    alloc_copy(&d_ray_origins, ray_origins, "ray origin upload");
    alloc_copy(&d_ray_to_net, ray_to_net, "ray map upload");
    alloc_copy(&d_points, points, "point upload");
    alloc_copy(&d_point_offsets, point_offsets, "point offset upload");
    alloc_copy(&d_h_segments, h_segments, "horizontal segment upload");
    alloc_copy(&d_h_offsets, h_offsets, "horizontal offset upload");
    alloc_copy(&d_v_segments, v_segments, "vertical segment upload");
    alloc_copy(&d_v_offsets, v_offsets, "vertical offset upload");

    int device_index = 0;
    check_hip(hipGetDevice(&device_index), "hipGetDevice");
    hipDevice_t device = 0;
    hipCtx_t hip_context = nullptr;
    check_hip(hipDeviceGet(&device, device_index), "hipDeviceGet");
    check_hip(hipCtxGetCurrent(&hip_context), "hipCtxGetCurrent");
    if (hip_context == nullptr) {
        check_hip(hipFree(nullptr), "initialize HIP context");
        check_hip(hipCtxGetCurrent(&hip_context), "hipCtxGetCurrent after initialization");
    }
    hiprtContextCreationInput context_input{};
    context_input.ctxt = hip_context;
    context_input.device = device;
    context_input.deviceType = hiprtDeviceAMD;
    auto& context = scene.context;
    check_hiprt(hiprtCreateContext(HIPRT_API_VERSION, context_input, context), "hiprtCreateContext");

    hiprtTriangleMeshPrimitive mesh{};
    mesh.vertices = d_cell_vertices;
    mesh.vertexCount = static_cast<unsigned int>(cell_vertices.size());
    mesh.vertexStride = sizeof(hiprtFloat3);
    mesh.triangleIndices = nullptr;
    mesh.triangleCount = static_cast<unsigned int>(total_cells);
    mesh.triangleStride = 0;
    hiprtGeometryBuildInput build_input{};
    build_input.type = hiprtPrimitiveTypeTriangleMesh;
    build_input.primitive.triangleMesh = mesh;
    hiprtBuildOptions build_options{};
    build_options.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    size_t temporary_size = 0;
    check_hiprt(hiprtGetGeometryBuildTemporaryBufferSize(
                    context, build_input, build_options, temporary_size),
                "hiprtGetGeometryBuildTemporaryBufferSize");
    unsigned char *d_temporary = nullptr;
    if (temporary_size != 0)
        memory.allocate(&d_temporary, temporary_size);
    auto& geometry = scene.geometry;
    check_hiprt(hiprtCreateGeometry(context, build_input, build_options, geometry),
                "hiprtCreateGeometry");
    check_hiprt(hiprtBuildGeometry(context, hiprtBuildOperationBuild, build_input, build_options,
                                  d_temporary, nullptr, geometry),
                "hiprtBuildGeometry");
    check_hip(hipDeviceSynchronize(), "BVH build synchronization");

    memory.release(d_temporary);
    memory.release(d_cell_vertices);
    vector<hiprtFloat3>().swap(cell_vertices);
    unsigned int *d_cell_batch_mask = nullptr, *d_net_used_mask = nullptr;
    unsigned int *d_cell_net_bits = nullptr, *d_predecessors = nullptr;
    int *d_cell_epoch = nullptr, *d_batch_id = nullptr, *d_max_batch_id = nullptr;
    const size_t cell_batch_bytes = static_cast<size_t>(total_cells) *
                                    HIPRT_BATCH_WORDS * sizeof(unsigned int);
    const size_t net_mask_bytes = static_cast<size_t>(total_nets) *
                                  HIPRT_BATCH_WORDS * sizeof(unsigned int);
    const size_t cell_net_bytes = static_cast<size_t>(total_cells) *
                                  HIPRT_GRID_PRED_WORDS * sizeof(unsigned int);
    const size_t predecessor_bytes = static_cast<size_t>(total_nets) *
                                     HIPRT_GRID_PRED_WORDS * sizeof(unsigned int);
    memory.allocate(&d_cell_batch_mask, cell_batch_bytes);
    memory.allocate(&d_net_used_mask, net_mask_bytes);
    memory.allocate(&d_cell_net_bits, cell_net_bytes);
    memory.allocate(&d_cell_epoch, static_cast<size_t>(total_cells) * sizeof(int));
    memory.allocate(&d_predecessors, predecessor_bytes);
    memory.allocate(&d_batch_id, static_cast<size_t>(total_nets) * sizeof(int));
    memory.allocate(&d_max_batch_id, sizeof(int));
    check_hip(hipMemset(d_cell_epoch, 0, static_cast<size_t>(total_cells) * sizeof(int)),
              "cell epoch initialization");
    check_hip(hipMemset(d_max_batch_id, 0xff, sizeof(int)), "maximum batch id initialization");

    HiprtGridColorParams color{};
    color.external_mask = d_net_used_mask;
    color.cell_batch_mask = d_cell_batch_mask;
    color.batch_id = d_batch_id;
    color.points = d_points;
    color.point_offsets = d_point_offsets;
    color.h_segments = d_h_segments;
    color.h_offsets = d_h_offsets;
    color.v_segments = d_v_segments;
    color.v_offsets = d_v_offsets;
    color.x_size = state.device.X;
    color.y_size = state.device.Y;

    int *d_batch_counts = nullptr;
    memory.allocate(&d_batch_counts, HIPRT_BATCH_WORDS * 32 * sizeof(int));
    check_hip(hipMemset(d_batch_counts, 0, HIPRT_BATCH_WORDS * 32 * sizeof(int)), "batch counts");
    color.batch_counts = d_batch_counts;
    color.batch_limit = limit;
    color.capacity_order = total_nets > limit;
    check_hip(hipDeviceSynchronize(), "before HIPRT batch interval");
    const auto batch_start = stamp();
    generate_batch_ids(color, geometry, d_ray_origins, d_ray_to_net,
                       d_cell_net_bits, d_cell_epoch, d_predecessors, d_max_batch_id,
                       ray_offsets, total_nets, cell_batch_bytes);
    const auto batch_end = stamp();

    vector<int> batch_ids(total_nets);
    check_hip(hipMemcpy(batch_ids.data(), d_batch_id,
                        static_cast<size_t>(total_nets) * sizeof(int), hipMemcpyDeviceToHost),
              "batch id download");
    int max_batch = *std::max_element(batch_ids.begin(), batch_ids.end());
    if (max_batch < 0 || max_batch >= HIPRT_BATCH_WORDS * 32) {
        std::cerr << "HIPRT grid batching produced an invalid batch id " << max_batch << std::endl;
        throw std::runtime_error("HIPRT batch color exhaustion or invalid assignment");
    }
    vector<vector<int>> batches(max_batch + 1);
    for (int i = 0; i < total_nets; ++i) {
        if (batch_ids[i] < 0 || batch_ids[i] > max_batch) {
            std::cerr << "HIPRT grid batching left net " << i << " unassigned" << std::endl;
            throw std::runtime_error("HIPRT batch color exhaustion or invalid assignment");
        }
        batches[batch_ids[i]].push_back(nets2route[i]);
    }

    scene.clear();
    memory.clear();
    if (state.validate_rtcore_batches) {
        auto cpu = cudb::generate_batches_rsmt(state, nets2route, limit, 1);
        const bool equal = cpu == batches;
        printf("BATCH_VALIDATION stage=%s nets=%zu hiprt_batches=%zu cpu_batches=%zu exact_lists_equal=%d\n", stage, nets2route.size(), batches.size(), cpu.size(), int(equal));
        if (!equal) throw std::runtime_error("HIPRT batches differ from CPU (members/order)");
    }
    printf("BATCH_SCOPE backend=hiprt scope=generate_batch_ids_call_wall\n");
    printf("BATCH_TIMING backend=hiprt start_us=%lld end_us=%lld elapsed_ms=%.6f nets=%zu batches=%zu used_for_routing=1\n", (long long)batch_start, (long long)batch_end, (batch_end-batch_start)/1000., nets2route.size(), batches.size());
    cout << setw(40) << "HIPRT Grid Batch" << setw(20) << "#Nets" << setw(20) << "#Batches" << endl;
    cout << setw(40) << "Generation" << setw(20) << nets2route.size() << setw(20) << batches.size() << endl;
    return batches;
}

} // namespace grotor::hiprt_batching
