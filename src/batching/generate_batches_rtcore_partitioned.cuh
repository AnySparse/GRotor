#pragma once
#include "generate_batches_rtcore_cached.cuh"
#include "generate_batches_rtcore.cuh"

namespace grotor::rtcore::partitioned {

// Completed scenes are summarized by the same directed occupancy predicate as
// the CPU planner. A cell stores the colors which a later query must avoid.
__device__ __forceinline__ void mark_cell(unsigned int* cells, int X, int Y,
                                         int x, int y, int color) {
    if (x >= 0 && x < X && y >= 0 && y < Y)
        atomicOr(cells + (size_t(x) * Y + y) * cached::mask_words + color / 32,
                 1u << (color % 32));
}

__global__ void mark_segments(const Triangle2D* triangles, size_t count, size_t first,
        const unsigned int* segment_ends, const int* colors, unsigned int* cells,
        int X, int Y, float margin) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto t = triangles[i];
    if (first + i >= segment_ends[t.slot]) return; // Point triangles are marked from rays below.
    const int color = colors[t.slot];
    if (t.a.x == t.b.x) {
        const int lo = __float2int_rn(t.a.x + margin), hi = __float2int_rn(t.c.x - margin);
        const int y = __float2int_rn(t.c.y);
        for (int x = lo; x <= hi; ++x) mark_cell(cells, X, Y, x, y, color);
    } else {
        const int lo = __float2int_rn(t.a.y + margin), hi = __float2int_rn(t.c.y - margin);
        const int x = __float2int_rn(t.c.x);
        for (int y = lo; y <= hi; ++y) mark_cell(cells, X, Y, x, y, color);
    }
}

__global__ void mark_points(const Ray2D* rays, size_t count, const int* colors,
                            unsigned int* cells, int X, int Y) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto p = rays[i];
    const int x = __float2int_rn(p.origin.x), y = __float2int_rn(p.origin.y), color = colors[p.slot];
    mark_cell(cells, X, Y, x, y, color);
    mark_cell(cells, X, Y, x - 1, y, color);
    mark_cell(cells, X, Y, x + 1, y, color);
    mark_cell(cells, X, Y, x, y - 1, color); // CPU mark_3x3 excludes upper y.
}

inline void copy_window(const PreparedGeometry& source, const std::vector<int>& order,
                         size_t begin, size_t end, PreparedGeometry& target) {
    target.config = source.config;
    target.config.partitioned = false;
    target.arenas.resize(1);
    target.spans.resize(end - begin);
    target.net_to_slot.resize(end - begin);
    auto& arena = target.arenas[0];
    size_t triangles = 0, rays = 0;
    for (size_t i = begin; i < end; ++i) {
        const auto span = source.spans[source.net_to_slot[order[i]]];
        triangles += span.triangles; rays += span.rays;
    }
    arena.triangles.reserve(triangles); arena.rays.reserve(rays);
    for (size_t i = begin; i < end; ++i) {
        const auto span = source.spans[source.net_to_slot[order[i]]];
        const auto& from = source.arenas[span.lane];
        const unsigned int slot = i - begin;
        target.net_to_slot[slot] = slot;
        target.spans[slot] = {static_cast<unsigned int>(arena.triangles.size()), span.triangles,
                              static_cast<unsigned int>(arena.rays.size()), span.rays, 0};
        for (size_t j = 0; j < span.triangles; ++j) {
            auto t = from.triangles[span.triangle_begin + j]; t.slot = slot;
            arena.triangles.push_back(t);
        }
        for (size_t j = 0; j < span.rays; ++j) {
            auto r = from.rays[span.ray_begin + j]; r.slot = slot;
            arena.rays.push_back(r);
        }
        arena.input_points += span.rays;
    }
}

inline void commit_scene(Resources& resources, const PreparedGeometry& geometry, const int* colors,
                          unsigned int* cells, int X, int Y, unsigned char* staging, size_t bytes) {
    const auto& arena = geometry.arenas[0];
    std::vector<unsigned int> ends(geometry.spans.size());
    for (size_t i = 0; i < ends.size(); ++i) {
        const auto span = geometry.spans[i];
        ends[i] = span.triangle_begin + span.triangles - span.rays * (geometry.config.compact_points ? 1 : 2);
    }
    auto* device_ends = resources.upload(ends);
    for (size_t first = 0; first < arena.triangles.size();) {
        const size_t count = std::min(bytes / sizeof(Triangle2D), arena.triangles.size() - first);
        cuda_check(cudaMemcpy(staging, arena.triangles.data() + first, count * sizeof(Triangle2D), cudaMemcpyHostToDevice), "stage occupancy segments");
        mark_segments<<<(count + 255) / 256, 256>>>(reinterpret_cast<Triangle2D*>(staging), count,
            first, device_ends, colors, cells, X, Y, geometry.config.margin);
        first += count;
    }
    for (size_t first = 0; first < arena.rays.size();) {
        const size_t count = std::min(bytes / sizeof(Ray2D), arena.rays.size() - first);
        cuda_check(cudaMemcpy(staging, arena.rays.data() + first, count * sizeof(Ray2D), cudaMemcpyHostToDevice), "stage occupancy points");
        mark_points<<<(count + 255) / 256, 256>>>(reinterpret_cast<Ray2D*>(staging), count, colors, cells, X, Y);
        first += count;
    }
    cuda_check(cudaDeviceSynchronize(), "commit scene occupancy");
    resources.memory.release(device_ends);
}

// Fixed production scene budgets. A smaller task limit is accepted only by
// this internal entry so component tests can exercise cross-scene boundaries.
inline BatchResult generate(cudb::RouterState& state, const std::vector<int>& order, int limit,
                            bool rounds, const PreparedPipeline& pipeline,
                            size_t scene_nets = 1000000) {
    if (!scene_nets || scene_nets > 1000000)
        throw std::invalid_argument("Partitioned scene task limit must be 1..1000000");
    BatchResult run;
    run.method = "partitioned";
    run.timing_scope = "partitioned_pipeline_prebuilt_wall";
    run.batches.resize(cached::mask_words * 32);
    {
        Resources global;
        const int X = state.device.X, Y = state.device.Y;
        const size_t cell_bytes = size_t(X) * Y * cached::mask_words * sizeof(unsigned int);
        auto* cells = global.allocate<unsigned int>(size_t(X) * Y * cached::mask_words);
        cuda_check(cudaMemset(cells, 0, cell_bytes), "initialize prior-scene occupancy");
        auto* full = global.allocate<unsigned int>(cached::mask_words);
        constexpr size_t staging_bytes = size_t(32) << 20;
        auto* staging = global.allocate<unsigned char>(staging_bytes);
        std::vector<int> counts(cached::mask_words * 32, 0);
        std::vector<unsigned int> forbidden(cached::mask_words, 0);
        const size_t net_limit = scene_nets;
        printf("RTCORE_SPACE model=partitioned point_model=%s z_gap=%.9g margin=%.9g\n",
               state.rtcore_geometry.config.compact_points ? "triangle" : "cross", double(state.rtcore_z_gap), double(state.rtcore_margin));
        printf("RTCORE_PARTITION_POLICY coloring=%s max_nets=%zu max_triangles=8000000 max_rays=4000000 occupancy_bytes=%zu batch_capacity=%d\n", rounds ? "rounds" : "cached", net_limit, cell_bytes, limit);
        for (size_t first = 0; first < order.size();) {
            size_t end = first, triangles = 0, rays = 0;
            while (end < order.size() && end - first < net_limit) {
                const auto span = state.rtcore_geometry.spans[state.rtcore_geometry.net_to_slot[order[end]]];
                if (triangles + span.triangles > 8000000 || rays + span.rays > 4000000) break;
                triangles += span.triangles; rays += span.rays; ++end;
            }
            if (end == first) throw std::runtime_error("Single net exceeds partitioned RT-Core scene budget");
            cudb::RouterState local;
            copy_window(state.rtcore_geometry, order, first, end, local.rtcore_geometry);
            std::vector<int> sequence(end - first);
            std::iota(sequence.begin(), sequence.end(), 0);
            if (rounds) {
                BatchWorkspace work;
                prepare_batches_rtcore(work, local, sequence, limit, &pipeline);
                work.params.cell_colors = cells;
                work.params.full_colors = full;
                work.params.grid_y = Y;
                cuda_check(cudaMemcpy(full, forbidden.data(), forbidden.size()*sizeof(unsigned int), cudaMemcpyHostToDevice), "rounds full colors");
                generate_batches_rtcore(work);
                std::vector<int> colors(sequence.size());
                cuda_check(cudaMemcpy(colors.data(), work.params.batch_id, colors.size()*sizeof(int), cudaMemcpyDeviceToHost), "round colors");
                size_t accepted = 0;
                for (; accepted < colors.size(); ++accepted) {
                    const int c = colors[accepted];
                    if (c < 0 || c >= int(counts.size())) throw std::runtime_error("Invalid round color");
                    if (counts[c] == limit) break;
                    run.batches[c].push_back(order[first+accepted]);
                    if (++counts[c] == limit) forbidden[c/32] |= 1u << (c%32);
                }
                if (!accepted) throw std::runtime_error("Round capacity replay stalled");
                if (accepted < sequence.size()) {
                    PreparedGeometry prefix;
                    copy_window(state.rtcore_geometry, order, first, first+accepted, prefix);
                    commit_scene(work.resources, prefix, work.params.batch_id, cells, X, Y, staging, staging_bytes);
                } else commit_scene(work.resources, local.rtcore_geometry, work.params.batch_id, cells, X, Y, staging, staging_bytes);
                first += accepted;
                printf("RTCORE_PARTITION completed=%zu total=%zu\n", first, order.size()); fflush(stdout);
                continue;
            }
            cached::CachedWorkspace work;
            cached::prepare_batches_rtcore_cached(work, local, sequence, limit, &pipeline);
            cached::initialize_coloring(work);
            work.params.cell_colors = cells;
            work.params.full_colors = full;
            work.params.grid_y = Y;
            const int chunk_nets = work.max_chunk_nets;
            std::vector<int> colors(chunk_nets);
            for (int begin = 0; begin < int(sequence.size());) {
                const int count = std::min(chunk_nets, int(sequence.size()) - begin);
                cuda_check(cudaMemcpy(full, forbidden.data(), forbidden.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "upload full colors");
                cached::color_chunk(work, begin, count);
                int error = 0;
                cuda_check(cudaMemcpy(&error, work.params.color_error, sizeof(int), cudaMemcpyDeviceToHost), "check partitioned colors");
                if (error) throw std::runtime_error("Partitioned RT-Core exhausted its 1024-batch mask");
                cuda_check(cudaMemcpy(colors.data(), work.params.batch_id + begin, count * sizeof(int), cudaMemcpyDeviceToHost), "download color prefix");
                int accepted = 0;
                for (; accepted < count; ++accepted) {
                    const int color = colors[accepted];
                    if (color < 0 || color >= int(counts.size())) throw std::runtime_error("Invalid partitioned color");
                    if (counts[color] == limit) break;
                    run.batches[color].push_back(order[first + begin + accepted]);
                    if (++counts[color] == limit) forbidden[color / 32] |= 1u << (color % 32);
                }
                if (!accepted) throw std::runtime_error("Partitioned capacity replay made no progress");
                // Colors are exact up to the first overflowing net. Replay only
                // the tail with the now-full colors forbidden, in priority order.
                if (accepted < count)
                    cuda_check(cudaMemset(work.params.batch_id + begin + accepted, 0xff, (count - accepted) * sizeof(int)), "reset overflowing color tail");
                begin += accepted;
            }
            commit_scene(work.resources, local.rtcore_geometry, work.params.batch_id, cells, X, Y, staging, staging_bytes);
            printf("RTCORE_PARTITION first=%zu end=%zu triangles=%zu rays=%zu\n", first, end, triangles, rays);
            fflush(stdout);
            first = end;
        }
    }
    while (!run.batches.empty() && run.batches.back().empty()) run.batches.pop_back();
    return run;
}
// Run a tiny independent scene before timing to load CUDA/OptiX launch resources.
inline void warm_pipeline(const PreparedPipeline& pipeline, bool rounds) {
    cudb::RouterState state;
    state.device.X = state.device.Y = 8;
    std::vector<int> order{0};
    cudb::net net;
    net.points = {9};
    state.rtcore_geometry.begin(order, 1, 1);
    state.rtcore_geometry.emit(0, 0, net, 8);
    if (rounds) {
        BatchWorkspace work;
        work.resources.quiet = true;
        prepare_batches_rtcore(work, state, order, 1000000, &pipeline);
        generate_batches_rtcore(work);
    } else {
        cached::CachedWorkspace work;
        work.resources.quiet = true;
        cached::prepare_batches_rtcore_cached(work, state, order, 1000000, &pipeline);
        cached::generate_batches_rtcore_cached(work);
    }
    cuda_check(cudaDeviceSynchronize(), "warm pipeline complete");
}
} // namespace grotor::rtcore::partitioned
