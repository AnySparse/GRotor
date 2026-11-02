#pragma once

#include <vector_types.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace grotor::rtcore {

struct GeometryConfig {
    float z_gap = 0x1p-16f, margin = .1f;
    bool coplanar = false, compact_points = true;
};

struct Triangle2D {
    float2 a, b, c;
    std::uint32_t slot;
};
struct Ray2D {
    float2 origin;
    std::uint32_t slot;
};
struct GeometrySpan {
    std::uint32_t triangle_begin = 0, triangles = 0, ray_begin = 0, rays = 0;
    std::uint32_t lane = std::numeric_limits<std::uint32_t>::max();
};
struct alignas(64) GeometryArena {
    std::vector<Triangle2D> triangles;
    std::vector<Ray2D> rays;
    std::vector<int> point_scratch;
    std::uint64_t input_points = 0;
};

// The stage owns contiguous, thread-local buffers. No per-net geometry allocation
// or GPU allocation is needed while CPU routing topology is being constructed.
struct PreparedGeometry {
    GeometryConfig config;
    std::vector<GeometryArena> arenas;
    std::vector<GeometrySpan> spans;
    std::vector<std::uint32_t> net_to_slot;

    void clear() {
        std::vector<GeometryArena>().swap(arenas);
        std::vector<GeometrySpan>().swap(spans);
        std::vector<std::uint32_t>().swap(net_to_slot);
    }

    void begin(const std::vector<int>& order, std::size_t total_nets, int lanes,
               GeometryConfig settings = {}) {
        clear();
        if (order.size() > 16000000 || lanes < 1)
            throw std::invalid_argument("RT-Core topology exceeds supported priority range");
        if (!std::isfinite(settings.z_gap) || settings.z_gap < 0x1p-24f || settings.z_gap > 64.f
            || !std::isfinite(settings.margin) || settings.margin < .01f || settings.margin > .4f)
            throw std::invalid_argument("Invalid RT-Core geometry configuration");
        if (!order.empty()) {
            const float z = (order.size() - 1) * settings.z_gap;
            if (!settings.coplanar && std::nextafter(z, std::numeric_limits<float>::infinity()) - z > settings.z_gap)
                throw std::invalid_argument("RT-Core priority layers are not distinct in float coordinates");
        }
        config = settings;
        arenas.resize(lanes);
        spans.resize(order.size());
        net_to_slot.assign(total_nets, std::numeric_limits<std::uint32_t>::max());
        for (std::size_t i = 0; i < order.size(); ++i) {
            const int id = order[i];
            if (id < 0 || std::size_t(id) >= total_nets || net_to_slot[id] != std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("Invalid or duplicated topology net ID");
            net_to_slot[id] = static_cast<std::uint32_t>(i);
        }
    }

    template <class Net>
    void emit(std::size_t slot, int lane, const Net& net, int Y, bool points_are_unique = false) {
        if (slot >= spans.size() || lane < 0 || std::size_t(lane) >= arenas.size() || net.points.empty())
            throw std::invalid_argument("Missing RT-Core topology geometry");
        auto& arena = arenas[lane];
        GeometrySpan span;
        span.lane = lane;
        span.triangle_begin = static_cast<std::uint32_t>(arena.triangles.size());
        span.ray_begin = static_cast<std::uint32_t>(arena.rays.size());
        const float m = config.margin;
        const auto triangle = [&](float2 a, float2 b, float2 c) {
            arena.triangles.push_back({a, b, c, static_cast<std::uint32_t>(slot)});
        };
        for (const auto& segment : net.rsmt_h_segments) {
            const float lo = std::min(segment.first / Y, segment.second / Y);
            const float hi = std::max(segment.first / Y, segment.second / Y), y = segment.first % Y;
            triangle({lo-m, y-m}, {lo-m, y+m}, {hi+m, y});
        }
        for (const auto& segment : net.rsmt_v_segments) {
            const float x = segment.first / Y;
            const float lo = std::min(segment.first % Y, segment.second % Y);
            const float hi = std::max(segment.first % Y, segment.second % Y);
            triangle({x+m, lo-m}, {x-m, lo-m}, {x, hi+m});
        }
        const std::vector<int>* unique = &net.points;
        if (!points_are_unique) {
            auto& scratch = arena.point_scratch;
            scratch.assign(net.points.begin(), net.points.end());
            std::sort(scratch.begin(), scratch.end());
            scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
            unique = &scratch;
        }
        arena.input_points += net.points.size();
        for (const int point : *unique) {
            const float x = point / Y, y = point % Y;
            if (config.compact_points) {
                // Exactly (x-1,y), (x,y), (x+1,y), (x,y-1) at integer queries.
                triangle({x-1.5f, y+.25f}, {x+1.5f, y+.25f}, {x, y-1.25f});
            } else {
                triangle({x-(1.f+m), y-m}, {x-(1.f+m), y+m}, {x+(1.f+m), y});
                triangle({x+m, y-(1.f+m)}, {x-m, y-(1.f+m)}, {x, y+m});
            }
            arena.rays.push_back({{x, y}, static_cast<std::uint32_t>(slot)});
        }
        if (arena.triangles.size() > std::numeric_limits<std::uint32_t>::max()
            || arena.rays.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::overflow_error("RT-Core geometry arena is too large");
        span.triangles = static_cast<std::uint32_t>(arena.triangles.size()) - span.triangle_begin;
        span.rays = static_cast<std::uint32_t>(arena.rays.size()) - span.ray_begin;
        spans[slot] = span;
    }
};

} // namespace grotor::rtcore
