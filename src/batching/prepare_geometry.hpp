#pragma once
#include "router_state.hpp"

namespace grotor::batching {

#ifdef GROTOR_ENABLE_RTCORE
inline bool partitioned(const cudb::RouterState& state, size_t count) {
    return rtcore::needs_partitioning(state.nets.size(), count);
}
#endif

inline bool geometry_fused(const cudb::RouterState& state) {
#ifdef GROTOR_ENABLE_RTCORE
    return state.rtcore_batching;
#else
    return false; // HIPRT packs the shared net geometry in its preparation interval.
#endif
}

// OptiX fuses geometry packing into topology preparation.
inline void prepare_geometry(cudb::RouterState& state, const std::vector<int>& order, int lanes) {
#ifdef GROTOR_ENABLE_RTCORE
    if (state.rtcore_batching)
        state.rtcore_geometry.begin(order, state.nets.size(), lanes,
            {state.rtcore_z_gap, state.rtcore_margin, state.rtcore_coplanar, state.rtcore_compact_points,
             partitioned(state, order.size())});
#endif
}

inline void emit_geometry(cudb::RouterState& state, size_t slot, int lane,
                          const cudb::net& net, bool detour = false) {
#ifdef GROTOR_ENABLE_RTCORE
    if (state.rtcore_batching)
        state.rtcore_geometry.emit(slot, lane, net, state.device.Y, detour);
#endif
}

inline void release_geometry(cudb::RouterState& state) {
#ifdef GROTOR_ENABLE_RTCORE
    state.rtcore_geometry.clear();
#endif
}
} // namespace grotor::batching
