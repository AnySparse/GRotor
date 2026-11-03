#pragma once
#include "router_state.hpp"

namespace grotor::batching {

// The CUDA-only build has no geometry types, traversal or storage.
inline void prepare_geometry(cudb::RouterState& state, const std::vector<int>& order, int lanes) {
#ifdef GROTOR_ENABLE_RTCORE
    if (state.rtcore_batching)
        state.rtcore_geometry.begin(order, state.nets.size(), lanes,
            {state.rtcore_z_gap, state.rtcore_margin, state.rtcore_coplanar, state.rtcore_compact_points});
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
