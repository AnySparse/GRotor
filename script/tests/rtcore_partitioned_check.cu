#include "batching/generate_batches.hpp"
#include <numeric>
#include <random>

namespace {
void check(const char* name, cudb::RouterState& state, int scene_nets, int capacity) {
    std::vector<int> order(state.nets.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random(81);
    std::shuffle(order.begin(), order.end(), random);
    auto expected = cudb::generate_batches_rsmt(state, order, capacity, 1);
    auto emission = order;
    std::reverse(emission.begin(), emission.end());
    state.rtcore_geometry.begin(emission, state.nets.size(), 2,
        {state.rtcore_z_gap, state.rtcore_margin, state.rtcore_coplanar, state.rtcore_compact_points, true});
    for (size_t slot = 0; slot < emission.size(); ++slot)
        state.rtcore_geometry.emit(slot, slot % 2, state.nets[emission[slot]], state.device.Y);
    for (bool rounds : {true, false}) {
        auto run = grotor::batching::generate_batches_partitioned(state, order, capacity, rounds, scene_nets);
        if (std::string(run.method) != "partitioned" || run.batches != expected)
            throw std::runtime_error(std::string("Partitioned batch mismatch: ") + name);
        if (run.milliseconds != (run.end_us - run.start_us) / 1000.)
            throw std::runtime_error("Partitioned timing span mismatch");
    }
    printf("PASS PARTITIONED name=%s nets=%zu scene_nets=%d capacity=%d batches=%zu\n",
           name, order.size(), scene_nets, capacity, expected.size());
}
}

int main() {
    try {
        using grotor::rtcore::needs_partitioning;
        if (needs_partitioning(16000000, 16000000)
            || !needs_partitioning(16000001, 3)
            || !needs_partitioning(3, 16000001)
            || !needs_partitioning(55315745, 8716217))
            throw std::runtime_error("Automatic large-design dispatch mismatch");
        {
            cudb::RouterState state;
            state.device.X = state.device.Y = 128;
            state.nets.resize(97);
            for (int i = 0; i < state.nets.size(); ++i)
                state.nets[i].points = {(i / 16 * 4) * 128 + i % 16 * 4};
            check("capacity-replay-and-scene-carry", state, 23, 7);
        }
        for (bool compact : {false, true}) for (bool coplanar : {false, true}) {
            cudb::RouterState state;
            state.device.X = state.device.Y = 64;
            state.rtcore_compact_points = compact; state.rtcore_coplanar = coplanar;
            state.nets.resize(193);
            std::mt19937 random(431);
            for (auto& net : state.nets) {
                for (int p = 0; p < 9; ++p) net.points.push_back(random() % (64 * 64));
                net.points.push_back(net.points.front());
                for (int s = 0; s < 4; ++s) {
                    int x0 = random() % 64, x1 = random() % 64, y0 = random() % 64, y1 = random() % 64;
                    net.rsmt_h_segments.emplace_back(std::min(x0, x1) * 64 + y0, std::max(x0, x1) * 64 + y0);
                    net.rsmt_v_segments.emplace_back(x0 * 64 + std::min(y0, y1), x0 * 64 + std::max(y0, y1));
                }
            }
            check("directed-segments-duplicates-and-order", state, 17, 7);
        }
        {
            cudb::RouterState state;
            state.device.X = state.device.Y = 128;
            state.nets.resize(16384 + 97);
            for (int i = 0; i < state.nets.size(); ++i) {
                const int cell = i % 1024;
                state.nets[i].points = {(cell / 32 * 4 + 1) * 128 + cell % 32 * 4 + 1};
            }
            check("dependency-chunk-boundary", state, 20000, 1000);
        }
        printf("RTCORE_PARTITIONED_CHECK_PASSED\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
