#include "batching/generate_batches.hpp"
#include <numeric>
#include <random>

namespace {
void check(const char* name, cudb::RouterState& state, std::vector<int> order) {
    auto expected = cudb::generate_batches_rsmt(state, order, 1000000, 1);
    for (bool coplanar : {false, true}) {
        for (bool compact : {false, true}) {
            state.rtcore_coplanar = coplanar;
            state.rtcore_compact_points = compact;
            for (const char* method : {"cached", "rounds"}) {
                auto emission_order = order;
                std::reverse(emission_order.begin(), emission_order.end());
                state.rtcore_geometry.begin(emission_order, state.nets.size(), 2,
                    {state.rtcore_z_gap, state.rtcore_margin, state.rtcore_coplanar, state.rtcore_compact_points});
                for (size_t slot = 0; slot < emission_order.size(); ++slot)
                    state.rtcore_geometry.emit(slot, slot%2, state.nets[emission_order[slot]], state.device.Y);
                for (int repeat = 0; repeat < 3; ++repeat) {
                    auto actual = std::string(method) == "cached" ? grotor::batching::generate_batches_rtcore_cached(state, order, 1000000)
                                                               : grotor::batching::generate_batches_rtcore(state, order, 1000000);
                    if (actual.milliseconds != (actual.end_us - actual.start_us) / 1000.0
                        || actual.milliseconds <= 0)
                        throw std::runtime_error("Batch timing must equal its contiguous timestamp span");
                    if (actual.batches != expected)
                        throw std::runtime_error(std::string("Batch mismatch: ") + name);
                }
                printf("PASS %s method=%s coplanar=%d compact=%d nets=%zu batches=%zu\n",
                       name, method, int(coplanar), int(compact), order.size(), expected.size());
            }
        }
    }
    state.rtcore_coplanar = false;
    state.rtcore_compact_points = false;
}
void setup(cudb::RouterState& state, int count, int side = 64) {
    state.device.X = state.device.Y = side;
    state.nets.resize(count);
}
std::vector<int> order(int n) {
    std::vector<int> result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
}
}
int main() {
    try {
        grotor::cuda_check(cudaFree(nullptr), "initialize batch checks");
        {
            cudb::RouterState state;
            setup(state, 2);
            state.nets[0].points = {4 * 64 + 4};
            state.nets[1].points = {4 * 64 + 5};
            check("exclusive-upper-y", state, {0, 1});
            check("directed-query-order", state, {1, 0});
            if (!grotor::batching::generate_batches_rtcore(state, {}, 1000000).batches.empty())
                throw std::runtime_error("Empty batch input failed");
        }
        {
            cudb::RouterState state;
            setup(state, 96);
            for (auto& net : state.nets) net.points = {0, 0, 63 * 64 + 63};
            check("clique-and-duplicate-probes", state, order(state.nets.size()));
        }
        for (unsigned int seed = 1; seed <= 4; ++seed) {
            cudb::RouterState state;
            setup(state, 193);
            std::mt19937 random(seed);
            for (auto& net : state.nets) {
                for (int p = 0; p < 9; ++p) net.points.push_back(random() % (64 * 64));
                net.points.push_back(net.points.front());
                for (int s = 0; s < 4; ++s) {
                    int x0 = random() % 64, x1 = random() % 64, y0 = random() % 64, y1 = random() % 64;
                    net.rsmt_h_segments.emplace_back(std::min(x0, x1) * 64 + y0, std::max(x0, x1) * 64 + y0);
                    net.rsmt_v_segments.emplace_back(x0 * 64 + std::min(y0, y1), x0 * 64 + std::max(y0, y1));
                }
            }
            auto sequence = order(state.nets.size());
            std::shuffle(sequence.begin(), sequence.end(), random);
            check("random-segments-and-order", state, sequence);
            if (seed == 1) {
                for (const float gap : {0x1p-20f, 0x1p-12f, 0x1p-4f, 16.f}) {
                    state.rtcore_z_gap = gap;
                    for (const float margin : {.01f, .25f, .4f}) {
                        state.rtcore_margin = margin;
                        check("geometry-spacing-and-margin", state, sequence);
                    }
                }
                state.rtcore_z_gap = 1e-3f;
                state.rtcore_margin = .1f;
            }
        }
        {
            cudb::RouterState state;
            setup(state, 16384 + 17, 128);
            for (int i = 0; i < state.nets.size(); ++i) {
                const int cell = i % 1024;
                state.nets[i].points = {(cell / 32 * 4 + 1) * 128 + cell % 32 * 4 + 1};
            }
            check("large-grid-and-tail", state, order(state.nets.size()));
        }
        {
            cudb::RouterState state;
            setup(state, 1027);
            for (auto& net : state.nets) net.points = {20 * 64 + 20};
            for (const char* method : {"cached", "rounds"}) {
            auto sequence = order(state.nets.size());
            state.rtcore_geometry.begin(sequence, state.nets.size(), 1);
            for (size_t slot = 0; slot < sequence.size(); ++slot)
                state.rtcore_geometry.emit(slot, 0, state.nets[sequence[slot]], state.device.Y);
            bool rejected = false;
            try {
                if (std::string(method) == "cached") grotor::batching::generate_batches_rtcore_cached(state, sequence, 1000000);
                else grotor::batching::generate_batches_rtcore(state, sequence, 1000000);
            }
            catch (const std::runtime_error& error) { rejected = std::string(error.what()).find("1024-batch mask") != std::string::npos; }
            if (!rejected) throw std::runtime_error("Color-capacity overflow was not rejected");
            printf("PASS color-capacity-error-propagation method=%s\n", method);
            }
        }
        printf("RTCORE_BATCH_CHECK_PASSED\n");
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "RT-Core batch check failed: %s\n", error.what());
        return 1;
    }
}
