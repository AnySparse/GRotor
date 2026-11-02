#pragma once
#include "launch_params.h"

namespace grotor::rtcore::synchronous {
// Batch IDs are immutable throughout tracing. Only the commit kernel publishes
// pending colors, after every ray and every assignment decision has completed.
struct LaunchParams {
    grotor::rtcore::LaunchParams query{};
    int* eligible = nullptr;
    int* pending = nullptr;
    int* assigned_round = nullptr;
    unsigned int* trace_counts = nullptr;
    int use_blocker = 0;
};
}
