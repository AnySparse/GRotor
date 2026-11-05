# AMD HIP build and benchmarking

HIP build rules live in the top-level `CMakeLists.txt`. Generated CMake files
and binaries stay in the selected `build/` directory; results stay in `results/`.
The maintained entry points are `script/benchmark_amd.py` and `script/plot_amd.py`.

The CUDA and HIP builds compile the same `src/main.cu`, pipeline, topology,
L-shape/FLT routing and cost/BIT kernels. `src/platform/gpu_runtime.hpp`
adapts runtime APIs and explicitly widens 32-bit participation masks for HIP's
64-bit mask API. It preserves the participating lanes and does not disable
device assertions. `flt_workspace.cuh` selects hipCUB or CUB.
Only ray tracing batching has an independent AMD implementation, in
`src/batching/hiprt/grid_chunk.cuh`, extracted from the supplied
`amd-gr-workspace-4` working tree. The old router was not copied.

## Build

A fresh build directory defaults to **single-GPU CUDA**. HIP is enabled only
by explicitly setting `-DGROTOR_GPU_BACKEND=HIP` at configure time. The HIP
branch returns before CUDA language/dependency setup, so one build never
compiles both backends. `GROTOR_BUILD_MULTIGPU` defaults to OFF.

Default CUDA build:

```bash
cmake -S . -B build/cuda
cmake --build build/cuda --target GRotor --parallel 2
```

For AMD, select HIP explicitly. `GROTOR_ENABLE_HIPRT=ON` separately enables
ray-tracing batching; without it, HIP routing uses CPU batching. Keep CUDA and
HIP in separate build directories: CMake caches the selected backend, and
omitting the option later does not reset an existing HIP cache to CUDA.

Requirements: CMake >= 3.21, C++17, a working HIP compiler/hipCUB installation,
and a 32-lane AMD GPU. CUDA remains the default backend and its manual build
still requires only CMake 3.18. HIP builds do not discover CUDA, OptiX, NCCL
or NVSHMEM. HIP multi-GPU is not implemented.

The tested machine has RX 9060 XT (`gfx1200`, 16 GiB), Intel i7-8700,
Ubuntu 26.04.1 and HIP 7.16.26385 / AMD clang 24. This differs from the paper's
historical RX 7900 XTX setup. This port does not reproduce or claim the paper's
historical 2.83x / 2.21x results.

HIPRT was built from the [official source repository](https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT),
commit `0629d08568fff39f9a36bf9ee800fc7bac5bb096`, version
`3.2.0.ceba893`, library `libhiprt0300264.so`. SDK sources and libraries are
external dependencies, not vendored into GRotor. To reproduce this dependency:

```bash
git clone https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git /path/to/HIPRT
git -C /path/to/HIPRT checkout 0629d08568fff39f9a36bf9ee800fc7bac5bb096
git -C /path/to/HIPRT submodule update --init --recursive
cmake -S /path/to/HIPRT -B /path/to/HIPRT/build \
  -DNO_UNITTEST=ON -DBITCODE=OFF -DFORCE_DISABLE_CUDA=ON -DHIP_PATH=/opt/rocm
cmake --build /path/to/HIPRT/build --parallel 2
```

From GRotor's repository root:

```bash
export HIPRT_ROOT=/path/to/HIPRT
export HIPRT_PATH="$HIPRT_ROOT"
cmake --preset hiprt
cmake --build --preset hiprt
```

The preset defaults to `gfx1200`. For another supported 32-lane GPU, specify
its architecture, for example `cmake --preset hiprt -DCMAKE_HIP_ARCHITECTURES=gfx1100`.
Only gfx1200 has been tested here; wave64 devices are rejected at startup.
Manual configuration, including explicit library selection:

```bash
cmake -S . -B build/hip \
  -DGROTOR_GPU_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx1200 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DGROTOR_ENABLE_RTCORE=OFF -DGROTOR_ENABLE_HIPRT=ON \
  -DHIPRT_ROOT="$HIPRT_ROOT" \
  -DHIPRT_LIBRARY="$HIPRT_ROOT/build/libhiprt0300264.so" \
  -DGROTOR_HOST_MEMORY_SAVING=OFF
cmake --build build/hip --parallel 2
```

The HIPRT library must match its headers. CMake discovers a unique built
library below `HIPRT_ROOT`, or requires explicit `HIPRT_LIBRARY` when ambiguous;
it records the library path and embeds the build runtime search path.
Source-built HIPRT may JIT-compile BVH builders on the first call; that cost is
outside the batching execution function and is not timed as batching.

For HIP routing without HIPRT, use `cmake --preset hip` and
`cmake --build --preset hip`. This creates `build/hip-cpu/GRotor` with no
HIPRT dependency. CPU batching remains available in the HIPRT build too.
`GROTOR_HOST_MEMORY_SAVING` is OFF by default in both branches. It controls
host lifetimes, not the device memory limit.

## Runtime and strategies

```bash
build/hip/GRotor -cap /data/mempool_tile.cap -net /data/mempool_tile.net \
  -out /tmp/tile.out --batching hiprt --validate-batches 1 --profile-stages 1
```

| Executable / option | Batching | Cost / prefix | L-shape / FLT scheduling | Purpose |
|---|---|---|---|---|
| `GRotor --batching hiprt` | HIPRT grid/chunk | Incremental / BIT | Warp / block per DAG | Optimized AMD benchmark |
| `grotor_amd_original --batching cpu` | CPU greedy | Full / full scans | Thread per DAG / global depth | Unoptimized AMD benchmark |
| `GRotor --batching cpu` | CPU greedy | Incremental / BIT | Warp / block per DAG | Backend correctness check |


`grotor_amd_original` is built by `GROTOR_BUILD_ROUTING_ABLATION=ON` (HIP
default). It explicitly enables **both** `GROTOR_FULL_COST_REFERENCE` and
`GROTOR_SERIAL_DAG`, so it disables BIT/incremental maintenance and the
optimized DAG schedulers together. Changing only cost maintenance would not
be this baseline. HIP `GRotor` can separately enable those two CMake options
for debugging; the experiment script checks actual logged configurations.

The AMD correctness reference is GRotor's own unoptimized strategy. No AMD
InstantGR2 source adapter, build target or runtime dependency is maintained.
Additional temporary development checks do not enter the AMD benchmark ratios.

HIPRT uses 512-net chunks, one triangle per grid cell, and 1024 colors. It
preserves GRotor CPU's existing asymmetric point marking, including the
`y < point_y + 1` boundary, rather than importing the old AMD workspace's
symmetric cross. Capacity is 1,000,000 for L-shape and 300,000 for FLT.
Chunks that cannot fill any batch color in that chunk retain parallel
conflict coloring. Near-full batches use net-order coloring to preserve CPU
capacity semantics. A 1024-color overflow fails explicitly; it neither writes
out of bounds nor silently switches the measured backend to CPU.
The historical iterative HIPRT blocker path is not part of this port.

`--validate-batches 1` compares **all batch members and their order**, before
routing reverses the batch execution order. It runs outside the reported
batching interval and fails the process on any difference. Do not use it in
performance samples.

## Benchmark runner

```bash
python3 script/benchmark_amd.py --data-root /data/ISPD24_tmp \
  --build-dir build/hip --results results/amd-benchmark-new \
  --repeats 3 --warmups 1
```

The results directory must be new. The default list is the ten supplied
input pairs through `mempool_group`, explicitly including separate rank
variants: `ariane133_51`, `ariane133_68`, `ariane_68_rank`, `bsg_chip`,
`bsg_chip_rank`, `nvdla`, `nvdla_rank`, `mempool_tile`,
`mempool_tile_rank`, `mempool_group`. No cluster case is run automatically.
Override with `--cases mempool_tile mempool_group` for a smaller review run.
All compared strategies use exactly the same named inputs and sparsity 1.

For each case the runner:

1. Runs separate original and optimized GRotor validation (the latter with
   exact HIPRT/CPU batch checks). It requires identical evaluator
   wirelength/via/overflow/total values, zero open/incomplete nets, and
   identical per-net segment multisets, preserving duplicate counts.
2. Sets `--report-score 0` symmetrically for both GRotor strategies, disabling
   only the slow intermediate atomic score-printing diagnostics. The default
   CLI behavior remains `--report-score 1`; routing decisions and output are
   unchanged, and independent evaluator scores remain required. It then warms
   both strategies and alternates the execution order for repeated
   measurements. Performance runs have batch validation disabled. Their
   output geometries are also compared, outside the timed process.
3. Retains validation solutions, every raw routing/evaluator log, exact
   commands and statuses, input/source/binary hashes, build cache, environment,
   sampled host RSS and device-wide VRAM. Performance solution files are
   deleted only after they have passed geometry comparison.
4. Emits `measurements.csv`, `summary.csv`, `geomean.json`, `status.json` and
   `environment.json`. Each failure stays visible. Any failed case returns
   nonzero and suppresses an incomplete geometric mean.

With Matplotlib installed, render the completed summary as a standalone
PDF and PNG:

```bash
python3 script/plot_amd.py results/amd-benchmark-new
```

`summary.csv` uses each strategy's median per-case runtime. Each stage's
geometric mean is over the **same full case list** of original/optimized
ratios. Rank variants remain distinct observations; use `--cases` to select a
paper-specific subset rather than mixing case names or silently dropping OOMs.

## Timing boundaries and limits


```cpp
prepare_scene_and_workspace();        // outside the timer
hipDeviceSynchronize();               // exclude earlier GPU work
const auto start = stamp();
generate_batch_ids(...);              // synchronizes before returning
const auto end = stamp();
collect_batches();                   // outside the timer
// batching_ms = (end - start) / 1000.0
```

- `batching_ms`: wall time measured immediately before and after **one function
  call**. CPU uses `generate_batches_rsmt(...)`. HIPRT uses
  `generate_batch_ids(...)` after its scene and workspace have been prepared.
  HIPRT's function synchronizes the device before it returns, so the call time
  includes completion of occupancy reset, query, rasterization, dependency
  collection, coloring and commit; it is not just asynchronous launch overhead.
- `dag_ms`: synchronized complete batched execution, including cost refresh,
  full or BIT prefix maintenance, DP, traceback, demand commits and applicable
  assembly/uploads inside that interval. It is not a sum of selected kernels.
- `algorithm_total_ms = batching_ms + dag_ms`: the primary AMD benchmark interval.
- HIPRT setup, BVH build, downloads/collection and validation are outside the
  batching call timer. Their runtimes and GPU event timings are not emitted or
  exported as experiment columns. `BATCH_SCOPE` identifies
  `generate_batch_ids_call_wall`; the runner rejects binaries missing this
  marker, so legacy logs cannot be silently labeled as function-call timing.
- Topology preprocessing, FLT preparation outside the batched execution,
  classification, output and cleanup are reflected in
  `router_wall_seconds` / full process wall time and raw stage logs, not
  silently included in the algorithm interval. Intermediate score diagnostics
  are off in both GRotor benchmark runs and on by default for ordinary CLI use.
- `router_wall_seconds` uses the program's routing clock, which starts after
  CLI parsing and HIP device-property validation. `process_wall_seconds`
  measures the entire subprocess, including that startup and final exit.
- Memory values are 20 ms samples: host VmHWM and aggregate device-wide AMD
  DRM VRAM usage. VRAM includes other users of those devices; sampling can
  miss short peaks. No clocks or power settings are changed by the runner.

This AMD implementation currently requires a full-prefix track to fit in
shared memory: `max(X,Y) * 8 <= sharedMemPerBlock` (8192 positions on this
64 KiB device). Main rejects larger grids with a clear message before device
routing allocations. Larger tracks need a different shared-memory strategy. This is separate
from device OOM and does not establish support for tera-sized routing.

## Retained results

The local run and its numerical results are recorded in
[AMD port validation report](../results/amd-report/report.md).
Raw artifacts live under `results/amd-dev` and `results/amd-benchmark` (ignored by
Git). `results/amd-report/source-manifest.json` records the supplied source
working trees and input hashes. The original AMD workspace was left untouched.
The CUDA paths remain present, but no NVIDIA GPU/toolkit is available on this
machine to rerun their device tests; AMD compilation is not evidence of a new
CUDA runtime validation.
