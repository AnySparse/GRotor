# GRotor

For the original NVSHMEM L-shape execution policy and its one-to-many GPU
scaling baseline, see [legacy L-shape execution](docs/legacy_lshape_execution.md).
It is the default `GRotor_multi` L-shape path; reports default to L-shape only.

Build GRotor and compare its L-shape and FLT routing runtimes and evaluator
scores with InstantGR2. The main test produces total, batching and routing DAG
tables, with speedups for both algorithms. Reports and logs are in English.

## Requirements and setup

Use Linux with:

- An NVIDIA CUDA GPU and a driver compatible with the installed CUDA toolkit.
  RT-Core batching additionally requires a supported RTX GPU.
- CUDA `nvcc`, a compatible C++17 host compiler, and CMake 3.18 or newer.
- NVIDIA OptiX SDK headers for RT-Core builds; Python 3.9 or newer, `nvidia-smi`, and GNU time at
  `/usr/bin/time`. The routing comparison uses only the Python standard library;
  the Figure 7, Figure 10 and Figure 11 plots additionally require Matplotlib.

The tested setup is RTX 4090 (`sm_89`), CUDA 12.4 and OptiX 7.7. Run the commands
below from the repository root. Set paths for your installation:

```bash
export NVCC=/usr/local/cuda-12.4/bin/nvcc
export OPTIX_ROOT_DIR=/path/to/NVIDIA-OptiX-SDK-7.7.0
```

Provide matching `<case>.cap` and `<case>.net` files in the input directory,
for example `../ISPD24_tmp/mempool_tile.cap` and
`../ISPD24_tmp/mempool_tile.net`. The default input directory is `../ISPD24_tmp`.
Relative input and result paths are resolved from the repository root.

## CUDA-only builds and CPU batching (A100/H100)

Use this configuration when RT-Core hardware or the OptiX SDK is unavailable.
Routing still runs on the GPU with warp/block scheduling, BIT and incremental
cost updates; only batch generation runs on the CPU. No OptiX package, RT-Core
geometry buffers or RT-Core PTX programs are required. CMake presets require
CMake 3.21 or newer; manual configuration continues to support CMake 3.18+.

| Preset | CUDA architecture | RT-Core | Build directory |
|---|---:|---|---|
| `cuda` | 89 | Off | `build/GRotor-cuda/sm_89` |
| `a100` | 80 | Off | `build/GRotor-cuda/sm_80` |
| `h100` | 90 | Off | `build/GRotor-cuda/sm_90` |
| `rtx4090` | 89 | On | `build/GRotor-rtcore/sm_89` |

For A100, from the repository root:

```bash
export CUDACXX=/path/to/cuda/bin/nvcc
cmake --preset a100
cmake --build --preset a100
python3 script/benchmark_grotor_system.py --batching cpu --cuda-arch 80 \
  --no-build --cases mempool_tile nvdla --repeats 2 \
  --output-dir results/GRotor-a100
```

For H100, replace the preset with `h100`, the architecture with `90`, and the
result directory with `results/GRotor-h100`. For the local RTX 4090 without
RT-Core support compiled in, use preset `cuda` and architecture `89`.
Keep separate build directories when switching architecture or RT-Core support.

The test can also configure and build automatically:

```bash
python3 script/benchmark_grotor_system.py --batching cpu --cuda-arch 80 \
  --nvcc "$CUDACXX" --cases mempool_tile nvdla --repeats 2 \
  --output-dir results/GRotor-a100 --clean-results
```

Without presets, the essential configuration is:

```bash
cmake -S . -B build/GRotor-cuda/sm_80 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DGROTOR_ENABLE_RTCORE=OFF -DGROTOR_FULL_COST_REFERENCE=OFF
cmake --build build/GRotor-cuda/sm_80 \
  --target GRotor instantgr2_reference grotor_evaluator --parallel 2
```

CPU-mode test options include `--cases CASE [CASE ...]`, `--data-dir DIR`,
`--cuda-arch ARCH`, `--nvcc COMPILER`, `--jobs N`, `--repeats N`, `--no-build`,
`--output-dir DIR` and `--clean-results`. By default it runs the six cases through
`mempool_group`, excluding `mempool_cluster`. It respects the existing
`CUDA_VISIBLE_DEVICES` setting, including a scheduler's allocation; optional
`--gpu INDEX` explicitly overrides that setting with one GPU index.

Both routers use CPU batching with sparsity 1. Each dataset runs InstantGR2
once and GRotor twice by default. Every run writes an actual routing file,
invokes the evaluator, requires identical wirelength/via/overflow/total cost
strings and zero open/incomplete nets, then deletes the file. No routing
reference or compressed output is retained.

The report contains `runtime.csv`, `batching.csv`, `dag.csv`, `totals.csv`,
`quality.csv`, `summary.md`, `summary.json`, and per-run router/evaluator logs.
Each stage total includes its actual CPU batching time plus routing DAG time.
GRotor times one continuous batch-loop interval; InstantGR2 retains its FLT
GPU-suffix accumulation. Input, source and executable hashes, exact commands,
hardware details and process wall times are included in `summary.json`.

The CUDA-only path is validated locally on RTX 4090. The A100/H100 configurations
are cross-compiled locally; hardware validation and multi-GPU execution remain
separate follow-up work. These
presets and the current runner select one GPU; they do not implement multi-GPU
routing. The RT-Core comparison below remains available with `--batching rtcore`
(the default mode of the system test script).

## NVSHMEM L-shape scaling (A100)

`script/run-fig13.py` runs **L-shape only**, stops before FLT, and reports
runtime and speedup for **the last executed L-shape batch**. All earlier
batches still execute to establish the correct demand/cost state. Every GPU
count, including one, uses the same original BFS node-parallel NVSHMEM
executor: fixed 1056×256 launches, full per-PE matrices, integer demand,
bool dirty flags, separate wire/via commits and full local presum scans.
BIT is disabled on every batch.

The multi-GPU build needs Open MPI, NCCL and NVSHMEM, including its device
library and CUDA relocatable device code. No OptiX or PyTorch is needed.
See [the execution/timing contract](docs/legacy_lshape_execution.md).

```bash
/home/caoweichen/env/myenv/bin/python script/run-fig13.py \
  --nvcc /usr/local/cuda-12.2/bin/nvcc --cuda-arch 80 \
  --mpirun /home/caoweichen/opt/openmpi/bin/mpirun \
  --data-dir /data0/cwc/ISPD24 --cases mempool_cluster \
  --devices 3,2,1,0 --counts 1 2 3 4 --repeats 2 \
  --symmetric-size 25G --output-dir results/GRotor-lshape-1to4
```

The build directory is `build/GRotor-multi-legacy/sm_80`; use `--no-build`
after building. Select currently available devices. Each N-GPU run uses the
first N entries in `--devices`. Add `--require-idle` for a strict check that
no other compute processes are present. Otherwise shared-device load and
clocks are recorded in per-run `*.gpu-load.jsonl` files.
For 1–8 GPU scaling, supply eight devices and `--counts 1 2 3 4 5 6 7 8`.
`tera_cluster` is opt-in.

The timer includes DP/traceback, commits, post-commit cost/presum updates and
required synchronization. It excludes CPU preparation, initial distribution,
final route/grid collection, output writing and evaluation. Reports use the
maximum participating rank time and median across repeats. `scaling.csv`,
`summary.json`, `quality.csv`, `fig13-scaling.{png,pdf,svg}` and
`last-batch-runtime.{png,pdf,svg}` retain timings, validation and plots.

L-shape routes are written and externally evaluated for zero open/incomplete
nets and matching costs across GPU counts. Byte-identical outputs reuse a
validated evaluator result after SHA-256 verification within the experiment.
This stage-only reference is the one-GPU L-shape result. Full-pipeline
InstantGR2 quality checks remain in `script/test_grotor_reference.py`.

The detour routing entry point is `Lshape_route_detour::route_detour`.
Running `GRotor_multi` directly without `--stop-after lshape` still executes
the full pipeline. The ordinary `GRotor` executable keeps its own single-GPU
solver; the scaling script always uses `GRotor_multi`, including at N=1.

## Compile

```bash
cmake -S . -B build/GRotor-rtcore/sm_89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGROTOR_ENABLE_RTCORE=ON \
  -DGROTOR_FULL_COST_REFERENCE=OFF \
  -DOPTIX_ROOT_DIR="$OPTIX_ROOT_DIR"

cmake --build build/GRotor-rtcore/sm_89 \
  --target GRotor instantgr2_reference grotor_evaluator --parallel 2
```

This produces `GRotor`, `instantgr2_reference`, `grotor_evaluator`,
`rtcore_batches.ptx` and `rtcore_cached.ptx` in the build directory. Keep both
PTX files there when running the executables. GRotor uses the repository's
lookup tables automatically.

For another GPU architecture, change `89`, `sm_89` and the test script's
`--cuda-arch` together. Normal performance tests enable RT-Core batching,
warp/block DAG solving, BIT and incremental cost updates; keep
`GROTOR_FULL_COST_REFERENCE=OFF`.

## Run the comparison

First check a small case using the build above:

```bash
python3 script/benchmark_grotor_system.py \
  --no-build \
  --data-dir ../ISPD24_tmp \
  --cases mempool_tile \
  --output-dir results/GRotor-smoke
```

Run all seven datasets:

```bash
python3 script/benchmark_grotor_system.py \
  --no-build \
  --data-dir ../ISPD24_tmp \
  --output-dir results/GRotor-system
```

The default cases are `mempool_tile`, `ariane133_51`, `ariane133_68`, `nvdla`,
`bsg_chip`, `mempool_group` and `mempool_cluster`. Each case runs InstantGR2 and
GRotor sequentially, writes an actual routing output, and invokes the external
evaluator. All four printed cost values must match, with zero open and
incomplete nets. The temporary output is deleted after evaluation, before
the next run. No `.out` or compressed routing reference is retained.

The script can also configure and compile automatically; omit `--no-build`:

```bash
python3 script/benchmark_grotor_system.py \
  --optix-root "$OPTIX_ROOT_DIR" \
  --nvcc "$NVCC" \
  --cases mempool_tile nvdla \
  --output-dir results/GRotor-subset
```

Command format:

```text
python3 script/benchmark_grotor_system.py
    [--data-dir INPUT_DIR] [--cases CASE [CASE ...]]
    [--output-dir RESULTS_DIR] [--gpu INDEX] [--cuda-arch ARCH]
    [--optix-root SDK_DIR] [--nvcc COMPILER] [--jobs N]
    [--no-build] [--clean-results] [--validate-batches]
```

| Option | Default / behavior |
|---|---|
| `--data-dir` | `../ISPD24_tmp` |
| `--cases` | All seven cases listed above; give names without extensions |
| `--output-dir` | `results/GRotor-system`; must be below repository `results/` |
| `--gpu` | Physical GPU index `0` |
| `--cuda-arch` | `89`; selects `build/GRotor-rtcore/sm_<ARCH>/` |
| `--optix-root` | Reads `OPTIX_ROOT_DIR` when omitted |
| `--nvcc` | Reads `NVCC`, otherwise finds `nvcc` on `PATH` |
| `--jobs` | `2` parallel compilation jobs |
| `--no-build` | Reuses a matching Release build with RT-Core and incremental updates enabled |
| `--clean-results` | Deletes and replaces only the selected report directory |
| `--validate-batches` | Also compares both GRotor batch lists with CPU results; off for performance runs |

Existing output directories are preserved by default. To repeat a test in the
same directory, explicitly replace its previous reports and logs:

```bash
python3 script/benchmark_grotor_system.py --no-build --clean-results \
  --data-dir ../ISPD24_tmp --output-dir results/GRotor-system
```

The script prints the completed tables and report path to the terminal.
A routing, evaluator or configuration failure returns a nonzero exit status;
inspect `summary.json`, `summary.md` and the case logs. Completed-case rows are
retained in the report. Run tests on an otherwise idle GPU for comparable
performance measurements.

## Generated files

All results go directly into the selected directory, without date subfolders.

| File | Contents |
|---|---|
| `summary.md` | L-shape and FLT tables for total time, batching and DAG solving, plus quality and memory |
| `runtime.csv` | One row per dataset/algorithm with all three runtime comparisons and speedups |
| `stage-runtime.csv` | InstantGR2/GRotor batching, DAG and total times; total speedup |
| `batching-runtime.csv` | CPU, CUDA-Core and RT-Core batching times; CPU/RT and CUDA/RT speedups |
| `dag-runtime.csv` | InstantGR2/GRotor DAG times and speedup |
| `quality.csv` | Matching wirelength, via, overflow and total evaluator costs; connectivity counts |
| `memory.csv` | GRotor CPU peak RSS and sampled GPU peak/minimum free memory |
| `summary.json` | Raw timings, exact commands, configuration, source/input/binary hashes and status |
| `configure.log`, `build.log` | Present when the script performs compilation |
| `<case>/baseline.routing.log`, `<case>/baseline.evaluator.log` | InstantGR2 routing and external evaluator logs |
| `<case>/rtcore.routing.log`, `<case>/rtcore.evaluator.log` | GRotor routing and external evaluator logs |
| `<case>/rtcore.host_memory.log`, `<case>/rtcore.gpu_memory.csv` | GNU time statistics and GPU memory samples |

Every runtime CSV uses an `algorithm` column containing `L-shape` or `FLT`.
There is no routing-backend or stage column in these tables. Runtime units are
milliseconds; speedups are unitless ratios.

## Reading the timings

```text
InstantGR2 total = CUDA-Core batching + InstantGR2 DAG execution
GRotor total    = RT-Core batching   + GRotor DAG execution
Speedup         = InstantGR2 time    / GRotor time
```

A speedup above `1` means GRotor is faster. The batching table additionally
reports CPU batching time and CPU/RT speedup. CPU batching supplies InstantGR2's
deterministic routing plan in dual mode; its time is excluded from algorithm
totals. Sampling is fixed to `1`. GRotor uses standard RT-Core batching for
L-shape and cached RT-Core batching for FLT, with no method-tuning arguments.

L-shape DAG time covers its complete batch execution loop. GRotor FLT uses one
continuous synchronized interval including compact upload, GPU batch assembly,
cost/prefix updates, solving, demand commits and waits. InstantGR2 FLT retains
its sum of per-batch GPU suffix times, excluding preceding CPU packing. These
scope differences are retained in the comparison.

Standard RT-Core timing covers the coloring loop; cached RT-Core timing covers
pipeline preparation through coloring completion. Earlier geometry/BVH work
and final batch collection are outside both intervals. CPU/CUDA-Core batching
retains its complete-function timing. Algorithm totals exclude earlier
preparation and file output. Complete GRotor phase and router times are also
recorded in routing logs and `summary.json`.

Performance tables contain single sequential runs with unlocked GPU clocks.
GPU memory is sampled device-wide every 100 ms; CPU peak RSS comes from GNU time
for the GRotor process. Evaluation runs after routing and is excluded from
algorithm timing.

## Regenerate tables without rerunning

Use a saved system benchmark `summary.json` to export all tables to a separate
directory. This requires no GPU, compilation, datasets or evaluator run:

```bash
python3 script/benchmark_grotor_system.py \
  --report-only results/GRotor-system/summary.json \
  --output-dir results/GRotor-tables
```

To replace a previously generated table directory, add `--clean-results`.
The input report must be outside the output directory. The new summary records
the source report path/hash and identifies the measurements as reused.

## Table 3: CUGR2 comparison

Compare CUGR2's CPU L-shape, detour and sparse-maze algorithms with GRotor's
CUGR2 routing pipeline on one dataset:

```bash
python3 script/compare_cugr2.py \
  --case mempool_tile \
  --nvcc "$NVCC" \
  --optix-root "$OPTIX_ROOT_DIR"
```

The script builds the CPU baseline, `grotor_cugr2` and
`grotor_evaluator`, then runs both routers sequentially. CUGR2 requires Boost
system, filesystem and program_options development libraries. Its retained
LEF/DEF parser libraries target Linux x86-64. To build explicitly:

```bash
cmake -S baseline/cu-gr-2/src -B build/CUGR2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/CUGR2 --target route --parallel 2
cmake -S . -B build/GRotor-cugr2/sm_89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGROTOR_ENABLE_RTCORE=ON \
  -DGROTOR_BUILD_CUGR2_ROUTER=ON \
  -DOPTIX_ROOT_DIR="$OPTIX_ROOT_DIR"
cmake --build build/GRotor-cugr2/sm_89 \
  --target grotor_cugr2 grotor_evaluator --parallel 2
```

Reuse these binaries and validate RT-Core batch lists against CPU batching:

```bash
python3 script/compare_cugr2.py \
  --case nvdla --no-build --validate-batches
```

Command format:

```text
python3 script/compare_cugr2.py --case CASE
  [--data-dir ../ISPD24_tmp]
  [--def-dir ../ISPD24_def/def]
  [--lef ../ISPD24_def/lef/Nangate.lef]
  [--def-file PATH] [--threads 8] [--cpu-log PATH | --reuse-cpu SUMMARY_JSON]
  [--cpu-timeout 3600] [--gpu-timeout 3600]
  [--gpu 0] [--cuda-arch 89] [--nvcc PATH] [--optix-root PATH]
  [--jobs 2] [--no-build] [--validate-batches] [--clean-results] [--report-only]
```

CUGR2 reads LEF/DEF; GRotor reads the corresponding `.cap/.net` pair.
`mempool_cluster` maps to `cluster.def`; other cases use `<case>.def`.
Pass `--def-file` for another filename mapping. The input mapping does not
select routing algorithms: all datasets run the same three stages. Each
stage reports both task counts, and speedups require nonzero counts within
a factor of ten. Exact evaluator score equality is not required.

Reports are written to `results/CUGR2/<case>/`:

| File | Contents |
|---|---|
| `summary.md` | Three-stage runtime, speedup, task counts and quality tables |
| `runtime.csv` | CPU runtime, GRotor batching/DAG components, speedups, workload ratios and excluded diagnostics |
| `table3.csv` | Compact Table 3 rows in seconds, with speedups and both task counts |
| `quality.csv` | GRotor ISPD24 evaluator costs and connectivity results |
| `summary.json` | Commands, source/input/binary hashes, timing data and native CUGR2 statistics |
| `cugr2.routing.log`, `grotor.routing.log` | Actual router output and process wall time/peak host RSS |
| `grotor.evaluator.log`, `gpu-memory.csv` | Actual output evaluation and sampled device memory |

GRotor writes a routing output and invokes the evaluator before deleting it.
CUGR2's guide is deleted after its run. Routing files are not retained.
Use `--clean-results` to replace a previous report for the same dataset.
Use `--report-only --case CASE` to regenerate tables from hash-verified saved
logs without rerunning either router.

CUGR2 time sums per-net solver and result-commit regions; maze includes sparse
search and the subsequent pattern-routing DP. GRotor time for all three
algorithms is measured RT-Core batching plus one continuous GPU
batch-execution/update interval. The GPU interval includes topology uploads,
GPU batch assembly, cost/BIT updates, DAG solving and demand commits.
Maze CPU search/topology preparation, quality checks and restoration are
excluded from this comparison total. Its full phase wall interval is retained
as `grotor_maze_phase_ms` for diagnostics; batching + DAG + CPU search +
`grotor_maze_other_ms` equals that diagnostic interval.
A bounded device checkpoint and deterministic internal-cost reduction protect
the previous result: worsening candidates restore the previous routes and
demand. Acceptance is recorded separately from the comparison runtime.
The acceptance objective is the router's internal cost; official quality
scores still come only from evaluating the actual routing output.
These component times exclude some setup and I/O; process wall times remain
in the logs. CUGR2 native quality uses its LEF/DEF resource model and is not
an ISPD24 evaluator score. Hardware, inputs and timing scopes differ from the
published table, so current measurements are not expected to match its
printed numbers. Timed-out CPU stages remain incomplete and have no speedup.
For an expensive CPU run that already has a saved log, `--cpu-log PATH`
explicitly imports its measurements while still rerunning GRotor and the
evaluator. Imported measurements are labeled as archived; legacy logs do not
prove source/input/binary identity. A previously skipped CPU maze remains
incomplete, with its planned net count shown separately and no speedup.

To rerun GRotor after a code change without repeating a long CPU run, use a
saved summary. Input, CPU-source, binary and log fingerprints must match:

```bash
python3 script/compare_cugr2.py --case mempool_group --no-build \
  --reuse-cpu results/CUGR2/mempool_group/summary.json --clean-results
```

The CPU record is retained in `build/CUGR2/cpu-cache/` and labeled as reused;
GRotor and its external evaluator run again. For sparse-maze storage and
checkpoint checks, configure with `-DGROTOR_BUILD_CUGR2_TESTS=ON`, build
`grotor_cugr2_check`, and run
`ctest --test-dir build/GRotor-cugr2/sm_89 -R cugr2_components --output-on-failure`.

## Figure 7: blocker ray-count microbenchmark

Run one L-shape dataset and produce its paired count table and curve:

```bash
python3 script/run-fig7.py --case bsg_chip \
  --data-dir ../ISPD24_tmp --optix-root "$OPTIX_ROOT_DIR" \
  --output-dir results/fig7
```

The script builds `grotor_rtcore_ray_benchmark` and `rtcore_synchronous.ptx` in `build/fig7/sm_89/`.
Add `--no-build` to reuse them. Matplotlib is required for plotting; install it
in your Python environment with `python3 -m pip install matplotlib` if needed.
GPU selection, compiler/architecture, build jobs and `--clean-results` work as
in the main benchmark. `--case` is required and accepts exactly one dataset.

Without `--output-dir`, files are written directly to
`results/fig7/<case>/`. Each invocation produces one dataset's table and one
plot, exported as PDF, SVG and PNG.

The selected result directory contains:

| File | Contents |
|---|---|
| `ray-counts.csv` | Exactly two columns: `without_blocker,with_blocker`; row 1 is iteration 0 |
| `ray-counts.md` | The same complete two-column table in Markdown |
| `round-stats.csv` | Iteration index, both trace counts and shared assignment progress |
| `checks.json` | Exact per-net assignment-round/color and CPU batch-list checks |
| `figure7.pdf`, `figure7.svg`, `figure7.png` | The two measured curves, including every iteration |
| `worker.log`, `host-memory.log`, `gpu-memory.csv` | Execution log and memory observations |

The same directory contains `summary.md` and `summary.json`, recording
checks, totals, inputs, source/binary hashes and commands.

This is a separate synchronous, uncached RT-Core method. Batch IDs are frozen
throughout ray tracing; assignment decisions are stored separately and committed
at the end of the round. Ray eligibility is also frozen. Both arms collect the
same blocker state, but the control never uses it to suppress launches. Actual
`optixTrace` calls are counted, rather than counting launch-grid threads.
Every net must receive the same color in the same iteration in both arms;
no rounds are padded or removed. Final batch lists must equal CPU batching.
The microbenchmark stops after L-shape batching and creates no routing `.out`
files or evaluator scores. Instrumented times are not performance results.

Recreate this dataset's figure from its saved CSV/check files without GPU work:

```bash
python3 script/run-fig7.py --case bsg_chip --plot-only --output-dir results/fig7
```

Run the independent GPU checks, including a chain whose predicted count is
`N, N-1, ..., 1` without blockers and `N, 1, ..., 1` with blockers:

```bash
cmake -S . -B build/fig7/sm_89 -DGROTOR_BUILD_RTCORE_BLOCKER_TESTS=ON
cmake --build build/fig7/sm_89 --target grotor_rtcore_blocker_check --parallel 2
ctest --test-dir build/fig7/sm_89 -R rtcore_blocker --output-on-failure
```

## Figure 10: per-batch routing DAG throughput

Build and run the L-shape experiment for one specified dataset:

```bash
python3 script/run-fig10.py --nvcc "$NVCC" \
  --case nvdla \
  --data-dir ../ISPD24_tmp --output-dir results/fig10
```

`--case` is required. The default output directory is `results/fig10/<case>/`.
This script needs CUDA and Matplotlib but does not need OptiX. It builds `grotor_dag_throughput` and
`grotor_evaluator` under `build/fig10/sm_89/`. Use `--no-build` for subsequent
runs, or start with a small smoke test:

```bash
python3 script/run-fig10.py --no-build --case mempool_tile \
  --output-dir results/fig10-smoke
```

Command format:

```text
python3 script/run-fig10.py
    --case CASE [--data-dir INPUT_DIR]
    [--output-dir RESULTS_DIR] [--gpu INDEX] [--cuda-arch ARCH]
    [--nvcc COMPILER] [--jobs N] [--no-build] [--clean-results]
```

Both variants reuse exactly the same CPU batch plan (sparsity `1`), net order
and RSMT topology, and start from identical demand and route state:

- **Without optimizations:** original thread-per-DAG algorithm with full cost
  refresh and full prefix scans, using explicit device memory.
- **With optimizations:** warp-per-DAG solving, BIT prefix maintenance and
  incremental cost updates.

Each batch is timed with synchronized CPU timestamps around cost/prefix
updates, DAG solving/traceback and wire/via demand commits. Throughput is
`batch net count / elapsed seconds`. CPU batching, topology construction,
allocations, initial uploads, file output and evaluation precede or follow the
measured intervals. The first batch includes cost/prefix initialization in
both variants; no samples are filtered out. These per-batch measurements add
synchronization overhead and are separate from the main benchmark's
continuous stage timers.

The selected result directory directly contains:

| File | Contents |
|---|---|
| `throughput.csv`, `throughput.md` | Exactly two throughput columns: `without_optimizations,with_optimizations`, in nets/s; first data row is batch 0 |
| `batch-runtime.csv` | Every batch's index, net/node count, shared plan hash, timestamps, both runtimes and throughput speedup |
| `without-optimizations.csv`, `with-optimizations.csv` | Raw measured batch intervals |
| `figure10.pdf`, `figure10.svg`, `figure10.png` | Complete per-batch curves with a logarithmic throughput axis |
| `*.evaluator.log`, `checks.json` | External evaluation and exact equality of four cost strings with zero open/incomplete nets |
| `measurement.json`, `worker.log`, `host-memory.log`, `gpu-memory.csv` | Measurement configuration, execution log and memory observations |

`summary.md/json` in the same directory records aggregate throughput, quality,
inputs, source/binary hashes and commands. Aggregate throughput uses total nets
divided by summed batch time, rather than averaging per-batch ratios.

Each arm writes a real L-shape routing output and runs the external evaluator
on it. The output is deleted before the next arm, with only one temporary
`.out` at a time under the repository's build directory. Scores are for the
L-shape result before FLT rerouting. The unoptimized arm is the original
scalar algorithm in the same framework, not a separate InstantGR2 executable.

Regenerate this dataset's table and plot from the same result directory,
without rerunning routing:

```bash
python3 script/run-fig10.py --case nvdla --plot-only --output-dir results/fig10
```

## Figure 11: prefix memory-access microbenchmark

Build and measure the three L-shape prefix variants:

```bash
python3 script/run-fig11.py --nvcc "$NVCC" \
  --case nvdla \
  --data-dir ../ISPD24_tmp --output-dir results/fig11
```

`--case` is required. Files go directly into `results/fig11/<case>/` by default.
CUDA and Matplotlib are required; OptiX is not.
The script builds `grotor_prefix_access_benchmark` and `grotor_evaluator` in `build/fig11/sm_89/`.
Use `--no-build` to reuse them. For a small smoke test:

```bash
python3 script/run-fig11.py --no-build --case mempool_tile \
  --output-dir results/fig11-smoke
```

The command accepts the same `--data-dir`, single-value `--case`, `--output-dir`,
`--gpu`, `--cuda-arch`, `--nvcc`, `--jobs`, `--no-build` and `--clean-results`
options as Figure 10. Existing result directories are preserved unless
`--clean-results` is given.

Each plotted value is **prefix update writes + wire-cost query reads** for
that batch. All three variants use the same CPU batch plan (sparsity `1`),
RSMT topology and warp-per-DAG solver, starting with identical demand:

- **Original:** full cost updates, a full conventional prefix scan, and two
  endpoint reads per interval query.
- **BIT:** incremental cost/dirty-node updates; independently sum both endpoint
  prefixes in the BIT and subtract.
- **BIT & Pruning:** identical BIT updates; stop both endpoint traversals at
  their common ancestor to omit their shared prefix.

The two direct-query BIT methods are retained as `bit_range_unpruned` and
`bit_range_pruned` in `src/experiments/prefix_queries.cuh`. Both experimental
arms use full BIT storage and skip dense prefix-cache generation. The normal
router currently generates that cache after its BIT updates and queries the
cache; Figure 11 is a separate direct-query ablation. Profiling is controlled by the CMake option `GROTOR_BUILD_PREFIX_ACCESS_BENCHMARK` (default
`OFF`). The script enables this option to build the standalone Figure 11
worker. Only the Figure 11 worker/tests receive the private compile definition
`GROTOR_PREFIX_ACCESS_COUNTS=1`; `GRotor`, InstantGR2 and Figure 10 do not.

Inside this profiling build, the warp-uniform child-segment query is executed
by lane 0 and broadcast with a shuffle. It performs one query rather than 32
identical lane queries. Other layer-specific queries remain individually
counted. No totals are divided or capped. The unprofiled solver retains its
existing implementation, with no added counters, branches or shuffles.

Counts are logical element loads/stores, not bytes, cache misses or physical
DRAM transactions. Repeated endpoint loads and stored zero-sentinel reads
count. Shared-memory operations, input cost arrays, dirty metadata, counter
traffic and demand-prefix scans do not. BIT maintenance reads are included in
the detailed CSV as a separate diagnostic, outside the requested sum.
Initialization writes in batch 0 are included; every batch remains in the
CSV and plot. Instrumented runtimes are not performance measurements.

Files in the selected result directory:

| File | Contents |
|---|---|
| `accesses.csv`, `accesses.md` | Three total-count columns: `original,bit,bit_pruning`; first data row is execution batch 0 |
| `batch-accesses.csv` | Batch/net/node counts, plan hash, common query count, all three totals, and separate query reads/update writes/maintenance reads |
| `original.csv`, `bit.csv`, `bit-pruning.csv` | Raw per-batch counters for each actual variant |
| `figure11.pdf`, `figure11.svg`, `figure11.png` | Three measured curves with a linear count axis starting at zero |
| `*.evaluator.log`, `measurement.json`, `checks.json` | External quality results, configuration and identical query/update-workload checks |
| `worker.log`, `host-memory.log`, `gpu-memory.csv` | Worker log and memory observations |

`summary.md/json` records aggregate counts, exact commands, quality, inputs
and source/executable hashes. Each variant writes a real L-shape-only routing output
and invokes the evaluator; all four costs must match with zero open/incomplete
nets. Outputs are deleted before the next variant. At most one temporary
`.out` exists under the repository's build directory.

Regenerate this dataset's table and figure without routing or GPU work:

```bash
python3 script/run-fig11.py --case nvdla --plot-only --output-dir results/fig11
```

The independent query check covers all intervals, including equal endpoints,
in non-power-of-two tracks and strided storage, validating both values and
exact read counts against integer-cost and popcount oracles. A second GPU
check requires the profiled warp solver to match the scalar oracle's query
calls/read counts as well as exact costs, paths and demand:

```bash
cmake --build build/fig11/sm_89 \
  --target grotor_prefix_query_check grotor_prefix_dag_check --parallel 2
ctest --test-dir build/fig11/sm_89 \
  -R 'prefix_queries|prefix_dag_broadcast' --output-on-failure
```

## Figure 12: cumulative routing ablation

Run the four configurations with a CPU-batching baseline:

```bash
python3 script/run-fig12.py --case nvdla --nvcc "$NVCC" \
  --optix-root "$OPTIX_ROOT_DIR" \
  --data-dir ../ISPD24_tmp --output-dir results/fig12
```

Select exactly one dataset with required `--case`. The default result directory
is `results/fig12/<case>/`. The `tera_cluster` family is excluded. CUDA, OptiX
SDK headers and Matplotlib are required.

The script builds four independent Release executables under
`build/fig12/sm_89/`, with compile-time switches isolated from normal GRotor:

| Legend | Enabled optimizations | Batching |
|---|---|---|
| `Original` | Full cost/prefix updates; thread-per-DAG L-shape; global-depth FLT kernels | CPU |
| `+BIT` | Incremental cost updates and BIT prefix maintenance | CPU |
| `+Schedule` | Previous features plus warp-centric L-shape and block-centric FLT | CPU |
| `+RT-Core` | Previous features plus standard L-shape / cached FLT RT-Core batching | RT-Core |

These are **cumulative** configurations. Common explicit-memory storage,
compact FLT GPU assembly, bounded staging and output overlap are retained in
all configurations. Figure 11 access counters are disabled in every ablation.

Reuse the compiled executables for a smaller test:

```bash
python3 script/run-fig12.py --no-build \
  --case mempool_tile --output-dir results/fig12-smoke
```

Command format:

```text
python3 script/run-fig12.py
    --case CASE [--data-dir INPUT_DIR]
    [--output-dir RESULTS_DIR] [--gpu INDEX] [--cuda-arch ARCH]
    [--nvcc COMPILER] [--optix-root SDK_DIR] [--jobs N]
    [--no-build] [--clean-results] [--validate-batches]
```

`--validate-batches` additionally compares RT-Core lists with CPU lists,
outside measured intervals. Leave it off for performance runs. Existing
results are preserved; use `--clean-results` to replace only the selected
result directory.

Each algorithm's time is `batching_ms + dag_ms`. L-shape and FLT both use one
continuous synchronized DAG interval. All four configurations are measured
independently. `speedup` divides Original time by the current total;
`incremental_speedup` divides the preceding configuration's time by the
current total. The plot uses the cumulative `speedup`, with Original at `1x`.
RT-Core retains the established scoped batching timers; algorithm totals
exclude earlier topology/geometry preparation, output and evaluation.
The generated summary describes the full timing boundaries.

Generated files:

| File | Contents |
|---|---|
| `figure12.pdf`, `figure12.svg`, `figure12.png` | One linear-axis chart for the selected dataset: L-shape and FLT groups, each with four ablation bars |
| `runtime.csv` | Every variant's batching, DAG and total milliseconds; cumulative, adjacent-step and DAG-only speedups |
| `speedup.csv` | One row per dataset/algorithm, with `original,plus_bit,plus_schedule,plus_rtcore` columns |
| `summary.md`, `summary.json` | Configuration table, runtime tables, evaluator quality, exact commands and provenance hashes |
| `<variant>.routing.log`, `<variant>.evaluator.log` | Actual routing and external evaluator logs for each configuration |
| `<variant>.host-memory.log`, `<variant>.gpu-memory.csv` | CPU RSS and device-wide GPU memory observations |

Every variant writes a real complete routing output and invokes the external
evaluator. All four decimal costs must match with zero open/incomplete nets.
Each temporary output is deleted before the next run; no `.out` is retained.

Regenerate this dataset's tables and chart from saved, hash-checked logs:

```bash
python3 script/run-fig12.py --case nvdla --plot-only --output-dir results/fig12
```

Validate the global-depth FLT baseline against independent scalar kernels,
including exact costs, paths and demand with resident and streamed inputs:

```bash
cmake --build build/fig12/sm_89 --target grotor_global_depth_check --parallel 2
ctest --test-dir build/fig12/sm_89 -R '^global_depth_scheduler$' --output-on-failure
```

## Optional correctness checks

Check exact CPU batch lists on NVDLA, in addition to evaluator scores:

```bash
python3 script/benchmark_grotor_system.py --no-build --validate-batches \
  --cases nvdla --output-dir results/GRotor-batch-check
```

Run reporting checks without a GPU:

```bash
python3 -m unittest discover -s script/tests -p 'test_*.py'
```

Build and run GPU cost, assembly and DAG scheduling checks separately from
performance measurements:

```bash
cmake -S . -B build/GRotor-rtcore/sm_89 \
  -DGROTOR_BUILD_COST_TESTS=ON \
  -DGROTOR_BUILD_FLT_TESTS=ON \
  -DGROTOR_BUILD_DAG_TESTS=ON
cmake --build build/GRotor-rtcore/sm_89 --parallel 2
ctest --test-dir build/GRotor-rtcore/sm_89 \
  -R 'cost_bit_updates|flt_workspace|dag_scheduling' --output-on-failure
```

See [baseline/README.md](baseline/README.md) for the InstantGR2 source version,
local modifications and publication reference, and [NOTICE.md](NOTICE.md) for
third-party licenses.

## GGR Z-shape comparison (Table 3)

This experiment runs one Z-shape stage. Native GGR reads LEF/DEF; GRotor reads
CAP/NET and uses RT-Core RSMT batching with disjoint-bounding-box refinement,
cooperative Z-shape DP, BIT maintenance and incremental cost updates. Both
net-task counts are included in the comparison because the input models differ.

GGR additionally requires CMake 3.22+, CUDA-enabled PyTorch, Python development
headers, Bison, Flex and zlib development headers. The bundled LEF/DEF static
libraries target Linux x86-64. The tested Python environment is Python 3.10
with PyTorch 2.2.0+cu121. Use that same Python interpreter for the build and run.

Build and compare one dataset:

```bash
python3 script/compare_ggr.py --case mempool_tile \
  --input-dir ../ISPD24_tmp \
  --def-dir ../ISPD24_def/def \
  --lef ../ISPD24_def/lef/Nangate.lef \
  --optix-root "$OPTIX_ROOT_DIR" --nvcc "$NVCC"
```

Subsequent runs can reuse the build. Repeated runs compare evaluator scores,
reusing and deleting the same temporary routing output:

```bash
python3 script/compare_ggr.py --case nvdla --no-build \
  --clean-results --validate-batches --repeat 2
```

Command format:

```text
python3 script/compare_ggr.py --case CASE
    [--input-dir DIR] [--def-dir DIR] [--lef FILE]
    [--results-dir DIR] [--optix-root SDK_DIR] [--nvcc COMPILER]
    [--cuda-arch ARCH] [--jobs N] [--threads N] [--timeout SECONDS]
    [--no-build] [--clean-results] [--validate-batches] [--repeat N]
    [--reuse-native SUMMARY_JSON]
```

Cases are `ariane133_51`, `ariane133_68`, `nvdla`, `bsg_chip`, `mempool_tile`
and `mempool_group`; `mempool_cluster` is excluded. Relative paths resolve
from the caller's directory. Builds go to `build/GGR` and
`build/GRotor-ggr/sm_<ARCH>`. Results go to `results/GGR/<case>` by default:

- `runtime.csv`: batching, routing DAG and total runtime comparisons/speedups.
- `table3.csv`: total-time comparison with net and batch counts.
- `quality.csv`: actual GRotor evaluator scores for each run.
- `ggr.log`, `grotor.log`, `grotor.evaluate.log`: native, optimized and evaluator logs.
- `summary.md`, `summary.json`: readable tables and input/source/binary fingerprints.

Total time is batching plus routing DAG time. GGR's inherited routing metric
sums cost/prefix update and DP/commit GPU times, excluding CPU preparation and
dense map clearing. GRotor measures its entire GPU batch loop with one pair of
synchronized timestamps, including prepared-input transfers, incremental
cost/BIT updates, node-map resets, DP, route conversion and demand commits.
GRotor batching includes coloring, result collection and bounding-box refinement;
OptiX scene construction and CPU topology encoding are outside the two reported
intervals. Full process wall times and peak host RSS remain in the logs.

GGR's native guides and cost model differ from CAP/NET; no common quality score
or byte-identical routing is claimed. GRotor writes an actual `.out`, runs the
evaluator and requires zero open/incomplete nets. The adapter retains its
deterministic promotion of congested wire segments to upper layers and merges
duplicate tree paths before output. This CPU postprocessing is logged separately
and excluded from the GPU timing. No Detour or Maze stage runs. Temporary native
guides and `.out` files are deleted before the next run; logs, CSV files and
summaries remain.

Use `--reuse-native results/GGR/<case>/summary.json` to reuse an existing native
GGR run while testing GRotor. The script verifies native inputs, sources,
module hashes and log timings before accepting it. This also works together
with `--clean-results` for the same dataset directory.
