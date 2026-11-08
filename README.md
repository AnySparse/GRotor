# GRotor

GPU parallel global routing with routing DAG abstraction, RT-Core batching and parallel GPU DAG execution.

## 1. Build

### 1.1 Requirements

- Linux, a C++17 compiler, Make, [CMake 3.21+](https://cmake.org/download/) and Python 3.9+.
- An NVIDIA GPU, [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)
  (`nvcc` on `PATH`) and a compatible NVIDIA driver. RT-Core batching requires an RTX GPU.
- [NVIDIA OptiX SDK 7.7+](https://developer.nvidia.com/designworks/optix/downloads/legacy)
  for RT-Core batching. Blackwell requires CUDA 12.8+ and an OptiX SDK supporting Blackwell.

Download the [OptiX 7.7 Linux installer](https://developer.nvidia.com/downloads/designworks/optix/secure/7.7/nvidia-optix-sdk-7.7.0-linux64-x86_64.sh/)
(NVIDIA account required), then install:

```bash
export OPTIX_ROOT_DIR="$HOME/optix-7.7"
mkdir -p "$OPTIX_ROOT_DIR"
sh NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh \
  --prefix="$OPTIX_ROOT_DIR" --exclude-subdir --skip-license
```

### 1.2 Build Instructions

From the repository root (RTX 4090):

```bash
cmake --preset rtx4090
cmake --build --preset rtx4090
```

| Preset | Architecture | Batching path | Build directory |
|---|---:|---|---|
| `rtx4090` | 89 | RT-Core | `build/GRotor/sm_89` |
| `rtxpro6000` | 120 | RT-Core | `build/GRotor/sm_120` |
| `a100` | 80 | CPU fallback | `build/GRotor/sm_80` |
| `h100` | 90 | CPU fallback | `build/GRotor/sm_90` |

The presets build `GRotor`, `instantgr2_reference` and `resources/evaluator`.

Equivalent configuration without a preset:

```bash
cmake -S . -B build/GRotor/sm_89 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGROTOR_ENABLE_RTCORE=ON
cmake --build build/GRotor/sm_89 -j2
```

Set `CMAKE_CUDA_ARCHITECTURES` for the target GPU and `GROTOR_ENABLE_RTCORE=OFF`
for CPU batching. For a custom build directory, pass `--build-dir <path>` to the
run script. Retain generated PTX files in the RTX build directory.

[Additional build options](docs/build-options.md).

### 1.3 Baselines

InstantGR2 (also included in the NVIDIA presets):

```bash
mkdir -p build/InstantGR2/sm_89
nvcc -x cu -std=c++17 -O3 -arch=sm_89 \
  baseline/InstantGR2/src/main.cpp \
  -o build/InstantGR2/sm_89/instantgr2_reference
```

CUGR2 (CPU; requires Boost system, filesystem and program_options):

```bash
cmake -S baseline/cu-gr-2/src -B build/CUGR2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/CUGR2 --target route -j2
```

GGR (CMake 3.22+, CUDA-enabled PyTorch, Python development headers, Bison, Flex and zlib):

```bash
cmake -S baseline/GGR -B build/GGR -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build/GGR --target gpugr flute_cpp -j2
```

Baseline source revisions, modifications and licenses: [baseline/README.md](baseline/README.md).

## 2. Run

### 2.1 Datasets

Download the [Nangate45 benchmarks](https://drive.google.com/drive/folders/1ocChoQupNxlLBH2hgqkPwR0-7D3ocTwm?usp=sharing):
`Simple_inputs/` contains matching `<case>.cap` and `<case>.net` inputs for
GRotor/InstantGR2. `lef/` and `def/` contain technology and design inputs for CUGR2/GGR.

Specify input locations with `--data-dir` for CAP/NET, or `--lef` and
`--def-dir` for LEF/DEF. DEF filenames follow `<case>.def`,
`mempool_cluster` uses `cluster.def`.

### 2.2 Routing

Run GRotor with RT-Core batching on RTX 4090:

```bash
python3 script/run_grotor.py --case mempool_tile \
  --data-dir /path/to/cap-net-files
```

Results are saved under `results/run/grotor-rtx/mempool_tile/`: the routing
solution `mempool_tile.out`, runtime log `routing.log`, evaluator scores in
`evaluation.log`, and execution summary `summary.json`.

Change `--case` and `--data-dir` for other datasets. Select the target GPU with
`--cuda-arch`; A100/H100 also use `--variant cuda`. Custom build and output
directories are specified with `--build-dir` and `--output-dir`.

### 2.3 Evaluation

The framework-wide source and executable are `resources/evaluator.cpp` and
`resources/evaluator`. The evaluator retains InstantGR2's
[BSD 3-Clause license](third_party/LICENSE).
It can be built independently of any GPU toolkit:

```bash
g++ -O3 -UNDEBUG -std=c++17 resources/evaluator.cpp -o resources/evaluator
python3 script/evaluate.py --case mempool_tile --data-dir /path/to/cap-net-files \
  --out results/run/grotor-rtx/mempool_tile/mempool_tile.out
```

The evaluator reports wirelength, via, overflow and total cost, plus open and
incomplete net counts. The raw CLI is
`resources/evaluator <case.cap> <case.net> <solution.out>`.

## 3. Benchmarks

### 3.1 Overall Routing Performance

Benchmark scripts require `/usr/bin/time`, `nvidia-smi` and Matplotlib in addition to the build dependencies. They build their targets unless `--no-build` is given.
Use an idle GPU. Each stage's runtime is **batching + routing DAG execution**.

```bash
python3 script/experiments/compare_instantgr2.py --data-dir /path/to/cap-net-files
```

This runs all 12 non-tera paper cases, with one warmup and three paired repeats
per case. Sample/rank inputs remain separate:

| Paper IDs | Sample | Rank |
|---|---|---|
| B1 / B2 | `ariane133_68` | `ariane_68_rank` |
| B3 / B4 | `nvdla` | `nvdla_rank` |
| B5 / B6 | `bsg_chip` | `bsg_chip_rank` |
| B7 / B8 | `mempool_tile` | `mempool_tile_rank` |
| B9 / B10 | `mempool_group` | `mempool_group_rank` |
| B11 / B12 | `mempool_cluster` | `mempool_cluster_ranking` |

`results/instantgr2-comparison/` contains runtime tables, evaluator scores, raw
logs and plots. Reported components are medians across repeats. Each routing
output is checked for matching evaluator scores and zero open/incomplete nets,
then removed.

| Current paper reference | Output |
|---|---|
| Table 2: L-shape / FLT overall comparison | `total-runtime.csv`, `total.pdf/png/svg` |
| Figure 8: batching | `batching-runtime.csv`, `batching.pdf/png/svg` (CPU / CUDA / RT cores) |
| Figure 9: routing DAG | `dag-runtime.csv`, `dag.pdf/png/svg` |

Use `--cases <case1> <case2> ...` to select datasets and `--no-build` to reuse
compiled programs. Set `--cuda-arch` for the target GPU listed in Section 1.2
(default: `89` for RTX 4090; `120` for RTX PRO 6000 Blackwell).

B13/B14 (`tera_cluster`, `tera_cluster_rank`) require an RTX GPU with at least
80 GB VRAM and are selected with the same `--cases` option.
For these cases, enable `GROTOR_HOST_MEMORY_SAVING=ON` and
`INSTANTGR2_HOST_MEMORY_SAVING=ON` during CMake configuration and rebuild both
routers, see [host-memory options](docs/build-options.md#cost-and-host-memory-options).

### 3.2 Other Routing Baselines

```bash
python3 script/experiments/compare_cugr2.py --data-dir /path/to/cap-net-files --lef /path/to/technology.lef --def-dir /path/to/def-files
python3 script/experiments/compare_ggr.py --data-dir /path/to/cap-net-files --lef /path/to/technology.lef --def-dir /path/to/def-files
```

Both default to B1/B3/B5/B7/B9/B11. CUGR2 compares **L-shape, Detour and Maze**;
GGR compares **Z-shape**. Results are runtime tables only. Native LEF/DEF and matching CAP/NET inputs are required. Outputs and logs go to `results/cugr2-comparison`
and `results/ggr-comparison`， temporary `.guide/.out` files are removed.

Use `--case mempool_tile` or `--cases ...` for a subset, `--repeats N` for repetition, and `--native-build-dir` / `--build-dir` to reuse compiled native and GRotor adapter targets with `--no-build`. `--native-timeout` defaults to 10800 seconds.

### 3.3 Microbenchmarks

Names describe the experiment, paper numbering is confined to this mapping.
Each command selects one case, builds its worker and exports CSV, logs and
PDF/PNG/SVG plots under `results/<experiment>/<case>/`.

| Current paper | Experiment | Command |
|---|---|---|
| Figure 7 | Blocker ray-launch counts | `python3 script/experiments/run_blocker.py --case bsg_chip --data-dir /path/to/cap-net-files` |
| Figure 10 | Per-batch routing DAG throughput | `python3 script/experiments/run_dag_throughput.py --case bsg_chip --data-dir /path/to/cap-net-files` |
| Figure 11 | Prefix/BIT memory-access counts | `python3 script/experiments/run_prefix_access.py --case bsg_chip --data-dir /path/to/cap-net-files` |
| Figure 12 | Cumulative routing ablation | `python3 script/experiments/run_ablation.py --case bsg_chip --data-dir /path/to/cap-net-files` |

Use `--case nvdla`, `mempool_group` or `mempool_cluster` to generate the other paper panels. Shared options include `--data-dir`, `--output-dir`,
`--build-dir`, `--cuda-arch`, `--nvcc`, `--no-build`, `--clean-results` and
`--plot-only` (replot completed measurements without a GPU).



## 4. Extensions

### 4.1 Multi-GPU Scaling

Requirements: NVLink-connected NVIDIA GPUs with native peer atomics, CUDA,
Open MPI, NCCL and NVSHMEM 3.x host/device libraries.

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export NCCL_HOME=/path/to/nccl
cmake --preset multi-a100
cmake --build --preset multi-a100
python3 script/experiments/run_multigpu.py --cuda-arch 80 --data-dir /path/to/cap-net-files \
  --cases mempool_cluster \
  --devices 0,1,2,3,4,5,6,7 --counts 1 2 3 4 5 6 7 8 --repeats 2
```

For H100 use preset `multi-h100` and architecture `90`. MPI must be discoverable
by CMake and `mpirun` available on PATH. `scaling.csv` reports median time, speedup and
parallel efficiency; evaluator costs must match the one-GPU result.
NVSHMEM heap size uses `--symmetric-size` or `NVSHMEM_SYMMETRIC_SIZE` (default `25G`).
[Multi-GPU setup and usage](docs/multigpu.md).

### 4.2 AMD Portability

Requirements: a 32-lane-wave AMD GPU, HIP compiler/runtime and hipCUB.
Ray-tracing batching also requires HIPRT SDK 3.2.0 with matching headers and library;
see [HIPRT installation](docs/amd.md#build). Select devices with `HIP_VISIBLE_DEVICES`.

Compare GRotor's unoptimized and HIPRT-optimized strategies;
no AMD InstantGR2 baseline is used. Validation checks exact batch lists,
evaluator scores and per-net geometry before timed runs.

The primary AMD configuration is RX 7900 XTX (`gfx1100`). For the HIPRT experiment:

```bash
export HIPRT_ROOT=/path/to/HIPRT
export HIPRT_PATH="$HIPRT_ROOT"
cmake --preset rx7900 -DGROTOR_BUILD_ROUTING_ABLATION=ON
cmake --build build/hip --target GRotor grotor_amd_original grotor_evaluator -j2
python3 script/experiments/run_amd.py --data-dir /path/to/cap-net-files \
  --results results/amd --repeats 3 --warmups 1
```

For other AMD GPUs, see the [preset list](docs/amd.md#build) or override
`CMAKE_HIP_ARCHITECTURES` during configuration.
Extension builds do not compile baselines.
The runner saves all samples, per-case medians, geometric means and
`amd-speedup.pdf/png`. `--plot-only --results <directory>` regenerates the
plot. [HIPRT setup and backend limits](docs/amd.md).
