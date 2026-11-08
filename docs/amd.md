# AMD HIP / HIPRT

The HIP backend compiles the shared GRotor routing pipeline with hipCUB.
HIPRT provides AMD ray-tracing batching.

## Build

Requirements: CMake 3.21+, C++17, HIP compiler/runtime and hipCUB, plus a
32-lane-wave AMD GPU. The primary configuration is RX 7900 XTX (`gfx1100`).
Wave64 is rejected at startup.
Keep CUDA and HIP in separate build directories.

| GPU | HIPRT preset | Architecture |
|---|---|---|
| RX 7900 XTX / XT | `rx7900` | `gfx1100` |
| RX 9060 XT | `hiprt` | `gfx1200` |

HIPRT must be built separately with matching headers and library. The validated
version is 3.2.0.ceba893, commit `0629d08568fff39f9a36bf9ee800fc7bac5bb096`:

```bash
git clone https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git /path/to/HIPRT
git -C /path/to/HIPRT checkout 0629d08568fff39f9a36bf9ee800fc7bac5bb096
git -C /path/to/HIPRT submodule update --init --recursive
cmake -S /path/to/HIPRT -B /path/to/HIPRT/build \
  -DNO_UNITTEST=ON -DBITCODE=OFF -DFORCE_DISABLE_CUDA=ON -DHIP_PATH=/opt/rocm
cmake --build /path/to/HIPRT/build -j2
export HIPRT_ROOT=/path/to/HIPRT
export HIPRT_PATH="$HIPRT_ROOT"
cmake --preset rx7900
cmake --build --preset rx7900
```

This writes `build/hip/GRotor` and `resources/evaluator`. If HIPRT library
selection is ambiguous, pass `-DHIPRT_LIBRARY=/path/to/libhiprt0300264.so`.
HIP builds discover no CUDA, OptiX, MPI, NVSHMEM or NCCL dependencies.
For another AMD architecture, override `-DCMAKE_HIP_ARCHITECTURES=<arch>`
when configuring.

## Routing and Performance Comparison

```bash
python3 script/run_grotor.py --variant hiprt --case mempool_tile --data-dir /path/to/benchmarks --validate-batches
cmake --preset rx7900 -DGROTOR_BUILD_ROUTING_ABLATION=ON
cmake --build build/hip --target GRotor grotor_amd_original grotor_evaluator -j2
python3 script/experiments/run_amd.py --data-dir /path/to/benchmarks \
  --results results/amd --repeats 3 --warmups 1
```

`grotor_amd_original` uses CPU batching, full cost/prefix updates and
thread/global-depth DAG solving. Optimized `GRotor` uses HIPRT batching,
incremental cost/BIT and warp/block DAG solving. The runner validates exact
batch lists, evaluator costs and per-net segment multisets. Warmups and
validation are separate from performance runs; execution order alternates.

The default nine inputs are `ariane133_68`, `ariane_68_rank`,
`bsg_chip`, `bsg_chip_rank`, `nvdla`, `nvdla_rank`, `mempool_tile`,
`mempool_tile_rank`, `mempool_group`. Override with `--cases`; rank variants
remain distinct. Failed cases remain visible and suppress incomplete geometric
means. `--plot-only --results <directory>` regenerates completed plots.

## Timing and limits

- Algorithm runtime = batching function-call wall time + the complete
  synchronized routing DAG interval, including cost/prefix updates and commits.
- HIPRT uses grid/chunk batching with 512-net chunks and 1024 colors. A color
  overflow fails explicitly. `--validate-batches` checks exact CPU/HIPRT lists
  and should be disabled for performance measurements.
- The current backend requires `max(X,Y)*8 <= sharedMemPerBlock` (8192 positions
  on a 64 KiB device). Larger tracks are rejected. This does not establish
  tera-scale support.
- Host-memory saving defaults to OFF. Sampled GPU memory is device-wide and
  can include other processes. Clocks and power settings are not changed.
