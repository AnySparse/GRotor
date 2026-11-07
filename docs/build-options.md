# Additional build options

Run configure/build commands from the repository root. Main build and run
commands are in [README](../README.md). Presets need CMake 3.21+; manual CUDA
configuration supports CMake 3.18+.

## Manual configuration

RTX example:

```bash
cmake -S . -B build/GRotor/sm_89 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DGROTOR_ENABLE_RTCORE=ON
cmake --build build/GRotor/sm_89 \
  --target GRotor instantgr2_reference grotor_evaluator -j2
```

For CUDA without ray tracing, set `GROTOR_ENABLE_RTCORE=OFF`. A100 and H100 presets
use `build/GRotor/sm_80` and `build/GRotor/sm_90`, respectively. The `cuda` and
`rtx4090` presets share `build/GRotor/sm_89`; reconfigure before switching between them.

For HIP, set `GROTOR_GPU_BACKEND=HIP`,
`CMAKE_HIP_COMPILER` and `CMAKE_HIP_ARCHITECTURES` instead of CUDA variables.
`HIPRT_ROOT` and optional `HIPRT_LIBRARY` select matching HIPRT headers/library.

CUDA is discovered from `PATH`. To select a non-default compiler, set `CUDACXX`
before the initial configuration or pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
`OPTIX_ROOT_DIR`, `NVSHMEM_ROOT` and `NCCL_ROOT` can be passed with `-D`.
The corresponding environment hints are `OPTIX_ROOT_DIR`, `NVSHMEM_HOME` and
`NCCL_HOME`. Only the selected backend's dependencies are discovered.

The `grotor_evaluator` target writes `resources/evaluator`.
Generated executables are not tracked by Git.

## Cost and host-memory options

| CMake option | Default | Purpose |
|---|---|---|
| `GROTOR_HOST_MEMORY_SAVING` | `OFF` | Release unused host data early and limit preparation overlap |
| `INSTANTGR2_HOST_MEMORY_SAVING` | `OFF` | Equivalent host-memory policy for InstantGR2 |
| `GROTOR_FULL_COST_REFERENCE` | `OFF` | Full cost/prefix updates for validation |

For B13/B14 comparisons, enable host-memory saving for both routers and rebuild:

```bash
cmake --preset rtxpro6000 \
  -DGROTOR_HOST_MEMORY_SAVING=ON \
  -DINSTANTGR2_HOST_MEMORY_SAVING=ON
cmake --build --preset rtxpro6000
```

These are build-time settings, not routing command-line arguments. CMake retains
cached options; set an option explicitly to `OFF` or use a fresh build directory
to disable a previously selected policy.

## Specialized targets

All experiment switches below default to `OFF`, unless indicated otherwise. Set them
with `-D<OPTION>=ON`, then build the named target.

| Option | Targets / behavior |
|---|---|
| `GROTOR_BUILD_INSTANTGR2` | `instantgr2_reference`; CUDA default `ON`, multi presets `OFF`; unavailable on HIP |
| `GROTOR_BUILD_CUGR2_ROUTER` | `grotor_cugr2`, GRotor's CUGR2-style pipeline |
| `GROTOR_BUILD_GGR_ROUTER` | `grotor_ggr`, GRotor's Z-shape pipeline |
| `GROTOR_BUILD_ROUTING_ABLATION` | CUDA: four `grotor_ablation_*` variants, requires RT-Core; HIP: `grotor_amd_original` (manual HIP default `ON`, basic HIP presets `OFF`) |
| `GROTOR_BUILD_DAG_THROUGHPUT_BENCHMARK` | `grotor_dag_throughput` |
| `GROTOR_BUILD_PREFIX_ACCESS_BENCHMARK` | `grotor_prefix_access_benchmark`  |
| `GROTOR_BUILD_RTCORE_RAY_BENCHMARK` | `grotor_rtcore_ray_benchmark`, requires RT-Core |
| `GROTOR_MULTI_GRID` | Spatial distributed grid; multi-GPU default `ON` |
| `GROTOR_MULTI_LEGACY_LSHAPE` | Original NVSHMEM L-shape executor; multi-GPU default `ON`, requires spatial grid, NCCL and NVSHMEM device library |
| `GROTOR_SERIAL_DAG` | HIP validation switch: thread L-shape / global-depth FLT scheduling |

For an adapter's basic run, use its executable directly with the same
`-cap`, `-net`, `-out`, `--batching` arguments as `GRotor`.
Native CUGR2/GGR baseline scripts use their own input formats.

