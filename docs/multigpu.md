# Multi-GPU Routing

## Requirements

- NVLink-connected NVIDIA GPUs with native peer atomics.
- CUDA Toolkit, Open MPI, NCCL and NVSHMEM 3.x host/device libraries.
- Python 3.9+ and Matplotlib for plots (`python3 -m pip install matplotlib`).

Make `nvcc` and `mpirun` available on `PATH`. Run the following commands from
the repository root.

## Build

```bash
export NVSHMEM_HOME=/path/to/nvshmem
export NCCL_HOME=/path/to/nccl
cmake --preset multi-a100
cmake --build --preset multi-a100
```

This builds `build/GRotor-multi/sm_80/GRotor_multi` and `resources/evaluator`.
For H100, use preset `multi-h100` and architecture `90` in the run commands.
OptiX is not required for the multi-GPU build.

## Run Routing and Evaluation

```bash
python3 script/run_grotor.py --variant multi --cuda-arch 80 \
  --gpus 4 --devices 0,1,2,3 \
  --case mempool_cluster --data-dir /path/to/cap-net-files
```

The script launches one MPI rank per GPU and evaluates the generated routing
output. Results are saved under `results/run/grotor-multi/mempool_cluster/`:
`mempool_cluster.out`, `routing.log`, `evaluation.log` and `summary.json`.
Use `--output-dir <path>` to change the destination.

`--gpus` selects the GPU count; `--devices` lists visible devices and must cover
all requested ranks. Omit `--devices` to preserve `CUDA_VISIBLE_DEVICES`.
The NVSHMEM heap defaults to `25G` per GPU; adjust it with `--symmetric-size`
or `NVSHMEM_SYMMETRIC_SIZE` to fit device memory.

## Run the Scaling Experiment

```bash
python3 script/experiments/run_multigpu.py --cuda-arch 80 \
  --data-dir /path/to/cap-net-files --cases mempool_cluster \
  --devices 0,1,2,3 --counts 1 2 3 4 --repeats 2
```

Include `1` in `--counts` as the speedup and quality reference. The script
checks evaluator costs across GPU counts and writes `scaling.csv`,
`summary.md`, `summary.json`, plots and per-run logs to
`results/GRotor-multigpu/`. The CSV reports median runtime, speedup and
parallel efficiency. Use `--no-build` to reuse a matching build and
`--output-dir` to select a separate experiment directory.
