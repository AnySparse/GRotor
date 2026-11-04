# Legacy L-shape execution in GRotor

The default `GRotor_multi` L-shape executor ports the original
`GRaptor_multi/RouterSolver_P2P.cuh` execution policy. The same implementation
runs at every GPU count, including one. The ordinary `GRotor` target keeps its
separate optimized single-GPU implementation.

The port retains:

- The original warp-per-DAG kernel, stable BFS node layout, and fixed
  1056 blocks × 256 threads. A warp processes trees in a grid-stride loop and
  reuses its maximum-tree scratch storage. The ordinary GRotor executor now also parallelizes nodes by depth, but
  uses ordered sibling merges and a different scratch/commit policy. This
  legacy path retains its own executor at the one-GPU baseline.
- Full-size matrices on every PE, with disjoint X ownership and raw P2P
  pointers indexed by global coordinates, including at N=1. No compact-shard
  decoder or special one-GPU pointer path is used.
- Integer demand in doubled units: a wire adds 2 and half a via adds 1.
  Dirty wire/via flags are bool stores, not bitmap `atomicOr` operations.
- Separate DP/traceback, block-cooperative wire commit, and via commit kernels.
  Wire commits are not fused into the solver.
- Dirty wire-cost update, complete local inclusive prefix scans, then dirty
  via-cost update after each batch's commits. BIT is disabled.
- One persistent distributed grid for the entire L-shape batch loop. CPU
  topology construction/distribution, initial uploads/costs, and allocation
  precede the loop. Final NCCL route reduction and demand collection follow it.
- The original route capacity of 27 integers per pin. The final collected
  routes are used by the real output writer and, in full-pipeline runs, FLT.

Only the last executed L-shape batch is reported for scaling. Every rank uses
CUDA events around solve, commits, cost maintenance and synchronization. The
reported time is the maximum participating rank's time. Per-batch/per-rank
records remain in `LEGACY_LSHAPE_BATCH` log lines.

## Compatibility and synchronization

InstantGR2's cost weights, overflow formulas, cross-point terms, and routing
input are retained. Doubled integer demand converts exactly to float half
units when these formulas are evaluated or the final grid is collected.
Via landing costs read the current owner at an X partition boundary.
Zero-length wire queries return zero, as required by InstantGR2.

Two synchronization dependencies are made explicit: all wire/timestamp writes
complete before via kernels start, and all updated costs complete before the
next batch reads a peer's costs. These waits are **inside** the CUDA-event
interval at every GPU count. Synchronizing the unrelated NCCL stream alone
does not establish the first dependency. Small-case single/dual-GPU evaluator
checks verify identical InstantGR2 costs with these dependencies in place.

`GROTOR_LEGACY_VALIDATE_COSTS=1` enables expensive per-batch diagnostics for
cost caches, interval queries and demand fingerprints. It is rejected by the
performance runner and must be unset for reported timings.

## Run

```bash
/home/caoweichen/env/myenv/bin/python script/run-fig13.py \
  --nvcc /usr/local/cuda-12.2/bin/nvcc --cuda-arch 80 \
  --mpirun /home/caoweichen/opt/openmpi/bin/mpirun \
  --data-dir /data0/cwc/ISPD24 --cases mempool_cluster \
  --devices 3,2,1,0 --counts 1 2 3 4 --repeats 2 \
  --symmetric-size 25G \
  --output-dir results/GRotor-legacy-lshape-new
```

The runner builds in `build/GRotor-multi-legacy/sm_80`. NCCL, the NVSHMEM device
library, and CUDA relocatable device code are required, as in the old project.
Use `--no-build` for an existing matching build. `--require-idle` is optional;
omit it only when sharing relatively idle devices is authorized. Background
load must be reported when devices are shared.

The runner passes `--stop-after lshape` and stops before congestion
classification, rip-up, FLT topology construction or FLT device allocations.
All earlier L-shape batches execute; only the final batch contributes to the
scaling report. The script deliberately exposes no FLT, compact-executor or
root-BIT benchmark mode. Historical comparison tooling remains separate.

Real L-shape routes are independently evaluated for zero open/incomplete
nets and identical costs across GPU counts. The comparison reference is the
one-GPU L-shape result. Byte-identical outputs reuse the evaluator result
only after SHA-256 verification within the same experiment. A stage-only
result is not compared with an InstantGR2 full-pipeline result, which includes
FLT. Run `script/test_grotor_reference.py` for that separate quality check.

GPU utilization, process IDs, memory use and clocks are sampled every five
seconds during router execution into per-run `*.gpu-load.jsonl` files.

The detour routing entry point is `Lshape_route_detour::route_detour`. Direct router runs
without `--stop-after lshape` retain the full pipeline; the default stop point
is `routing`. Ordinary single-GPU GRotor uses its existing solver, while the
scaling runner always launches `GRotor_multi`, including at N=1.

`preparation_ms` covers preparation before the whole L-shape loop;
`finalization_ms` covers collection after it; `end_to_end_ms` covers the entire
phase, including earlier batches. These fields are not included in the final
batch scaling time. `snapshot_ms` is the historical field name for the measured
post-commit cost/prefix maintenance. `solve_ms` excludes the separate commits.
