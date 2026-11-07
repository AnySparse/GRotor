# Baselines

## InstantGR2

`InstantGR2/` contains the TCAD version of InstantGR, based on upstream commit
`476570c491ca9d6871fa0180eb827a0e9932b3b7` (`tcad version`). The pinned revision
is also recorded in `InstantGR2/UPSTREAM_REVISION` for the experiment scripts.
The source was recovered from a local Git copy of the upstream repository.
It is tracked directly by GRotor; no submodule checkout is required.

This copy includes the following local changes:

- Deterministic congestion reduction: one thread reduces all layers for each
  grid cell, replacing concurrent writes to the same congestion cell.
- CPU, GPU and dual batching modes, plus an optional sparsity override. Dual
  mode measures both implementations and uses the CPU plan for routing.
- Synchronized batching and routing-DAG timing. L-shape measures its batch loop; FLT sums the GPU suffix time of each batch.
- Release of temporary GPU batching buffers and removal of a non-English
  comment.

The original project documentation is retained in
[InstantGR2/README.md](InstantGR2/README.md). See the
[main README](../README.md) for building and running the comparison scripts.
The evaluator is built from `InstantGR2/run/evaluator.cpp`; the bundled
executable is excluded from version control.

### Publication and License

Liang Xiao, Shiju Lin, Jinwei Liu, Qinkai Duan, Tsung-Yi Ho, and Evangeline
Young, ["InstantGR: Scalable GPU Parallelization for 3-D Global Routing"](https://ieeexplore.ieee.org/document/11015529),
IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems
(TCAD), 2025.

Copyright (c) 2024, CUHK EDA. Distributed under the
[BSD 3-Clause license](InstantGR2/LICENSE). FLUTE and robin_hood retain their
own notices in the source headers. See [NOTICE.md](../NOTICE.md) for GRotor's
third-party notices.

## CUGR2

`cu-gr-2/` vendors CUGR 2.0 from upstream commit
`76297e58225d6398825472921d6c3600cc448ae9`, recorded in
`cu-gr-2/UPSTREAM_REVISION`. Source and required Rsyn/FLUTE dependencies are
included; optional detailed-router executables and evaluation suites are
omitted. No nested Git repository or network checkout is required.

Local instrumentation measures solver plus route commit time and reports
the number of nontrivial L-shape DAGs, detour nets and sparse-maze nets.
Full stage wall times remain available as diagnostics. Progress is printed
after each stage so a timeout cannot turn an unfinished maze into a zero-time
result. Routing algorithms and default solver parameters are retained.
The per-net CPU solver loop remains sequential, as in upstream; the thread
setting is passed through unchanged.

Build the CPU baseline from the GRotor root:

```bash
cmake -S baseline/cu-gr-2/src -B build/CUGR2 -DCMAKE_BUILD_TYPE=Release
cmake --build build/CUGR2 --target route --parallel 2
```

CUGR2 requires Boost system, filesystem and program_options development
libraries. Upstream Linux x86-64 LEF/DEF parser static libraries are retained
under `cu-gr-2/rsyn/lib/linux/`; these are input dependencies, not local build
outputs. Run the router with `cu-gr-2/flute/` as its working directory so its
lookup tables resolve. The comparison script handles this automatically.

For the GRotor CUGR2 adapter, build and run commands are documented in
the main README's Table 3 section. L-shape uses warp scheduling; detour and
maze layer assignment use block scheduling. All use incremental cost/BIT
updates. All three comparison times include measured RT-Core batching plus
the complete GPU batch-execution/update interval. Maze topology search uses
eight CPU workers. Worsening candidates restore a bounded checkpoint of the
previous routes and demand.
Inputs and intermediate topologies differ
from the CPU baseline; compare stage task scales rather than output equality.

Publication: Jinwei Liu and Evangeline F. Y. Young, *EDGE: Efficient DAG-Based
Global Routing Engine*, DAC 2023. See [cu-gr-2/README.md](cu-gr-2/README.md) for
upstream information and [cu-gr-2/LICENSE](cu-gr-2/LICENSE) for its CU-SD license.

## GGR

`GGR/` contains the Z-shape router integrated in XPlace and the parser/build
requirements needed to run it on LEF/DEF inputs. The local snapshot revision,
retained licenses, inherited instrumentation and build instructions are
recorded in [GGR/README.md](GGR/README.md). No placement application or nested
Git checkout is included. Run `python3 script/compare_ggr.py --case mempool_tile`
from the repository root to build and compare its Z-shape stage with GRotor.
Maze routing is disabled in this experiment.
