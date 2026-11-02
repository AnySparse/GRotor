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
- Synchronized batching and routing-DAG timing. L-shape measures its batch
  loop; FLT sums the GPU suffix time of each batch, excluding CPU DAG packing.
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
