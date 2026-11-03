# GGR Z-shape Baseline

This is the GGR router integrated in XPlace, imported from the workspace's
`baseline/Xplace-main` snapshot. The local source revision is recorded in
`UPSTREAM_REVISION`; it identifies that snapshot, not an independently fetched
upstream revision. The router, parser, FLUTE and required dependencies are
vendored directly without nested Git metadata. Placement modules are omitted.

The original GGR documentation and publication references are in
[cpp_to_py/gpugr/README.md](cpp_to_py/gpugr/README.md). GGR's native input is
LEF/DEF. This comparison sets `rrrIters=0`, running Z-shape pattern routing only.
It does not replace GGR with a CAP/NET adapter.

Local changes inherited from the earlier experiment include Z-shape timing and
task counts, a larger cost-scan limit, CUDA error checks, and Z-shape-only
workspace allocation when Maze is disabled. The baseline retains managed
backpointer storage; GRotor's implementation uses explicit device allocation.
The standalone CMake build includes only the required XPlace modules and writes
libraries to `build/GGR/modules`.

Requirements: Linux x86-64, CMake 3.22+, C++17, CUDA, a CUDA-enabled PyTorch
installation, Python development headers, Bison, Flex and zlib development
headers. The bundled LEF/DEF dependencies include their original Linux x86-64
static libraries. Tested with Python 3.10, PyTorch 2.2.0+cu121 and CUDA 12.4.

From the GRotor root:

```bash
cmake -S baseline/GGR -B build/GGR -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DPYTHON_EXECUTABLE="$(command -v python3)"
cmake --build build/GGR --target gpugr flute_cpp --parallel 2
python3 script/compare_ggr.py --case mempool_tile --optix-root "$OPTIX_ROOT_DIR"
```

Native routing time includes full wire/via cost and prefix updates plus DP and
route commits, summed across batches. It excludes CPU topology preparation and
dense DP-map initialization. Native guides use GGR's cost model; they are not
scored by the CAP/NET evaluator. See the main README for GRotor's timing scope,
evaluator validation and result files.

The BSD 3-Clause notice is retained in [LICENSE](LICENSE). Bundled dependencies
retain their respective license files and source notices. See also the
repository [NOTICE.md](../../NOTICE.md).
