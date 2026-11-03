# License Notices

## InstantGR

`baseline/InstantGR2` contains InstantGR source code based on upstream commit
`476570c491ca9d6871fa0180eb827a0e9932b3b7` (`tcad version`). It retains the
BSD 3-Clause license and Copyright (c) 2024, CUHK EDA notice in
[baseline/InstantGR2/LICENSE](baseline/InstantGR2/LICENSE).

Local modifications include deterministic congestion reduction, CPU/GPU/dual
batching selection, a sparsity override, routing runtime instrumentation, and
cleanup of temporary GPU batching allocations. See
[baseline/README.md](baseline/README.md) for baseline details and publication
references.

Portions of GRotor's `src/` and `include/router_state.hpp` are also derived
from InstantGR and retain the BSD 3-Clause license and copyright notice in
[third_party/LICENSE](third_party/LICENSE).

## Other Dependencies

FLUTE and robin_hood retain their copyright and license notices in
`third_party/flute.hpp` and `third_party/robin_hood.h`, including the copies
under `baseline/InstantGR2/src/`.

## CUGR 2.0

`baseline/cu-gr-2` contains CUGR 2.0 source code based on upstream commit
`76297e58225d6398825472921d6c3600cc448ae9` from
<https://github.com/cuhk-eda/cu-gr-2>. Copyright (c) 2023 by The Chinese
University of Hong Kong. Its CU-SD license is retained in
[baseline/cu-gr-2/LICENSE](baseline/cu-gr-2/LICENSE) and the upstream README.

Local changes add solver/commit timing, stage task counts, incremental
progress records, a maze-skip option, and build compatibility updates.
GRotor's sparse maze adaptation uses a direction-collapsed cost view and CPU
sparse topology search followed by GPU DAG layer assignment.

The bundled Rsyn dependency retains its Apache 2.0 license in
[baseline/cu-gr-2/rsyn/LICENSE](baseline/cu-gr-2/rsyn/LICENSE); its third-party
headers retain their own notices. The bundled FLUTE license is retained in
[baseline/cu-gr-2/flute/license.txt](baseline/cu-gr-2/flute/license.txt).

## GGR / XPlace

`baseline/GGR` contains GGR and its required XPlace parser modules from local
snapshot `1f057cbd1deb7328e77bd377c555be07620bddad`. The snapshot retains the
Copyright (c) 2022 BSD 3-Clause notice in
[baseline/GGR/LICENSE](baseline/GGR/LICENSE). This local revision is recorded
for provenance; it is not claimed as a fetched upstream release.

GRotor's Z-shape recurrence and adapter routines in `src/routing/zshape.cuh`
and `src/routing/zshape_topology.hpp`, and the test oracle in
`script/tests/zshape_reference.cuh`, are derived from GGR and the earlier
local routing adapter. They remain subject to that BSD 3-Clause notice.
Local baseline modifications and the relevant publications are documented in
[baseline/GGR/README.md](baseline/GGR/README.md) and its retained GGR README.

The vendored FLUTE, pybind11, LEF/DEF and SPEF parser dependencies retain their
license files and copyright notices under `baseline/GGR/thirdparty` and
`baseline/GGR/cpp_to_py/common/lib`.
