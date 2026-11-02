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
