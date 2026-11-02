#!/usr/bin/env python3
"""Export recorded geometry trials without changing their native durations."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True, help="Completed geometry-sweep summary.json files")
    parser.add_argument("--baseline", default="results/GRotor-rtcore-contiguous/summary.json")
    parser.add_argument("--output-dir", default="results/GRotor-rtcore-geometry")
    args = parser.parse_args()
    local = lambda value: (ROOT / value).resolve()
    baseline_path = local(args.baseline)
    baseline = json.loads(baseline_path.read_text())
    if baseline["status"] != "passed":raise RuntimeError("Incomplete baseline")
    controls = {c["name"]: c for c in baseline["cases"] if c["status"] == "passed"}
    rows, provenance = [], {}
    for name in args.sessions:
        path = local(name)
        session = json.loads(path.read_text())
        if session["status"] != "passed":raise RuntimeError("Incomplete sweep: " + str(path))
        provenance[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for run in session["runs"]:
            dataset = run["dataset"]
            if (run["status"] != "passed" or not run["matches_instantgr2"]
                    or run["metrics"] != controls[dataset]["baseline"]["metrics"]
                    or session["oracles"][dataset]["inputs"] != controls[dataset]["inputs"]):
                raise RuntimeError("Sweep quality or input mismatch")
            cpu_cuda = {t["backend"]: t["elapsed_ms"] for t in controls[dataset]["baseline"]["batch_timings"] if t["stage"] == "detour"}
            log = path.parent / dataset / f"run{run['round']}.routing.log"
            candidates = []
            for record in re.findall(r"^GEOMETRY_TRIAL (.+)$", log.read_text(), re.MULTILINE):
                fields = dict(token.split("=", 1) for token in record.split())
                elapsed = float(fields["elapsed_ms"])
                if (fields["exact_lists_equal"] != "1" or not math.isfinite(elapsed) or elapsed <= 0
                        or abs(elapsed - (int(fields["end_us"]) - int(fields["start_us"])) / 1000) > .001):
                    raise RuntimeError("Invalid native geometry timing or batch list")
                row = dict(dataset=dataset, suite=path.parent.name, candidate=fields["name"],
                           z_gap=float(fields["z_gap"]), margin=float(fields["margin"]),
                           **{key:float(fields.get(key, default)) for key, default in
                              (("x_scale", 1), ("fast_trace", 0), ("single_anyhit", 0),
                               ("compact_points", 0), ("sorted_rays", 0), ("coplanar", 0))},
                           elapsed_ms=elapsed, start_us=int(fields["start_us"]), end_us=int(fields["end_us"]),
                           batches=int(fields["batches"]), exact_cpu_lists=True,
                           cpu_ms=cpu_cuda["cpu"], cuda_core_ms=cpu_cuda["cuda_core"],
                           speedup_vs_cpu=cpu_cuda["cpu"] / elapsed,
                           speedup_vs_cuda_core=cpu_cuda["cuda_core"] / elapsed,
                           source_log=str(log.relative_to(ROOT)))
                candidates.append(row)
            paired = [r["elapsed_ms"] for r in candidates if r["candidate"] in ("control", "control-end")]
            if len(paired) != 2:raise RuntimeError("Missing paired controls")
            for row in candidates:
                row.update(control_ms=statistics.median(paired), speedup_vs_control=statistics.median(paired) / row["elapsed_ms"])
            rows.extend(candidates)
    folder = local(args.output_dir)
    if ROOT / "results" not in folder.parents:raise RuntimeError("Output must be below results/")
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "geometry_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines = ["# Standard RT-Core RT-Core Geometry Trials", "",
             "All durations are native contiguous coloring-loop wall intervals: end minus start, after GPU completion. No per-round or component sums are used.", "",
             "Geometry/BVH/pipeline preparation and optional ray sorting precede the measured interval. Build strategies and sorting may add preparation work. CPU/CUDA controls retain complete-function timing, so ratios describe these scopes, not whole-router speedups.", "",
             "Every candidate's complete batch lists match CPU. Each sweep evaluates one actual routing output per design, then deletes it. Candidate sweeps do not generate separate routing files for each geometry.", "",
             "CPU/CUDA controls are reused from the hash-referenced baseline. Each sweep brackets candidates with original controls; control normalization uses the median of those two durations. Trials are single runs with unlocked GPU clocks.", "",
             "| Dataset | Suite | Candidate | RT ms | CPU / RT | CUDA / RT | Control / RT |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['suite']} | {r['candidate']} | {r['elapsed_ms']:.3f} | {r['speedup_vs_cpu']:.2f}x | {r['speedup_vs_cuda_core']:.2f}x | {r['speedup_vs_control']:.2f}x |")
    lines += ["", "## Provenance", "", f"Baseline SHA-256: `{hashlib.sha256(baseline_path.read_bytes()).hexdigest()}`.", ""]
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in provenance.items())
    (folder / "trials.md").write_text("\n".join(lines) + "\n")
    print(folder / "trials.md")


if __name__ == "__main__":main()
