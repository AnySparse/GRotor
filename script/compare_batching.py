#!/usr/bin/env python3
"""Compare recorded CPU/CUDA-Core batching with validated RT-Core timings."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from test_instantgr2 import ROOT, BASELINE, sha256
from test_grotor_reference import local_path


def restore_stage_labels(relative, text):
    """Reverse only the historical batching-log label removal, without editing sources."""
    calls = {"src/Lshape_route.hpp": (1000000, "lshape"),
             "src/Lshape_route_detour.hpp": (300000, "detour")}
    if relative in calls:
        limit, stage = calls[relative]
        return text.replace(f"generate_batches_rsmt_selected(nets2route, {limit})",
                            f'generate_batches_rsmt_selected(nets2route, {limit}, "{stage}")')
    if relative == "src/database_cuda.hpp":
        return (text.replace(
            "vector<vector<int>> generate_batches_rsmt_selected(vector<int> &nets2route, int max_batch_size) {",
            "vector<vector<int>> generate_batches_rsmt_selected(vector<int> &nets2route, int max_batch_size,\n"
            "                                                  const char *stage) {")
            .replace("BATCH_TIMING backend=%s", "BATCH_TIMING stage=%s backend=%s")
            .replace('               cpu ? "cpu" : "cuda_core", static_cast<long long>(start_us),',
                     '               stage, cpu ? "cpu" : "cuda_core", static_cast<long long>(start_us),'))
    return None


def compatible_source(relative, historical_hash, recorded_hash):
    if historical_hash == recorded_hash:
        return True
    path = BASELINE / relative
    if not path.is_file() or sha256(path) != recorded_hash:
        return False
    restored = restore_stage_labels(relative, path.read_text())
    return restored is not None and hashlib.sha256(restored.encode()).hexdigest() == historical_hash


def compare(instant, rt_results):
    if (instant["status"] != "passed" or instant["mode"] != "dual" or instant["sparsity"] != 1
            or instant["batching_backend"] != "cpu" or not instant["source_sha256"]):
        raise ValueError("InstantGR2 must be a passed sparsity-1 dual run using CPU routing")
    records = {}
    native_hashes = None
    timing_mode = None
    for result in rt_results:
        config = result["configuration"]
        if (result["status"] != "passed" or config["sparsity"] != 1
                or config["cuda_arch"] != instant["cuda_arch"] or config["gpu"] != instant["gpu"]
                or not config["validate_batches"]):
            raise ValueError("RT-Core results must pass CPU-plan validation with matching sparsity/GPU/architecture")
        if any(not compatible_source(k, v, config["sources"].get("baseline/InstantGR2/" + k))
               for k, v in instant["source_sha256"].items()):
            raise ValueError("InstantGR2 source differences extend beyond verified batching-log label removal")
        mode = config.get("timing", "region")
        if timing_mode is not None and mode != timing_mode:
            raise ValueError("Do not combine different RT-Core timing scopes")
        timing_mode = mode
        hashes = dict(result["binary_sha256"])
        hashes["executed_router"] = result.get("executed_router", {}).get("sha256", hashes["GRotor"])
        if native_hashes is not None and hashes != native_hashes:
            raise ValueError("RT-Core sessions used different binaries or PTX")
        native_hashes = hashes
        for run in result["runs"]:
            if run["status"] != "passed" or not run["matches_instantgr2"]:
                raise ValueError("RT-Core comparison contains a failed or mismatching run")
            records.setdefault(run["dataset"], []).append((run, result["oracles"][run["dataset"]]))

    rows = []
    for case in instant["cases"]:
        dataset = case["name"]
        if case["status"] != "passed" or dataset not in records:
            raise ValueError("Missing passed measurements for " + dataset)
        if any(r["status"] != "passed" for r in case["runs"]):
            raise ValueError("Incomplete InstantGR2 runs for " + dataset)
        expected = case["runs"][0]["metrics"]
        if any(r["metrics"] != expected for r in case["runs"]):
            raise ValueError("InstantGR2 evaluator results differ across runs")
        for run, oracle in records[dataset]:
            if oracle["inputs"] != case["inputs"] or run["metrics"] != expected:
                raise ValueError("Inputs or evaluator costs differ for " + dataset)
        for stages, algorithm in ((("lshape",), "L-shape"), (("detour",), "FLT"),
                                  (("lshape", "detour"), "Both")):
            samples = {"cpu": [], "cuda_core": [], "rt_core": []}
            for run in case["runs"]:
                for backend in ("cpu", "cuda_core"):
                    timings = [t for t in run["batch_timings"] if t["stage"] in stages and t["backend"] == backend]
                    if (len(timings) != len(stages) or {t["stage"] for t in timings} != set(stages)
                            or any(t["elapsed_ms"] <= 0 for t in timings)):
                        raise ValueError("Missing, duplicate or invalid InstantGR2 timing")
                    samples[backend].append(sum(t["elapsed_ms"] for t in timings))
            for run, _ in records[dataset]:
                timings = [t for t in run["batch_timings"] if t["stage"] in stages and t["backend"] == "rt_core"]
                if (len(timings) != len(stages) or {t["stage"] for t in timings} != set(stages)
                        or any(t["elapsed_ms"] <= 0 for t in timings)):
                    raise ValueError("Missing, duplicate or invalid RT-Core timing")
                # The CPU/RT-Core plans must describe the same amount of work.
                for timing in timings:
                    plans = [t for r in case["runs"] for t in r["batch_timings"]
                             if t["backend"] == "cpu" and t["stage"] == timing["stage"]]
                    if any((p["nets"], p["batches"]) != (timing["nets"], timing["batches"]) for p in plans):
                        raise ValueError("CPU and RT-Core work counts differ")
                samples["rt_core"].append(sum(t["elapsed_ms"] for t in timings))
            medians = {k: statistics.median(v) for k, v in samples.items()}
            rows.append({"dataset": dataset, "algorithm": algorithm,
                         "cpu_runs": len(samples["cpu"]), "cuda_core_runs": len(samples["cuda_core"]),
                         "rtcore_runs": len(samples["rt_core"]), "cpu_ms": medians["cpu"],
                         "cuda_core_ms": medians["cuda_core"], "rtcore_ms": medians["rt_core"],
                         "speedup_vs_cpu": medians["cpu"] / medians["rt_core"],
                         "speedup_vs_cuda_core": medians["cuda_core"] / medians["rt_core"],
                         "samples_ms": samples})
    return rows


def write_report(folder, rows, inputs, source_differences, timing_mode):
    folder.mkdir(parents=True, exist_ok=True)
    result = {"aggregation": "median of completed run times; Both sums stages within each run before taking the median",
              "ratio": "comparison backend milliseconds / RT-Core milliseconds",
              "input_summaries": {str(p): sha256(p) for p in inputs}, "rows": rows,
              "comparison_script_sha256": sha256(Path(__file__)),
              "rtcore_timing_mode": timing_mode,
              "verified_stage_label_only_source_differences": source_differences}
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (folder / "speedup.csv").open("w", newline="") as stream:
        fields = [k for k in rows[0] if k != "samples_ms"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    lines = ["# Batching Runtime Comparison", "",
             "- All datasets use sparsity 1, matching CUDA architectures/GPU indices and identical inputs.",
             "- Baseline sources are identical except verified removal of batching stage-label arguments/prints: reconstructing the labels reproduces the historical source hashes exactly. Algorithms and timing scopes are unchanged.",
             "- All four evaluator costs match; RT-Core batch lists were checked against CPU in each stage.",
             "- CPU and CUDA-Core times come from recorded InstantGR2 dual runs; only their CPU plans feed routing.",
             "- RT-Core times come from GRotor; its RT-Core plans feed routing.",
             "- Values are medians. Speedup = CPU or CUDA-Core time / RT-Core time; above 1 is faster, below 1 is slower.",
             ("- RT-Core scopes depend on method: queued GPU intervals for rounds, BVH build plus coloring event for cached. These are not equal-scope end-to-end comparisons."
              if timing_mode == "method" else
              "- RT-Core uses one contiguous wall interval per method: standard coloring loop, cached pipeline preparation through coloring completion. CPU control/transfers/waits inside the interval are included."
              if timing_mode == "contiguous_wall" else
              "- RT-Core includes the complete batching function: preparation, OptiX/GAS/compilation, coloring, readback and resource cleanup."
              if timing_mode == "full" else
              "- RT-Core sums queued GPU round event intervals, excluding host enqueue/readback gaps but including GPU scheduling gaps and OptiX internal work. This is not a profiler sum of individual kernels."
              if timing_mode == "gpu" else
              "- RT-Core measures the complete coloring event region, including CPU round-control gaps."),
             ("- RT-Core preparation and cleanup are included in primary time." if timing_mode == "full" else "- Geometry/BVH preparation and final batch download/cleanup are excluded. Cached pipeline preparation is included." if timing_mode == "contiguous_wall" else "- RT-Core initialization, geometry/ray preparation, GAS building, compilation, final readback and cleanup are excluded."),
             "- CPU/CUDA-Core use existing synchronized function-wall timings, including their preparation and cleanup.",
             ("- These ratios compare complete batching functions; routing DAG solving and the rest of the router are outside this comparison." if timing_mode == "full" else "- These ratios describe the requested reported scopes, not equivalent whole-function or end-to-end speedups."),
             "- CUDA-Core batching is nondeterministic and can produce different batch counts from CPU/RT-Core.",
             "- These are separate recorded sessions, not paired measurements with locked GPU clocks; small differences are indicative.", ""]
    for algorithm in ("L-shape", "FLT", "Both"):
        lines += ["## " + ("Combined Batching" if algorithm == "Both" else algorithm), "",
                  "| Dataset | CPU ms | CUDA-Core ms | RT-Core ms | CPU / RT | CUDA / RT | Runs CPU/CUDA/RT |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        for row in rows:
            if row["algorithm"] == algorithm:
                lines.append(f"| {row['dataset']} | {row['cpu_ms']:.3f} | {row['cuda_core_ms']:.3f} | "
                             f"{row['rtcore_ms']:.3f} | {row['speedup_vs_cpu']:.2f}x | {row['speedup_vs_cuda_core']:.2f}x | "
                             f"{row['cpu_runs']}/{row['cuda_core_runs']}/{row['rtcore_runs']} |")
        lines.append("")
    (folder / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instantgr2-summary", default="results/InstantGR2/summary.json")
    parser.add_argument("--rtcore-summary", nargs="+", default=["results/GRotor-rtcore/summary.json"])
    parser.add_argument("--output-dir", default="results/batching-comparison")
    args = parser.parse_args()
    baseline = local_path(args.instantgr2_summary)
    rt_paths = [local_path(p) for p in args.rtcore_summary]
    if len(set(rt_paths)) != len(rt_paths):
        parser.error("Do not include the same RT-Core summary twice")
    folder = local_path(args.output_dir)
    if ROOT / "results" not in folder.parents:
        parser.error("Output must be below this repository's results/")
    instant = json.loads(baseline.read_text())
    rt_results = [json.loads(p.read_text()) for p in rt_paths]
    rows = compare(instant, rt_results)
    differences = [{"summary": str(path), "file": k, "historical_sha256": v,
                    "rtcore_oracle_sha256": result["configuration"]["sources"].get("baseline/InstantGR2/" + k)}
                   for path, result in zip(rt_paths, rt_results) for k, v in instant["source_sha256"].items()
                   if result["configuration"]["sources"].get("baseline/InstantGR2/" + k) != v]
    write_report(folder, rows, [baseline, *rt_paths], differences,
                 rt_results[0]["configuration"].get("timing", "region"))
    print(folder / "summary.md")


if __name__ == "__main__":
    main()
