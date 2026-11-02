#!/usr/bin/env python3
"""Compare GRotor and InstantGR2 with alternating, paired CPU-batching runs."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import resource
import shutil
import statistics
import subprocess
import tempfile

from test_instantgr2 import (ROOT, BASELINE, evaluator_metrics, parse_batch_timings,
                            parse_routing_timings, require_cuda, run_logged, sha256)
from test_grotor_reference import source_manifest, validate_luts

POLICY = "paired-cpu-batching-performance-v1"
IMPLEMENTATIONS = ("InstantGR2", "GRotor")


def measured_runs(result, dataset, implementation):
    return [r for r in result["runs"] if r["dataset"] == dataset
            and r["implementation"] == implementation and not r["warmup"] and r["status"] == "passed"]


def samples(runs, algorithm, component):
    if algorithm == "Router":
        return [r[component] for r in runs]
    return [a[component] for r in runs for a in r["algorithms"] if a["algorithm"] == algorithm]


def comparison_rows(result):
    rows = []
    for dataset in result["configuration"]["cases"]:
        for algorithm, component in [("Router", "wall_ms"), ("Router", "other_wall_ms"),
                                     ("Router", "cpu_user_ms"), ("Router", "cpu_system_ms")]:
            rows.append((dataset, algorithm, component))
        for algorithm in ("L-shape", "FLT"):
            rows.extend((dataset, algorithm, c) for c in ("cpu_batch_ms", "dag_solve_ms", "cpu_path_ms"))
    output = []
    for dataset, algorithm, component in rows:
        groups = {name: measured_runs(result, dataset, name) for name in IMPLEMENTATIONS}
        values = {name: samples(runs, algorithm, component) for name, runs in groups.items()}
        if not all(values.values()):
            continue
        row = {"dataset": dataset, "algorithm": algorithm, "component": component}
        for name, prefix in (("InstantGR2", "instantgr2"), ("GRotor", "grotor")):
            row.update({prefix + "_runs": len(values[name]), prefix + "_median_ms": statistics.median(values[name]),
                        prefix + "_min_ms": min(values[name]), prefix + "_max_ms": max(values[name])})
        base, candidate = row["instantgr2_median_ms"], row["grotor_median_ms"]
        row["change_pct"] = 100 * (candidate / base - 1) if base else None
        paired = []
        for run in groups["InstantGR2"]:
            other = next((r for r in groups["GRotor"] if r["round"] == run["round"]), None)
            if other:
                a, b = samples([run], algorithm, component)[0], samples([other], algorithm, component)[0]
                if a:
                    paired.append(100 * (b / a - 1))
        row["paired_median_change_pct"] = statistics.median(paired) if paired else None
        output.append(row)
    return output


def write_summary(folder, result):
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    comparisons = comparison_rows(result)
    with (folder / "comparison.csv").open("w", newline="") as stream:
        fields = ["dataset", "algorithm", "component", "instantgr2_runs", "grotor_runs",
                  "instantgr2_median_ms", "grotor_median_ms", "change_pct", "paired_median_change_pct",
                  "instantgr2_min_ms", "instantgr2_max_ms", "grotor_min_ms", "grotor_max_ms"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(comparisons)
    with (folder / "runtime.csv").open("w", newline="") as stream:
        fields = ["dataset", "run", "implementation", "order", "algorithm", "nets", "batches",
                  "cpu_batch_ms", "dag_solve_ms", "cpu_path_ms", "router_wall_ms", "other_wall_ms",
                  "cpu_user_ms", "cpu_system_ms"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for run in result["runs"]:
            if run["warmup"] or run["status"] != "passed":
                continue
            for algorithm in run["algorithms"]:
                writer.writerow({"dataset": run["dataset"], "run": run["round"],
                                 "implementation": run["implementation"], "order": run["order"],
                                 **algorithm, "router_wall_ms": run["wall_ms"],
                                 **{k: run[k] for k in ("other_wall_ms", "cpu_user_ms", "cpu_system_ms")}})
    with (folder / "quality.csv").open("w", newline="") as stream:
        fields = ["dataset", "run", "implementation", "warmup", "matches", "wirelength", "via",
                  "overflow", "total", "open_nets", "incomplete_nets"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for run in result["runs"]:
            if run["status"] == "passed":
                writer.writerow({"dataset": run["dataset"], "run": run["round"],
                                 "implementation": run["implementation"], "warmup": run["warmup"],
                                 "matches": run["metrics_match"], **run["metrics"]})
    lines = ["# GRotor Performance Comparison", "", f"- Status: {result['status']}",
             "- Both programs use CPU batching, sparsity 1, the same inputs/LUTs, and matching compiler flags.",
             "- InstantGR2 is the repository baseline with the deterministic congestion repair; its routing algorithms and managed storage are retained.",
             f"- Measured pairs per dataset: {result['configuration']['repeats']}; first implementation alternates by round and dataset.",
             "- One warm-up run per implementation is excluded. Values below are medians; raw values and ranges are in the CSV files.",
             "- Positive change means GRotor takes longer; negative change means GRotor takes less time.",
             "- L-shape DAG scope: complete batch loop, including incremental cost/BIT updates and demand commits.",
             "- GRotor FLT DAG scope: one continuous interval including topology upload, GPU assembly, updates and solving; preceding CPU topology construction/planning and allocation are excluded. InstantGR2 retains its per-batch GPU suffix sum.",
             "- CPU path = CPU batching + scoped DAG. This is a CPU-batching comparison, not the CUDA-batching algorithm total.",
             "- Router wall time covers the whole subprocess, including CUDA setup, input, packing, output and process teardown. Evaluator time is excluded.",
             "- Other wall time = router wall minus the two CPU-batching times and two DAG times. It includes CPU, I/O and unscoped GPU work; it is not a pure CPU measurement.",
             "- GPU clocks are not locked. Runs are sequential; no profiler is active during measured runs.",
             "- Every actual .out is evaluated and immediately deleted. No routing output is retained.", "",
             "| Dataset | Measurement | InstantGR2 ms | GRotor ms | Change |",
             "|---|---|---:|---:|---:|"]
    for row in comparisons:
        algorithm, component = row["algorithm"], row["component"]
        if component not in ("wall_ms", "other_wall_ms", "dag_solve_ms", "cpu_path_ms"):
            continue
        label = ("Router wall" if component == "wall_ms" else "Other wall" if component == "other_wall_ms"
                 else algorithm + (" DAG" if component == "dag_solve_ms" else " CPU path"))
        lines.append(f"| {row['dataset']} | {label} | {row['instantgr2_median_ms']:.3f} | "
                     f"{row['grotor_median_ms']:.3f} | {row['change_pct']:+.2f}% |")
    lines += ["", "## Evaluator Costs", "",
              "| Dataset | Wirelength | Via | Overflow | Total | Matches |", "|---|---:|---:|---:|---:|---|"]
    for dataset, expected in result["metrics"].items():
        runs = [r for r in result["runs"] if r["dataset"] == dataset and r["status"] == "passed"]
        lines.append("| " + dataset + " | " + " | ".join(expected[k] for k in ("wirelength", "via", "overflow", "total"))
                     + " | " + str(all(r["metrics_match"] for r in runs)) + " |")
    if result.get("error"):
        lines += ["", "## Error", "", result["error"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def run_one(args, result, folder, build, output, dataset, implementation, round_id, order, warmup):
    run = {"dataset": dataset, "implementation": implementation, "round": round_id,
           "order": order, "warmup": warmup, "status": "running"}
    result["runs"].append(run)
    write_summary(folder, result)
    label = "warmup" if warmup else f"run{round_id}"
    logs = folder / dataset
    logs.mkdir(exist_ok=True)
    prefix = logs / f"{label}.{implementation.lower()}"
    inputs = [args.data_dir / f"{dataset}.{suffix}" for suffix in ("cap", "net")]
    try:
        binary = build / ("GRotor" if implementation == "GRotor" else "instantgr2_reference")
        command = [str(binary), "-cap", str(inputs[0]), "-net", str(inputs[1]), "-out", str(output), "--sparsity", "1"]
        if implementation == "InstantGR2":
            command += ["--batching", "cpu"]
        run["router_command"] = command
        print(f"{dataset} {label}: {implementation} ({order}/2)", flush=True)
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        status, seconds = run_logged(command, Path(str(prefix) + ".routing.log"), cwd=BASELINE / "run" if implementation == "InstantGR2" else ROOT)
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        run.update({"router_exit_status": status, "wall_ms": seconds * 1000,
                    "cpu_user_ms": (usage_after.ru_utime - usage_before.ru_utime) * 1000,
                    "cpu_system_ms": (usage_after.ru_stime - usage_before.ru_stime) * 1000})
        if status or not output.is_file() or not output.stat().st_size:
            raise RuntimeError(f"Router failed: {prefix}")
        text = Path(str(prefix) + ".routing.log").read_text()
        if re.findall(r"^BATCH_SPARSITY backend=(\w+) mode=\d+ value=(\d+)$", text, re.MULTILINE) != [("cpu", "1")] * 2:
            raise RuntimeError("Both stages must use CPU batching with sparsity 1")
        batches = parse_batch_timings(text, "determinism", "cpu")
        run["batch_timings"] = batches
        run["dag_timings"], algorithms = parse_routing_timings(text, batches)
        run["algorithms"] = [{k: a[k] for k in ("algorithm", "nets", "batches", "cpu_batch_ms", "dag_solve_ms")}
                             | {"cpu_path_ms": a["cpu_batch_ms"] + a["dag_solve_ms"]} for a in algorithms]
        run["other_wall_ms"] = run["wall_ms"] - sum(a["cpu_path_ms"] for a in run["algorithms"])
        plan = [{k: a[k] for k in ("algorithm", "nets", "batches")} for a in run["algorithms"]]
        if dataset in result["plans"] and result["plans"][dataset] != plan:
            raise RuntimeError("Different routing workload: net or batch counts differ")
        result["plans"][dataset] = plan
        command = [str(build / "grotor_evaluator"), *map(str, inputs), str(output)]
        run["evaluator_command"] = command
        status, seconds = run_logged(command, Path(str(prefix) + ".evaluator.log"))
        run["evaluator_exit_status"], run["evaluator_seconds"] = status, seconds
        run["metrics"] = evaluator_metrics(Path(str(prefix) + ".evaluator.log"), status)
        expected = result["metrics"].setdefault(dataset, run["metrics"])
        run["metrics_match"] = run["metrics"] == expected
        if not run["metrics_match"]:
            raise RuntimeError(f"Evaluator score mismatch: {dataset} {implementation}")
        run["status"] = "passed"
        print(f"PASS {dataset} {label} {implementation}: wall={run['wall_ms']:.3f} ms; total={run['metrics']['total']}", flush=True)
    except BaseException as error:
        run["status"], run["error"] = "failed", str(error)
        raise
    finally:
        output.unlink(missing_ok=True)
        write_summary(folder, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["mempool_tile", "nvdla", "bsg_chip", "mempool_group"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-dir", default="../ISPD24_tmp")
    parser.add_argument("--output-dir", default="results/GRotor-performance")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 3 or not args.cuda_arch.isdigit() or len(set(args.cases)) != len(args.cases) or not args.cases:
        parser.error("Use at least three pairs, a numeric CUDA architecture, and unique datasets")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in args.cases):
        parser.error("Invalid dataset name")
    args.data_dir = (ROOT / args.data_dir).resolve()
    folder = (ROOT / args.output_dir).resolve()
    if ROOT / "results" not in folder.parents:
        parser.error("Results must be below this repository's results/ directory")
    inputs = {str(args.data_dir / f"{n}.{ext}"): sha256(args.data_dir / f"{n}.{ext}")
              for n in args.cases for ext in ("cap", "net")}
    validate_luts()
    build = ROOT / "build/GRotor" / f"sm_{args.cuda_arch}"
    build.mkdir(parents=True, exist_ok=True)
    sources = source_manifest()
    sources[str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    config = {"policy": POLICY, "cases": args.cases, "repeats": args.repeats,
              "cuda_arch": args.cuda_arch, "gpu": args.gpu, "inputs": inputs, "sources": sources}
    names = ("GRotor", "instantgr2_reference", "grotor_evaluator")
    if args.resume:
        result = json.loads((folder / "summary.json").read_text())
        if result["configuration"] != config or any(sha256(build / n) != result["binary_sha256"][n] for n in names):
            parser.error("Cannot resume with changed sources, inputs, configuration or binaries")
        if any(r.get("metrics_match") is False for r in result["runs"]):
            parser.error("A quality mismatch cannot be discarded")
        for run in result["runs"]:
            if run["status"] != "passed":
                label = "warmup" if run["warmup"] else f"run{run['round']}"
                for suffix in ("routing", "evaluator"):
                    (folder / run["dataset"] / f"{label}.{run['implementation'].lower()}.{suffix}.log").unlink(missing_ok=True)
        result["runs"] = [r for r in result["runs"] if r["status"] == "passed"]
        result.pop("error", None)
        result["status"] = "running"
    else:
        folder.mkdir(parents=True, exist_ok=False)
        result = {"status": "running", "configuration": config, "runs": [], "metrics": {}, "plans": {}}
    write_summary(folder, result)
    try:
        if not args.resume:
            nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
            if not nvcc:
                raise RuntimeError("nvcc is required")
            commands = [["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                         f"-DCMAKE_CUDA_COMPILER={Path(nvcc).resolve()}", f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}"],
                        ["cmake", "--build", str(build), "--parallel", "2"]]
            result["build_commands"] = commands
            for label, command in zip(("configure.log", "build.log"), commands):
                status, _ = run_logged(command, folder / label)
                if status:
                    raise RuntimeError(f"Build failed: {label}")
            result["binary_sha256"] = {n: sha256(build / n) for n in names}
            result["compiler_flags"] = {n: (build / f"CMakeFiles/{n}.dir/flags.make").read_text() for n in names}
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        require_cuda()
        info = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,clocks.sm,temperature.gpu,power.limit", "--format=csv"], capture_output=True, text=True)
        result["gpu_details"] = info.stdout.strip()
        active = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], capture_output=True, text=True)
        result["other_compute_processes_at_start"] = active.stdout.strip()
        if active.stdout.strip():
            raise RuntimeError("Another CUDA compute process is active; run this benchmark on an idle GPU")
        schedule = [(args.cases[0], 0, name, index + 1, True) for index, name in enumerate(IMPLEMENTATIONS)]
        for round_id in range(1, args.repeats + 1):
            for case_index, dataset in enumerate(args.cases):
                order = IMPLEMENTATIONS if (round_id + case_index) % 2 else IMPLEMENTATIONS[::-1]
                schedule.extend((dataset, round_id, name, index + 1, False) for index, name in enumerate(order))
        with tempfile.TemporaryDirectory(prefix="performance-eval-", dir=build) as temporary:
            output = Path(temporary) / "routing.out"
            for dataset, round_id, name, order, warmup in schedule:
                if any(r["dataset"] == dataset and r["round"] == round_id and r["implementation"] == name
                       and r["warmup"] == warmup and r["status"] == "passed" for r in result["runs"]):
                    continue
                run_one(args, result, folder, build, output, dataset, name, round_id, order, warmup)
        result["status"] = "passed"
    except BaseException as error:
        result["status"], result["error"] = "failed", str(error)
        print(f"ERROR: {error}", flush=True)
    finally:
        write_summary(folder, result)
        print("Summary:", folder / "summary.md", flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
