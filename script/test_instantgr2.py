#!/usr/bin/env python3
"""Measure InstantGR2 batching backends and verify CPU-routed repeatability."""
import argparse
import ctypes
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline/InstantGR2"
TIMING_POLICY = "cuda-batching-plus-scoped-dag-wall-v3"


def repository_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command, log_path, cwd=None):
    print(shlex.join(command), flush=True)
    start = time.perf_counter()
    with log_path.open("x") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode, time.perf_counter() - start


def evaluator_metrics(log_path, returncode):
    if returncode not in (0, 1):
        raise RuntimeError(f"Evaluator failed with exit status {returncode}: {log_path}")
    text = log_path.read_text()
    def one(pattern):
        values = re.findall(pattern, text, re.MULTILINE)
        if len(values) != 1:
            raise RuntimeError(f"Missing or ambiguous evaluator result: {pattern}")
        return values[0]
    metrics = {key: one(rf"^{key} cost (\d+\.\d+)$")
               for key in ("wirelength", "via", "overflow", "total")}
    metrics["open_nets"] = int(one(r"^Number of open nets : (\d+)$"))
    metrics["incomplete_nets"] = int(one(r"^Number of incompleted nets : (\d+)$"))
    one(r"^FOR_STAT (.+)$")
    if metrics["open_nets"] or metrics["incomplete_nets"]:
        raise RuntimeError("Evaluator found open or incomplete nets")
    return metrics


def parse_batch_timings(text, mode, routing_backend):
    pattern = (r"^BATCH_TIMING backend=(cpu|cuda_core) "
               r"start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+) "
               r"nets=(\d+) batches=(\d+) used_for_routing=([01])$")
    records = re.findall(pattern, text, re.MULTILINE)
    backends = ["cuda_core", "cpu"] if mode == "dual" else ["cpu" if routing_backend == "cpu" else "cuda_core"]
    if [record[0] for record in records] != backends * 2:
        raise RuntimeError("Missing, duplicated, or out-of-order batching measurements")
    timings = []
    for index, (backend, start, end, elapsed, nets, batches, used) in enumerate(records):
        # Routing calls batching first for L-shape, then for FLT; this is reporting metadata only.
        stage = ("lshape", "detour")[index // len(backends)]
        row = {"stage": stage, "backend": backend, "start_us": int(start), "end_us": int(end),
               "elapsed_ms": float(elapsed), "nets": int(nets), "batches": int(batches),
               "used_for_routing": bool(int(used))}
        if row["end_us"] < row["start_us"] or abs(row["elapsed_ms"] - (int(end) - int(start)) / 1000) > 0.00001:
            raise RuntimeError("Invalid batching timestamps")
        if timings and row["start_us"] < timings[-1]["end_us"]:
            raise RuntimeError("Overlapping or out-of-order batching measurements")
        timings.append(row)
    selected = "cpu" if routing_backend == "cpu" else "cuda_core"
    if any(t["used_for_routing"] != (t["backend"] == selected) for t in timings):
        raise RuntimeError("Routing must use the configured batch backend")
    for stage in ("lshape", "detour"):
        if len({t["nets"] for t in timings if t["stage"] == stage}) != 1:
            raise RuntimeError("Batch backends received different net counts")
    return timings


def parse_routing_timings(text, batch_timings):
    pattern = (r"^DAG_TIMING stage=(lshape|detour) batch_index=(-?\d+) "
               r"start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+)$")
    timings = []
    for stage, index, start, end, elapsed, nets, batches in re.findall(pattern, text, re.MULTILINE):
        row = {"stage": stage, "batch_index": int(index), "start_us": int(start), "end_us": int(end),
               "elapsed_ms": float(elapsed), "nets": int(nets), "batches": int(batches)}
        span_ms = (int(end) - int(start)) / 1000
        if (span_ms < 0 or row["elapsed_ms"] < 0 or row["elapsed_ms"] > span_ms + 0.00001
                or (stage == "lshape" and abs(row["elapsed_ms"] - span_ms) > 0.00001)):
            raise RuntimeError("Invalid DAG timestamps")
        if timings and row["start_us"] < timings[-1]["end_us"]:
            raise RuntimeError("Overlapping or out-of-order DAG measurements")
        timings.append(row)
    selected = {t["stage"]: t for t in batch_timings if t["used_for_routing"]}
    expected = [("lshape", -1), ("detour", -1)]
    if [(t["stage"], t["batch_index"]) for t in timings] != expected:
        raise RuntimeError("Missing, duplicated, or out-of-order DAG measurements")
    algorithms = []
    for stage, algorithm in (("lshape", "L-shape"), ("detour", "FLT")):
        regions = [t for t in timings if t["stage"] == stage]
        plan = selected[stage]
        if sum(t["nets"] for t in regions) != plan["nets"] or sum(t["batches"] for t in regions) != plan["batches"]:
            raise RuntimeError("DAG measurements do not cover the routing batch plan")
        if any(t["start_us"] < plan["end_us"] for t in regions):
            raise RuntimeError("DAG timing precedes batch generation")
        cuda = [t["elapsed_ms"] for t in batch_timings if t["stage"] == stage and t["backend"] == "cuda_core"]
        cpu = [t["elapsed_ms"] for t in batch_timings if t["stage"] == stage and t["backend"] == "cpu"]
        dag_ms = sum(t["elapsed_ms"] for t in regions)
        algorithms.append({"stage": stage, "algorithm": algorithm,
                           "cuda_batch_ms": cuda[0] if cuda else None,
                           "cpu_batch_ms": cpu[0] if cpu else None,
                           "dag_solve_ms": dag_ms,
                           "algorithm_total_ms": cuda[0] + dag_ms if cuda else None,
                           "routing_backend": plan["backend"], "nets": plan["nets"], "batches": plan["batches"]})
    return timings, algorithms


def require_cuda():
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError("CUDA driver library is unavailable; use --build-only on a CPU host") from error
    status = driver.cuInit(0)
    if status:
        raise RuntimeError(f"CUDA initialization failed (driver status {status}); check GPU access")
    count = ctypes.c_int()
    status = driver.cuDeviceGetCount(ctypes.byref(count))
    if status or count.value == 0:
        raise RuntimeError("No CUDA GPU is available")


def write_summary(folder, result):
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (folder / "batching-runtime.csv").open("w", newline="") as output:
        fields = ["dataset", "run", "stage", "backend", "start_us", "end_us", "elapsed_ms",
                  "nets", "batches", "used_for_routing"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            for run in case["runs"]:
                for timing in run.get("batch_timings", []):
                    writer.writerow({"dataset": case["name"], "run": run["round"], **timing,
                                     "used_for_routing": int(timing["used_for_routing"])})
    with (folder / "dag-runtime.csv").open("w", newline="") as output:
        fields = ["dataset", "run", "stage", "batch_index", "start_us", "end_us", "elapsed_ms", "nets", "batches"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            for run in case["runs"]:
                for timing in run.get("dag_timings", []):
                    writer.writerow({"dataset": case["name"], "run": run["round"], **timing})
    with (folder / "runtime.csv").open("w", newline="") as output:
        fields = ["dataset", "run", "algorithm", "nets", "batches",
                  "cuda_batch_ms", "cpu_batch_ms", "dag_solve_ms", "algorithm_total_ms"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            for run in case["runs"]:
                if run["status"] == "passed":
                    for runtime in run.get("algorithm_runtime", []):
                        row = {"dataset": case["name"], "run": run["round"], **runtime}
                        writer.writerow({field: row[field] for field in fields})
    lines = ["# InstantGR2 Test Results", "",
             f"- Status: {result['status']}", f"- Mode: {result['mode']}",
             f"- Routing batch backend: {result['batching_backend']}",
             "- Congestion reduction: one thread per 2D cell; all outputs initialized",
             f"- Runs per dataset: {result['repeats']}",
             f"- Upstream commit: `{result['source_commit']}` (vendored source hashes recorded in summary.json)",
             f"- CUDA architecture: `sm_{result['cuda_arch']}`",
             f"- Batch sparsity: {result['sparsity'] or 'original backend defaults'}",
             "- Retained outputs: one compressed reference per dataset", "",
             "| Dataset | Status | Runs | Scores match | Total cost | Open | Incomplete |",
             "|---|---|---:|---|---:|---:|---:|"]
    for case in result["cases"]:
        runs = case["runs"]
        comparisons = runs[1:]
        metrics = case.get("metrics", {})
        scores = str(all(r.get("metrics_match", False) for r in comparisons)) if comparisons else "not tested"
        completed = sum(r["status"] == "passed" for r in runs)
        lines.append(f"| {case['name']} | {case['status']} | {completed}/{result['repeats']} | "
                     f"{scores} | {metrics.get('total', '-')} | "
                     f"{metrics.get('open_nets', '-')} | {metrics.get('incomplete_nets', '-')} |")
    lines += ["", "## Routing Algorithm Runtime (ms)", "",
              "Algorithm total = CUDA-core batching + scoped routing-DAG solve time. CPU batching is excluded from this total.",
              "In dual mode DAG solving consumes the CPU plan. This sum combines CUDA batching measurements with deterministic CPU-plan routing; it is not end-to-end router wall time.",
              "L-shape: the complete batch loop (update_cost through wire/via commits), after setup and transfers.",
              "FLT: sum of each batch's GPU suffix, starting at cudaMemcpy(net_ids, ...) through final wire/via commits. Includes transfers, initialization, edge precompute, forward solve, backtrace, and synchronization; excludes preceding CPU packing and wrapper allocations.",
              "All regions use synchronized steady-clock wall time. FLT emits one aggregate after its batch loop: elapsed_ms sums GPU suffixes; start_us/end_us enclose the loop and include excluded CPU gaps. Means use completed runs; per-run values are in runtime.csv and stage-level DAG timestamps in dag-runtime.csv. Missing CUDA measurements are shown as '-' rather than replaced by CPU timings.", "",
              "| Dataset | Algorithm | CUDA batching | DAG solve | Algorithm total |",
              "|---|---|---:|---:|---:|"]
    for case in result["cases"]:
        for algorithm in ("L-shape", "FLT"):
            rows = [t for run in case["runs"] if run["status"] == "passed"
                    for t in run.get("algorithm_runtime", []) if t["algorithm"] == algorithm]
            cells = []
            for key in ("cuda_batch_ms", "dag_solve_ms", "algorithm_total_ms"):
                values = [row[key] for row in rows if row[key] is not None]
                cells.append(f"{sum(values) / len(values):.6f}" if values else "-")
            lines.append("| " + case["name"] + " | " + algorithm + " | " + " | ".join(cells) + " |")
    lines += ["", "## Batching Runtime (ms)", "",
              "Synchronized full-function wall time, including allocation, packing, transfers, batch generation, result collection, and temporary-buffer cleanup.",
              "Means below use completed runs. Per-run process-relative steady-clock timestamps and batch counts are in batching-runtime.csv.",
              "Dual mode runs CUDA first, discards its plan, then runs CPU; routing always consumes the CPU plan.", "",
              "| Dataset | L-shape CPU | L-shape CUDA-core | Detour CPU | Detour CUDA-core |",
              "|---|---:|---:|---:|---:|"]
    for case in result["cases"]:
        cells = []
        for stage, backend in (("lshape", "cpu"), ("lshape", "cuda_core"),
                               ("detour", "cpu"), ("detour", "cuda_core")):
            values = [t["elapsed_ms"] for run in case["runs"] if run["status"] == "passed"
                      for t in run.get("batch_timings", []) if t["stage"] == stage and t["backend"] == backend]
            cells.append(f"{sum(values) / len(values):.6f}" if values else "-")
        lines.append("| " + case["name"] + " | " + " | ".join(cells) + " |")
    lines += ["", "## Reference Evaluator Costs", "",
              "| Dataset | Wirelength cost | Via cost | Overflow cost | Total cost |",
              "|---|---:|---:|---:|---:|"]
    for case in result["cases"]:
        metrics = case.get("metrics", {})
        lines.append("| " + case["name"] + " | " + " | ".join(
            metrics.get(key, "-") for key in ("wirelength", "via", "overflow", "total")) + " |")
    lines += ["", "Each run uses an independent router process. Full decimal evaluator costs must match exactly in dual and determinism modes.",
              "Evaluator-only cases compare the four costs and connectivity; route bytes, route contents, and raw hashes are not checked.",
              "Repeated-run outputs are deleted after evaluation. Only the first output remains as <dataset>.out.gz.",
              "Router wall time includes input, CUDA initialization, routing, and output. Evaluation and compression run afterward; per-run timings are in summary.json.",
              "In dual mode the router wall time includes BOTH planners and is not a measured single-backend end-to-end runtime. Batching timings use a fixed CUDA-then-CPU order.",
              "Passing repeat tests is evidence for these inputs and this environment, not a universal determinism proof."]
    if result["status"] == "build-only":
        lines += ["", "Build completed; no routing test was run."]
    if result.get("error"):
        lines += ["", "## Error", "", "```text", result["error"], "```"]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def compress_reference(source, destination):
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        with gzip.GzipFile(filename="", mode="wb", compresslevel=1,
                           fileobj=output_file, mtime=0) as compressed:
            shutil.copyfileobj(input_file, compressed, length=1024**2)


def test_case(args, result, name, output_dir, build_dir):
    folder = output_dir / name
    folder.mkdir()
    inputs = [args.data_dir / f"{name}.{suffix}" for suffix in ("cap", "net")]
    case = {"name": name, "status": "running", "runs": [],
            "inputs": {str(p): sha256(p) for p in inputs}}
    case["comparison_policy"] = "evaluator-only"
    result["cases"].append(case)
    retained = folder / f"{name}.out.gz"
    with tempfile.TemporaryDirectory(prefix="repeat-check-", dir=build_dir) as temporary:
        temporary = Path(temporary)
        raw_output = temporary / "candidate.out"
        for round_id in range(1, args.repeats + 1):
            run = {"round": round_id, "status": "running"}
            case["runs"].append(run)
            log = folder / f"run{round_id}.routing.log"
            evaluation_log = folder / f"run{round_id}.evaluator.log"
            try:
                command = [str(build_dir / "InstantGR2"), "-cap", str(inputs[0]),
                           "-net", str(inputs[1]), "-out", str(raw_output),
                           "--batching", "dual" if args.mode == "dual" else result["batching_backend"]]
                if args.sparsity:
                    command += ["--sparsity", str(args.sparsity)]
                run["router_command"] = command
                print(f"Routing {name}: run {round_id}/{args.repeats}", flush=True)
                run["router_exit_status"], run["router_seconds"] = run_logged(
                    command, log, cwd=BASELINE / "run")
                if run["router_exit_status"] or not raw_output.is_file() or not raw_output.stat().st_size:
                    raise RuntimeError(f"Routing failed: {name}, run {round_id}")
                text = log.read_text()
                backends = re.findall(r"^BATCH_BACKEND=(cpu|gpu)$", text, re.MULTILINE)
                if backends != [result["batching_backend"]] * 2 or "CONGESTION_REDUCTION=per-cell" not in text:
                    raise RuntimeError("Both stages must use the selected backend and congestion repair")
                if args.sparsity:
                    values = re.findall(r"^BATCH_SPARSITY backend=(\w+) mode=\d+ value=(\d+)$", text, re.MULTILINE)
                    expected = ([('gpu', str(args.sparsity)), ('cpu', str(args.sparsity))] * 2
                                if args.mode == "dual" else [(result["batching_backend"], str(args.sparsity))] * 2)
                    if values != expected:
                        raise RuntimeError("Both stages must use the selected sparsity")
                run["batch_timings"] = parse_batch_timings(text, args.mode, result["batching_backend"])
                run["dag_timings"], run["algorithm_runtime"] = parse_routing_timings(text, run["batch_timings"])
                run["batch_generation"] = [{"nets": t["nets"], "batches": t["batches"]}
                                           for t in run["batch_timings"] if t["used_for_routing"]]
                command = [str(build_dir / "evaluator"), *map(str, inputs), str(raw_output)]
                run["evaluator_command"] = command
                run["evaluator_exit_status"], run["evaluator_seconds"] = run_logged(command, evaluation_log)
                run["metrics"] = evaluator_metrics(evaluation_log, run["evaluator_exit_status"])
                if round_id == 1:
                    case["metrics"] = run["metrics"]
                    compress_reference(raw_output, retained)
                    case["output"] = str(retained.relative_to(output_dir))
                else:
                    first = case["runs"][0]
                    run["metrics_match"] = run["metrics"] == first["metrics"]
                    if not run["metrics_match"]:
                        raise RuntimeError(f"Determinism mismatch: {name}, run {round_id}")
                run["status"] = "passed"
                print(f"PASS {name} run {round_id}: total cost {run['metrics']['total']}", flush=True)
                for runtime in run["algorithm_runtime"]:
                    cuda_ms, total_ms = runtime["cuda_batch_ms"], runtime["algorithm_total_ms"]
                    cuda_label = f"{cuda_ms:.3f}" if cuda_ms is not None else "-"
                    total_label = f"{total_ms:.3f}" if total_ms is not None else "-"
                    print(f"  {runtime['algorithm']}: CUDA batch={cuda_label} ms, "
                          f"DAG={runtime['dag_solve_ms']:.3f} ms, total={total_label} ms", flush=True)
            except BaseException as error:
                run["status"], run["error"] = "failed", str(error)
                case["status"] = "failed"
                raise
            finally:
                raw_output.unlink(missing_ok=True)
                write_summary(output_dir, result)
    case["status"] = "passed"
    write_summary(output_dir, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="baseline/InstantGR2/benchmarks")
    parser.add_argument("--cases", nargs="+", default=["mempool_tile_rank"])
    parser.add_argument("--mode", choices=("dual", "performance", "determinism"), default="dual")
    parser.add_argument("--repeats", type=int, help="determinism runs per dataset (default: 3)")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--sparsity", type=int, default=1,
                        help="sampling step; default 1; 0 keeps original backend defaults")
    parser.add_argument("--output-dir", default="results/InstantGR2")
    parser.add_argument("--build-only", action="store_true")
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", action="store_true",
                          help="resume an interrupted run with unchanged source/configuration")
    existing.add_argument("--clean-results", action="store_true",
                          help="remove this script's previous output directory under repository results/")
    args = parser.parse_args()
    if args.repeats is None:
        args.repeats = 1 if args.mode == "performance" else 3
    if args.mode != "performance" and args.repeats < 2:
        parser.error("CPU-routed repeat tests require at least two independent runs")
    if args.mode == "performance" and args.repeats != 1:
        parser.error("Performance mode uses one run; repeatability checks use --mode determinism")
    if not args.cuda_arch.isdigit() or not 0 <= args.sparsity <= 2147483647:
        parser.error("Invalid CUDA architecture or sparsity")
    if len(set(args.cases)) != len(args.cases) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", case) for case in args.cases
    ):
        parser.error("Case names must be unique and contain only letters, digits, '_' or '-'")
    args.data_dir = repository_path(args.data_dir)
    if not args.build_only:
        for name in args.cases:
            for suffix in ("cap", "net"):
                if not (args.data_dir / f"{name}.{suffix}").is_file():
                    parser.error(f"Missing input: {name}.{suffix}")
    output_dir = repository_path(args.output_dir)
    if args.clean_results and output_dir.exists():
        if ROOT / "results" not in output_dir.parents or not (output_dir / "summary.json").is_file():
            parser.error("--clean-results requires a previous test directory below this repository's results/")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    build_dir = ROOT / "build/InstantGR2" / f"sm_{args.cuda_arch}"
    build_dir.mkdir(parents=True, exist_ok=True)
    commit = (BASELINE / "UPSTREAM_REVISION").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        parser.error("Invalid InstantGR2 UPSTREAM_REVISION; expected a full upstream commit hash")
    result = {"status": "running", "source_commit": commit,
              "mode": args.mode, "batching_backend": "gpu" if args.mode == "performance" else "cpu",
              "congestion_fix": True, "repeats": args.repeats, "sparsity": args.sparsity or None,
              "comparison_policy": "evaluator-only",
              "timing_policy": TIMING_POLICY,
              "runner_sha256": sha256(Path(__file__)),
              "cuda_arch": args.cuda_arch, "gpu": args.gpu, "cases": [],
              "source_sha256": {str(p.relative_to(BASELINE)): sha256(p) for p in sorted((BASELINE / "src").iterdir()) if p.is_file()}}
    if args.resume:
        previous = json.loads((output_dir / "summary.json").read_text())
        for key in ("source_commit", "source_sha256", "mode", "batching_backend", "repeats", "sparsity", "cuda_arch", "gpu", "timing_policy", "runner_sha256"):
            if previous.get(key) != result[key]:
                parser.error(f"Cannot resume with a changed {key}")
        for name in ("InstantGR2", "evaluator"):
            if sha256(build_dir / name) != previous["binary_sha256"][name]:
                parser.error("Cannot resume with changed binaries")
        for case in previous["cases"]:
            if case["name"] not in args.cases:
                parser.error("Resume cases must include the previously tested datasets")
            for run in case["runs"]:
                if run.get("canonical_match") is False or run.get("metrics_match") is False:
                    parser.error("A real determinism mismatch cannot be discarded by --resume")
            expected = {str(args.data_dir / f"{case['name']}.{ext}"): sha256(args.data_dir / f"{case['name']}.{ext}")
                        for ext in ("cap", "net")}
            if case["inputs"] != expected:
                parser.error("Cannot resume with changed inputs")
            if case["status"] == "passed" and not (output_dir / case["output"]).is_file():
                parser.error("A completed reference output is missing")
            case["comparison_policy"] = "evaluator-only"
            for run in case["runs"]:
                for key in ("canonical", "canonical_match", "raw_sha256", "raw_bytes_equal"):
                    run.pop(key, None)
        incomplete = [c for c in previous["cases"] if c["status"] != "passed"]
        completed_names = {c["name"] for c in previous["cases"] if c["status"] == "passed"}
        for name in args.cases:
            if name not in completed_names and (output_dir / name).exists():
                shutil.rmtree(output_dir / name)
        previous.setdefault("resume_history", []).append({
            "restarted_cases": incomplete,
            "note": "Interrupted datasets restarted from run 1; their old reference removed. Completed evaluator results and batching timings retained. All cases use evaluator-only comparison; historical route-comparison fields removed."})
        previous["cases"] = [c for c in previous["cases"] if c["status"] == "passed"]
        previous["status"] = "running"
        previous.pop("score_only_cases", None)
        previous["comparison_policy"] = "evaluator-only"
        previous.pop("error", None)
        result = previous
    write_summary(output_dir, result)
    try:
        nvcc, cxx = shutil.which(args.nvcc), shutil.which(args.cxx)
        if not nvcc or not cxx:
            raise RuntimeError("nvcc and a C++ compiler are required")
        result["compiler_versions"] = {"nvcc": subprocess.check_output([nvcc, "--version"], text=True),
                                       "cxx": subprocess.check_output([cxx, "--version"], text=True)}
        commands = [[nvcc, str(BASELINE / "src/main.cpp"), "-o", str(build_dir / "InstantGR2"),
                     "-std=c++17", "-x", "cu", "-O3", f"-arch=sm_{args.cuda_arch}"],
                    [cxx, str(BASELINE / "run/evaluator.cpp"), "-o", str(build_dir / "evaluator"),
                     "-O3", "-std=c++17"]]
        result["build_commands"] = commands
        if args.resume:
            for name in ("InstantGR2", "evaluator"):
                if sha256(build_dir / name) != result["binary_sha256"][name]:
                    raise RuntimeError("Cannot resume with changed binaries")
            print("Resuming with verified, unchanged routing/evaluator binaries", flush=True)
        else:
            for name, command in zip(("router-build.log", "evaluator-build.log"), commands):
                status, _ = run_logged(command, output_dir / name)
                if status:
                    raise RuntimeError(f"Build failed: {name}")
            result["binary_sha256"] = {name: sha256(build_dir / name) for name in ("InstantGR2", "evaluator")}
        if args.build_only:
            result["status"] = "build-only"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            require_cuda()
            if shutil.which("nvidia-smi"):
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                     text=True, capture_output=True)
                result["gpu_details"] = gpu.stdout.strip()
            for name in args.cases:
                if not any(c["name"] == name and c["status"] == "passed" for c in result["cases"]):
                    test_case(args, result, name, output_dir, build_dir)
            result["status"] = "passed"
    except BaseException as error:
        result["status"], result["error"] = "failed", str(error)
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        write_summary(output_dir, result)
        print(f"Summary: {output_dir / 'summary.md'}", flush=True)
    return 0 if result["status"] in ("passed", "build-only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
