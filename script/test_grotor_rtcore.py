#!/usr/bin/env python3
"""Validate RT-Core batch lists, evaluate routing, and report scoped runtimes."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from test_instantgr2 import (ROOT, BASELINE, evaluator_metrics, parse_routing_timings,
                            require_cuda, run_logged, sha256)
from test_grotor_reference import local_path, source_manifest, validate_luts

from rtcore_reporting import parse_spatial_settings

POLICY = "rtcore-cooperative-dag-v13"


def parse_rtcore_timings(text, validate=True):
    pattern = (r"^BATCH_TIMING backend=(rt_core) start_us=(\d+) end_us=(\d+) "
               r"elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+) used_for_routing=(1)$")
    records = re.findall(pattern, text, re.MULTILINE)
    if len(records) != 2 or re.findall(r"^BATCH_BACKEND=(\w+)$", text, re.MULTILINE) != ["rtcore"] * 2:
        raise RuntimeError("Both routing stages must consume RT-Core batches")
    timings = []
    for stage, values in zip(("lshape", "detour"), records):
        backend, start, end, elapsed, nets, batches, used = values
        row = dict(stage=stage, backend=backend, start_us=int(start), end_us=int(end),
                   elapsed_ms=float(elapsed), nets=int(nets), batches=int(batches), used_for_routing=True)
        span = (row["end_us"] - row["start_us"]) / 1000
        if (span < 0 or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0
                or abs(row["elapsed_ms"] - span) > .001 or row["nets"] < row["batches"]
                or (row["nets"] > 0 and row["batches"] == 0)):
            raise RuntimeError("Invalid RT-Core contiguous wall timing or batch counts")
        if timings and row["start_us"] < timings[-1]["end_us"]:
            raise RuntimeError("Out-of-order RT-Core timings")
        timings.append(row)
    checks = re.findall(r"^BATCH_VALIDATION nets=(\d+) rtcore_batches=(\d+) cpu_batches=(\d+) exact_lists_equal=([01])$",
                        text, re.MULTILINE)
    if validate and checks != [(str(t["nets"]), str(t["batches"]), str(t["batches"]), "1") for t in timings]:
        raise RuntimeError("Missing or failed exact CPU/RT-Core batch-list comparison")
    return timings


def parse_execution_timings(text, timings):
    from rtcore_reporting import parse_totals
    for batch, total in zip(timings, parse_totals(text, timings)):
        batch.update(total)


def saved_oracle(path, dataset, inputs):
    """Reuse evaluated baseline costs only when their input/source/LUT hashes match."""
    result = json.loads(path.read_text())
    recorded = result["configuration"]["sources"]
    sources = source_manifest()
    relevant = {k: v for k, v in sources.items()
                if k.startswith(("baseline/InstantGR2/src/", "resources/flute/"))
                or k == "baseline/InstantGR2/run/evaluator.cpp"}
    if not relevant or any(recorded.get(k) != v for k, v in relevant.items()):
        raise RuntimeError("Saved oracle baseline sources, evaluator or LUTs have changed")
    cases = [c for c in result["cases"] if c["name"] == dataset and c["status"] == "passed"]
    if len(cases) != 1 or cases[0]["inputs"] != inputs or result.get("sparsity") != 1:
        raise RuntimeError("Saved oracle must be a passed sparsity-1 case with identical inputs")
    runs = [r for r in cases[0]["runs"] if r["implementation"] == "InstantGR2" and r["status"] == "passed"]
    if len(runs) != 1:
        raise RuntimeError("Saved oracle has no unique successful InstantGR2 run")
    metrics = runs[0]["metrics"]
    if metrics["open_nets"] or metrics["incomplete_nets"]:
        raise RuntimeError("Saved oracle has invalid connectivity")
    return metrics


def gpu_monitor(command, log, path, gpu):
    """Sample device-wide memory during routing only; never overlap two routers."""
    with path.open("x") as stream:
        monitor = subprocess.Popen(["nvidia-smi", "--id=" + gpu,
                                   "--query-gpu=timestamp,index,memory.used,memory.free",
                                   "--format=csv,noheader,nounits", "--loop-ms=100"],
                                  stdout=stream, stderr=subprocess.STDOUT)
        try:
            status, seconds = run_logged(command, log, cwd=ROOT)
        finally:
            monitor.terminate()
            monitor.wait()
    samples = []
    for row in csv.reader(path.read_text().splitlines()):
        if len(row) == 4:
            try:
                samples.append((int(row[2]), int(row[3])))
            except ValueError:
                pass
    if not samples:
        raise RuntimeError("No GPU memory samples were collected")
    return status, seconds, {"samples": len(samples), "sample_interval_ms": 100,
                             "sampled_peak_used_mib": max(x[0] for x in samples),
                             "sampled_min_free_mib": min(x[1] for x in samples)}


def write_summary(folder, result):
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (folder / "runtime.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["dataset", "run", "algorithm", "nets", "batches",
                                        "rtcore_method", "timing_scope", "spatial_model", "point_model", "z_gap", "margin", "rtcore_batch_ms", "dag_solve_ms", "algorithm_total_ms"])
        writer.writeheader()
        for run in result["runs"]:
            if run["status"] != "passed":
                continue
            for row in run["algorithm_runtime"]:
                writer.writerow({"dataset": run["dataset"], "run": run["round"], **row})
    with (folder / "quality.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["dataset", "run", "matches_instantgr2", "wirelength", "via",
                                        "overflow", "total", "open_nets", "incomplete_nets"])
        writer.writeheader()
        for run in result["runs"]:
            if run["status"] == "passed":
                writer.writerow({"dataset": run["dataset"], "run": run["round"],
                                 "matches_instantgr2": run["matches_instantgr2"], **run["metrics"]})
    lines = ["# GRotor RT-Core Validation", "", f"- Status: {result['status']}",
             "- RT-Core plans are used for routing in both stages; no CPU fallback.",
             f"- Exact batch-list validation against CPU: {result['configuration']['validate_batches']}.",
             f"- Selected RT-Core timing: {result['configuration']['timing']}.",
             "- Each method measures one contiguous steady-clock interval, including host work, transfers and waits inside it; no per-round accumulation or component sums.",
             "- Standard RT-Core scope: coloring loop through GPU completion. Cached scope: pipeline preparation through coloring GPU completion.",
             "- Geometry/BVH preparation, final batch download/assembly, CPU validation and evaluation are outside the measured intervals.",
             "- L-shape uses standard batching; FLT uses cached batching for every design.",
             "- Algorithm total = selected RT-Core batching scope + scoped DAG solving; it is not complete router wall time.",
             "- CPU validation is outside both reported algorithm times, but inside router wall time.",
             "- Memory values are device-wide 100 ms samples, not exact instantaneous allocation peaks.",
             "- Every actual routing .out is read by the external evaluator and deleted afterward; no route bytes are compared.", "",
             "| Dataset | Run | Algorithm | RT-Core batching ms | DAG solve ms | Total ms |",
             "|---|---:|---|---:|---:|---:|"]
    for run in result["runs"]:
        if run["status"] == "passed":
            for row in run["algorithm_runtime"]:
                lines.append(f"| {run['dataset']} | {run['round']} | {row['algorithm']} | "
                             f"{row['rtcore_batch_ms']:.3f} | {row['dag_solve_ms']:.3f} | {row['algorithm_total_ms']:.3f} |")
    lines += ["", "## Quality and Memory", "",
              "| Dataset | Run | Evaluator total | All four costs match | Peak used MiB | Min free MiB |",
              "|---|---:|---:|---|---:|---:|"]
    for run in result["runs"]:
        if run["status"] == "passed":
            memory = run["gpu_memory"]
            lines.append(f"| {run['dataset']} | {run['round']} | {run['metrics']['total']} | "
                         f"{run['matches_instantgr2']} | {memory['sampled_peak_used_mib']} | {memory['sampled_min_free_mib']} |")
    if result.get("error"):
        lines += ["", "## Error", "", result["error"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["mempool_tile", "nvdla", "mempool_cluster"])
    parser.add_argument("--data-dir", default="../ISPD24_tmp")
    parser.add_argument("--output-dir", default="results/GRotor-rtcore")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--optix-root", default=os.environ.get("OPTIX_ROOT_DIR"))
    parser.add_argument("--oracle-summary", help="Reuse a validated CPU-reference summary with matching hashes")
    parser.add_argument("--router-binary", help="Use an explicitly recorded alternative router executable")
    parser.add_argument("--no-validate-batches", action="store_true", help="Skip CPU batch-list checks for performance runs")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if (args.repeats < 1 or not args.cuda_arch.isdigit() or not args.gpu.isdigit()
            or len(set(args.cases)) != len(args.cases)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in args.cases)):
        parser.error("Use positive repeats, numeric GPU/architecture, and unique simple dataset names")
    args.data_dir = local_path(args.data_dir)
    folder = local_path(args.output_dir)
    if ROOT / "results" not in folder.parents:
        parser.error("Output directory must be below this repository's results/")
    build = ROOT / "build/GRotor-rtcore" / f"sm_{args.cuda_arch}"
    build.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "configuration": {
        "policy": POLICY, "sources": {**source_manifest(), str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
        "datasets": args.cases, "repeats": args.repeats, "sparsity": 1, "gpu": args.gpu,
        "validate_batches": not args.no_validate_batches, "cuda_arch": args.cuda_arch,
        "timing": "contiguous_wall", "cost_update": "incremental", "prefix": "bit", "flt_execution": "contiguous", "dag_lshape": "warp", "dag_flt": "block",
        "space": "layered", "point_model": "triangle",
        "z_gap": 2**-16, "margin": .1, "method_policy": "explicit_standard_lshape_cached_flt"}, "oracles": {}, "runs": []}
    try:
        validate_luts()
        nvcc = shutil.which(args.nvcc)
        if not nvcc:
            raise RuntimeError("nvcc is required")
        commands = [["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                     f"-DCMAKE_CUDA_COMPILER={Path(nvcc).resolve()}", f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
                     "-DGROTOR_ENABLE_RTCORE=ON", "-DGROTOR_FULL_COST_REFERENCE=OFF"]]
        if args.optix_root:
            commands[0].append(f"-DOPTIX_ROOT_DIR={Path(args.optix_root).resolve()}")
        commands.append(["cmake", "--build", str(build), "--parallel", "2"])
        result["build_commands"] = commands
        for command, label in zip(commands, ("configure.log", "build.log")):
            status, _ = run_logged(command, folder / label)
            if status:
                raise RuntimeError(f"Build failed: {label}")
        result["binary_sha256"] = {n: sha256(build / n) for n in
                                   ("GRotor", "instantgr2_reference", "grotor_evaluator", "rtcore_batches.ptx", "rtcore_cached.ptx")}
        router = local_path(args.router_binary) if args.router_binary else build / "GRotor"
        result["executed_router"] = {"path": str(router), "sha256": sha256(router)}
        if args.build_only:
            result["status"] = "built"
            return 0
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        require_cuda()
        result["gpu_details"] = subprocess.check_output(["nvidia-smi", "--id=" + args.gpu,
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version", "--format=csv"], text=True)
        oracle_path = local_path(args.oracle_summary) if args.oracle_summary else None
        if oracle_path:
            result["oracle_summary"] = {"path": str(oracle_path), "sha256": sha256(oracle_path)}
        for dataset in args.cases:
            inputs = [args.data_dir / f"{dataset}.{ext}" for ext in ("cap", "net")]
            hashes = {str(p): sha256(p) for p in inputs}
            case_folder = folder / dataset
            case_folder.mkdir()
            with tempfile.TemporaryDirectory(prefix="rtcore-evaluation-", dir=build) as temporary:
                output = Path(temporary) / (dataset + ".out")
                if oracle_path:
                    metrics = saved_oracle(oracle_path, dataset, hashes)
                    result["oracles"][dataset] = {"metrics": metrics, "inputs": hashes, "source": "saved CPU reference"}
                else:
                    command = [str(build / "instantgr2_reference"), "-cap", str(inputs[0]), "-net", str(inputs[1]),
                               "-out", str(output), "--sparsity", "1", "--batching", "cpu"]
                    status, seconds = run_logged(command, case_folder / "instantgr2.routing.log", cwd=BASELINE / "run")
                    if status or not output.is_file() or not output.stat().st_size:
                        raise RuntimeError("InstantGR2 oracle routing failed")
                    evaluator = [str(build / "grotor_evaluator"), *map(str, inputs), str(output)]
                    log = case_folder / "instantgr2.evaluator.log"
                    eval_status, _ = run_logged(evaluator, log)
                    metrics = evaluator_metrics(log, eval_status)
                    result["oracles"][dataset] = {"metrics": metrics, "inputs": hashes,
                        "router_command": command, "router_seconds": seconds, "evaluator_command": evaluator}
                    output.unlink()
                for repeat in range(1, args.repeats + 1):
                    run = {"dataset": dataset, "round": repeat, "status": "running"}
                    result["runs"].append(run)
                    write_summary(folder, result)
                    try:
                        command = [str(router), "-cap", str(inputs[0]), "-net", str(inputs[1]), "-out", str(output),
                                   "--batching", "rtcore", "--sparsity", "1", "--validate-batches", "0" if args.no_validate_batches else "1"]
                        run["router_command"] = command
                        log = case_folder / f"run{repeat}.routing.log"
                        run["router_exit_status"], run["router_seconds"], run["gpu_memory"] = gpu_monitor(
                            command, log, case_folder / f"run{repeat}.gpu_memory.csv", args.gpu)
                        if run["router_exit_status"] or not output.is_file() or not output.stat().st_size:
                            raise RuntimeError(f"GRotor routing failed: {log}")
                        text = log.read_text()
                        if "DAG_SCHEDULING lshape=warp flt=block" not in text:
                            raise RuntimeError("Unexpected DAG scheduling")
                        modes = re.findall(r"^GROTOR_CONFIG batching=rtcore cost=(full|incremental) prefix=(full|bit) memory=explicit$", text, re.MULTILINE)
                        if (modes not in ([('full', 'full')], [('incremental', 'bit')])
                                or "CONGESTION_REDUCTION=per-cell" not in text):
                            raise RuntimeError("Missing RT-Core explicit-memory pipeline or deterministic congestion reduction")
                        run['cost_update'], run['prefix'] = modes[0]
                        result['configuration']['cost_update'], result['configuration']['prefix'] = modes[0]
                        run["batch_timings"] = parse_rtcore_timings(text, not args.no_validate_batches)
                        parse_execution_timings(text, run["batch_timings"])
                        run["dag_timings"], algorithms = parse_routing_timings(text, run["batch_timings"])
                        flt = run["dag_timings"][1]
                        if abs(flt['elapsed_ms'] - (flt['end_us'] - flt['start_us']) / 1000) > .00001:
                            raise RuntimeError("FLT execution must use one contiguous interval")
                        run["space"] = parse_spatial_settings(text)
                        run["algorithm_runtime"] = []
                        for batch, row, space in zip(run["batch_timings"], algorithms, run["space"]):
                            run["algorithm_runtime"].append({"algorithm": row["algorithm"], "nets": row["nets"], "batches": row["batches"],
                                "rtcore_method": batch["method"], "timing_scope": batch["timing_scope"], **space,
                                "rtcore_batch_ms": batch["elapsed_ms"], "dag_solve_ms": row["dag_solve_ms"],
                                "algorithm_total_ms": batch["elapsed_ms"] + row["dag_solve_ms"]})
                        write_summary(folder, result)
                        evaluator = [str(build / "grotor_evaluator"), *map(str, inputs), str(output)]
                        run["evaluator_command"] = evaluator
                        eval_log = case_folder / f"run{repeat}.evaluator.log"
                        run["evaluator_exit_status"], run["evaluator_seconds"] = run_logged(evaluator, eval_log)
                        run["metrics"] = evaluator_metrics(eval_log, run["evaluator_exit_status"])
                        run["matches_instantgr2"] = run["metrics"] == metrics
                        if not run["matches_instantgr2"]:
                            raise RuntimeError(f"Evaluator costs differ from CPU reference: {dataset}")
                        run["status"] = "passed"
                        print(f"PASS {dataset} run {repeat}: {run['metrics']['total']}", flush=True)
                    except BaseException as error:
                        run["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                        run["error"] = str(error)
                        raise
                    finally:
                        output.unlink(missing_ok=True)
                        write_summary(folder, result)
        result["status"] = "passed"
        return 0
    except BaseException as error:
        result["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        result["error"] = str(error) or type(error).__name__
        return 1
    finally:
        write_summary(folder, result)


if __name__ == "__main__":
    raise SystemExit(main())
