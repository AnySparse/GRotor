#!/usr/bin/env python3
"""Compare legacy FLT packing, contiguous GPU assembly, and incremental costs."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import tempfile

from test_instantgr2 import ROOT, run_logged, sha256, evaluator_metrics, parse_routing_timings
from test_grotor_reference import CASES, local_path, source_manifest
from test_grotor_rtcore import gpu_monitor, parse_rtcore_timings, saved_oracle

VARIANTS = ("legacy_full", "assembled_full", "assembled_bit")


def write_summary(folder, result):
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    runtime, memory, quality = [], [], []
    for case in result["cases"]:
        for label, run in case.get("runs", {}).items():
            if run.get("status") != "passed":continue
            memory.append(dict(dataset=case["name"], variant=label, cpu_peak_mib=run["cpu_peak_mib"],
                gpu_peak_mib=run["gpu_memory"]["sampled_peak_used_mib"],
                gpu_min_free_mib=run["gpu_memory"]["sampled_min_free_mib"],
                input_mode=run.get("storage", {}).get("mode", "legacy")))
            quality.append(dict(dataset=case["name"], variant=label, matches_baseline=True, **run["metrics"]))
        if case["status"] != "passed":continue
        a, b, c = (case["runs"][v] for v in VARIANTS)
        runtime.append(dict(dataset=case["name"], legacy_gpu_sum_ms=a["dag_timings"][1]["elapsed_ms"],
            legacy_contiguous_ms=a["execution_ms"], assembled_full_ms=b["execution_ms"], assembled_bit_ms=c["execution_ms"],
            engineering_vs_legacy_sum=a["dag_timings"][1]["elapsed_ms"]/b["execution_ms"],
            engineering_vs_legacy_wall=a["execution_ms"]/b["execution_ms"],
            bit_speedup=b["execution_ms"]/c["execution_ms"],
            legacy_flt_phase_ms=a["flt_phase_ms"], assembled_full_flt_phase_ms=b["flt_phase_ms"],
            assembled_bit_flt_phase_ms=c["flt_phase_ms"],
            legacy_router_seconds=a["program_seconds"], assembled_full_router_seconds=b["program_seconds"],
            assembled_bit_router_seconds=c["program_seconds"]))
    for name, rows in (("runtime.csv", runtime), ("memory.csv", memory), ("quality.csv", quality)):
        if rows:
            with (folder / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines = ["# FLT Execution Comparison", "", f"Status: {result['status']}", "",
        "- Legacy full: the pre-refactor router with full cost/prefix updates. Its original GPU suffix sum is retained, and an instrumentation-only outer timer also measures CPU packing plus execution continuously.",
        "- Assembled full: compact topology, GPU batch assembly, full cost/prefix updates.",
        "- Assembled BIT: the identical execution pipeline with incremental costs/BIT enabled.",
        "- Both assembled execution times use one outer steady-clock interval, from before topology upload through the final demand commit and GPU completion. They include GPU assembly, transfers, staging waits and launch gaps. Preparation/allocation and cleanup are reported in complete FLT phase/router time.",
        "- The legacy GPU suffix sum excludes per-batch CPU packing. Its ratio to a new continuous interval compares different scopes and is shown separately from the like-for-like continuous-wall comparison.",
        "- All runs use deterministic RT-Core batch plans, sparsity 1, CPU batch checks disabled, one router at a time. GPU memory is sampled device-wide at 100 ms; CPU maximum RSS is collected by GNU time for the router process.",
        "- Each actual routing output is externally evaluated against hash-checked fixed baseline scores and deleted before the next run. No output bytes are compared. These are single runs with unlocked GPU clocks.", "",
        "## Execution Runtime (ms)", "",
        "| Dataset | Legacy GPU sum | Legacy wall | Assembled full | Assembled BIT | Legacy sum / full | Legacy wall / full | Full / BIT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runtime:
        lines.append(f"| {r['dataset']} | {r['legacy_gpu_sum_ms']:.3f} | {r['legacy_contiguous_ms']:.3f} | {r['assembled_full_ms']:.3f} | {r['assembled_bit_ms']:.3f} | {r['engineering_vs_legacy_sum']:.2f}x | {r['engineering_vs_legacy_wall']:.2f}x | {r['bit_speedup']:.2f}x |")
    lines += ["", "## Peak Memory", "", "| Dataset | Variant | CPU RSS MiB | GPU used MiB | GPU free MiB | Input mode |", "|---|---|---:|---:|---:|---|"]
    for r in memory:
        lines.append(f"| {r['dataset']} | {r['variant']} | {r['cpu_peak_mib']:.1f} | {r['gpu_peak_mib']} | {r['gpu_min_free_mib']} | {r['input_mode']} |")
    lines += ["", "## Evaluator Quality", "", "| Dataset | Variant | Total | All costs match | Open / incomplete |", "|---|---|---:|---|---|"]
    for r in quality:lines.append(f"| {r['dataset']} | {r['variant']} | {r['total']} | True | {r['open_nets']} / {r['incomplete_nets']} |")
    if result.get("error"):lines += ["", "## Error", "", result["error"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--data-dir", default="../ISPD24_tmp")
    parser.add_argument("--output-dir", default="results/GRotor-flt-execution")
    parser.add_argument("--legacy-router", default="build/GRotor-flt-before/build/GRotor")
    parser.add_argument("--oracle-summary", default="results/GRotor/summary.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--optix-root", default=os.environ.get("OPTIX_ROOT_DIR"))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    if (not args.gpu.isdigit() or not args.cuda_arch.isdigit() or len(set(args.cases)) != len(args.cases)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", c) for c in args.cases)):
        parser.error("Use numeric GPU/architecture and unique simple dataset names")
    legacy = local_path(args.legacy_router)
    if not legacy.is_file():parser.error("Supply the instrumented pre-refactor executable with --legacy-router")
    folder = local_path(args.output_dir)
    if ROOT / "results" not in folder.parents:parser.error("Output must be below results/")
    folder.mkdir(parents=True, exist_ok=False)
    full_build = ROOT / "build/GRotor-flt-full"
    bit_build = ROOT / "build/GRotor-rtcore" / ("sm_" + args.cuda_arch)
    data, oracle = local_path(args.data_dir), local_path(args.oracle_summary)
    result = dict(status="running", sources=source_manifest(), script_sha256=sha256(Path(__file__)),
        gpu=args.gpu, cuda_arch=args.cuda_arch, oracle=dict(path=str(oracle), sha256=sha256(oracle)), cases=[])
    try:
        if not args.no_build:
            for label, build, cost in (("full", full_build, "ON"), ("bit", bit_build, "OFF")):
                command = ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch, "-DGROTOR_ENABLE_RTCORE=ON", "-DGROTOR_FULL_COST_REFERENCE=" + cost]
                if args.optix_root:command.append("-DOPTIX_ROOT_DIR=" + str(Path(args.optix_root).resolve()))
                for cmd, suffix in ((command, "configure"), (["cmake", "--build", str(build), "--target", "GRotor", "grotor_evaluator", "--parallel", "2"], "build")):
                    status, _ = run_logged(cmd, folder / f"{label}.{suffix}.log")
                    if status:raise RuntimeError(f"{label} {suffix} failed")
        binaries = {"legacy_full": legacy, "assembled_full": full_build / "GRotor", "assembled_bit": bit_build / "GRotor"}
        result["binaries"] = {k:dict(path=str(v), sha256=sha256(v)) for k,v in binaries.items()}
        result["evaluator_sha256"] = sha256(bit_build / "grotor_evaluator")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ["LC_ALL"] = "C"
        for name in args.cases:
            case = dict(name=name, status="running", runs={});result["cases"].append(case)
            files = [data / (name + "." + ext) for ext in ("cap", "net")]
            case["inputs"] = {str(p):sha256(p) for p in files}
            case["expected"] = saved_oracle(oracle, name, case["inputs"])
            cf = folder / name;cf.mkdir()
            with tempfile.TemporaryDirectory(prefix="flt-execution-", dir=ROOT / "build") as temporary:
                output = Path(temporary) / (name + ".out")
                for label in VARIANTS:
                    run = dict(status="running");case["runs"][label] = run;write_summary(folder, result)
                    command = [str(binaries[label]), "-cap", str(files[0]), "-net", str(files[1]), "-out", str(output),
                        "--batching", "rtcore", "--sparsity", "1", "--validate-batches", "0", "--profile-stages", "1"]
                    run["router_command"] = command
                    rss = cf / (label + ".host_memory.log")
                    wrapped = ["/usr/bin/time", "-v", "-o", str(rss), *command]
                    log = cf / (label + ".routing.log")
                    run["exit_status"], run["process_seconds"], run["gpu_memory"] = gpu_monitor(wrapped, log, cf / (label + ".gpu_memory.csv"), args.gpu)
                    if run["exit_status"] or not output.is_file() or not output.stat().st_size:raise RuntimeError(f"Routing failed: {name} {label}")
                    text = log.read_text()
                    expected_mode = "cost=incremental prefix=bit" if label == "assembled_bit" else "cost=full prefix=full"
                    if f"GROTOR_CONFIG batching=rtcore {expected_mode} memory=explicit" not in text:
                        raise RuntimeError("Unexpected cost/prefix configuration")
                    run["cpu_peak_mib"] = int(re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", rss.read_text())[1]) / 1024
                    run["batch_timings"] = parse_rtcore_timings(text, False)
                    if label != "legacy_full":
                        old = case["runs"]["legacy_full"]["batch_timings"]
                        if [(r["nets"], r["batches"]) for r in old] != [(r["nets"], r["batches"]) for r in run["batch_timings"]]:
                            raise RuntimeError("Routing batch workloads differ between variants")
                    run["dag_timings"], _ = parse_routing_timings(text, run["batch_timings"])
                    run["program_seconds"] = float(re.search(r"^GROTOR_COMPLETE wall_seconds=([\d.]+)$", text, re.M)[1])
                    phase = re.search(r"^PIPELINE_TIMING (.+)$", text, re.M)[1]
                    run["flt_phase_ms"] = float(dict(t.split("=",1) for t in phase.split())["flt_ms"])
                    if label == "legacy_full":
                        a,b,ms = re.search(r"^FLT_EXECUTION_WALL start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+)$", text, re.M).groups()
                        run["execution_ms"] = float(ms)
                        if abs(float(ms)-(int(b)-int(a))/1000) > .00001:raise RuntimeError("Invalid legacy wall interval")
                    else:
                        dag = run["dag_timings"][1];run["execution_ms"] = dag["elapsed_ms"]
                        if abs(dag["elapsed_ms"]-(dag["end_us"]-dag["start_us"])/1000) > .00001:raise RuntimeError("FLT interval is not contiguous")
                        storage = re.search(r"^FLT_STORAGE (.+)$", text, re.M)[1]
                        run["storage"] = dict(t.split("=",1) for t in storage.split())
                    evaluator = [str(bit_build / "grotor_evaluator"), *map(str, files), str(output)]
                    run["evaluator_command"] = evaluator
                    elog = cf / (label + ".evaluator.log")
                    status, run["evaluation_seconds"] = run_logged(evaluator, elog)
                    run["metrics"] = evaluator_metrics(elog, status)
                    output.unlink()
                    if run["metrics"] != case["expected"]:raise RuntimeError(f"Evaluator costs differ: {name} {label}")
                    run["status"] = "passed";write_summary(folder, result)
                    print("PASS", name, label, run["execution_ms"], run["metrics"]["total"], flush=True)
            case["status"] = "passed";write_summary(folder, result)
        result["status"] = "passed";return 0
    except BaseException as error:
        result["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        result["error"] = str(error) or type(error).__name__
        return 1
    finally:write_summary(folder, result)


if __name__ == "__main__":raise SystemExit(main())
