#!/usr/bin/env python3
"""Evaluate stage-aware RT-Core batching using each method's contiguous wall interval."""
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

from test_instantgr2 import (ROOT, BASELINE, run_logged, sha256, evaluator_metrics,
                            parse_batch_timings, parse_routing_timings, require_cuda)
from test_grotor_reference import local_path, source_manifest, validate_luts
from test_grotor_rtcore import gpu_monitor, parse_rtcore_timings

CASES = ["mempool_tile", "ariane133_51", "ariane133_68", "nvdla", "bsg_chip", "mempool_group", "mempool_cluster"]
from rtcore_reporting import parse_totals, parse_spatial_settings


def routing_comparison(case):
    dag_rows, stage_rows = [], []
    for i, algorithm in enumerate(("L-shape", "FLT")):
        old_dag = case["baseline"]["dag_timings"][i]["elapsed_ms"]
        new_dag = case["rtcore"]["dag_timings"][i]["elapsed_ms"]
        old_batch = next(b["elapsed_ms"] for b in case["baseline"]["batch_timings"]
            if b["stage"] == ("lshape", "detour")[i] and b["backend"] == "cuda_core")
        new_batch = case["rtcore"]["totals"][i]["elapsed_ms"]
        dag_rows.append(dict(dataset=case["name"], algorithm=algorithm, instantgr2_dag_ms=old_dag,
            grotor_dag_ms=new_dag, speedup=old_dag/new_dag))
        stage_rows.append(dict(dataset=case["name"], algorithm=algorithm,
            instantgr2_batch_ms=old_batch, instantgr2_dag_ms=old_dag, instantgr2_total_ms=old_batch+old_dag,
            grotor_batch_ms=new_batch, grotor_dag_ms=new_dag, grotor_total_ms=new_batch+new_dag,
            speedup=(old_batch+old_dag)/(new_batch+new_dag)))
    return dag_rows, stage_rows


def write_summary(folder, result):
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rows, quality, dag_rows, stage_rows, memory_rows = [], [], [], [], []
    for case in result["cases"]:
        if case["status"] != "passed":
            continue
        for i, (stage, algorithm) in enumerate((("lshape", "L-shape"), ("detour", "FLT"))):
            native = case["rtcore"]["totals"][i]
            base = {b["backend"]: b["elapsed_ms"] for b in case["baseline"]["batch_timings"] if b["stage"] == stage}
            rows.append(dict(dataset=case["name"], algorithm=algorithm, rtcore_method=native["method"],
                **case["rtcore"]["space"][i],
                design_nets=native["design_nets"], input_nets=native["nets"], cpu_ms=base["cpu"], cuda_core_ms=base["cuda_core"],
                timing_scope=native["timing_scope"], start_us=native["start_us"], end_us=native["end_us"],
                topology_geometry_ms=case["rtcore"]["topology_timings"][i]["elapsed_ms"],
                rtcore_ms=native["elapsed_ms"], preparation_ms=native["preparation_ms"],
                collection_ms=native["collection_ms"], complete_batch_call_ms=native["call_ms"],
                speedup_vs_cpu=base["cpu"] / native["elapsed_ms"],
                speedup_vs_cuda_core=base["cuda_core"] / native["elapsed_ms"],
                dag_solve_ms=case["rtcore"]["dag_timings"][i]["elapsed_ms"],
                algorithm_total_ms=native["elapsed_ms"] + case["rtcore"]["dag_timings"][i]["elapsed_ms"]))
        case_dag, case_stages = routing_comparison(case)
        dag_rows.extend(case_dag); stage_rows.extend(case_stages)
        memory_rows.append(dict(dataset=case["name"], cpu_peak_mib=case["rtcore"].get("cpu_peak_mib"),
            gpu_sampled_peak_mib=case["rtcore"]["gpu_memory"]["sampled_peak_used_mib"],
            gpu_sampled_min_free_mib=case["rtcore"]["gpu_memory"]["sampled_min_free_mib"]))
        quality.append(dict(dataset=case["name"], evaluator_matches=True, cpu_batch_lists_equal=True if result["configuration"].get("validate_batches", True) else "not_run",
                            **case["rtcore"]["metrics"]))
    for filename, data in (("runtime.csv", rows), ("quality.csv", quality), ("dag-runtime.csv", dag_rows), ("stage-runtime.csv", stage_rows), ("memory.csv", memory_rows)):
        if data:
            with (folder / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, list(data[0])); writer.writeheader(); writer.writerows(data)
    lines = ["# RT-Core Contiguous Wall-Time Benchmark", "", f"Status: {result['status']}", "",
        "- The pipeline caller brackets the complete generation function with one steady-clock interval and GPU synchronization. Generation functions contain no timing operations. Preparation and result collection are separate calls.",
        "- Standard RT-Core: start immediately before the coloring loop; end after every round and final GPU synchronization. Parameter uploads, counter readbacks, host control, GPU work and waits inside the interval are included.",
        "- Cached: start immediately before pipeline preparation; end after all chunks, error readback and final GPU synchronization. Pipeline/SBT setup, coloring-buffer allocation/initialization, transfers and all chunk control/work inside the interval are included.",
        "- Both scopes exclude earlier geometry/ray preparation and BVH construction, and subsequent final batch-ID download, CPU batch-list assembly and cleanup. Standard RT-Core also excludes pipeline preparation and inactive warm-up.",
        "- Large-design partitioned batching compiles/warms a shared pipeline before one continuous call interval containing bitmap initialization, geometry copies, BVH construction, all coloring work, occupancy publication and collection. L-shape uses rounds; FLT uses cached.",
        "- CPU/CUDA-Core retain their existing complete-function timers in the sparsity-1 dual baseline; its CPU plan feeds routing. These scope-specific ratios are not whole-router speedups.",
        "- Smaller designs use standard L-shape/cached FLT batching. Large designs use partitioned batching with bounded local-priority scenes and a prior-scene occupancy bitmap.",
        "- RT-Core milliseconds equal (end_us - start_us) / 1000 in both native logs and this table. The caller simply forwards the selected method's returned duration.",
        "- Speedup = CPU or CUDA-Core milliseconds / RT-Core milliseconds. Values below 1 indicate a slowdown; ratios are not constrained to a target.",
        "- These are single sequential runs with unlocked GPU clocks. CPU batch-list checks are enabled only when --validate-batches is supplied; the recorded configuration identifies this choice.",
        "- Every new-router routing file is read by the external evaluator and deleted before the next run. Optional saved-router A/B outputs are deleted before the new run without re-evaluating them. CPU validation and evaluation are outside the measured intervals. Memory is sampled device-wide every 100 ms.", ""]
    if result.get("baseline_summary"):
        lines += ["- CPU/CUDA-Core controls are reused from the validated summary referenced in summary.json; RT-Core is freshly run. Inputs, baseline sources/binaries and GPU configuration must match.", ""]
    for algorithm in ("L-shape", "FLT"):
        lines += ["## " + algorithm, "", "| Dataset | RT method | CPU ms | CUDA-Core ms | RT-Core reported ms | CPU / RT | CUDA / RT |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["algorithm"] == algorithm:
                lines.append(f"| {row['dataset']} | {row['rtcore_method']} | {row['cpu_ms']:.3f} | {row['cuda_core_ms']:.3f} | {row['rtcore_ms']:.3f} | {row['speedup_vs_cpu']:.2f}x | {row['speedup_vs_cuda_core']:.2f}x |")
        lines.append("")
    lines += ["## Routing DAG Runtime", "",
        "GRotor uses warp-per-DAG L-shape solving, block-per-DAG FLT solving and incremental cost/BIT maintenance. Its timers cover one continuous execution interval. FLT includes compact uploads, GPU assembly, staging waits, cost/prefix updates and demand commits. CPU topology construction/planning and workspace allocation precede this interval. InstantGR2 retains the full L-shape loop and the sum of FLT GPU suffix intervals, excluding per-batch CPU packing. These scope differences are deliberate and are not whole-phase wall-time comparisons.", "",
        "| Dataset | Algorithm | InstantGR2 DAG ms | GRotor DAG ms | Speedup |", "|---|---|---:|---:|---:|"]
    for row in dag_rows:
        lines.append(f"| {row['dataset']} | {row['algorithm']} | {row['instantgr2_dag_ms']:.3f} | {row['grotor_dag_ms']:.3f} | {row['speedup']:.2f}x |")
    lines += ["", "## Batching Plus DAG Runtime", "",
        "Algorithm total = measured batching + measured DAG execution. InstantGR2 uses CUDA-Core batching time, with the deterministic CPU plan used for routing in dual mode. GRotor uses standard L-shape/cached FLT batching or the partitioned large-design path. These totals exclude earlier preparation and file output; complete phase and router times are separately recorded when available.", "",
        "| Dataset | Algorithm | InstantGR2 batching ms | InstantGR2 total ms | GRotor batching ms | GRotor total ms | Speedup |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in stage_rows:
        lines.append(f"| {row['dataset']} | {row['algorithm']} | {row['instantgr2_batch_ms']:.3f} | {row['instantgr2_total_ms']:.3f} | {row['grotor_batch_ms']:.3f} | {row['grotor_total_ms']:.3f} | {row['speedup']:.2f}x |")
    lines.append("")
    lines += ["## Quality and Memory", "", "| Dataset | Evaluator total | Exact CPU batches | Costs match | Peak used MiB | Min free MiB |", "|---|---:|---|---|---:|---:|"]
    for case in result["cases"]:
        if case["status"] == "passed":
            rt = case["rtcore"]; mem = rt["gpu_memory"]
            lines.append(f"| {case['name']} | {rt['metrics']['total']} | {'True' if result['configuration'].get('validate_batches', True) else 'Not run'} | True | {mem['sampled_peak_used_mib']} | {mem['sampled_min_free_mib']} |")
    dag_rows = []
    for case in result["cases"]:
        if case["status"] != "passed" or "before" not in case:continue
        for algorithm, old, new in zip(("L-shape", "FLT"), case["before"]["dag_timings"], case["rtcore"]["dag_timings"]):
            dag_rows.append(dict(dataset=case["name"], algorithm=algorithm,
                before_ms=old["elapsed_ms"], after_ms=new["elapsed_ms"], speedup=old["elapsed_ms"]/new["elapsed_ms"]))
    if dag_rows:
        with (folder / "dag-comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(dag_rows[0]));writer.writeheader();writer.writerows(dag_rows)
        lines += ["", "## Routing DAG Before/After", "",
            "L-shape covers the complete batch loop. Current GRotor FLT uses one continuous interval including topology upload, GPU assembly, cost/prefix refresh and demand commits. CPU topology construction/planning and allocation precede it. An archived before-binary may retain the older suffix-sum scope; inspect its provenance. CPU batch validation is outside these intervals.", "",
            "| Dataset | Algorithm | Before ms | After ms | Before / after |", "|---|---|---:|---:|---:|"]
        for row in dag_rows:
            lines.append(f"| {row['dataset']} | {row['algorithm']} | {row['before_ms']:.3f} | {row['after_ms']:.3f} | {row['speedup']:.2f}x |")
    end_to_end = []
    for case in result["cases"]:
        if case["status"] != "passed" or "before" not in case:continue
        old, new = case["before"], case["rtcore"]
        end_to_end.append(dict(dataset=case["name"], before_seconds=old["program_seconds"],
            after_seconds=new["program_seconds"], speedup=old["program_seconds"]/new["program_seconds"],
            before_peak_mib=old["gpu_memory"]["sampled_peak_used_mib"],
            after_peak_mib=new["gpu_memory"]["sampled_peak_used_mib"]))
    if end_to_end:
        with (folder / "end_to_end.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, list(end_to_end[0]));w.writeheader();w.writerows(end_to_end)
        lines += ["", "## Complete Router Wall Time", "",
            "Before/after are independent complete routing runs including input, topology/geometry, preparation, optional CPU batch validation, routing, downloads and output. Preparation moved outside the generation interval remains included here. Old outputs are removed before the new run; only the new output is evaluated in this A/B run, against the already evaluated baseline costs.", "",
            "| Dataset | Before s | After s | Before / after | Before peak MiB | After peak MiB |",
            "|---|---:|---:|---:|---:|---:|"]
        for r in end_to_end:
            lines.append(f"| {r['dataset']} | {r['before_seconds']:.3f} | {r['after_seconds']:.3f} | {r['speedup']:.3f}x | {r['before_peak_mib']} | {r['after_peak_mib']} |")
    if result.get("error"):
        lines += ["", "## Error", "", result["error"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def prepare_output_folder(folder, clean=False, protected=()):
    if ROOT / "results" not in folder.parents:
        raise ValueError("Output must be below this repository's results/")
    if any(path == folder or folder in path.parents for path in protected):
        raise ValueError("Output must be separate from the input report directory")
    if folder.exists():
        if not folder.is_dir():
            raise ValueError("Output path is not a directory")
        if not clean:
            raise ValueError("Output exists; select another --output-dir or use --clean-results")
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def main(argv=None, *, report_writer=write_summary,
         default_output_dir="results/GRotor-rtcore-contiguous", description=__doc__, entry_script=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--data-dir", default="../ISPD24_tmp")
    parser.add_argument("--output-dir", default=default_output_dir, help="Repository-relative report directory below results/")
    parser.add_argument("--before-router", help="Measure a saved pre-change router on each input before the new router" if entry_script is None else argparse.SUPPRESS)
    parser.add_argument("--baseline-summary", help="Reuse validated baseline runs with matching inputs, source hashes and hardware" if entry_script is None else argparse.SUPPRESS)
    parser.add_argument("--validate-batches", action="store_true", help="Enable CPU batch checks; disabled for performance measurements")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--optix-root", default=os.environ.get("OPTIX_ROOT_DIR"))
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"), help="CUDA compiler name or path")
    parser.add_argument("--jobs", type=int, default=2, help="Parallel compilation jobs (default: 2)")
    parser.add_argument("--no-build", action="store_true", help="Use the existing Release RT-Core build for this CUDA architecture")
    parser.add_argument("--clean-results", action="store_true", help="Replace the selected report directory below results/")
    parser.add_argument("--report-only", metavar="SUMMARY_JSON", help="Generate tables from saved measurements without building or routing")
    args = parser.parse_args(argv)
    if (args.jobs < 1 or not args.gpu.isdigit() or not args.cuda_arch.isdigit()
            or len(set(args.cases)) != len(args.cases) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in args.cases)):
        parser.error("Use positive jobs, numeric GPU/architecture and unique simple case names")
    folder = local_path(args.output_dir)
    report_source = local_path(args.report_only) if args.report_only else None
    try:
        prepare_output_folder(folder, args.clean_results, (report_source,) if report_source else ())
        if report_source:
            result = json.loads(report_source.read_text())
            result["report_generation"] = dict(source=str(report_source), source_sha256=sha256(report_source),
                reporter_sha256=sha256(Path(entry_script or __file__)))
            report_writer(folder, result)
            return 0 if result["status"] == "passed" else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))
    data = local_path(args.data_dir); build = ROOT / "build/GRotor-rtcore" / ("sm_" + args.cuda_arch)
    result = dict(status="running", configuration=dict(method_policy="standard_lshape_cached_flt_partitioned_large", geometry_build="topology_fused_2d",
        sparsity=1, gpu=args.gpu, cuda_arch=args.cuda_arch, timing="contiguous_wall", sources=source_manifest(),
        validate_batches=args.validate_batches, cost_update="incremental", prefix="bit",
        dag_lshape="warp", dag_flt="block", flt_execution="contiguous",
        space="layered", point_model="triangle", z_gap=2**-16, margin=.1,
        script_sha256=sha256(Path(__file__)),
        reporting_sha256=sha256(ROOT / "script/rtcore_reporting.py"),
        rtcore_runner_sha256=sha256(ROOT / "script/test_grotor_rtcore.py")), cases=[])
    if entry_script:
        result["configuration"]["entry_script"] = dict(path=str(Path(entry_script).resolve()), sha256=sha256(Path(entry_script)))
    result["configuration"]["build_performed"] = not args.no_build
    result["configuration"]["build_directory"] = str(build)
    try:
        validate_luts()
        config = ["cmake", "-S", str(ROOT), "-B", str(build), "-DGROTOR_ENABLE_RTCORE=ON", "-DGROTOR_FULL_COST_REFERENCE=OFF", "-DCMAKE_BUILD_TYPE=Release",
                  "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch]
        if args.optix_root:
            config.append("-DOPTIX_ROOT_DIR=" + str(Path(args.optix_root).resolve()))
        if not args.no_build:
            nvcc = shutil.which(args.nvcc)
            if not nvcc:raise RuntimeError("nvcc not found; put CUDA bin on PATH or pass --nvcc")
            config.append("-DCMAKE_CUDA_COMPILER=" + str(Path(nvcc).resolve()))
            commands = ((config, "configure.log"), (["cmake", "--build", str(build), "--target",
                "GRotor", "instantgr2_reference", "grotor_evaluator", "--parallel", str(args.jobs)], "build.log"))
            result["build_commands"] = [c for c, _ in commands]
            for command, name in commands:
                status, _ = run_logged(command, folder / name)
                if status:raise RuntimeError("Build failed: " + name)
        cache_path = build / "CMakeCache.txt"
        cache = dict((line.split("=",1)[0].split(":",1)[0], line.split("=",1)[1])
            for line in cache_path.read_text().splitlines() if "=" in line and not line.startswith(("#", "//")))
        for key, expected in (("CMAKE_BUILD_TYPE", "Release"), ("GROTOR_ENABLE_RTCORE", "ON"),
                              ("GROTOR_FULL_COST_REFERENCE", "OFF"), ("CMAKE_CUDA_ARCHITECTURES", args.cuda_arch)):
            if cache.get(key) != expected:raise RuntimeError(f"Build requires {key}={expected}; rerun without --no-build")
        result["configuration"]["cmake_cache_sha256"] = sha256(cache_path)
        result["binary_sha256"] = {n: sha256(build / n) for n in ("GRotor", "instantgr2_reference", "grotor_evaluator", "rtcore_batches.ptx", "rtcore_cached.ptx")}
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu; os.environ["LC_ALL"] = "C"; require_cuda()
        result["gpu_details"] = subprocess.check_output(["nvidia-smi", "--id=" + args.gpu,
            "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True)
        baseline_summary = None
        if args.baseline_summary:
            bp = local_path(args.baseline_summary)
            baseline_summary = json.loads(bp.read_text())
            cfg = baseline_summary["configuration"]
            if (baseline_summary["status"] != "passed" or cfg["sparsity"] != 1
                    or cfg["gpu"] != args.gpu or cfg["cuda_arch"] != args.cuda_arch
                    or baseline_summary["gpu_details"] != result["gpu_details"]
                    or baseline_summary["binary_sha256"]["grotor_evaluator"] != result["binary_sha256"]["grotor_evaluator"]
                    or baseline_summary["binary_sha256"]["instantgr2_reference"] != result["binary_sha256"]["instantgr2_reference"]):
                raise RuntimeError("Baseline summary is not a compatible evaluated dual run")
            current = result["configuration"]["sources"]
            for key, value in current.items():
                if key.startswith(("baseline/InstantGR2/", "resources/flute/")) and cfg["sources"].get(key) != value:
                    raise RuntimeError("Baseline source or lookup-table hash changed")
            result["baseline_summary"] = dict(path=str(bp), sha256=sha256(bp))
        before_router = local_path(args.before_router) if args.before_router else None
        if before_router:
            result["before_router"] = dict(path=str(before_router), sha256=sha256(before_router))
            for name in ("rtcore_batches.ptx", "rtcore_cached.ptx"):
                saved_ptx = before_router.parent / name
                if not saved_ptx.is_file() or sha256(saved_ptx) != sha256(build / name):
                    raise RuntimeError("Saved router requires matching device PTX; use an isolated old build")
        for name in args.cases:
            case = dict(name=name, status="running"); result["cases"].append(case); report_writer(folder, result)
            cf = folder / name; cf.mkdir()
            inputs = [data / (name + "." + ext) for ext in ("cap", "net")]
            case["inputs"] = {str(p):sha256(p) for p in inputs}
            with tempfile.TemporaryDirectory(prefix="hybrid-evaluation-", dir=build) as temporary:
                output = Path(temporary) / (name + ".out")
                if baseline_summary:
                    candidates = [c for c in baseline_summary["cases"] if c["name"] == name and c["status"] == "passed"]
                    if len(candidates) != 1 or candidates[0]["inputs"] != case["inputs"]:
                        raise RuntimeError("Baseline dataset hashes do not match")
                    source_case = candidates[0]
                    metrics = source_case["baseline"]["metrics"]
                    if metrics["open_nets"] or metrics["incomplete_nets"]:
                        raise RuntimeError("Baseline connectivity is invalid")
                    case["baseline"] = source_case["baseline"]
                    copied = cf / "baseline.evaluator.log"
                    original = local_path(args.baseline_summary).parent / name / "baseline.evaluator.log"
                    copied.write_text(original.read_text())
                if before_router:
                    old = {}; case["before"] = old
                    command = [str(before_router), "-cap", str(inputs[0]), "-net", str(inputs[1]), "-out", str(output),
                               "--sparsity", "1", "--batching", "rtcore", "--validate-batches", "1" if args.validate_batches else "0"]
                    old["router_command"] = command
                    old["router_exit_status"], old["process_seconds"], old["gpu_memory"] = gpu_monitor(command,
                        cf / "before.routing.log", cf / "before.gpu_memory.csv", args.gpu)
                    if old["router_exit_status"] or not output.is_file() or not output.stat().st_size:
                        raise RuntimeError("Saved router failed")
                    old_text = (cf / "before.routing.log").read_text()
                    old["batch_timings"] = parse_rtcore_timings(old_text, args.validate_batches)
                    old["dag_timings"], _ = parse_routing_timings(old_text, old["batch_timings"])
                    complete = re.findall(r"^GROTOR_COMPLETE wall_seconds=([\d.]+)$", old_text, re.MULTILINE)
                    if len(complete) != 1:raise RuntimeError("Missing saved-router wall time")
                    old["program_seconds"] = float(complete[0])
                    old["evaluated_in_this_run"] = False
                    output.unlink()
                for implementation in (("rtcore",) if baseline_summary else ("baseline", "rtcore")):
                    run = {}; case[implementation] = run
                    command = [str(build / ("instantgr2_reference" if implementation == "baseline" else "GRotor")),
                               "-cap", str(inputs[0]), "-net", str(inputs[1]), "-out", str(output), "--sparsity", "1",
                               "--batching", "dual" if implementation == "baseline" else "rtcore"]
                    if implementation == "rtcore":
                        command += ["--validate-batches", "1" if args.validate_batches else "0", "--profile-stages", "1"]
                    run["router_command"] = command
                    log = cf / (implementation + ".routing.log")
                    if implementation == "baseline":
                        run["router_exit_status"], run["router_seconds"] = run_logged(command, log, cwd=BASELINE / "run")
                    else:
                        rss_log = cf / "rtcore.host_memory.log"
                        run["router_exit_status"], run["router_seconds"], run["gpu_memory"] = gpu_monitor(
                            ["/usr/bin/time", "-v", "-o", str(rss_log), *command], log, cf / "rtcore.gpu_memory.csv", args.gpu)
                        run["cpu_peak_mib"] = int(re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", rss_log.read_text())[1])/1024
                    if run["router_exit_status"] or not output.is_file() or not output.stat().st_size:
                        raise RuntimeError("Routing failed: " + str(log))
                    text = log.read_text()
                    run["batch_timings"] = (parse_batch_timings(text, "dual", "cpu") if implementation == "baseline"
                                            else parse_rtcore_timings(text, args.validate_batches))
                    run["dag_timings"], _ = parse_routing_timings(text, run["batch_timings"])
                    if implementation == "rtcore":
                        if "GROTOR_CONFIG batching=rtcore cost=incremental prefix=bit memory=explicit" not in text:
                            raise RuntimeError("Expected incremental costs and BIT maintenance")
                        if "DAG_SCHEDULING lshape=warp flt=block" not in text:
                            raise RuntimeError("Unexpected DAG scheduling configuration")
                        for timing in run["dag_timings"]:
                            if abs(timing["elapsed_ms"]-(timing["end_us"]-timing["start_us"])/1000) > 1e-5:
                                raise RuntimeError("GRotor DAG timers must be contiguous")
                        phases = re.search(r"^PIPELINE_TIMING (.+)$", text, re.MULTILINE)
                        if phases:run["pipeline_ms"] = {k:float(v) for k,v in (t.split("=",1) for t in phases[1].split())}
                        complete = re.findall(r"^GROTOR_COMPLETE wall_seconds=([\d.]+)$", text, re.MULTILINE)
                        if len(complete) != 1:raise RuntimeError("Missing complete-router wall time")
                        run["program_seconds"] = float(complete[0])
                        run["totals"] = parse_totals(text, run["batch_timings"])
                        run["space"] = parse_spatial_settings(text)
                        topology = re.findall(r"^TOPOLOGY_TIMING stage=(lshape|flt) elapsed_ms=([\d.]+) geometry_fused=(1)$", text, re.MULTILINE)
                        if [r[0] for r in topology] != ["lshape", "flt"]:raise RuntimeError("Missing fused topology timing")
                        run["topology_timings"] = [dict(stage=stage, elapsed_ms=float(ms)) for stage,ms,_ in topology]
                    evaluator = [str(build / "grotor_evaluator"), *map(str, inputs), str(output)]
                    run["evaluator_command"] = evaluator; elog = cf / (implementation + ".evaluator.log")
                    run["evaluator_exit_status"], run["evaluator_seconds"] = run_logged(evaluator, elog)
                    run["metrics"] = evaluator_metrics(elog, run["evaluator_exit_status"])
                    output.unlink()
                    if implementation == "rtcore":
                        for index, stage in enumerate(("lshape", "detour")):
                            control = next(t for t in case["baseline"]["batch_timings"] if t["stage"] == stage and t["backend"] == "cpu")
                            native = run["batch_timings"][index]
                            if (control["nets"], control["batches"]) != (native["nets"], native["batches"]):
                                raise RuntimeError("Baseline and GRotor batch workloads differ")
                    if implementation == "rtcore" and run["metrics"] != case["baseline"]["metrics"]:
                        raise RuntimeError("Evaluator costs differ: " + name)
            case["status"] = "passed"; report_writer(folder, result)
            print("PASS", name, case["rtcore"]["metrics"]["total"], flush=True)
        result["status"] = "passed"
        return 0
    except BaseException as error:
        result["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        result["error"] = str(error) or type(error).__name__
        if result["cases"] and result["cases"][-1]["status"] == "running":result["cases"][-1]["status"] = result["status"]
        return 1
    finally:
        report_writer(folder, result)


if __name__ == "__main__":
    raise SystemExit(main())
