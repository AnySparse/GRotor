#!/usr/bin/env python3
"""Compare CPU, CUDA-Core and both RT-Core FLT batching implementations."""
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
                            parse_batch_timings, require_cuda)
from test_grotor_reference import local_path, source_manifest, validate_luts
from test_grotor_rtcore import gpu_monitor, parse_rtcore_timings
from rtcore_reporting import parse_spatial_settings

CASES = ["mempool_tile", "ariane133_51", "ariane133_68", "nvdla", "bsg_chip", "mempool_group", "mempool_cluster"]

COMPARISON = r'''
        auto batch_standard = grotor::batching::generate_batches_rtcore(state, nets2route, 300000);
        printf("FLT_VARIANT method=rounds scope=%s elapsed_ms=%.6f start_us=%lld end_us=%lld nets=%zu batches=%zu\n",
               batch_standard.timing_scope, batch_standard.milliseconds, batch_standard.start_us, batch_standard.end_us, nets2route.size(), batch_standard.batches.size());
        auto batch_run = grotor::batching::generate_batches_rtcore_cached(state, nets2route, 300000);
        printf("FLT_VARIANT method=cached scope=%s elapsed_ms=%.6f start_us=%lld end_us=%lld nets=%zu batches=%zu\n",
               batch_run.timing_scope, batch_run.milliseconds, batch_run.start_us, batch_run.end_us, nets2route.size(), batch_run.batches.size());
        const bool equal = batch_standard.batches == batch_run.batches;
        printf("FLT_PLAN_CHECK exact_lists_equal=%d routing_method=cached\n", int(equal));
        if (!equal) throw std::runtime_error("Cached and RT-Core FLT lists differ");
'''



def prepare_worker(folder):
    source = folder / "source"
    source.mkdir(parents=True)
    for name in ("src", "include"):
        shutil.copytree(ROOT / name, source / name)
    shutil.copy2(ROOT / "CMakeLists.txt", source / "CMakeLists.txt")
    for name in ("baseline", "resources", "third_party", "script"):
        (source / name).symlink_to(ROOT / name, target_is_directory=True)
    dispatch = source / "src/routing/flt_thread.cuh"
    code = dispatch.read_text()
    anchor = "        auto batch_run = grotor::batching::generate_batches_rtcore_cached(state, nets2route, 300000);"
    if code.count(anchor) != 1:raise RuntimeError("Cannot locate FLT batching call")
    dispatch.write_text(code.replace(anchor, COMPARISON, 1))
    files = [source / "CMakeLists.txt"]
    for name in ("src", "include"):
        files += [p for p in (source / name).rglob("*") if p.is_file()]
    return source, {str(p.relative_to(ROOT)): sha256(p) for p in files}


def parse_comparison(text, routed):
    records = re.findall(r"^FLT_VARIANT (.+)$", text, re.MULTILINE)
    if len(records) != 2 or re.findall(r"^FLT_PLAN_CHECK exact_lists_equal=(\d+) routing_method=(\w+)$", text, re.MULTILINE) != [("1", "cached")]:
        raise RuntimeError("Missing or failed FLT plan comparison")
    rows = []
    for expected, record in zip(("rounds", "cached"), records):
        f = dict(token.split("=", 1) for token in record.split())
        row = dict(method=f["method"], scope=f["scope"], elapsed_ms=float(f["elapsed_ms"]),
                   start_us=int(f["start_us"]), end_us=int(f["end_us"]), nets=int(f["nets"]), batches=int(f["batches"]))
        scope = "coloring_loop_wall" if expected == "rounds" else "pipeline_to_coloring_wall"
        if (row["method"] != expected or row["scope"] != scope or not math.isfinite(row["elapsed_ms"])
                or row["elapsed_ms"] <= 0 or abs(row["elapsed_ms"] - (row["end_us"]-row["start_us"])/1000) > .001
                or (row["nets"], row["batches"]) != (routed["nets"], routed["batches"])):
            raise RuntimeError("Invalid FLT variant timing, scope or workload")
        rows.append(row)
    if rows[0]["end_us"] > rows[1]["start_us"] or abs(rows[1]["elapsed_ms"]-routed["elapsed_ms"]) > .001:
        raise RuntimeError("FLT variants overlap or routing did not consume cached batches")
    return rows


def write_summary(folder, result):
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    rows, quality = [], []
    for c in result["cases"]:
        if c["status"] != "passed":continue
        baseline = {t["backend"]:t["elapsed_ms"] for t in c["baseline"]["batch_timings"] if t["stage"] == "detour"}
        variants = {t["method"]:t for t in c["rtcore"]["variants"]}
        cached, standard = variants["cached"], variants["rounds"]
        rows.append(dict(dataset=c["name"], algorithm="FLT", cpu_ms=baseline["cpu"], cuda_core_ms=baseline["cuda_core"],
            rtcore_cached_ms=cached["elapsed_ms"], rtcore_ms=standard["elapsed_ms"],
            cached_vs_cpu=baseline["cpu"]/cached["elapsed_ms"], cached_vs_cuda=baseline["cuda_core"]/cached["elapsed_ms"],
            standard_vs_cpu=baseline["cpu"]/standard["elapsed_ms"], standard_vs_cuda=baseline["cuda_core"]/standard["elapsed_ms"],
            cached_scope=cached["scope"], standard_scope=standard["scope"],
            cached_start_us=cached["start_us"], cached_end_us=cached["end_us"],
            standard_start_us=standard["start_us"], standard_end_us=standard["end_us"],
            nets=cached["nets"], batches=cached["batches"],
            peak_used_mib=c["rtcore"]["gpu_memory"]["sampled_peak_used_mib"],
            min_free_mib=c["rtcore"]["gpu_memory"]["sampled_min_free_mib"]))
        quality.append(dict(dataset=c["name"], cached_equals_standard_equals_cpu=True, evaluator_matches=True, **c["rtcore"]["metrics"]))
    for name, data in (("runtime.csv", rows), ("quality.csv", quality)):
        if data:
            with (folder / name).open("w", newline="") as f:
                w = csv.DictWriter(f, list(data[0]));w.writeheader();w.writerows(data)
    lines = ["# FLT Batching Comparison", "", f"Status: {result['status']}", "",
        "- Every design uses the same fixed geometry: layered, 2^-16 spacing, single point triangle, padding 0.1, sparsity 1.",
        "- One routing process measures standard then cached FLT batching on the identical work set. GPU resources from the first call are released before the second call. Cached batches feed the actual routing DAG solver.",
        "- The caller brackets each complete generation function with one contiguous steady-clock interval. There are no per-round or postprocessed component sums.",
        "- Standard RT-Core scope: coloring loop through GPU completion, including host control, transfers and waits. Cached scope: pipeline preparation through coloring GPU completion, including allocation/initialization and chunk control/transfers. Both exclude geometry/BVH construction and final batch download/assembly/cleanup.",
        "- CPU/CUDA-Core retain original complete-function timers. Ratios are scope-specific batching comparisons, not whole-router speedups. These are single sequential runs with unlocked GPU clocks.",
        "- Complete cached and standard lists must match; the actual routing lists must also match CPU. One actual .out per design is evaluated and deleted. No routing files are compared byte-for-byte.",
        "- Memory is sampled device-wide every 100 ms across the complete routing process, not separately per FLT variant. The dependency graph is chunked: at most 16,384 nets, 32 MiB dependency bits and 2 MiB color masks per chunk; no full-design conflict matrix is retained.", ""]
    if result.get("baseline_summary"):
        lines += ["CPU/CUDA controls are reused from the hash-referenced baseline after matching inputs, source/LUT/binary hashes and GPU configuration. Both RT-Core variants are freshly measured.", ""]
    lines += ["## Runtime (ms)", "", "| Dataset | CPU | CUDA-Core | RT cached | RT standard |", "|---|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['dataset']} | {r['cpu_ms']:.3f} | {r['cuda_core_ms']:.3f} | {r['rtcore_cached_ms']:.3f} | {r['rtcore_ms']:.3f} |")
    lines += ["", "## Speedup", "", "| Dataset | CPU / cached | CUDA / cached | CPU / standard | CUDA / standard |", "|---|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['dataset']} | {r['cached_vs_cpu']:.2f}x | {r['cached_vs_cuda']:.2f}x | {r['standard_vs_cpu']:.2f}x | {r['standard_vs_cuda']:.2f}x |")
    lines += ["", "Speedup = baseline milliseconds / RT-Core milliseconds; below 1 means slower.", "", "## Quality and Memory", "", "| Dataset | Evaluator total | Lists/costs match | Peak MiB | Min free MiB |", "|---|---:|---|---:|---:|"]
    for c,r in zip((c for c in result["cases"] if c["status"] == "passed"), rows):
        lines.append(f"| {c['name']} | {c['rtcore']['metrics']['total']} | True | {r['peak_used_mib']} | {r['min_free_mib']} |")
    if result.get("error"):lines += ["", "## Error", "", result["error"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--data-dir", default="../ISPD24_tmp")
    parser.add_argument("--output-dir", default="results/GRotor-flt-batching")
    parser.add_argument("--baseline-summary", help="Reuse compatible evaluated CPU/CUDA dual controls")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--optix-root", default=os.environ.get("OPTIX_ROOT_DIR"))
    args = parser.parse_args()
    if (not args.gpu.isdigit() or not args.cuda_arch.isdigit() or len(set(args.cases)) != len(args.cases)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in args.cases)):
        parser.error("Use numeric GPU/architecture and unique simple dataset names")
    folder = local_path(args.output_dir)
    if ROOT / "results" not in folder.parents:parser.error("Output must be below results/")
    folder.mkdir(parents=True, exist_ok=False)
    work = ROOT / "build/GRotor-flt-compare" / ("sm_"+args.cuda_arch) / folder.name
    native = ROOT / "build/GRotor-rtcore" / ("sm_"+args.cuda_arch)
    result = dict(status="running", configuration=dict(sparsity=1, gpu=args.gpu, cuda_arch=args.cuda_arch,
        geometry="fixed-layered-point-triangle", timing="contiguous_wall", routing_flt="cached",
        sources=source_manifest(), script_sha256=sha256(Path(__file__))), cases=[])
    try:
        validate_luts()
        source, hashes = prepare_worker(work)
        result["comparison_sources"] = hashes
        binary = work / "bin"
        nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
        if not nvcc:raise RuntimeError("nvcc is required")
        common = ["-DGROTOR_ENABLE_RTCORE=ON", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES="+args.cuda_arch,
                  "-DCMAKE_CUDA_COMPILER="+str(Path(nvcc).resolve())]
        if args.optix_root:common += ["-DOPTIX_ROOT_DIR="+str(Path(args.optix_root).resolve())]
        commands = [( ["cmake", "-S", str(ROOT), "-B", str(native), *common], "native-configure.log"),
                    ( ["cmake", "--build", str(native), "--target", "instantgr2_reference", "grotor_evaluator", "--parallel", "2"], "native-build.log"),
                    ( ["cmake", "-S", str(source), "-B", str(binary), *common], "compare-configure.log"),
                    ( ["cmake", "--build", str(binary), "--target", "GRotor", "--parallel", "2"], "compare-build.log")]
        result["build_commands"] = [c for c,_ in commands]
        for command,name in commands:
            status,_ = run_logged(command, folder/name)
            if status:raise RuntimeError("Build failed: "+name)
        result["binary_sha256"] = {"comparison_router":sha256(binary/"GRotor"),
            **{n:sha256(native/n) for n in ("instantgr2_reference", "grotor_evaluator")},
            **{n:sha256(binary/n) for n in ("rtcore_batches.ptx", "rtcore_cached.ptx")}}
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu;require_cuda()
        result["gpu_details"] = subprocess.check_output(["nvidia-smi", "--id="+args.gpu,
            "--query-gpu=name,memory.total,driver_version", "--format=csv"], text=True)
        saved = None
        if args.baseline_summary:
            bp=local_path(args.baseline_summary);saved=json.loads(bp.read_text());cfg=saved["configuration"]
            if (saved["status"] != "passed" or cfg["sparsity"] != 1 or cfg["gpu"] != args.gpu
                    or cfg["cuda_arch"] != args.cuda_arch or saved["gpu_details"] != result["gpu_details"]
                    or any(saved["binary_sha256"][n] != result["binary_sha256"][n] for n in ("instantgr2_reference","grotor_evaluator"))
                    or any(cfg["sources"].get(k) != v for k,v in result["configuration"]["sources"].items()
                           if k.startswith(("baseline/InstantGR2/", "resources/flute/")))):
                raise RuntimeError("Incompatible CPU/CUDA baseline")
            result["baseline_summary"] = dict(path=str(bp), sha256=sha256(bp))
        for name in args.cases:
            case=dict(name=name,status="running");result["cases"].append(case);write_summary(folder,result)
            cf=folder/name;cf.mkdir()
            inputs=[local_path(args.data_dir)/(name+"."+ext) for ext in ("cap","net")]
            case["inputs"]={str(p):sha256(p) for p in inputs}
            with tempfile.TemporaryDirectory(prefix="evaluation-",dir=work) as temporary:
                output=Path(temporary)/(name+".out")
                if saved:
                    matched=[c for c in saved["cases"] if c["name"]==name and c["status"]=="passed" and c["inputs"]==case["inputs"]]
                    if len(matched)!=1:raise RuntimeError("Baseline input mismatch")
                    case["baseline"]=matched[0]["baseline"]
                    shutil.copy2(local_path(args.baseline_summary).parent/name/"baseline.evaluator.log",cf/"baseline.evaluator.log")
                else:
                    run={};case["baseline"]=run
                    command=[str(native/"instantgr2_reference"),"-cap",str(inputs[0]),"-net",str(inputs[1]),"-out",str(output),"--sparsity","1","--batching","dual"]
                    run["router_command"]=command
                    run["router_exit_status"],run["router_seconds"]=run_logged(command,cf/"baseline.routing.log",cwd=BASELINE/"run")
                    if run["router_exit_status"] or not output.is_file() or not output.stat().st_size:raise RuntimeError("Baseline routing failed")
                    run["batch_timings"]=parse_batch_timings((cf/"baseline.routing.log").read_text(),"dual","cpu")
                    evaluator=[str(native/"grotor_evaluator"),*map(str,inputs),str(output)];run["evaluator_command"]=evaluator
                    status,_=run_logged(evaluator,cf/"baseline.evaluator.log")
                    run["metrics"]=evaluator_metrics(cf/"baseline.evaluator.log",status);output.unlink()
                run={};case["rtcore"]=run
                command=[str(binary/"GRotor"),"-cap",str(inputs[0]),"-net",str(inputs[1]),"-out",str(output),"--sparsity","1","--batching","rtcore","--validate-batches","1"]
                run["router_command"]=command
                run["router_exit_status"],run["router_seconds"],run["gpu_memory"]=gpu_monitor(command,cf/"rtcore.routing.log",cf/"rtcore.gpu_memory.csv",args.gpu)
                if run["router_exit_status"] or not output.is_file() or not output.stat().st_size:raise RuntimeError("Comparison routing failed")
                text=(cf/"rtcore.routing.log").read_text();run["batch_timings"]=parse_rtcore_timings(text)
                run["variants"]=parse_comparison(text,run["batch_timings"][1]);run["space"]=parse_spatial_settings(text)
                for stage,batch in zip(("lshape","detour"),run["batch_timings"]):
                    control=next(t for t in case["baseline"]["batch_timings"] if t["stage"]==stage and t["backend"]=="cpu")
                    if (control["nets"],control["batches"])!=(batch["nets"],batch["batches"]):raise RuntimeError("Baseline workloads differ")
                evaluator=[str(native/"grotor_evaluator"),*map(str,inputs),str(output)];run["evaluator_command"]=evaluator
                run["evaluator_exit_status"],run["evaluator_seconds"]=run_logged(evaluator,cf/"rtcore.evaluator.log")
                run["metrics"]=evaluator_metrics(cf/"rtcore.evaluator.log",run["evaluator_exit_status"]);output.unlink()
                if run["metrics"]!=case["baseline"]["metrics"]:raise RuntimeError("Evaluator costs differ")
            case["status"]="passed";write_summary(folder,result)
            print("PASS",name,"FLT",{r["method"]:r["elapsed_ms"] for r in run["variants"]},flush=True)
        result["status"]="passed";return 0
    except BaseException as error:
        result["status"]="interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
        result["error"]=str(error) or type(error).__name__
        if result["cases"] and result["cases"][-1]["status"]=="running":result["cases"][-1]["status"]=result["status"]
        return 1
    finally:write_summary(folder,result)


if __name__ == "__main__":raise SystemExit(main())
