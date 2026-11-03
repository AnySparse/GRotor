#!/usr/bin/env python3
"""Build and compare the CUGR2 baseline and GRotor CUGR2 routing pipeline."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

from cugr2_reporting import STAGES, comparison_rows, parse_cpu, parse_grotor, pending_cpu_tasks, write_report
from test_instantgr2 import ROOT, evaluator_metrics, sha256
from test_grotor_reference import local_path, source_manifest, validate_luts

CPU_SOURCE = ROOT / "baseline/cu-gr-2"


def relative_cpu_sources(sources):
    """Compare vendor contents independently of the baseline directory name."""
    relative = {}
    for name, digest in sources.items():
        path = Path(name)
        if len(path.parts) > 2 and path.parts[0] == "baseline":
            path = Path(*path.parts[2:])
        key = path.as_posix()
        if key in relative:
            raise ValueError("Duplicate CPU source fingerprint: " + key)
        relative[key] = digest
    return relative


def run(command, log, cwd, timeout):
    print(shlex.join(map(str, command)), flush=True)
    start = time.perf_counter()
    with log.open("x") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if not isinstance(error, subprocess.TimeoutExpired):
                raise
            status = "timeout"
    return status, time.perf_counter() - start


def resources(log):
    records = re.findall(r"^PROCESS_RESOURCES wall_s=([\d.]+) max_rss_kib=(\d+)$",
                         log.read_text(), re.MULTILINE)
    return dict(wall_s=float(records[-1][0]), max_rss_kib=int(records[-1][1])) if records else {}


def build(args, folder, gpu_build, cpu_build):
    gpu_build.mkdir(parents=True, exist_ok=True)
    cpu_build.mkdir(parents=True, exist_ok=True)
    if not args.no_build:
        if not args.optix_root:
            raise ValueError("Set OPTIX_ROOT_DIR or pass --optix-root to build RT-Core batching")
        commands = [
            (["cmake", "-S", ROOT, "-B", gpu_build, "-DCMAKE_BUILD_TYPE=Release",
              "-DGROTOR_ENABLE_RTCORE=ON", "-DGROTOR_BUILD_CUGR2_ROUTER=ON",
              "-DGROTOR_FULL_COST_REFERENCE=OFF", "-DGROTOR_BUILD_ROUTING_ABLATION=OFF",
              "-DCMAKE_CUDA_COMPILER=" + str(Path(shutil.which(args.nvcc) or args.nvcc).resolve()),
              "-DCMAKE_CUDA_ARCHITECTURES=" + args.cuda_arch,
              "-DOPTIX_ROOT_DIR=" + str(local_path(args.optix_root))], "grotor.configure.log"),
            (["cmake", "--build", gpu_build, "--target", "grotor_cugr2", "grotor_evaluator",
              "--parallel", str(args.jobs)], "grotor.build.log"),
            (["cmake", "-S", CPU_SOURCE / "src", "-B", cpu_build,
              "-DCMAKE_BUILD_TYPE=Release"], "cugr2.configure.log"),
            (["cmake", "--build", cpu_build, "--target", "route", "--parallel", str(args.jobs)],
             "cugr2.build.log")]
        for command, name in commands:
            if args.cpu_log and name.startswith("cugr2."):
                continue
            status, _ = run(command, folder / name, ROOT, 1800)
            if status != 0:
                raise RuntimeError("Build failed: " + str(folder / name))
    flags = (gpu_build / "CMakeFiles/grotor_cugr2.dir/flags.make").read_text()
    if any("-D" + flag in flags for flag in ("GROTOR_PREFIX_ACCESS_COUNTS", "GROTOR_FULL_COST_REFERENCE", "GROTOR_SERIAL_DAG")):
        raise ValueError("Comparison requires incremental costs/BIT, cooperative scheduling and disabled profiling counters")
    for flag in ("GROTOR_CUGR2_ROUTING=1", "GROTOR_ENABLE_RTCORE=1"):
        if "-D" + flag not in flags:
            raise ValueError("Comparison build is missing " + flag)


def native_quality(text):
    rows = {}
    for key, pattern in (("wirelength_metric", r"wire length \(metric\):\s+(\d+)"),
                         ("via_count", r"total via count:\s+(\d+)"),
                         ("wire_overflow", r"total wire overflow:\s+(\d+)")):
        values = re.findall(pattern, text)
        if values:
            rows[key] = int(values[-1])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="One cap/net dataset name without an extension")
    parser.add_argument("--data-dir", default="../ISPD24_tmp")
    parser.add_argument("--def-dir", default="../ISPD24_def/def")
    parser.add_argument("--lef", default="../ISPD24_def/lef/Nangate.lef")
    parser.add_argument("--def-file", help="Matching DEF; defaults to <case>.def (mempool_cluster maps to cluster.def)")
    parser.add_argument("--cpu-log", help="Explicitly import an archived CPU log; provenance and skipped stages remain labeled")
    parser.add_argument("--reuse-cpu", help="Reuse a completed summary after verifying its input, CPU source, binary and log fingerprints")
    parser.add_argument("--output-dir", help="Default: results/CUGR2/<case>")
    parser.add_argument("--threads", type=int, default=8, help="CUGR2 thread setting; GRotor topology preparation uses eight threads")
    parser.add_argument("--cpu-timeout", type=float, default=3600, help="CPU process limit in seconds; completed stages remain in the report")
    parser.add_argument("--gpu-timeout", type=float, default=3600)
    parser.add_argument("--max-net-ratio", type=float, default=10, help="Maximum count ratio for reporting a workload-comparable speedup")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--optix-root", default=os.environ.get("OPTIX_ROOT_DIR"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--clean-results", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Regenerate tables from saved, hash-verified logs without routing")
    parser.add_argument("--validate-batches", action="store_true", help="Compare RT-Core plans against CPU plans outside measured intervals")
    args = parser.parse_args()
    if args.cpu_log and args.reuse_cpu:
        parser.error("Choose --cpu-log or --reuse-cpu")
    if (not re.fullmatch(r"[A-Za-z0-9_-]+", args.case) or not args.gpu.isdigit()
            or not args.cuda_arch.isdigit() or args.threads < 1 or args.jobs < 1
            or not 0 < args.cpu_timeout < float("inf") or not 0 < args.gpu_timeout < float("inf")
            or not 1 <= args.max_net_ratio < float("inf")):
        parser.error("Use a simple case name, numeric GPU/architecture, positive limits and a count ratio of at least one")
    folder = local_path(args.output_dir or f"results/CUGR2/{args.case}")
    if ROOT / "results" not in folder.parents:
        parser.error("Output must be below this repository's results directory")
    if args.cpu_log and args.clean_results and folder in local_path(args.cpu_log).parents:
        parser.error("The imported CPU log must be outside the output directory being replaced")
    if args.report_only:
        if args.clean_results:
            parser.error("--report-only cannot delete its inputs with --clean-results")
        result = json.loads((folder / "summary.json").read_text())
        if result.get("dataset") != args.case or result.get("status") == "running":
            parser.error("Use a finished report matching --case")
        for name, expected in result["log_sha256"].items():
            if sha256(folder / name) != expected:
                raise ValueError("Saved log changed: " + name)
        result["cpu"]["stages"] = parse_cpu((folder / "cugr2.routing.log").read_text())
        result["cpu"].setdefault("measurement_origin", "fresh")
        result["cpu"]["pending_nets"] = pending_cpu_tasks((folder / "cugr2.routing.log").read_text())
        if "metrics" in result.get("grotor", {}):
            result["grotor"]["stages"] = parse_grotor((folder / "grotor.routing.log").read_text())
            result["grotor"]["metrics"] = evaluator_metrics(folder / "grotor.evaluator.log",
                result["grotor"]["evaluator_exit_status"])
        result["report_generation"] = dict(script_sha256=sha256(Path(__file__)),
            reporting_sha256=sha256(ROOT / "script/cugr2_reporting.py"))
        write_report(folder, result)
        print("Report:", folder / "summary.md")
        return 0 if result["status"] == "passed" else 1
    data, lef = local_path(args.data_dir), local_path(args.lef)
    def_file = local_path(args.def_file) if args.def_file else local_path(args.def_dir) / (
        ("cluster" if args.case == "mempool_cluster" else args.case) + ".def")
    inputs = [data / (args.case + ".cap"), data / (args.case + ".net"), lef, def_file]
    for path in inputs:
        if not path.is_file():
            parser.error("Missing input: " + str(path))
    reused = None
    if args.reuse_cpu:
        saved_path = local_path(args.reuse_cpu)
        saved = json.loads(saved_path.read_text())
        saved_log = saved_path.parent / "cugr2.routing.log"
        cpu_sources = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(CPU_SOURCE.rglob("*"))
            if p.is_file() and p.relative_to(CPU_SOURCE).parts[0] not in ("build", "run", ".git")}
        if (saved.get("dataset") != args.case or saved.get("cpu", {}).get("exit_status") != 0
                or list(saved["cpu"].get("stages", {})) != list(STAGES)
                or saved["configuration"]["threads"] != args.threads
                or relative_cpu_sources(saved["configuration"]["cpu_sources"]) != relative_cpu_sources(cpu_sources)
                or saved["inputs"] != {str(p): sha256(p) for p in inputs}
                or saved["log_sha256"]["cugr2.routing.log"] != sha256(saved_log)
                or saved["binary_sha256"].get("cugr2") != sha256(ROOT / "build/CUGR2/route")):
            parser.error("CPU reuse requires matching completed source/input/binary/log fingerprints")
        cache = ROOT / "build/CUGR2/cpu-cache"
        cache.mkdir(parents=True, exist_ok=True)
        record = cache / (sha256(saved_path) + ".json")
        log = cache / (sha256(saved_log) + ".log")
        shutil.copyfile(saved_path, record)
        shutil.copyfile(saved_log, log)
        reused = dict(cpu=saved["cpu"], record=str(record), record_sha256=sha256(record),
                      binary_sha256=saved["binary_sha256"]["cugr2"])
        args.cpu_log = str(log)
    if folder.exists():
        previous = folder / "summary.json"
        if not args.clean_results or not previous.is_file() or json.loads(previous.read_text()).get("dataset") != args.case:
            parser.error("Existing output requires --clean-results and a matching previous comparison report")
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    gpu_build, cpu_build = ROOT / "build/GRotor-cugr2" / ("sm_" + args.cuda_arch), ROOT / "build/CUGR2"
    result = dict(status="running", dataset=args.case, configuration=dict(
        gpu=args.gpu, cuda_arch=args.cuda_arch, threads=args.threads, max_net_ratio=args.max_net_ratio,
        sparsity=1, validate_batches=args.validate_batches,
        cpu_timeout=args.cpu_timeout, gpu_timeout=args.gpu_timeout,
        cpu_upstream_commit=(CPU_SOURCE / "UPSTREAM_REVISION").read_text().strip(),
        cpu_sources={str(p.relative_to(ROOT)): sha256(p) for p in sorted(CPU_SOURCE.rglob("*"))
            if p.is_file() and p.relative_to(CPU_SOURCE).parts[0] not in ("build", "run", ".git")},
        grotor_sources=source_manifest(), script_sha256=sha256(Path(__file__)),
        reporting_sha256=sha256(ROOT / "script/cugr2_reporting.py")),
        inputs={str(p): sha256(p) for p in inputs})
    write_report(folder, result)
    guide, output = folder / "cugr2.guide", folder / "grotor.out"
    try:
        validate_luts()
        build(args, folder, gpu_build, cpu_build)
        result["binary_sha256"] = {name: sha256(path) for name, path in (
            ("grotor", gpu_build / "grotor_cugr2"), ("evaluator", gpu_build / "grotor_evaluator"))}
        if not args.cpu_log:
            result["binary_sha256"]["cugr2"] = sha256(cpu_build / "route")
        elif reused:
            result["binary_sha256"]["cugr2"] = reused["binary_sha256"]
        cpu_command = ["/usr/bin/time", "-f", "\nPROCESS_RESOURCES wall_s=%e max_rss_kib=%M",
            cpu_build / "route", "-lef", lef, "-def", def_file, "-threads", str(args.threads), "-output", guide]
        cpu_log = folder / "cugr2.routing.log"
        result["cpu"] = dict(command=list(map(str, cpu_command)), measurement_origin="fresh")
        try:
            if args.cpu_log:
                archived = local_path(args.cpu_log)
                text = archived.read_text()
                for label, expected in (("lef file", lef), ("def file", def_file)):
                    values = re.findall(rf"{label}:\s*(.+)", text)
                    if len(values) != 1 or Path(values[0].strip()).name != expected.name:
                        raise ValueError("Archived CPU input filename does not match " + str(expected))
                if "Terminated." not in text or not parse_cpu(text):
                    raise ValueError("Archived CPU log has no completed routing run")
                shutil.copyfile(archived, cpu_log)
                result["cpu"] = dict(measurement_origin="archived", imported_log=str(archived),
                    imported_log_sha256=sha256(archived), historical_fingerprints_available=False)
                status, wall = 0, None
                if reused:
                    result["cpu"] = dict(reused["cpu"], measurement_origin="reused-verified",
                        reused_record=reused["record"], reused_record_sha256=reused["record_sha256"])
                    wall = reused["cpu"].get("wall_s")
            else:
                status, wall = run(cpu_command, cpu_log, CPU_SOURCE / "flute", args.cpu_timeout)
            result["cpu"].update(exit_status=status, wall_s=wall, stages=parse_cpu(cpu_log.read_text()),
                pending_nets=pending_cpu_tasks(cpu_log.read_text()),
                quality=native_quality(cpu_log.read_text()), resources=resources(cpu_log))
            if status == 0 and not args.cpu_log:
                if not guide.is_file() or guide.stat().st_size == 0:
                    raise RuntimeError("CUGR2 produced no routing guide")
                result["cpu"]["guide_bytes"] = guide.stat().st_size
        finally:
            guide.unlink(missing_ok=True)
        write_report(folder, result)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        gpu_command = ["/usr/bin/time", "-f", "\nPROCESS_RESOURCES wall_s=%e max_rss_kib=%M",
            gpu_build / "grotor_cugr2", "-cap", inputs[0], "-net", inputs[1], "-out", output,
            "--batching", "rtcore", "--sparsity", "1", "--validate-batches", str(int(args.validate_batches))]
        gpu_log = folder / "grotor.routing.log"
        with (folder / "gpu-memory.csv").open("x") as stream:
            monitor = subprocess.Popen(["nvidia-smi", "--id=" + args.gpu,
                "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"], stdout=stream)
            try:
                status, wall = run(gpu_command, gpu_log, ROOT, args.gpu_timeout)
            finally:
                monitor.terminate()
                monitor.wait()
        result["grotor"] = dict(exit_status=status, wall_s=wall, command=list(map(str, gpu_command)),
            resources=resources(gpu_log))
        if status != 0:
            raise RuntimeError("GRotor routing failed; see " + str(gpu_log))
        result["grotor"]["stages"] = parse_grotor(gpu_log.read_text())
        samples = [float(v) for v in (folder / "gpu-memory.csv").read_text().splitlines() if v.strip()]
        result["grotor"]["peak_device_mib"] = max(samples) if samples else None
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Router produced no routing output")
        result["grotor"]["output_bytes"] = output.stat().st_size
        evaluation = [gpu_build / "grotor_evaluator", inputs[0], inputs[1], output]
        status, _ = run(evaluation, folder / "grotor.evaluator.log", ROOT, args.gpu_timeout)
        if status == "timeout":
            raise RuntimeError("Evaluator timed out")
        result["grotor"]["metrics"] = evaluator_metrics(folder / "grotor.evaluator.log", status)
        result["grotor"]["evaluator_command"] = list(map(str, evaluation))
        result["grotor"]["evaluator_exit_status"] = status
        completed = result["cpu"].get("exit_status") == 0 and list(result["cpu"].get("stages", {})) == list(STAGES)
        comparable = all(row["workload_status"] in ("comparable", "empty") for row in comparison_rows(result))
        result["status"] = "passed" if completed and comparable else "incomplete" if not completed else "workload-mismatch"
        if not completed:
            result["error"] = "CUGR2 did not complete all stages. Only completed stage timings are comparable."
    except Exception as error:
        result["status"], result["error"] = "failed", str(error)
        raise
    finally:
        guide.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        result["log_sha256"] = {p.name: sha256(p) for p in sorted(folder.glob("*.log"))}
        write_report(folder, result)
        print("Report:", folder / "summary.md", flush=True)
    for row in comparison_rows(result):
        speedup = "-" if row["speedup"] is None else f"{row['speedup']:.2f}x"
        print(f"{row['algorithm']}: CUGR2 nets={row['cugr2_nets']}, GRotor nets={row['grotor_nets']}, speedup={speedup}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
