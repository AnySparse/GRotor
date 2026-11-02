#!/usr/bin/env python3
"""Build GRotor and compare repeated CPU-reference runs with a fresh InstantGR2 run."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from test_instantgr2 import (ROOT, BASELINE, compress_reference, evaluator_metrics,
                            parse_batch_timings, parse_routing_timings, require_cuda,
                            run_logged, sha256)

POLICY = "grotor-explicit-memory-cpu-reference-v1"
CASES = ["mempool_tile", "ariane133_51", "ariane133_68", "nvdla", "bsg_chip",
         "mempool_group", "mempool_cluster"]


def local_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def source_manifest():
    files = [ROOT / "CMakeLists.txt", Path(__file__), ROOT / "script/test_instantgr2.py"]
    for folder in (ROOT / "src", ROOT / "include", ROOT / "third_party", ROOT / "resources",
                   BASELINE / "src", BASELINE / "run"):
        files.extend(p for p in folder.rglob("*") if p.is_file()
                     and p.suffix in (".cpp", ".cu", ".cuh", ".hpp", ".h", ".dat"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}


def write_summary(folder, result):
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    quality_fields = ["dataset", "implementation", "run", "matches_instantgr2",
                      "wirelength", "via", "overflow", "total", "open_nets", "incomplete_nets"]
    runtime_fields = ["dataset", "implementation", "run", "algorithm", "nets", "batches",
                      "cpu_batch_ms", "dag_solve_ms", "reference_total_ms"]
    with (folder / "quality.csv").open("w", newline="") as quality, \
            (folder / "runtime.csv").open("w", newline="") as runtime:
        quality_writer = csv.DictWriter(quality, quality_fields)
        runtime_writer = csv.DictWriter(runtime, runtime_fields)
        quality_writer.writeheader()
        runtime_writer.writeheader()
        for case in result["cases"]:
            for run in case["runs"]:
                if run["status"] != "passed":
                    continue
                identity = {"dataset": case["name"], "implementation": run["implementation"], "run": run["round"]}
                quality_writer.writerow({**identity, **run["metrics"],
                                         "matches_instantgr2": run["matches_instantgr2"]})
                for timing in run["algorithm_runtime"]:
                    runtime_writer.writerow({**identity, "algorithm": timing["algorithm"],
                                             "nets": timing["nets"], "batches": timing["batches"],
                                             "cpu_batch_ms": timing["cpu_batch_ms"],
                                             "dag_solve_ms": timing["dag_solve_ms"],
                                             "reference_total_ms": timing["cpu_batch_ms"] + timing["dag_solve_ms"]})
    lines = ["# GRotor Reference Validation", "", f"- Status: {result['status']}",
             "- GRotor: explicit device memory; CPU batching; warp/block DAG solve; incremental costs/BIT",
             "- Oracle: a fresh, independent InstantGR2 CPU-batching run per dataset",
             f"- GRotor repetitions: {result['repeats']}; sparsity: {result['sparsity']}",
             "- Comparison: all four evaluator decimal cost strings and zero open/incomplete nets",
             "- The external evaluator reads an actual routing output on every run.",
             "- One GRotor output is retained per dataset; other outputs are deleted after evaluation.", "",
             "| Dataset | Status | GRotor runs | InstantGR2 total | GRotor total | All costs match |",
             "|---|---|---:|---:|---:|---|"]
    for case in result["cases"]:
        oracle = next((r for r in case["runs"] if r["implementation"] == "InstantGR2" and r["status"] == "passed"), {})
        candidates = [r for r in case["runs"] if r["implementation"] == "GRotor" and r["status"] == "passed"]
        lines.append(f"| {case['name']} | {case['status']} | {len(candidates)}/{result['repeats']} | "
                     f"{oracle.get('metrics', {}).get('total', '-')} | "
                     f"{candidates[0]['metrics']['total'] if candidates else '-'} | "
                     f"{all(r['matches_instantgr2'] for r in candidates) if candidates else '-'} |")
    lines += ["", "## GRotor Reference Runtime (ms)", "",
              "Reference total = CPU batching + scoped DAG solve. This CPU-reference measurement is distinct from the InstantGR2 dual-mode CUDA-batching total.",
              "L-shape measures the complete batch loop. GRotor FLT measures one continuous interval including topology upload, GPU batch assembly, demand/cost updates and DAG solving. CPU topology construction and allocation precede it. Router wall time includes preparation, downloads and output; see summary.json.", "",
              "| Dataset | Algorithm | CPU batching | DAG solve | Reference total |",
              "|---|---|---:|---:|---:|"]
    for case in result["cases"]:
        for algorithm in ("L-shape", "FLT"):
            rows = [t for r in case["runs"] if r["implementation"] == "GRotor" and r["status"] == "passed"
                    for t in r["algorithm_runtime"] if t["algorithm"] == algorithm]
            values = [sum(r[k] for r in rows) / len(rows) for k in ("cpu_batch_ms", "dag_solve_ms")] if rows else []
            cells = [f"{v:.6f}" for v in (*values, sum(values))] if rows else ["-"] * 3
            lines.append("| " + case["name"] + " | " + algorithm + " | " + " | ".join(cells) + " |")
    lines += ["", "Full evaluator costs and per-run timings are in quality.csv and runtime.csv. Source, input, LUT, runner and executable hashes and exact commands are recorded in summary.json. Route bytes are not compared."]
    if result.get("error"):
        lines += ["", "## Error", "", result["error"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")


def validate_luts():
    for name in ("POWV9.dat", "POST9.dat"):
        if sha256(ROOT / "resources/flute" / name) != sha256(BASELINE / "run" / name):
            raise RuntimeError(f"Reference LUT differs from InstantGR2: {name}")
    for folder in (ROOT / "src", ROOT / "include"):
        for path in folder.rglob("*"):
            if path.is_file() and re.search(r"\b(__managed__|cudaMallocManaged)\b", path.read_text()):
                raise RuntimeError(f"Managed memory is forbidden in GRotor: {path}")


def test_case(args, result, name, folder, build):
    inputs = [args.data_dir / f"{name}.{suffix}" for suffix in ("cap", "net")]
    case_folder = folder / name
    case_folder.mkdir()
    case = {"name": name, "status": "running", "inputs": {str(p): sha256(p) for p in inputs}, "runs": []}
    result["cases"].append(case)
    with tempfile.TemporaryDirectory(prefix="evaluation-", dir=build) as temporary:
        raw_output = Path(temporary) / f"{name}.out"
        for implementation, round_id in [("InstantGR2", 1)] + [("GRotor", r) for r in range(1, args.repeats + 1)]:
            run = {"implementation": implementation, "round": round_id, "status": "running"}
            case["runs"].append(run)
            label = "instantgr2" if implementation == "InstantGR2" else f"run{round_id}"
            try:
                binary = build / ("instantgr2_reference" if implementation == "InstantGR2" else "GRotor")
                command = [str(binary), "-cap", str(inputs[0]), "-net", str(inputs[1]),
                           "-out", str(raw_output), "--sparsity", str(args.sparsity)]
                if implementation == "InstantGR2":
                    command += ["--batching", "cpu"]
                run["router_command"] = command
                print(f"Routing {name}: {implementation} run {round_id}", flush=True)
                log = case_folder / f"{label}.routing.log"
                run["router_exit_status"], run["router_seconds"] = run_logged(
                    command, log, cwd=BASELINE / "run" if implementation == "InstantGR2" else ROOT)
                if run["router_exit_status"] or not raw_output.is_file() or not raw_output.stat().st_size:
                    raise RuntimeError(f"Router failed: {log}")
                text = log.read_text()
                if implementation == "GRotor" and "memory=explicit" not in text:
                    raise RuntimeError("GRotor did not identify its explicit-memory reference pipeline")
                if "CONGESTION_REDUCTION=per-cell" not in text:
                    raise RuntimeError("Missing deterministic congestion reduction")
                if re.findall(r"^BATCH_SPARSITY backend=(\w+) mode=\d+ value=(\d+)$", text, re.MULTILINE) != [("cpu", str(args.sparsity))] * 2:
                    raise RuntimeError("Both routing stages must use CPU batching with the selected sparsity")
                run["batch_timings"] = parse_batch_timings(text, "determinism", "cpu")
                run["dag_timings"], run["algorithm_runtime"] = parse_routing_timings(text, run["batch_timings"])
                evaluation_log = case_folder / f"{label}.evaluator.log"
                command = [str(build / "grotor_evaluator"), *map(str, inputs), str(raw_output)]
                run["evaluator_command"] = command
                run["evaluator_exit_status"], run["evaluator_seconds"] = run_logged(command, evaluation_log)
                run["metrics"] = evaluator_metrics(evaluation_log, run["evaluator_exit_status"])
                oracle = case["runs"][0]
                run["matches_instantgr2"] = run["metrics"] == oracle["metrics"]
                if not run["matches_instantgr2"]:
                    raise RuntimeError(f"Evaluator cost mismatch: {name}, GRotor run {round_id}")
                if implementation == "GRotor" and round_id == 1:
                    retained = case_folder / f"{name}.out.gz"
                    compress_reference(raw_output, retained)
                    case["output"] = str(retained.relative_to(folder))
                run["status"] = "passed"
                print(f"PASS {name} {implementation} run {round_id}: {run['metrics']['total']}", flush=True)
            except BaseException as error:
                run["status"], run["error"] = "failed", str(error)
                case["status"] = "failed"
                raise
            finally:
                raw_output.unlink(missing_ok=True)
                write_summary(folder, result)
    case["status"] = "passed"
    write_summary(folder, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../ISPD24_tmp")
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sparsity", type=int, default=1)
    parser.add_argument("--cuda-arch", default="89")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "nvcc"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default="results/GRotor")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 2 or not 1 <= args.sparsity <= 2147483647 or not args.cuda_arch.isdigit():
        parser.error("Use at least two repeats, a positive sparsity and a numeric CUDA architecture")
    if len(set(args.cases)) != len(args.cases) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in args.cases):
        parser.error("Dataset names must be unique and contain letters, digits, '_' or '-'")
    args.data_dir = local_path(args.data_dir)
    folder = local_path(args.output_dir)
    if ROOT / "results" not in folder.parents:
        parser.error("Results must be stored below this repository's results/ directory")
    build = ROOT / "build/GRotor" / f"sm_{args.cuda_arch}"
    build.mkdir(parents=True, exist_ok=True)
    config = {"policy": POLICY, "repeats": args.repeats, "sparsity": args.sparsity,
              "cuda_arch": args.cuda_arch, "gpu": args.gpu, "data_dir": str(args.data_dir),
              "datasets": args.cases, "sources": source_manifest()}
    validate_luts()
    binaries = ("GRotor", "instantgr2_reference", "grotor_evaluator")
    if args.resume:
        result = json.loads((folder / "summary.json").read_text())
        if result["configuration"] != config:
            parser.error("Cannot resume with changed source, inputs location, configuration or datasets")
        if any(sha256(build / n) != result["binary_sha256"][n] for n in binaries):
            parser.error("Cannot resume with changed binaries")
        for case in result["cases"]:
            if any(r.get("matches_instantgr2") is False for r in case["runs"]):
                parser.error("A quality mismatch cannot be discarded by --resume")
            if case["inputs"] != {str(args.data_dir / f"{case['name']}.{ext}"): sha256(args.data_dir / f"{case['name']}.{ext}") for ext in ("cap", "net")}:
                parser.error("Cannot resume with changed inputs")
            if case["status"] == "passed" and not (folder / case["output"]).is_file():
                parser.error("Retained GRotor output is missing")
            if case["status"] != "passed":
                shutil.rmtree(folder / case["name"])
        result["cases"] = [c for c in result["cases"] if c["status"] == "passed"]
        result.pop("error", None)
        result["status"] = "running"
    else:
        folder.mkdir(parents=True, exist_ok=False)
        result = {"status": "running", "configuration": config, "repeats": args.repeats,
                  "sparsity": args.sparsity, "cases": []}
    write_summary(folder, result)
    try:
        if not args.build_only:
            for name in args.cases:
                for suffix in ("cap", "net"):
                    if not (args.data_dir / f"{name}.{suffix}").is_file():
                        raise RuntimeError(f"Missing input: {name}.{suffix}")
        if not args.resume:
            nvcc = shutil.which(args.nvcc)
            if not nvcc:
                raise RuntimeError("nvcc is required")
            # Resolve the toolchain symlink so CMake locates CUDA's library root.
            commands = [["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                         f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}", f"-DCMAKE_CUDA_COMPILER={Path(nvcc).resolve()}"],
                        ["cmake", "--build", str(build), "--parallel", "2"]]
            result["build_commands"] = commands
            for label, command in zip(("configure.log", "build.log"), commands):
                status, _ = run_logged(command, folder / label)
                if status:
                    raise RuntimeError(f"Build failed: {label}")
            result["binary_sha256"] = {n: sha256(build / n) for n in binaries}
            result["compiler_version"] = subprocess.check_output([nvcc, "--version"], text=True)
        if args.build_only:
            result["status"] = "build-only"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            require_cuda()
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True, capture_output=True)
            result["gpu_details"] = gpu.stdout.strip()
            for name in args.cases:
                if not any(c["name"] == name and c["status"] == "passed" for c in result["cases"]):
                    test_case(args, result, name, folder, build)
            result["status"] = "passed"
    except BaseException as error:
        result["status"], result["error"] = "failed", str(error)
        print(f"ERROR: {error}", file=sys.stderr)
    finally:
        write_summary(folder, result)
        print(f"Summary: {folder / 'summary.md'}", flush=True)
    return 0 if result["status"] in ("passed", "build-only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
