"""Shared utilities for the maintained AE runners."""
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "baseline/InstantGR2"
EVALUATOR = ROOT / "resources/evaluator"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()

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
        raise RuntimeError("CUDA driver library is unavailable") from error
    status = driver.cuInit(0)
    if status:
        raise RuntimeError(f"CUDA initialization failed (driver status {status}); check GPU access")
    count = ctypes.c_int()
    status = driver.cuDeviceGetCount(ctypes.byref(count))
    if status or count.value == 0:
        raise RuntimeError("No CUDA GPU is available")
def local_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def source_manifest(extra_folders=()):
    files = [ROOT / "CMakeLists.txt", ROOT / "CMakePresets.json"]
    for folder in (ROOT / "src", ROOT / "include", ROOT / "third_party", ROOT / "resources",
                   ROOT / "script", BASELINE / "src", BASELINE / "run", *extra_folders):
        files.extend(p for p in folder.rglob("*") if p.is_file()
                     and (p.suffix in (".cpp", ".cu", ".cuh", ".hpp", ".h", ".dat", ".py", ".a", ".txt")
                          or p.name == "UPSTREAM_REVISION"))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}
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


def run_logged(command, log_path, cwd=ROOT, timeout=None):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    start = time.perf_counter()
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            status = process.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return status, time.perf_counter() - start


def validate_luts():
    for name in ('POWV9.dat', 'POST9.dat'):
        path = ROOT / 'resources/flute' / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError('Missing FLUTE lookup table: ' + str(path))


def configure_build(args, build, folder, targets, options=(), source=ROOT, label=''):
    if args.no_build:
        return
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        raise RuntimeError('nvcc not found; set CUDACXX or pass --nvcc')
    configure = ['cmake', '-S', str(source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_CUDA_COMPILER=' + str(Path(nvcc).resolve()),
                 '-DCMAKE_CUDA_ARCHITECTURES=' + args.cuda_arch, *options]
    if source == ROOT and args.optix_root:
        configure.append('-DOPTIX_ROOT_DIR=' + str(Path(args.optix_root).resolve()))
    commands = [configure, ['cmake', '--build', str(build), '--target', *targets, '-j', str(args.jobs)]]
    for command, suffix in zip(commands, ('configure', 'build')):
        status, _ = run_logged(command, folder / (label + suffix + '.log'))
        if status:
            raise RuntimeError('Build failed; see ' + str(folder / (label + suffix + '.log')))


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
