"""Export measured routing components without changing their timing scopes."""
import csv
import json
import hashlib
from pathlib import Path
import math

ALGORITHMS = (("lshape", "L-shape", "rounds"), ("detour", "FLT", "cached"))
COSTS = ("wirelength", "via", "overflow", "total")
RUNTIME_FIELDS = ["dataset", "algorithm", "instantgr2_batch_ms", "grotor_batch_ms", "batching_speedup",
    "instantgr2_dag_ms", "grotor_dag_ms", "dag_speedup", "instantgr2_total_ms", "grotor_total_ms", "total_speedup"]
BATCH_FIELDS = ["dataset", "algorithm", "grotor_method", "grotor_timing_scope", "instantgr2_cpu_ms", "instantgr2_cuda_core_ms", "grotor_rt_core_ms",
    "speedup_vs_cpu", "speedup_vs_cuda_core"]
DAG_FIELDS = ["dataset", "algorithm", "instantgr2_dag_ms", "grotor_dag_ms", "speedup"]
TOTAL_FIELDS = ["dataset", "algorithm", "instantgr2_batch_ms", "instantgr2_dag_ms", "instantgr2_total_ms",
    "grotor_batch_ms", "grotor_dag_ms", "grotor_total_ms", "speedup"]


def duration(record):
    value = float(record["elapsed_ms"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("Runtime must be a finite, nonnegative number")
    return value


def speedup(before, after):
    return before / after if after > 0 else None


def one(records, stage, backend=None):
    matches = [r for r in records if r["stage"] == stage and (backend is None or r["backend"] == backend)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {stage} {backend or 'DAG'} timing")
    return matches[0]


def continuous(record):
    value = duration(record)
    if abs(value - (record["end_us"] - record["start_us"]) / 1000) > 1e-5:
        raise ValueError("GRotor duration must equal its contiguous timestamp interval")
    return value


def comparison_rows(case):
    baseline, native = case["baseline"], case["rtcore"]
    for key in COSTS:
        if baseline["metrics"][key] != native["metrics"][key]:
            raise ValueError(f"Evaluator {key} differs for {case['name']}")
    for run in (baseline, native):
        if run["metrics"]["open_nets"] or run["metrics"]["incomplete_nets"]:
            raise ValueError(f"Incomplete routing for {case['name']}")
    rows, batches, dags, totals = [], [], [], []
    for index, (stage, algorithm, method) in enumerate(ALGORITHMS):
        cpu = one(baseline["batch_timings"], stage, "cpu")
        cuda = one(baseline["batch_timings"], stage, "cuda_core")
        rt = one(native["batch_timings"], stage, "rt_core")
        old_dag = one(baseline["dag_timings"], stage)
        new_dag = one(native["dag_timings"], stage)
        for record in (rt, old_dag, new_dag):
            if (record["nets"], record["batches"]) != (cpu["nets"], cpu["batches"]):
                raise ValueError(f"CPU routing-plan workload differs for {case['name']} {algorithm}")
        if cuda["nets"] != cpu["nets"] or not cpu["used_for_routing"] or cuda["used_for_routing"] or not rt["used_for_routing"]:
            raise ValueError("Expected InstantGR2 dual CPU plan and GRotor RT-Core plan")
        selected = native["totals"][index]
        if selected["method"] not in (method, "partitioned") or abs(duration(selected) - duration(rt)) > 1e-5:
            raise ValueError("RT-Core method or returned duration differs from the measured interval")
        scope = "coloring_loop_wall" if method == "rounds" else "pipeline_to_coloring_wall"
        if selected["method"] == "partitioned":
            scope = selected.get("timing_scope")
            if scope not in ("partitioned_batch_wall", "partitioned_pipeline_prebuilt_wall"):
                raise ValueError("Partitioned timing must be a continuous full batching interval")
        cpu_ms, cuda_ms, rt_ms = continuous(cpu), continuous(cuda), continuous(rt)
        old_ms = continuous(old_dag) if stage == "lshape" else duration(old_dag)
        if old_ms > (old_dag["end_us"] - old_dag["start_us"]) / 1000 + 1e-5:
            raise ValueError("InstantGR2 FLT suffix sum exceeds its enclosing interval")
        new_ms = continuous(new_dag)
        identity = dict(dataset=case["name"], algorithm=algorithm)
        batch_ratio, dag_ratio = speedup(cuda_ms, rt_ms), speedup(old_ms, new_ms)
        old_total, new_total = cuda_ms + old_ms, rt_ms + new_ms
        total_ratio = speedup(old_total, new_total)
        rows.append(dict(**identity, instantgr2_batch_ms=cuda_ms, grotor_batch_ms=rt_ms, batching_speedup=batch_ratio,
            instantgr2_dag_ms=old_ms, grotor_dag_ms=new_ms, dag_speedup=dag_ratio,
            instantgr2_total_ms=old_total, grotor_total_ms=new_total, total_speedup=total_ratio))
        batches.append(dict(**identity, grotor_method=selected["method"], grotor_timing_scope=scope,
            instantgr2_cpu_ms=cpu_ms, instantgr2_cuda_core_ms=cuda_ms,
            grotor_rt_core_ms=rt_ms, speedup_vs_cpu=speedup(cpu_ms, rt_ms), speedup_vs_cuda_core=batch_ratio))
        dags.append(dict(**identity, instantgr2_dag_ms=old_ms, grotor_dag_ms=new_ms, speedup=dag_ratio))
        totals.append(dict(**identity, instantgr2_batch_ms=cuda_ms, instantgr2_dag_ms=old_ms, instantgr2_total_ms=old_total,
            grotor_batch_ms=rt_ms, grotor_dag_ms=new_ms, grotor_total_ms=new_total, speedup=total_ratio))
    return rows, batches, dags, totals


def formatted(value, ratio=False):
    if value is None:
        return "-"
    return f"{value:.2f}x" if ratio else f"{value:.3f}"


def write_summary(folder, result):
    # Validate all completed measurements before replacing any report files.
    runtime, batching, dag, total, quality, memory = [], [], [], [], [], []
    seen = set()
    for case in result["cases"]:
        if case["status"] != "passed":
            continue
        if case["name"] in seen:
            raise ValueError("Duplicate completed dataset")
        seen.add(case["name"])
        groups = comparison_rows(case)
        for target, rows in zip((runtime, batching, dag, total), groups):
            target.extend(rows)
        quality.append(dict(dataset=case["name"], evaluator_matches=True,
            **case["rtcore"]["metrics"]))
        native = case["rtcore"]
        memory.append(dict(dataset=case["name"], cpu_peak_mib=native.get("cpu_peak_mib"),
            gpu_sampled_peak_mib=native["gpu_memory"]["sampled_peak_used_mib"],
            gpu_sampled_min_free_mib=native["gpu_memory"]["sampled_min_free_mib"]))
    cfg = result["configuration"]
    if seen and any(cfg.get(k) != v for k, v in (("sparsity", 1), ("cost_update", "incremental"),
        ("prefix", "bit"), ("dag_lshape", "warp"), ("dag_flt", "block"))):
        raise ValueError("Expected sparsity-1 warp/block routing with incremental costs and BIT")
    exports = (("runtime.csv", runtime, RUNTIME_FIELDS), ("batching-runtime.csv", batching, BATCH_FIELDS),
        ("dag-runtime.csv", dag, DAG_FIELDS), ("stage-runtime.csv", total, TOTAL_FIELDS),
        ("quality.csv", quality, ["dataset", "evaluator_matches", *COSTS, "open_nets", "incomplete_nets"]),
        ("memory.csv", memory, ["dataset", "cpu_peak_mib", "gpu_sampled_peak_mib", "gpu_sampled_min_free_mib"]))
    for name, rows, fields in exports:
        with (folder / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
    result["report_format"] = dict(version=2, reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# GRotor and InstantGR2 Benchmark", "", f"Status: {result['status']}",
        f"Completed datasets: {len(seen)}", "", "All runtimes are in milliseconds. Speedup = InstantGR2 / GRotor; values below 1 indicate a slowdown. A zero denominator is reported as '-'.",
        "Total = measured batching + measured routing DAG execution. InstantGR2 uses CUDA-Core batching time and the deterministic CPU batch plan for routing. GRotor uses standard L-shape/cached FLT batching on smaller designs and partitioned RT-Core batching on large designs, with warp/block DAG solving and incremental costs/BIT.", ""]
    if result.get("report_generation"):
        source = result["report_generation"]["source"]
        lines += [f"Tables regenerated from saved measurements: `{source}`. No routing or evaluation was rerun.", ""]
    for title, rows, old_key, new_key in (("Total Runtime", total, "instantgr2_total_ms", "grotor_total_ms"),
            ("Routing DAG Runtime", dag, "instantgr2_dag_ms", "grotor_dag_ms")):
        lines += ["## " + title, ""]
        for _, algorithm, _ in ALGORITHMS:
            lines += ["### " + algorithm, "", "| Dataset | InstantGR2 ms | GRotor ms | Speedup |", "|---|---:|---:|---:|"]
            for r in rows:
                if r["algorithm"] == algorithm:
                    lines.append(f"| {r['dataset']} | {formatted(r[old_key])} | {formatted(r[new_key])} | {formatted(r['speedup'], True)} |")
            lines.append("")
    lines += ["## Batching Runtime", "", "CPU is an additional InstantGR2 control. Algorithm totals use the CUDA-Core column.", ""]
    for _, algorithm, _ in ALGORITHMS:
        lines += ["### " + algorithm, "", "| Dataset | RT method | CPU ms | CUDA-Core ms | RT-Core ms | CPU / RT | CUDA / RT |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in batching:
            if r["algorithm"] == algorithm:
                values = [formatted(r[k]) for k in ("instantgr2_cpu_ms", "instantgr2_cuda_core_ms", "grotor_rt_core_ms")]
                values += [formatted(r[k], True) for k in ("speedup_vs_cpu", "speedup_vs_cuda_core")]
                lines.append("| " + r["dataset"] + " | " + r["grotor_method"] + " | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## Evaluator Quality", "", "All four evaluator cost strings match for every completed case. Open and incomplete net counts must both be zero.", "",
        "| Dataset | Wirelength | Via | Overflow | Total | Open / incomplete |", "|---|---:|---:|---:|---:|---|"]
    for r in quality:
        lines.append(f"| {r['dataset']} | {r['wirelength']} | {r['via']} | {r['overflow']} | {r['total']} | {r['open_nets']} / {r['incomplete_nets']} |")
    lines += ["", "## Memory", "", "CPU maximum RSS is from GNU time. GPU memory is sampled device-wide every 100 ms; these are sampled peaks.", "",
        "| Dataset | CPU peak MiB | GPU peak MiB | GPU minimum free MiB |", "|---|---:|---:|---:|"]
    for r in memory:
        lines.append(f"| {r['dataset']} | {formatted(r['cpu_peak_mib'])} | {r['gpu_sampled_peak_mib']} | {r['gpu_sampled_min_free_mib']} |")
    lines += ["", "## Measurement Scope", "",
        "- L-shape DAG time covers the batch execution loop, including cost/prefix refresh and demand commits.",
        "- GRotor FLT uses one continuous synchronized interval, including compact upload, GPU assembly, staging waits, cost/BIT updates, DP, traceback and demand commits. CPU topology construction/planning and workspace allocation precede it.",
        "- InstantGR2 FLT retains its sum of per-batch GPU suffix intervals, excluding preceding CPU packing. These recorded scope differences are retained when calculating speedups.",
        "- Standard RT-Core batching covers the complete coloring loop. Cached RT-Core batching covers pipeline preparation through coloring completion. Both exclude earlier geometry/BVH construction and later final batch collection/cleanup. CPU/CUDA-Core controls retain their complete-function timers.",
        "- Partitioned RT-Core batching instead includes bitmap initialization, local geometry copies, scene/BVH construction, coloring, occupancy publication, batch-list assembly and cleanup in one continuous interval. For partitioned_pipeline_prebuilt_wall, shared pipeline compilation and warm-up precede the interval. L-shape retains rounds; FLT retains cached. Method and scope are retained in batching-runtime.csv and summary.json.",
        "- Algorithm totals exclude earlier preparation and file output. Complete GRotor phase/router measurements remain in summary.json and routing logs.",
        f"- CPU batch-list validation enabled: {cfg.get('validate_batches', False)}. It is disabled by default for performance tests.",
        "- Each new run writes one real routing output, evaluates it externally, then deletes it before the next run. No routing outputs are retained.",
        "- Performance measurements are single sequential runs with unlocked GPU clocks. Source, input, binary hashes and exact commands are retained in summary.json.", ""]
    if result.get("error"):
        lines += ["## Error", "", result["error"], ""]
    (folder / "summary.md").write_text("\n".join(lines))
