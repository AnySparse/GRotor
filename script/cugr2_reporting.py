"""Validate and report the CPU and GPU CUGR2 routing comparison."""
import csv
import json
import math
import re

STAGES = ("lshape", "detour", "maze")
LABELS = {"lshape": "L-shape", "detour": "Detour", "maze": "Maze"}
TIMING_POLICY = "rtcore-batching-plus-contiguous-dag-v1"


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Runtime must be finite and nonnegative")
    return value


def parse_cpu(text):
    rows = {}
    for stage, solver, full, nets in re.findall(
            r"^CUGR_STAGE_RUNTIME stage=(lshape|detour|maze) solver_ms=([\d.]+) "
            r"full_ms=([\d.]+) nets=(\d+)$", text, re.MULTILINE):
        if stage in rows:
            raise ValueError("Duplicate CPU stage: " + stage)
        rows[stage] = dict(solver_ms=number(solver), full_ms=number(full), nets=int(nets))
        if rows[stage]["solver_ms"] > rows[stage]["full_ms"] + .01:
            raise ValueError("CPU solve time exceeds its enclosing stage")
    if not rows:
        metrics = re.findall(r"^CUGR_RUNTIME_METRICS ([\d.]+) ([\d.]+) ([\d.]+)$", text, re.MULTILINE)
        counts = re.findall(r"^CUGR_TASK_COUNTS (\d+) (\d+) (\d+)$", text, re.MULTILINE)
        full = re.findall(r"^CUGR_FULL_STAGE_METRICS ([\d.]+) ([\d.]+) ([\d.]+)$", text, re.MULTILINE)
        if metrics or counts or full:
            if len(metrics) != 1 or len(counts) != 1 or len(full) != 1:
                raise ValueError("Archived CPU log lacks unique timing/count records")
            skipped = "stage 3: maze routing skipped" in text
            for i, stage in enumerate(STAGES):
                if stage == "maze" and skipped:
                    continue
                rows[stage] = dict(solver_ms=1000 * number(metrics[0][i]),
                    full_ms=1000 * number(full[0][i]), nets=int(counts[0][i]))
    if list(rows) != list(STAGES[:len(rows)]):
        raise ValueError("CPU stages are out of order")
    return rows


def pending_cpu_tasks(text):
    planned = re.findall(r"^CUGR_STAGE_BEGIN stage=maze nets=(\d+)$", text, re.MULTILINE)
    if planned:
        return {"maze": int(planned[-1])}
    if "stage 3: maze routing skipped" in text:
        overflow = re.findall(r"(\d+) / \d+ nets have overflows\.", text)
        if overflow:
            return {"maze": int(overflow[-1])}
    return {}


def parse_grotor(text):
    starts = list(re.finditer(r"^ROUTING_STAGE_BEGIN stage=(\w+) nets=(\d+)$", text, re.MULTILINE))
    if [m[1] for m in starts] != list(STAGES):
        raise ValueError("Expected exactly three GRotor stages in pipeline order")
    rows = {}
    for i, start in enumerate(starts):
        stage, nets = start[1], int(start[2])
        region = text[start.end():starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        if len(re.findall(rf"^ROUTING_STAGE_END stage={stage}$", region, re.MULTILINE)) != 1:
            raise ValueError("Incomplete GRotor stage: " + stage)
        batches = re.findall(r"^BATCH_TIMING backend=(\w+) start_us=(\d+) end_us=(\d+) "
            r"elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+) used_for_routing=1$", region, re.MULTILINE)
        dags = re.findall(rf"^DAG_TIMING stage={stage} batch_index=-1 start_us=(\d+) end_us=(\d+) "
            r"elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+)$", region, re.MULTILINE)
        search = re.findall(r"^MAZE_SEARCH elapsed_ms=([\d.]+) nets=(\d+) threads=8 sparse_interval=10$",
                            region, re.MULTILINE)
        row = dict(nets=nets, batches=0, batch_ms=0., dag_ms=0., maze_search_ms=0., maze_other_ms=0., maze_phase_ms=0.)
        if nets:
            if len(batches) != 1 or len(dags) != 1:
                raise ValueError("Missing or duplicated batching/DAG measurements: " + stage)
            backend, begin, end, elapsed, batch_nets, count = batches[0]
            dag_begin, dag_end, dag_elapsed, dag_nets, dag_count = dags[0]
            if (backend != "rt_core" or int(batch_nets) != nets or int(dag_nets) != nets
                    or count != dag_count or not 0 < int(count) <= nets):
                raise ValueError("Stage work does not match the routed RT-Core batch plan")
            for first, last, milliseconds in ((begin, end, elapsed), (dag_begin, dag_end, dag_elapsed)):
                if int(last) < int(first) or abs((int(last) - int(first)) / 1000 - number(milliseconds)) > .001:
                    raise ValueError("Stage runtime is not its contiguous timestamp span")
            if int(dag_begin) < int(end):
                raise ValueError("DAG solving precedes batching")
            row.update(batches=int(count), batch_ms=number(elapsed), dag_ms=number(dag_elapsed))
            if stage == "maze":
                if len(search) != 1 or int(search[0][1]) != nets:
                    raise ValueError("Maze diagnostics require a CPU search measurement")
                row["maze_search_ms"] = number(search[0][0])
        elif batches or dags or search:
            raise ValueError("Empty stages must not contain solver work")
        row["total_ms"] = row["batch_ms"] + row["dag_ms"]
        if stage == "maze":
            phases = re.findall(r"^MAZE_PHASE_TIMING start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+)$", region, re.MULTILINE)
            if len(phases) != 1:
                raise ValueError("Maze requires one complete phase wall interval")
            begin, end, elapsed = phases[0]
            component_ms = row["total_ms"] + row["maze_search_ms"]
            if (int(end) < int(begin) or abs((int(end) - int(begin)) / 1000 - number(elapsed)) > .001
                    or number(elapsed) + .001 < component_ms):
                raise ValueError("Maze wall interval does not enclose its component work")
            row["maze_other_ms"] = max(0, number(elapsed) - component_ms)
            row["maze_phase_ms"] = number(elapsed)
            gates = re.findall(r"^MAZE_QUALITY_GATE accepted=([01]) before_score=(\S+) candidate_score=(\S+)$", region, re.MULTILINE)
            if nets and len(gates) != 1:
                raise ValueError("Maze requires a deterministic quality acceptance decision")
            if gates:
                row["candidate_accepted"] = gates[0][0] == "1"
                row["before_score"], row["candidate_score"] = gates[0][1:]
        rows[stage] = row
    return rows


def comparison_rows(result):
    rows = []
    for stage in STAGES:
        cpu = result.get("cpu", {}).get("stages", {}).get(stage)
        gpu = result.get("grotor", {}).get("stages", {}).get(stage)
        cpu_nets = cpu["nets"] if cpu else result.get("cpu", {}).get("pending_nets", {}).get(stage)
        gpu_nets = gpu["nets"] if gpu else None
        ratio = cpu_nets / gpu_nets if cpu_nets and gpu_nets else None
        if cpu is None or gpu is None:
            workload = "incomplete"
        elif not cpu_nets and not gpu_nets:
            workload = "empty"
        elif ratio is None:
            workload = "one-side-empty"
        else:
            limit = result["configuration"]["max_net_ratio"]
            workload = "comparable" if 1 / limit <= ratio <= limit else "different-scale"
        total_ms = gpu["batch_ms"] + gpu["dag_ms"] if gpu else None
        speedup = cpu["solver_ms"] / total_ms if (
            workload == "comparable" and total_ms > 0) else None
        rows.append(dict(dataset=result["dataset"], algorithm=LABELS[stage],
            cugr2_nets=cpu_nets, grotor_nets=gpu_nets, net_ratio=ratio,
            cugr2_ms=cpu["solver_ms"] if cpu else None,
            grotor_batch_ms=gpu["batch_ms"] if gpu else None,
            grotor_dag_ms=gpu["dag_ms"] if gpu else None,
            grotor_maze_search_ms=gpu["maze_search_ms"] if gpu else None,
            grotor_maze_other_ms=gpu["maze_other_ms"] if gpu else None,
            grotor_maze_phase_ms=gpu["maze_phase_ms"] if gpu else None,
            maze_candidate_accepted=gpu.get("candidate_accepted") if gpu else None,
            grotor_total_ms=total_ms,
            speedup=speedup, per_net_speedup=speedup / ratio if speedup is not None else None,
            cpu_task_status="completed" if cpu else "planned" if cpu_nets is not None else "unavailable",
            workload_status=workload))
    return rows


def write_report(folder, result):
    result["reporting_timing_policy"] = TIMING_POLICY
    result.setdefault("report_generation", {})["current_layout"] = dict(
        cpu_source="baseline/cu-gr-2", pipeline_header="src/pipeline_cugr2.hpp",
        grotor_target="grotor_cugr2", grotor_build_directory="build/GRotor-cugr2")
    rows = comparison_rows(result)
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (folder / "runtime.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (folder / "table3.csv").open("w", newline="") as stream:
        fields = ["dataset", "algorithm", "cugr2_s", "grotor_s", "speedup", "cugr2_nets", "grotor_nets"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(dataset=row["dataset"], algorithm=row["algorithm"],
                cugr2_s=row["cugr2_ms"] / 1000 if row["cugr2_ms"] is not None else None,
                grotor_s=row["grotor_total_ms"] / 1000 if row["grotor_total_ms"] is not None else None,
                speedup=row["speedup"], cugr2_nets=row["cugr2_nets"], grotor_nets=row["grotor_nets"]))
    def cell(value, digits=3):
        return "-" if value is None else f"{value:.{digits}f}"
    lines = ["# CUGR2 Routing Comparison", "",
        f"- Dataset: {result['dataset']}; status: {result['status']}",
        "- CPU measurement origin: " + result.get("cpu", {}).get("measurement_origin", "pending") + ".",
        "- Current layout: CPU sources in `baseline/cu-gr-2/`; GRotor target `grotor_cugr2` in `build/GRotor-cugr2/`.",
        "- Historical measurement commands, source paths and fingerprints are retained as originally recorded.",
        "- CPU input: LEF/DEF. GRotor input: ISPD24 cap/net. Task counts and quality are reported separately.",
        "- CUGR2: sum of per-net solver plus route commit regions; Maze includes sparse search and subsequent layer-assignment DP.",
        "- GRotor reported time for all three stages: measured RT-Core batching plus one continuous GPU batch-execution/update interval.",
        "- GRotor Maze CPU search/topology preparation, quality checks and restoration are excluded from comparison totals; full-phase timings remain diagnostic measurements.",
        "- Graph construction and I/O scopes differ; these are algorithm-component comparisons, not whole-router wall-time speedups.",
        f"- Comparable work requires both nonzero task counts within a factor of {result['configuration']['max_net_ratio']:g}; other speedups are omitted.", "",
        "| Algorithm | CUGR2 nets | GRotor nets | CUGR2 (ms) | GRotor (ms) | Speedup | Workload |",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        cpu_count = str(row['cugr2_nets']) if row['cugr2_nets'] is not None else '-'
        if row["cpu_task_status"] == "planned": cpu_count += " (planned)"
        lines.append(f"| {row['algorithm']} | {cpu_count} | "
            f"{row['grotor_nets'] if row['grotor_nets'] is not None else '-'} | {cell(row['cugr2_ms'])} | "
            f"{cell(row['grotor_total_ms'])} | {cell(row['speedup'], 2)} | {row['workload_status']} |")
    lines += ["", "## GRotor Components", "",
        "Reported runtime = RT-Core batching + DAG solve/update. CPU search/topology, other Maze work and full phase wall time are diagnostics only.", "",
        "| Algorithm | RT-Core batching (ms) | DAG solve/update (ms) | CPU search/topology (excluded ms) | Other Maze work (excluded ms) | Full Maze phase (diagnostic ms) |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['algorithm']} | {cell(row['grotor_batch_ms'])} | {cell(row['grotor_dag_ms'])} | {cell(row['grotor_maze_search_ms'])} | {cell(row['grotor_maze_other_ms'])} | {cell(row['grotor_maze_phase_ms'])} |")
    maze = result.get("grotor", {}).get("stages", {}).get("maze", {})
    if "candidate_accepted" in maze:
        lines += ["", "Maze candidate accepted: " + str(maze["candidate_accepted"]) +
            ". Acceptance uses the router's deterministic internal cost, not the external evaluator score. "
            "A rejected candidate restores the previous routes and demand. Candidate batching and GPU DAG execution remain included; "
            "quality checks and restoration are recorded only in the excluded full-phase diagnostics."]
    metrics = result.get("grotor", {}).get("metrics")
    with (folder / "quality.csv").open("w", newline="") as stream:
        fields = ["dataset", "implementation", "wirelength", "via", "overflow", "total", "open_nets", "incomplete_nets"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        if metrics:
            writer.writerow(dict(dataset=result["dataset"], implementation="GRotor", **metrics))
    if metrics:
        lines += ["", "## Quality", "",
            "GRotor's actual routing output was processed by the ISPD24 evaluator before deletion.", "",
            "| Wirelength cost | Via cost | Overflow cost | Total cost | Open nets | Incomplete nets |",
            "|---:|---:|---:|---:|---:|---:|",
            "| " + " | ".join(str(metrics[k]) for k in
                ("wirelength", "via", "overflow", "total", "open_nets", "incomplete_nets")) + " |"]
    native = result.get("cpu", {}).get("quality")
    if native:
        lines += ["", "CUGR2 native statistics (different input/resource model; not ISPD24 evaluator costs): "
            + ", ".join(f"{k}={v}" for k, v in native.items()) + "."]
    if result.get("cpu", {}).get("measurement_origin") == "archived":
        lines += ["", "CPU timings were imported from an existing log, not rerun. "
            "The original run did not record source, input or binary hashes; current-file fingerprints do not establish its exact historical configuration. "
            "Planned CPU task counts are distinguished from completed tasks. Skipped stages have no runtime or speedup."]
    if result.get("cpu", {}).get("measurement_origin") == "reused-verified":
        lines += ["", "CPU measurements were reused after matching input, CPU-source, binary and log fingerprints; "
            "GRotor and its evaluator were rerun. The original CPU record is retained in the local build cache."]
    if result.get("error"):
        lines += ["", "## Error", "", result["error"]]
    (folder / "summary.md").write_text("\n".join(lines) + "\n")
