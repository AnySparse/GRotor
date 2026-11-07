"""Parse native and optimized CUGR2 algorithm runtimes."""
import re
import math
STAGES=("lshape","detour","maze")

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
