"""Parse native and optimized GGR algorithm runtimes."""
import re
import math

def single(pattern, text):
    records = re.findall(pattern, text, re.MULTILINE)
    if len(records) != 1:
        raise ValueError(f"Expected one measurement for {pattern!r}, found {len(records)}")
    return records[0]

def parse_native(text):
    batch, routing, maze_first, maze_total = map(float, single(
        r"^GGR_RUNTIME_METRICS ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)$", text))
    if maze_first or maze_total:
        raise ValueError("GGR comparison must disable Maze routing")
    batches, nets = map(int, single(r"number of batches: (\d+);\s+number of nets: (\d+)", text))
    if batch <= 0 or routing <= 0 or not nets or not batches:
        raise ValueError("Empty or invalid native GGR measurements")
    return dict(batch_ms=batch*1000, dag_ms=routing*1000, total_ms=(batch+routing)*1000,
                nets=nets, batches=batches)

def interval(start, end, elapsed):
    start, end, elapsed = int(start), int(end), float(elapsed)
    if end <= start or not math.isfinite(elapsed) or abs((end-start)/1000-elapsed) > 1.e-5:
        raise ValueError("Invalid continuous timing interval")
    return elapsed

def parse_grotor(text):
    backend, start, end, batch, nets, batches = single(
        r"^BATCH_TIMING backend=(rt_core|cpu) start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+) used_for_routing=1$", text)
    batch_ms = interval(start,end,batch)
    dag_start, dag_end, dag, dag_nets, dag_batches = single(
        r"^DAG_TIMING stage=zshape batch_index=-1 start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+)$",text)
    dag_ms = interval(dag_start,dag_end,dag)
    if int(dag_start)<int(end) or (nets,batches)!=(dag_nets,dag_batches):
        raise ValueError("Inconsistent Z-shape batch/DAG measurements")
    if "ROUTING_STAGE_END stage=zshape" not in text or "GROTOR_COMPLETE" not in text:
        raise ValueError("Routing did not complete")
    return dict(batch_ms=batch_ms, dag_ms=dag_ms, total_ms=batch_ms+dag_ms,
                nets=int(nets),batches=int(batches),batching=backend)
