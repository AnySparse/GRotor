"""Strict parsers and tables for the single-stage GGR comparison."""
import csv
import json
import math
from pathlib import Path
import re


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


def table_rows(case, native, optimized):
    if not .5 <= optimized['nets']/native['nets'] <= 2:
        raise ValueError("Native and optimized net task counts differ by more than 2x")
    return [dict(dataset=case, component=name, ggr_ms=native[key], grotor_ms=optimized[key],
                 speedup=native[key]/optimized[key])
            for name,key in [('batching','batch_ms'),('routing_dag','dag_ms'),('total','total_ms')]]


def write_report(folder, record):
    folder=Path(folder)
    rows=table_rows(record['dataset'],record['native']['timing'],record['runs'][0]['timing'])
    with (folder/'runtime.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    native=record['native']['timing']; gpu=record['runs'][0]['timing']
    table3=dict(dataset=record['dataset'],ggr_nets=native['nets'],grotor_nets=gpu['nets'],
        ggr_batches=native['batches'],grotor_batches=gpu['batches'],
        ggr_total_ms=native['total_ms'],grotor_total_ms=gpu['total_ms'],speedup=native['total_ms']/gpu['total_ms'])
    with (folder/'table3.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table3));writer.writeheader();writer.writerow(table3)
    scores=[dict(dataset=record['dataset'],run=i+1,**run['quality']) for i,run in enumerate(record['runs'])]
    with (folder/'quality.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    lines=[f"# GGR Z-shape comparison: {record['dataset']}",'',
        '| Component | GGR (ms) | GRotor (ms) | Speedup |',
        '|---|---:|---:|---:|']
    lines += [f"| {r['component']} | {r['ggr_ms']:.3f} | {r['grotor_ms']:.3f} | {r['speedup']:.2f}x |" for r in rows]
    lines += ['',f"Routed net tasks: GGR {native['nets']:,}; GRotor {gpu['nets']:,}.",
        f"Batches: GGR {native['batches']:,}; GRotor {gpu['batches']:,}.",'',
        'Total = batching + routing DAG. Tables use the first run; additional GRotor runs check evaluator-score determinism.',
        'GGR uses LEF/DEF and its native cost model. GRotor uses CAP/NET and its ISPD24 cost model; quality scores are not interchangeable.',
        'GGR routing = full cost/prefix updates + DP/commit GPU times summed across batches (CPU preparation and dense-map clearing excluded).',
        'GRotor routing = one continuous synchronized batch-loop interval, including input copies, incremental cost/BIT updates, node-map resets, DP, conversion, and demand commits.',
        'GRotor batching includes RT-Core coloring, collection, and required disjoint-bounding-box refinement. OptiX scene construction and CPU topology encoding are outside these two intervals.',
        '', '| Run | Wirelength | Via | Overflow | Total cost | Open | Incomplete |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {r['run']} | {r['wirelength']} | {r['via']} | {r['overflow']} | {r['total']} | {r['open_nets']} | {r['incomplete_nets']} |" for r in scores]
    lines += ['', 'GRotor retains deterministic upper-layer legalization and duplicate-path merging from the adapter, outside GPU timing. No Detour or Maze stage runs.', '', 'Every GRotor score comes from evaluating an actual routing file. Native guides and GRotor routing files are deleted after use; logs and summaries are retained.','']
    (folder/'summary.md').write_text('\n'.join(lines))
    (folder/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
