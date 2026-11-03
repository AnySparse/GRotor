"""Validate CPU-batching comparisons and export routing performance tables."""
import csv
import json
import math

from test_instantgr2 import parse_batch_timings, parse_routing_timings


def parse_cpu_run(text, optimized):
    batches = parse_batch_timings(text, "determinism", "cpu")
    dags, algorithms = parse_routing_timings(text, batches)
    if optimized:
        if "GROTOR_CONFIG batching=cpu cost=incremental prefix=bit memory=explicit" not in text:
            raise ValueError("Expected CPU batching, explicit storage and incremental cost/BIT")
        if "GROTOR_COMPLETE" not in text or "RTCORE_STAGE " in text or "RTCORE_SPACE " in text:
            raise ValueError("Incomplete run or unexpected RT-Core work")
        for dag in dags:
            if abs(dag['elapsed_ms'] - (dag['end_us'] - dag['start_us']) / 1000) > 1e-5:
                raise ValueError("GRotor must time one continuous interval per routing algorithm")
    return dict(batch_timings=batches, dag_timings=dags, algorithms=algorithms)


def comparison_rows(case):
    baseline = case['baseline']
    rows = []
    for run_number, run in enumerate(case['runs'], 1):
        if any(r['metrics'][key] for r in (baseline, run) for key in ('open_nets', 'incomplete_nets')):
            raise ValueError("Open or incomplete nets in evaluator result")
        if run['metrics'] != baseline['metrics']:
            raise ValueError("Evaluator scores differ from InstantGR2")
        for old, new in zip(baseline['algorithms'], run['algorithms']):
            if (old['algorithm'], old['nets'], old['batches']) != (new['algorithm'], new['nets'], new['batches']):
                raise ValueError("CPU batch-plan workload differs")
            before = old['cpu_batch_ms'] + old['dag_solve_ms']
            after = new['cpu_batch_ms'] + new['dag_solve_ms']
            if any(not math.isfinite(v) or v <= 0 for v in (before, after)):
                raise ValueError("Invalid total runtime")
            rows.append(dict(dataset=case['name'], run=run_number, algorithm=old['algorithm'],
                nets=new['nets'], batches=new['batches'],
                instantgr2_batch_ms=old['cpu_batch_ms'], grotor_batch_ms=new['cpu_batch_ms'],
                batch_speedup=old['cpu_batch_ms'] / new['cpu_batch_ms'] if new['cpu_batch_ms'] else None,
                instantgr2_dag_ms=old['dag_solve_ms'], grotor_dag_ms=new['dag_solve_ms'],
                dag_speedup=old['dag_solve_ms'] / new['dag_solve_ms'] if new['dag_solve_ms'] else None,
                instantgr2_total_ms=before, grotor_total_ms=after, total_speedup=before / after))
    return rows


def write_summary(folder, result):
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    rows, quality = [], []
    for case in result['cases']:
        if case.get('status') != 'passed':
            continue
        rows.extend(comparison_rows(case))
        for name, run_number, run in [('InstantGR2', 1, case['baseline'])] + [
                ('GRotor', i, run) for i, run in enumerate(case['runs'], 1)]:
            quality.append(dict(dataset=case['name'], implementation=name, run=run_number,
                **run['metrics']))
    columns = ['dataset', 'run', 'algorithm', 'nets', 'batches', 'instantgr2_batch_ms',
        'grotor_batch_ms', 'batch_speedup', 'instantgr2_dag_ms', 'grotor_dag_ms', 'dag_speedup',
        'instantgr2_total_ms', 'grotor_total_ms', 'total_speedup']
    tables = [('runtime.csv', columns, rows)]
    for component in ('batch', 'dag', 'total'):
        fields = ['dataset', 'run', 'algorithm', f'instantgr2_{component}_ms',
            f'grotor_{component}_ms', f'{component}_speedup']
        tables.append((dict(batch='batching.csv', dag='dag.csv', total='totals.csv')[component],
            fields, [{key: row[key] for key in fields} for row in rows]))
    tables.append(('quality.csv', ['dataset', 'implementation', 'run', 'wirelength', 'via',
        'overflow', 'total', 'open_nets', 'incomplete_nets'], quality))
    for filename, fields, values in tables:
        with (folder / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)
    lines = ['# GRotor CUDA-only comparison', '', f"Status: {result['status']}", '',
        'Both routers use CPU batching and sparsity 1. GRotor retains warp/block scheduling, BIT and incremental costs.',
        'Total = measured CPU batching + routing DAG time. GRotor uses one synchronized interval per algorithm; InstantGR2 retains its original FLT GPU-suffix accumulation.',
        'Every run writes an actual routing file and invokes the external evaluator. All four evaluator costs must match exactly, with zero open/incomplete nets. Routing files are deleted immediately after evaluation.', '',
        '| Dataset | Run | Algorithm | InstantGR2 total (ms) | GRotor total (ms) | Speedup |',
        '|---|---:|---|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['run']} | {r['algorithm']} | {r['instantgr2_total_ms']:.3f} | {r['grotor_total_ms']:.3f} | {r['total_speedup']:.2f}x |" for r in rows]
    lines += ['', 'Batching and DAG components are in runtime.csv, batching.csv, dag.csv and totals.csv. Actual evaluator costs are in quality.csv. Commands, hardware, source/input/binary hashes and process wall times are recorded in summary.json.']
    if result.get('error'):
        lines += ['', 'Error: ' + result['error']]
    (folder / 'summary.md').write_text('\n'.join(lines) + '\n')
