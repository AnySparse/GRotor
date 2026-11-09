"""Validate paired routing measurements and export the AE comparison."""
import csv
import json
import math
import statistics
from pathlib import Path
ALGORITHMS=(("lshape","L-shape","rounds"),("detour","FLT","cached"))
COSTS=("wirelength","via","overflow","total")

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


def aggregate(case):
    pairs = [run for run in case['runs'] if not run.get('warmup')]
    samples = [comparison_rows(run)[0] for run in pairs]
    if not samples:
        return []
    result = []
    for index, algorithm in enumerate(('L-shape', 'FLT')):
        row = dict(dataset=case['name'], algorithm=algorithm)
        row['instantgr2_cpu_batch_ms'] = statistics.median(
            next(t['elapsed_ms'] for t in pair['baseline']['batch_timings']
                 if t['stage'] == ('lshape','detour')[index] and t['backend']=='cpu') for pair in pairs)
        for key in ('instantgr2_batch_ms', 'grotor_batch_ms', 'instantgr2_dag_ms', 'grotor_dag_ms'):
            row[key] = statistics.median(sample[index][key] for sample in samples)
        for prefix in ('instantgr2', 'grotor'):
            row[prefix + '_total_ms'] = row[prefix + '_batch_ms'] + row[prefix + '_dag_ms']
        for name, suffix in (('batching_speedup', 'batch_ms'), ('dag_speedup', 'dag_ms'), ('total_speedup', 'total_ms')):
            row[name] = speedup(row['instantgr2_' + suffix], row['grotor_' + suffix])
        row['repeats'] = len(samples)
        result.append(row)
    return result


def plot_results(folder, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(r['dataset'] for r in rows))
    for stem, column, ylabel in [('batching', 'batching_speedup', 'Batching speedup over CUDA cores'),
                                  ('dag', 'dag_speedup', 'Routing DAG speedup'),
                                  ('total', 'total_speedup', 'Batching + routing DAG speedup')]:
        fig, axes = plt.subplots(2, 1, figsize=(max(7, len(names)*.65), 6), sharex=True, constrained_layout=True)
        for ax, algorithm in zip(axes, ('L-shape', 'FLT')):
            if stem == 'batching':
                for offset, key, label in ((-1,'instantgr2_cpu_batch_ms','CPU'),
                                           (0,'instantgr2_batch_ms','CUDA cores'),
                                           (1,'grotor_batch_ms','RT cores')):
                    values = [next(r[key] for r in rows if r['dataset']==name and r['algorithm']==algorithm) for name in names]
                    ax.bar([i+offset*.25 for i in range(len(names))], values, width=.24, label=label)
                ax.set(title=algorithm,ylabel='Batching runtime (ms)',yscale='log')
                ax.set_xticks(range(len(names)),names,rotation=40,ha='right')
                ax.legend()
                continue
            values = [next(r[column] for r in rows if r['dataset'] == name and r['algorithm'] == algorithm) for name in names]
            ax.bar(range(len(names)), [v if v is not None else float('nan') for v in values])
            ax.axhline(1, color='gray', linestyle='--', linewidth=.8)
            ax.set(title=algorithm, ylabel=ylabel)
            ax.set_xticks(range(len(names)), names, rotation=40, ha='right')
        for suffix in ('pdf', 'png', 'svg'):
            fig.savefig(folder / (stem + '.' + suffix), dpi=200)
        plt.close(fig)


def write_summary(folder, result, plots=False):
    from _support import write_csv
    rows = [row for case in result['cases'] if case['status'] == 'passed' for row in aggregate(case)]
    write_csv(folder / 'runtime.csv', rows)
    for filename, columns in (
        ('total-runtime.csv', ('dataset','algorithm','instantgr2_batch_ms','instantgr2_dag_ms','instantgr2_total_ms','grotor_batch_ms','grotor_dag_ms','grotor_total_ms','total_speedup')),
        ('batching-runtime.csv', ('dataset','algorithm','instantgr2_cpu_batch_ms','instantgr2_batch_ms','grotor_batch_ms','batching_speedup')),
        ('dag-runtime.csv', ('dataset','algorithm','instantgr2_dag_ms','grotor_dag_ms','dag_speedup'))):
        write_csv(folder / filename, [{k: row[k] for k in columns} for row in rows], columns)
    quality = [dict(dataset=c['name'], **c['runs'][0]['baseline']['metrics']) for c in result['cases'] if c['status']=='passed']
    write_csv(folder / 'quality.csv', quality)
    complete = result['status']=='passed' and len(result['cases'])==len(result['configuration']['cases'])
    geomean = {}
    if complete:
        for algorithm in ('L-shape','FLT'):
            group = [r for r in rows if r['algorithm']==algorithm]
            geomean[algorithm] = {key: math.exp(sum(math.log(r[key]) for r in group)/len(group))
                for key in ('batching_speedup','dag_speedup','total_speedup') if group and all(r[key] and r[key]>0 for r in group)}
    result['geomean'] = geomean
    (folder / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines=['# GRotor / InstantGR2 comparison','', 'Status: '+result['status'], '',
        'Component times are per-component medians across paired measured runs. Total = batching + routing DAG. Warmups and evaluation are excluded.',
        'InstantGR2 dual mode times CUDA batching and routes with the deterministic CPU plan, sparsity 1. GRotor uses RT-Core batching, incremental costs/BIT and warp/block DAG scheduling.',
        'GRotor DAG timers are continuous; InstantGR2 FLT retains its GPU-suffix sum.', '',
        '| Dataset | Algorithm | InstantGR2 total ms | GRotor total ms | Total speedup | Batching speedup | DAG speedup |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[r['instantgr2_total_ms'],r['grotor_total_ms'],r['total_speedup'],r['batching_speedup'],r['dag_speedup']]
        lines.append('| '+r['dataset']+' | '+r['algorithm']+' | '+' | '.join('-' if v is None else f'{v:.3f}' for v in values)+' |')
    lines += ['', 'Geometric means (only for a complete selected case set): '+json.dumps(geomean), '',
        'Every output is externally evaluated before deletion. Four evaluator cost strings must match, with zero open/incomplete nets. Logs, exact commands, hashes and all samples are retained.']
    for c in result['cases']:
        if c['status']!='passed': lines += ['', c['name']+': '+c.get('error',c['status'])]
    if result.get('error'):lines += ['',result['error']]
    (folder / 'summary.md').write_text('\n'.join(lines)+'\n')
    if plots and rows:
        plot_results(folder, rows)
