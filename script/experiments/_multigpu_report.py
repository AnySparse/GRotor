"""Parse and plot measured final-batch NVSHMEM scaling results."""
import csv
import json
import math
import re
import statistics


def parse_run(text, gpus, stages=('lshape',)):
    devices = re.findall(r'^MULTIGPU_DEVICE rank=(\d+) pe=(\d+) device=(\d+) name=(.+)$', text, re.M)
    if len(devices) != gpus or {int(r[0]) for r in devices} != set(range(gpus)):
        raise ValueError('Missing or duplicate participating GPU ranks')
    if any(r != pe or r != device for r, pe, device, _ in devices):
        raise ValueError('MPI rank, NVSHMEM PE and visible GPU must agree')
    records = []
    for line in text.splitlines():
        if line.startswith('MULTIGPU_BATCH '):
            fields = dict(item.split('=', 1) for item in line.split()[1:])
            for key in ('gpus', 'batch_index', 'batches', 'nets'):
                fields[key] = int(fields[key])
            for key in ('elapsed_ms', 'preparation_ms', 'snapshot_ms', 'solve_ms', 'commit_ms'):
                fields[key] = float(fields[key])
                if not math.isfinite(fields[key]) or fields[key] < 0:
                    raise ValueError('Invalid multi-GPU timing')
            for key in ('finalization_ms', 'end_to_end_ms'):
                if key in fields:
                    fields[key] = float(fields[key])
                    if not math.isfinite(fields[key]) or fields[key] < 0:
                        raise ValueError('Invalid multi-GPU timing')
            if 'critical_rank' in fields:
                fields['critical_rank'] = int(fields['critical_rank'])
                if not 0 <= fields['critical_rank'] < gpus:
                    raise ValueError('Invalid critical GPU rank')
                if abs(sum(fields[k] for k in ('snapshot_ms', 'solve_ms', 'commit_ms')) - fields['elapsed_ms']) > .001:
                    raise ValueError('Batch timing components must cover the measured interval')
            if fields['gpus'] != gpus or fields['batch_index'] != fields['batches'] - 1 or fields['elapsed_ms'] <= 0:
                raise ValueError('Expected the final executed batch on the requested GPUs')
            stage = fields['stage']
            partitions = re.findall(rf'^MULTIGPU_PARTITION stage={stage} rank=(\d+) nets=(\d+) nodes=(\d+)$', text, re.M)
            if len(partitions) != gpus or {int(p[0]) for p in partitions} != set(range(gpus)):
                raise ValueError('Incomplete partition participation')
            if sum(int(p[1]) for p in partitions) != fields['nets']:
                raise ValueError('Partition net counts do not cover the final batch')
            fields['partitions'] = [dict(rank=int(p[0]), nets=int(p[1]), nodes=int(p[2])) for p in partitions]
            records.append(fields)
    if [r['stage'] for r in records] != list(stages):
        raise ValueError('Expected exactly one final-batch interval per requested stage: ' + ', '.join(stages))
    return records


def scaling_rows(result):
    rows = []
    for case in result['cases']:
        runs = [r for r in case['runs'] if r.get('status') == 'passed']
        for stage in result.get('configuration', {}).get('stages', ('lshape',)):
            reference = [b['elapsed_ms'] for r in runs if r['gpus'] == 1 for b in r['batches'] if b['stage'] == stage]
            if not reference:
                continue
            baseline = statistics.median(reference)
            for gpus in sorted({r['gpus'] for r in runs}):
                batches = [b for r in runs if r['gpus'] == gpus for b in r['batches'] if b['stage'] == stage]
                med = statistics.median(b['elapsed_ms'] for b in batches)
                rows.append(dict(dataset=case['name'], stage=stage, gpus=gpus, repeats=len(batches),
                    nets=batches[0]['nets'], batches=batches[0]['batches'],
                    elapsed_ms=med, min_ms=min(b['elapsed_ms'] for b in batches),
                    max_ms=max(b['elapsed_ms'] for b in batches), speedup=baseline / med,
                    efficiency=baseline / med / gpus,
                    preparation_ms=statistics.median(b['preparation_ms'] for b in batches),
                    finalization_ms=statistics.median(b.get('finalization_ms', 0) for b in batches),
                    snapshot_ms=statistics.median(b['snapshot_ms'] for b in batches),
                    solve_ms=statistics.median(b['solve_ms'] for b in batches),
                    commit_ms=statistics.median(b['commit_ms'] for b in batches)))
    return rows


def write_summary(folder, result, plot=True):
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    rows = scaling_rows(result)
    columns = ['dataset', 'stage', 'gpus', 'repeats', 'nets', 'batches', 'elapsed_ms', 'min_ms', 'max_ms',
               'speedup', 'efficiency', 'preparation_ms', 'finalization_ms', 'snapshot_ms', 'solve_ms', 'commit_ms']
    with (folder / 'scaling.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader(); writer.writerows(rows)
    quality = []
    for case in result['cases']:
        for run in case['runs']:
            if run.get('status') == 'passed':
                quality.append(dict(dataset=case['name'], implementation='GRotor_multi', gpus=run['gpus'], repeat=run['repeat'], **run['metrics']))
    with (folder / 'quality.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['dataset', 'implementation', 'gpus', 'repeat', 'wirelength', 'via',
                                        'overflow', 'total', 'open_nets', 'incomplete_nets'])
        writer.writeheader(); writer.writerows(quality)
    lines = ['# GRotor L-shape Final-Batch Scaling', '', f"Status: {result['status']}", '',
        'All L-shape batches execute with the same legacy BFS NVSHMEM implementation at every GPU count.',
        'Only the final executed batch is used for runtime and scaling. FLT is not executed.',
        'Its CUDA-event interval includes DP/traceback, separate wire/via commits, post-commit dirty cost updates,',
        'full local presum scans and required synchronization. BIT is disabled on all batches.',
        'elapsed_ms is the maximum participating rank time. Preparation and final collection are excluded.',
        'snapshot_ms names post-commit cost/presum maintenance; solve_ms excludes separate commits.',
        'Speedup = median T(1 GPU) / median T(N GPUs); this is not whole-router speedup.',
        'L-shape route outputs are independently evaluated for zero open/incomplete nets and identical costs across GPU counts.',
        'Byte-identical outputs reuse an evaluator result verified by SHA-256 within this experiment.',
        'The reference is the one-GPU L-shape output, not a full L-shape + FLT InstantGR2 result.',
        'Per-run GPU load/process/clock samples are recorded in *.gpu-load.jsonl.', '',
        '| Dataset | Stage | GPUs | Last-batch nets | Median ms | Speedup | Efficiency |',
        '|---|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['stage']} | {r['gpus']} | {r['nets']} | {r['elapsed_ms']:.3f} | {r['speedup']:.3f}x | {r['efficiency']:.1%} |" for r in rows]
    if result.get('error'):
        lines += ['', 'Error: ' + result['error']]
    (folder / 'summary.md').write_text('\n'.join(lines) + '\n')
    if plot and rows:
        plot_scaling(folder, rows)


def plot_scaling(folder, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, label, filename in [('speedup', 'Speedup over 1 GPU', 'multigpu-scaling'),
                                    ('elapsed_ms', 'Last-batch time (ms)', 'last-batch-runtime')]:
        stages = [s for s in ('lshape', 'flt') if any(r['stage'] == s for r in rows)]
        fig, axes = plt.subplots(1, len(stages), figsize=(4.7*len(stages), 3.5), constrained_layout=True, squeeze=False)
        for ax, stage in zip(axes[0], stages):
            title = {'lshape':'L-shape','flt':'FLT'}[stage]
            max_gpus = max(r['gpus'] for r in rows)
            if metric == 'speedup':
                ax.plot(range(1, max_gpus + 1), range(1, max_gpus + 1), '--', color='0.6', label='Ideal', linewidth=1)
            for dataset in dict.fromkeys(r['dataset'] for r in rows):
                data = [r for r in rows if r['dataset'] == dataset and r['stage'] == stage]
                ax.plot([r['gpus'] for r in data], [r[metric] for r in data], marker='o', label=dataset)
            ax.set(title=title, xlabel='Number of GPUs', ylabel=label, xticks=list(range(1, max_gpus + 1)))
            ax.grid(alpha=.25); ax.legend(fontsize=8)
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(folder / f'{filename}.{extension}', dpi=200)
        plt.close(fig)
