"""Validate cumulative CPU -> BIT -> scheduling -> RT-Core ablation results."""
import csv
import json
import math
from pathlib import Path
import re
from _support import parse_routing_timings
from _support import parse_rtcore_timings
from _rtcore_metrics import parse_totals

VARIANTS=('original','bit','schedule','rtcore')
LABELS=('Original','+BIT','+Schedule','+RT-Core')
ALGORITHMS=('L-shape','FLT')


def parse_run(text,variant):
    if variant not in VARIANTS:raise ValueError('Unknown ablation variant')
    expected_batch='rtcore' if variant=='rtcore' else 'cpu'
    cost='cost=full prefix=full' if variant=='original' else 'cost=incremental prefix=bit'
    schedule='lshape=thread flt=global_depth' if variant in ('original','bit') else 'lshape=warp flt=block'
    for expected in (f'GROTOR_CONFIG batching={expected_batch} {cost} memory=explicit',
                     f'DAG_SCHEDULING {schedule}',f'ABLATION_CONFIG variant={variant} counters=off batching={expected_batch}'):
        if text.splitlines().count(expected)!=1:raise ValueError('Unexpected ablation configuration: '+expected)
    if variant=='rtcore':
        batches=parse_rtcore_timings(text,False);parse_totals(text,batches)
    else:
        pattern=r'^BATCH_TIMING backend=(cpu) start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+) used_for_routing=(1)$'
        records=re.findall(pattern,text,re.M)
        if len(records)!=2 or re.findall(r'^BATCH_BACKEND=(\w+)$',text,re.M)!=['cpu','cpu']:
            raise ValueError('Both control stages must use CPU batching')
        batches=[]
        for stage,(_,a,b,ms,n,k,_) in zip(('lshape','detour'),records):
            row=dict(stage=stage,backend='cpu',start_us=int(a),end_us=int(b),elapsed_ms=float(ms),nets=int(n),batches=int(k),used_for_routing=True)
            if (not math.isfinite(row['elapsed_ms']) or row['elapsed_ms']<=0 or int(b)<=int(a)
                    or abs(float(ms)-(int(b)-int(a))/1000)>1e-5 or not 0<int(k)<=int(n)):
                raise ValueError('Invalid contiguous CPU batch timer')
            batches.append(row)
    dag,_=parse_routing_timings(text,batches)
    result=[]
    for algorithm,b,d in zip(ALGORITHMS,batches,dag):
        if (d['batch_index']!=-1 or d['end_us']<=d['start_us'] or not math.isfinite(d['elapsed_ms'])
                or abs(d['elapsed_ms']-(d['end_us']-d['start_us'])/1000)>1e-5):
            raise ValueError('Each DAG stage must use one synchronized continuous interval')
        result.append(dict(algorithm=algorithm,nets=b['nets'],batches=b['batches'],batching_ms=b['elapsed_ms'],
            dag_ms=d['elapsed_ms'],total_ms=b['elapsed_ms']+d['elapsed_ms']))
    return result


def table_rows(result):
    rows=[]
    for case in result['cases']:
        if case['status']!='passed':continue
        runs=case['runs']
        if set(runs)!=set(VARIANTS):raise ValueError('Each case must contain all four ablations')
        baseline=runs['original']['metrics']
        for variant in VARIANTS:
            run=runs[variant]
            if run['status']!='passed' or run['metrics']!=baseline or baseline['open_nets'] or baseline['incomplete_nets']:
                raise ValueError('All four actual routing evaluations must match')
            if [r['algorithm'] for r in run['timings']]!=list(ALGORITHMS):raise ValueError('Missing routing algorithm')
        for i,algorithm in enumerate(ALGORITHMS):
            controls=[runs[v]['timings'][i] for v in VARIANTS]
            if any((r['nets'],r['batches'])!=(controls[0]['nets'],controls[0]['batches']) for r in controls):
                raise ValueError('Ablations must route identical batch workloads')
            for j,(variant,t) in enumerate(zip(VARIANTS,controls)):
                if (any(not math.isfinite(t[k]) or t[k]<=0 for k in ('batching_ms','dag_ms','total_ms'))
                        or abs(t['total_ms']-t['batching_ms']-t['dag_ms'])>1e-6):
                    raise ValueError('Algorithm runtime must equal batching plus DAG execution')
                rows.append(dict(dataset=case['name'],algorithm=algorithm,variant=LABELS[j],nets=t['nets'],batches=t['batches'],
                    batching_ms=t['batching_ms'],dag_ms=t['dag_ms'],total_ms=t['total_ms'],
                    speedup=controls[0]['total_ms']/t['total_ms'],
                    incremental_speedup=controls[max(0,j-1)]['total_ms']/t['total_ms'],
                    dag_speedup=controls[0]['dag_ms']/t['dag_ms']))
    return rows


def plot_ablation(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(dict.fromkeys(r['dataset'] for r in rows))
    if not names:return
    if len(names)!=1:raise ValueError('Plot one specified dataset per invocation')
    name=names[0]
    colors=('#999999','#e5a641','#4088c5','#238b63')
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}):
        fig,axis=plt.subplots(figsize=(6.4,3.8))
        for j,label in enumerate(LABELS):
            values=[next(r['speedup'] for r in rows if r['algorithm']==algorithm and r['variant']==label) for algorithm in ALGORITHMS]
            x=[i+(j-1.5)*.19 for i in range(len(ALGORITHMS))]
            bars=axis.bar(x,values,width=.18,label=label,color=colors[j],edgecolor='white',linewidth=.4)
            axis.bar_label(bars,labels=[f'{v:.2f}' for v in values],fontsize=8,padding=2)
        axis.set(title=name,xlabel='Routing algorithm',ylabel='Speedup over Original')
        axis.set_xticks(range(len(ALGORITHMS)),ALGORITHMS)
        axis.axhline(1,color='#666666',linewidth=.7,linestyle='--');axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
        axis.margins(y=.18);axis.set_ylim(bottom=0)
        handles,labels=axis.get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.90))
        for suffix in ('pdf','svg','png'):
            fig.savefig(Path(str(output)+'.'+suffix),dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)


def write_summary(folder,result,plots=False):
    rows=table_rows(result)
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fields=['dataset','algorithm','variant','nets','batches','batching_ms','dag_ms','total_ms','speedup','incremental_speedup','dag_speedup']
    with (folder/'runtime.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    with (folder/'speedup.csv').open('w',newline='') as stream:
        fields=['dataset','algorithm','original','plus_bit','plus_schedule','plus_rtcore']
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        for c in result['cases']:
            if c['status']!='passed':continue
            for algorithm in ALGORITHMS:
                group=[r for r in rows if r['dataset']==c['name'] and r['algorithm']==algorithm]
                writer.writerow(dict(dataset=c['name'],algorithm=algorithm,**{key:r['speedup'] for key,r in zip(fields[2:],group)}))
    lines=['# Ablation: Cumulative Routing Ablation','',f"Status: {result['status']}",'',
        '| Variant | Cost / prefix update | L-shape DAG | FLT DAG | Batching |',
        '|---|---|---|---|---|','| Original | Full / dense scan | Thread per DAG | Global depth kernels | CPU |',
        '| +BIT | Incremental / BIT with dense query cache | Thread per DAG | Global depth kernels | CPU |',
        '| +Schedule | Incremental / BIT with dense query cache | Warp per DAG | Block per DAG | CPU |',
        '| +RT-Core | Incremental / BIT with dense query cache | Warp per DAG | Block per DAG | RT-Core |','',
        'These are cumulative configurations, not three independent single-feature runs. Original is the GRotor explicit-memory framework with all three algorithm optimizations disabled. Compact topology, bounded GPU assembly/staging, scalar node arithmetic and output overlap remain common engineering infrastructure. The global-depth baseline launches separate solve/trace kernels for every depth.','',
        'Algorithm time = measured batching + measured DAG execution. Speedup = Original algorithm time / variant algorithm time. Incremental speedup compares the preceding configuration to the current one. All four variants are measured independently; no batching or DAG components are copied between runs. L-shape and FLT each use one continuous synchronized DAG interval, without per-batch timer sums. FLT includes compact upload, GPU assembly, cost/prefix updates, solving, demand commits and waits.','',
        'RT-Core uses the standard L-shape method and cached FLT method for every dataset. Sparsity is 1. CPU batch validation is optional.','',
        'Prefix access profiling is disabled in all four executables. Single sequential runs use unlocked GPU clocks; GPU memory is sampled device-wide every 100 ms, and GNU time RSS is not a summed parent/child footprint. The tera_cluster family is excluded.','']
    for algorithm in ALGORITHMS:
        lines += [f'## {algorithm}','', '| Dataset | Variant | Batching ms | DAG ms | Total ms | Speedup | Previous / current |','|---|---|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['algorithm']==algorithm:
                lines.append(f"| {r['dataset']} | {r['variant']} | {r['batching_ms']:.3f} | {r['dag_ms']:.3f} | {r['total_ms']:.3f} | {r['speedup']:.3f}x | {r['incremental_speedup']:.3f}x |")
        lines.append('')
    lines += ['## Quality','','Every variant writes a real complete routing output and invokes the external evaluator. All four decimal costs must match with zero open/incomplete nets. Each output is removed before the next run; at most one temporary .out exists under the build directory.','',
        '| Dataset | Wirelength | Via | Overflow | Total | Open / incomplete |','|---|---:|---:|---:|---:|---|']
    for c in result['cases']:
        if c['status']=='passed':
            m=c['runs']['original']['metrics'];lines.append(f"| {c['name']} | {m['wirelength']} | {m['via']} | {m['overflow']} | {m['total']} | {m['open_nets']} / {m['incomplete_nets']} |")
    lines += ['','runtime.csv contains raw component times and both speedup definitions; speedup.csv is the wide plotting table. ablation.pdf/svg/png contains one linear-axis chart for the selected dataset, with L-shape and FLT as its two algorithm groups. Per-case routing/evaluator logs and memory observations are retained. summary.json records configuration, source/input/executable hashes, exact commands and per-run status.']
    if result.get('plot_generation'):lines += ['','Plots/tables regenerated from hash-checked saved routing/evaluator logs; no GPU work.']
    if result.get('error'):lines += ['','## Error','',result['error']]
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    if plots:plot_ablation(rows,folder/'ablation')
