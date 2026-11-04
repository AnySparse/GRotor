"""Validate three prefix-access ablations and export Figure 11."""
import csv
import json
from pathlib import Path
from test_instantgr2 import evaluator_metrics

VARIANTS=('original','bit','bit-pruning')
FIELDS=['original','bit','bit_pruning']
RAW_FIELDS=['batch','color_id','nets','nodes','plan_hash','query_calls','query_reads','prefix_writes','total_accesses','maintenance_reads']


def read_variant(path):
    with path.open() as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=RAW_FIELDS:raise ValueError('Unexpected Figure 11 counter columns')
        rows=[]
        for i,raw in enumerate(reader):
            if set(raw)!=set(RAW_FIELDS):raise ValueError('Malformed Figure 11 counter row')
            row={k:int(raw[k]) for k in RAW_FIELDS}
            if any(v<0 for v in row.values()) or row['batch']!=i or row['nets']<1 or row['nodes']<row['nets']:
                raise ValueError('Invalid batch count, index or prefix access count')
            if row['total_accesses']!=row['query_reads']+row['prefix_writes']:
                raise ValueError('Requested total must equal query reads + prefix update writes')
            rows.append(row)
    if not rows:raise ValueError('No measured L-shape batches')
    return rows


def load_measurements(folder):
    metadata=json.loads((folder/'measurement.json').read_text())
    required=dict(status='completed',stage='lshape',shared_batch_plan=True,same_initial_demand=True,sparsity=1,
        solver='warp',counter='logical_prefix_elements',total_scope='query_reads_plus_update_writes',
        bit_storage='full_shifted_bit_without_dense_cache',first_batch_initialization_included=True,routing_outputs_retained=0)
    if any(metadata.get(k)!=v for k,v in required.items()):raise ValueError('Invalid Figure 11 measurement configuration')
    variants=[read_variant(folder/(name+'.csv')) for name in VARIANTS]
    if any(len(rows)!=metadata['batches'] for rows in variants):raise ValueError('Missing or extra variant batches')
    if sum(r['nets'] for r in variants[0])!=metadata['nets']:raise ValueError('Batches do not cover all routing nets')
    rows=[]
    for i,arms in enumerate(zip(*variants)):
        a,b,c=arms
        identity=('batch','color_id','nets','nodes','plan_hash','query_calls')
        if any(any(r[k]!=a[k] for k in identity) for r in arms[1:]):raise ValueError('Ablations must use identical DAGs, batch order and query workloads')
        if a['color_id']!=metadata['batches']-1-i:raise ValueError('Unexpected L-shape execution order')
        if a['prefix_writes']!=metadata['cells'] or a['query_reads']!=2*a['query_calls'] or a['maintenance_reads']:
            raise ValueError('Original must write the full prefix grid and read two endpoints per query')
        if any(r['prefix_writes']>metadata['cells'] for r in (b,c)):
            raise ValueError('BIT node should be rebuilt at most once per batch')
        if i==0 and any(r['prefix_writes']!=metadata['cells'] for r in (b,c)):
            raise ValueError('First batch must include complete BIT initialization')
        if b['prefix_writes']!=c['prefix_writes'] or b['maintenance_reads']!=c['maintenance_reads']:
            raise ValueError('BIT arms must have identical incremental update workloads')
        if c['query_reads']>b['query_reads']:raise ValueError('Pruning cannot add BIT query reads')
        row={k:a[k] for k in identity}
        for key,arm in zip(FIELDS,arms):
            row[key]=arm['total_accesses']
            for name in ('query_reads','prefix_writes','maintenance_reads'):row[key+'_'+name]=arm[name]
        rows.append(row)
    codes=metadata['evaluator_exit_status']
    if len(codes)!=3:raise ValueError('Every actual routing output must be externally evaluated')
    metrics=[evaluator_metrics(folder/(name+'.evaluator.log'),code) for name,code in zip(VARIANTS,codes)]
    if any(m!=metrics[0] for m in metrics[1:]):raise ValueError('L-shape evaluator costs differ between prefix variants')
    checks=dict(status='passed',stage='lshape',batches=len(rows),nets=metadata['nets'],shared_batch_plan=True,
        identical_query_workloads=True,identical_bit_update_workloads=True,all_four_evaluator_costs_equal=True,
        metrics=metrics[0],routing_outputs_retained=0)
    for key in FIELDS:
        checks[key+'_total']=sum(r[key] for r in rows)
        for name in ('query_reads','prefix_writes','maintenance_reads'):
            checks[key+'_'+name]=sum(r[key+'_'+name] for r in rows)
    return rows,checks


def plot_accesses(name,rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}):
        fig,axis=plt.subplots(figsize=(4.8,3.6))
        x=[r['batch'] for r in rows];markers=max(1,len(rows)//30)
        for key,label,color,marker,style in zip(FIELDS,('Original','BIT','BIT & Pruning'),('black','#1754df','#00875a'),('s','^','o'),('-','--','-')):
            axis.plot(x,[r[key] for r in rows],label=label,color=color,marker=marker,linestyle=style,
                markerfacecolor='white',markersize=3,markevery=markers,linewidth=1.1)
        axis.set(title=name,xlabel='Batch',ylabel='Prefix accesses (reads + writes)',yscale='linear')
        axis.set_ylim(bottom=0)
        axis.set_xlim((0,len(rows)-1) if len(rows)>1 else (-.5,.5))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
        axis.grid(axis='y',which='major',linestyle='--',linewidth=.6,alpha=.5)
        handles,labels=axis.get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.91))
        for suffix in ('pdf','svg','png'):
            fig.savefig(Path(str(output)+'.'+suffix),dpi=300,facecolor='white',bbox_inches='tight')
        plt.close(fig)


def export_case(name,folder):
    rows,checks=load_measurements(folder)
    with (folder/'accesses.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,FIELDS,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    with (folder/'batch-accesses.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[f'# Figure 11: {name}','','Each entry is query reads + prefix update writes. Row 1 is execution batch 0; all batches are retained.','',
        '| Original | BIT | BIT & Pruning |','|---:|---:|---:|']
    lines += [f"| {r['original']} | {r['bit']} | {r['bit_pruning']} |" for r in rows]
    (folder/'accesses.md').write_text('\n'.join(lines)+'\n')
    (folder/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    plot_accesses(name,rows,folder/'figure11')
    return checks


def write_summary(folder,result):
    (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Figure 11: Per-Batch Prefix Memory Accesses','',f"Status: {result['status']}",'',
        'Total per batch = writes to the prefix structure during its update + reads from that structure during wire-cost queries. Counters measure logical source-level element accesses, not bytes, cache misses, coalesced transactions or DRAM traffic. Repeated and zero-sentinel reads count; shared-memory scans, input costs, metadata, dirty masks and instrumentation traffic do not. BIT maintenance reads are recorded separately and excluded from the requested total.','',
        'Original uses full cost updates and a full dense prefix scan. BIT and BIT & Pruning use identical incremental cost/dirty-node updates and a full shifted BIT without materializing a dense prefix cache. BIT independently accumulates both endpoint prefixes and subtracts; pruning stops both traversals at their common ancestor. Only query mode differs between the two BIT arms.','',
        'All three arms use warp-per-DAG solving, one immutable CPU batch plan with sparsity 1, shared RSMT topology, and identical reset initial demand. Counts describe executed traversals and are never divided or capped. Keeping DAG scheduling fixed isolates prefix data structures. Initialization writes in batch 0 are included; every batch is measured and plotted, with no interpolation or point removal. Instrumented runtimes are not performance results.','',
        '| Dataset | Batches | Original accesses | BIT accesses | BIT & Pruning accesses |','|---|---:|---:|---:|---:|']
    for c in result['cases']:
        if c['status']=='passed':
            m=c['checks'];lines.append(f"| {c['name']} | {m['batches']} | {m['original_total']} | {m['bit_total']} | {m['bit_pruning_total']} |")
    lines += ['','## Quality','','Each arm writes an actual L-shape-only output, invokes the external evaluator and deletes the output before the next arm. All four decimal evaluator costs must match, with zero open/incomplete nets. No route bytes are compared. These scores precede FLT rerouting.','',
        '| Dataset | Wirelength | Via | Overflow | Total | Open / incomplete |','|---|---:|---:|---:|---:|---|']
    for c in result['cases']:
        if c['status']=='passed':
            m=c['checks']['metrics'];lines.append(f"| {c['name']} | {m['wirelength']} | {m['via']} | {m['overflow']} | {m['total']} | {m['open_nets']} / {m['incomplete_nets']} |")
    lines += ['','## Files and interpretation','',
        'The selected result directory contains accesses.csv/md (three total columns), batch-accesses.csv (batch identity plus reads/writes/maintenance reads), raw original.csv/bit.csv/bit-pruning.csv, evaluator logs, measurement.json, checks.json, worker.log, host-memory.log, gpu-memory.csv and figure11.pdf/svg/png. Every chart uses a linear vertical axis starting at zero. Saved raw counts and evaluator logs are checked before plot-only regeneration.','',
        'This is an isolated direct-query ablation. The normal router currently maintains a dense query cache after BIT updates; that cache is deliberately absent in both experimental BIT arms. Normal routing and Figure 10 do not enable the compile-time access instrumentation. Counts describe the current implementation and need not reproduce numerical values in an earlier paper draft.','',
        'Memory sampling covers the whole worker including external CPU evaluation. GPU memory samples are device-wide every 100 ms. GNU time RSS is not a sum of simultaneously resident parent and child memory.']
    if result.get('plot_generation'):lines += ['','Plots regenerated without routing or GPU work.']
    if result.get('error'):lines += ['','## Error','',result['error']]
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
