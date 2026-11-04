#!/usr/bin/env python3
"""Compare an archived L-shape executor with GRotor and InstantGR2."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from test_instantgr2 import ROOT, BASELINE, evaluator_metrics, run_logged, sha256
from test_grotor_reference import local_path, source_manifest
from cpu_reporting import parse_cpu_run

CASES = ['mempool_tile', 'ariane133_51', 'ariane133_68', 'nvdla', 'bsg_chip', 'mempool_group']


def tables(folder, result):
    rows = []
    for case in result['cases']:
        if case.get('status') != 'passed':
            continue
        row = dict(dataset=case['name'])
        for kind in ('instantgr2', 'before', 'after'):
            runs = [r for r in case['runs'] if r['implementation'] == kind]
            row[kind+'_dag_ms'] = statistics.median(r['algorithms'][0]['dag_solve_ms'] for r in runs)
            row[kind+'_batch_ms'] = statistics.median(r['algorithms'][0]['cpu_batch_ms'] for r in runs)
            row[kind+'_total_ms'] = row[kind+'_dag_ms'] + row[kind+'_batch_ms']
            row[kind+'_wall_s'] = statistics.median(r['wall_seconds'] for r in runs)
        row['dag_speedup_vs_before'] = row['before_dag_ms']/row['after_dag_ms']
        row['dag_speedup_vs_instantgr2'] = row['instantgr2_dag_ms']/row['after_dag_ms']
        row['total_speedup_vs_before'] = row['before_total_ms']/row['after_total_ms']
        row['total_speedup_vs_instantgr2'] = row['instantgr2_total_ms']/row['after_total_ms']
        rows.append(row)
    if rows:
        with (folder/'runtime.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    quality = [dict(dataset=c['name'], implementation=r['implementation'], run=r['run'], **r['metrics'])
        for c in result['cases'] for r in c['runs']]
    if quality:
        with (folder/'quality.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(quality[0])); writer.writeheader(); writer.writerows(quality)
    lines = ['# L-shape warp scheduling comparison', '', 'Status: '+result['status'], '',
        'CPU batching, sparsity 1. Before and after use the same incremental cost/BIT pipeline. Before uses layer-state-parallel warps; after uses warp-per-DAG execution, ordered sibling merges and stable parallel traceback.',
        'DAG timings include cost/BIT maintenance, solving and demand commits; they are not isolated DP-kernel times. InstantGR2 retains full cost/prefix updates. Stage total adds actual CPU batching.',
        'Tables use medians over repeated before/after runs. Runs are sequential, using baseline, before, after, after, before order at two repeats. GPU clocks are not locked.', '',
        '| Dataset | InstantGR2 DAG (ms) | Before DAG (ms) | GRotor DAG (ms) | vs before | vs InstantGR2 |',
        '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {r['dataset']} | {r['instantgr2_dag_ms']:.3f} | {r['before_dag_ms']:.3f} | {r['after_dag_ms']:.3f} | {r['dag_speedup_vs_before']:.2f}x | {r['dag_speedup_vs_instantgr2']:.2f}x |" for r in rows]
    lines += ['', 'Every implementation writes one actual routing output and invokes the evaluator. All four evaluator cost strings must match the fresh InstantGR2 result with zero open/incomplete nets. No route bytes are compared; no raw or compressed routing outputs are retained.',
        'Depth tagging and ordered merging are inside the after routing-DAG interval. Existing layer-range storage holds depth tags temporarily; no global floating-point message cache or per-design depth arrays are added.',
        'Full batching, stage totals and process wall times are in runtime.csv. All evaluator scores are in quality.csv. Input/binary/source fingerprints and commands are in summary.json.']
    if result.get('error'): lines += ['', 'Error: '+result['error']]
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    (folder/'summary.json').write_text(json.dumps(result, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-router', required=True)
    parser.add_argument('--after-router', default='build/GRotor-lshape-node/sm_89/GRotor')
    parser.add_argument('--instantgr2', default='build/GRotor-lshape-before/sm_89/instantgr2_reference')
    parser.add_argument('--evaluator', default='build/GRotor-lshape-node/sm_89/grotor_evaluator')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=CASES)
    parser.add_argument('--data-dir', default='../ISPD24_tmp')
    parser.add_argument('--output-dir', default='results/GRotor-lshape-node')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--clean-results', action='store_true')
    args = parser.parse_args()
    if args.repeats < 2 or len(set(args.cases)) != len(args.cases): parser.error('Use at least two repeats and unique cases')
    binaries = {key: local_path(value) for key,value in dict(before=args.before_router, after=args.after_router,
        instantgr2=args.instantgr2, evaluator=args.evaluator).items()}
    for path in binaries.values():
        if not path.is_file(): raise FileNotFoundError(path)
    data, folder = local_path(args.data_dir), local_path(args.output_dir)
    for case in args.cases:
        for ext in ('cap', 'net'):
            if not (data/(case+'.'+ext)).is_file(): raise FileNotFoundError(data/(case+'.'+ext))
    if ROOT/'results' not in folder.parents: parser.error('Results must be under the repository results directory')
    if folder.exists():
        if not args.clean_results: parser.error('Results exist; use --clean-results')
        import shutil
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    result = dict(status='running', configuration=dict(repeats=args.repeats, batching='cpu', sparsity=1,
        sources_after=source_manifest(), runner_sha256=sha256(Path(__file__))),
        binary_sha256={key:sha256(path) for key,path in binaries.items()}, cases=[])
    checkpoint = binaries['before'].parent.parent/'checkpoint.json'
    if checkpoint.is_file():
        record = json.loads(checkpoint.read_text())
        if record['binary_sha256']['GRotor'] != result['binary_sha256']['before']:
            raise ValueError('Before binary differs from its archived source checkpoint')
        result['before_checkpoint'] = record
    output = folder/'routing.out'
    try:
        for name in args.cases:
            cf = folder/name; cf.mkdir()
            paths = [data/(name+'.'+ext) for ext in ('cap', 'net')]
            case = dict(name=name, status='running', inputs={str(p):sha256(p) for p in paths}, runs=[])
            result['cases'].append(case)
            plan = [('instantgr2',1)]
            for repeat in range(1,args.repeats+1):
                plan += [('before',repeat),('after',repeat)] if repeat%2 else [('after',repeat),('before',repeat)]
            expected = None
            for kind, repeat in plan:
                label = f'{kind}{repeat}'
                command = [str(binaries[kind]), '-cap',str(paths[0]), '-net',str(paths[1]), '-out',str(output),
                    '--batching','cpu','--sparsity','1']
                code, wall = run_logged(command,cf/(label+'.routing.log'), BASELINE/'run' if kind=='instantgr2' else ROOT)
                if code or not output.is_file() or not output.stat().st_size: raise RuntimeError('Router failed: '+label)
                text = (cf/(label+'.routing.log')).read_text()
                if kind=='after' and 'DAG_SCHEDULING lshape=warp flt=block' not in text:
                    raise ValueError('After binary must enable warp/block DAG scheduling')
                run = dict(implementation=kind,run=repeat,wall_seconds=wall,command=command,
                    **parse_cpu_run(text,kind!='instantgr2'))
                evaluation = cf/(label+'.evaluator.log')
                code,_ = run_logged([str(binaries['evaluator']),*map(str,paths),str(output)],evaluation)
                run['metrics'] = evaluator_metrics(evaluation,code)
                output.unlink()
                if expected is None: expected=run['metrics']
                if run['metrics']!=expected: raise ValueError(f'Evaluator mismatch: {name} {label}')
                reference = case['runs'][0] if case['runs'] else run
                if [(a['nets'],a['batches']) for a in run['algorithms']] != [(a['nets'],a['batches']) for a in reference['algorithms']]:
                    raise ValueError('Routing batch workload differs')
                case['runs'].append(run)
                print('PASS',name,label,run['metrics']['total'],flush=True)
                tables(folder,result)
            case['status']='passed'; tables(folder,result)
        result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error)); print('FAILED',error,flush=True)
    finally:
        output.unlink(missing_ok=True); tables(folder,result)
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    raise SystemExit(main())
