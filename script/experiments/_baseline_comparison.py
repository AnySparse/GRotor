"""Run native and GRotor adapter pipelines; report algorithm runtimes without evaluation."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

from _datasets import BASELINE_CASES
from _support import ROOT, configure_build, local_path, prepare_output_folder, run_logged, sha256, source_manifest, write_csv
import _cugr2_metrics as cugr2
import _ggr_metrics as ggr


def main(kind):
    parser=argparse.ArgumentParser(description=f'Compare native {kind.upper()} with its GRotor routing branch; runtime only.')
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--cases',nargs='+',default=None)
    selection.add_argument('--case',help='Run one dataset instead of the paper subset')
    parser.add_argument('--data-dir',required=True,help='Directory containing matching <case>.cap/.net inputs')
    parser.add_argument('--lef',required=True,help='Technology LEF file')
    parser.add_argument('--def-dir',required=True,help='Directory containing <case>.def files')
    parser.add_argument('--output-dir',default='results/'+kind+'-comparison')
    parser.add_argument('--build-dir',type=Path,help='GRotor adapter build directory')
    parser.add_argument('--native-build-dir',type=Path)
    parser.add_argument('--cuda-arch',default='89')
    parser.add_argument('--nvcc',default=os.environ.get('CUDACXX',os.environ.get('NVCC','nvcc')))
    parser.add_argument('--optix-root',default=os.environ.get('OPTIX_ROOT_DIR'))
    parser.add_argument('--gpu',type=int,help='Physical GPU index; otherwise preserve the allocation')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--native-timeout',type=int,default=10800)
    parser.add_argument('--grotor-timeout',type=int,default=3600)
    parser.add_argument('--no-build',action='store_true')
    parser.add_argument('--clean-results',action='store_true')
    args=parser.parse_args()
    cases=[args.case] if args.case else args.cases or BASELINE_CASES
    if min(args.jobs,args.threads,args.repeats,args.native_timeout,args.grotor_timeout)<1 or not args.cuda_arch.isdigit() or len(cases)!=len(set(cases)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',c) for c in cases):
        parser.error('Use positive counts and unique dataset names')
    if args.gpu is not None:
        if args.gpu<0:parser.error('--gpu must be nonnegative')
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ['LC_ALL']='C'
    data=local_path(args.data_dir)
    defs=local_path(args.def_dir)
    lef=local_path(args.lef)
    files={case:[data/(case+'.cap'),data/(case+'.net'),lef,
                 defs/(('cluster' if case=='mempool_cluster' else case)+'.def')] for case in cases}
    for values in files.values():
        for p in values:
            if not p.is_file():parser.error('Missing input: '+str(p))
    folder=local_path(args.output_dir);prepare_output_folder(folder,args.clean_results)
    build=(args.build_dir or ROOT/('build/GRotor-'+kind)/('sm_'+args.cuda_arch)).resolve()
    native=(args.native_build_dir or ROOT/('build/'+('CUGR2' if kind=='cugr2' else 'GGR'))).resolve()
    result=dict(status='running',baseline=kind,configuration=dict(cases=cases,repeats=args.repeats,cuda_arch=args.cuda_arch,
        timing='native algorithm versus GRotor batching + DAG; no external evaluation'),cases=[])
    rows=[]

    def save():
        write_csv(folder/'runtime.csv',rows)
        (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        lines=[f'# {kind.upper()} / GRotor routing runtime','', 'Status: '+result['status'],'',
            '| Dataset | Algorithm | Baseline ms | GRotor ms | Speedup | Status |','|---|---|---:|---:|---:|---|']
        for row in rows:
            vals=[row['baseline_ms'],row['grotor_ms'],row['speedup']]
            lines.append('| '+row['dataset']+' | '+row['algorithm']+' | '+' | '.join('-' if v is None else f'{v:.3f}' for v in vals)+' | '+row['status']+' |')
        lines += ['', 'Times are medians across repeats. GRotor runtime = batching + routing DAG execution. Native CUGR2 measures solver/commit; native GGR includes batching and its GPU routing kernel sum.',
            'CUGR2 Maze CPU topology search and quality restoration are outside the GRotor comparison interval. Native and adapted inputs use different resource models; task counts remain in runtime.csv.',
            'Incomplete or timed-out stages have no runtime/speedup. External evaluator scores are neither run nor compared.']
        for case in result['cases']:
            if case.get('error'):lines += ['',case['name']+': '+case['error']]
        if result.get('error'):lines += ['',result['error']]
        (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    try:
        if not args.no_build:
            if kind=='cugr2':
                for command,label in [(['cmake','-S',str(ROOT/'baseline/cu-gr-2/src'),'-B',str(native),'-DCMAKE_BUILD_TYPE=Release'],'native-configure'),
                                      (['cmake','--build',str(native),'--target','route','-j',str(args.jobs)],'native-build')]:
                    code,_=run_logged(command,folder/(label+'.log'))
                    if code:raise RuntimeError('Native build failed: '+label)
            else:
                configure_build(args,native,folder,['gpugr','flute_cpp'],
                    ['-DPYTHON_EXECUTABLE='+sys.executable],source=ROOT/'baseline/GGR',label='native-')
        configure_build(args,build,folder,['grotor_'+kind],
            ['-DGROTOR_ENABLE_RTCORE=ON','-DGROTOR_FULL_COST_REFERENCE=OFF','-DGROTOR_BUILD_INSTANTGR2=OFF',
             '-DGROTOR_BUILD_'+kind.upper()+'_ROUTER=ON'],label='grotor-')
        binaries=[build/('grotor_'+kind)]
        binaries += [native/'route'] if kind=='cugr2' else list((native/'modules').glob('*.so'))
        result['binary_sha256']={str(p):sha256(p) for p in binaries}
        result['source_sha256']=source_manifest((ROOT/'baseline'/('cu-gr-2' if kind=='cugr2' else 'GGR'),))
        for name in cases:
            cap,net,lef,def_file=files[name]
            case=dict(name=name,status='running',inputs={str(p):sha256(p) for p in files[name]},runs=[])
            result['cases'].append(case)
            cf=folder/name;cf.mkdir()
            with tempfile.TemporaryDirectory(prefix='solutions-',dir=cf) as temp:
                temp=Path(temp)
                for index in range(args.repeats):
                    run=dict(repeat=index+1);case['runs'].append(run)
                    guide,output=temp/'native.guide',temp/'grotor.out'
                    command=([str(native/'route'),'-lef',str(lef),'-def',str(def_file),'-threads',str(args.threads),'-output',str(guide)] if kind=='cugr2' else
                        [sys.executable,str(ROOT/'script/_ggr.py'),'--module-dir',str(native/'modules'),'--lef',str(lef),'--def',str(def_file),'--design',name,'--guide',str(guide),'--threads',str(args.threads),'--rrr-iters','0'])
                    log=cf/f'native-{index+1}.log';run['native_command']=command
                    try:
                        code,seconds=run_logged(command,log,ROOT/'baseline/cu-gr-2/flute' if kind=='cugr2' else ROOT,args.native_timeout)
                        run.update(native_exit_status=code,native_wall_seconds=seconds)
                        if code or not guide.is_file() or not guide.stat().st_size:raise RuntimeError('Native routing failed: '+str(log))
                        run['native']=cugr2.parse_cpu(log.read_text()) if kind=='cugr2' else dict(zshape=ggr.parse_native(log.read_text()))
                        if kind=='cugr2' and set(run['native'])!=set(cugr2.STAGES):raise RuntimeError('Native CUGR2 did not complete all routing stages')
                    except subprocess.TimeoutExpired:
                        run['native_timeout']=True
                        run['native']=cugr2.parse_cpu(log.read_text()) if kind=='cugr2' else {}
                        case['error']='Native routing timed out; completed stages retained'
                    command=[str(build/('grotor_'+kind)),'-cap',str(cap),'-net',str(net),'-out',str(output),'--batching','rtcore','--sparsity','1']
                    run['grotor_command']=command;log=cf/f'grotor-{index+1}.log'
                    code,seconds=run_logged(command,log,ROOT,args.grotor_timeout)
                    run.update(grotor_exit_status=code,grotor_wall_seconds=seconds)
                    if code or not output.is_file() or not output.stat().st_size:raise RuntimeError('GRotor adapter failed: '+str(log))
                    text=log.read_text()
                    if 'GROTOR_CONFIG batching=rtcore cost=incremental prefix=bit memory=explicit' not in text:raise RuntimeError('Expected fully optimized adapter')
                    run['grotor']=cugr2.parse_grotor(text) if kind=='cugr2' else dict(zshape=ggr.parse_grotor(text))
                    guide.unlink(missing_ok=True);output.unlink()
            for stage,label in ([('lshape','L-shape'),('detour','Detour'),('maze','Maze')] if kind=='cugr2' else [('zshape','Z-shape')]):
                controls=[r['native'][stage] for r in case['runs'] if stage in r['native']]
                optimized=[r['grotor'][stage] for r in case['runs']]
                old=statistics.median(r['solver_ms'] if kind=='cugr2' else r['total_ms'] for r in controls) if len(controls)==args.repeats else None
                batch=statistics.median(r['batch_ms'] for r in optimized);dag=statistics.median(r['dag_ms'] for r in optimized);new=batch+dag
                rows.append(dict(dataset=name,algorithm=label,baseline_ms=old,grotor_ms=new,grotor_batch_ms=batch,grotor_dag_ms=dag,
                    speedup=old/new if old is not None and new>0 else None,
                    baseline_tasks=controls[0]['nets'] if controls else None,grotor_tasks=optimized[0]['nets'],
                    status='complete' if old is not None else 'incomplete'))
            case['status']='incomplete' if case.get('error') else 'passed';save()
        result['status']='passed' if all(c['status']=='passed' for c in result['cases']) else 'incomplete'
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        if result['cases'] and result['cases'][-1]['status']=='running':result['cases'][-1].update(status='failed',error=str(error))
    finally:
        save();print('Report:',folder/'summary.md',flush=True)
    return 0 if result['status']=='passed' else 1
