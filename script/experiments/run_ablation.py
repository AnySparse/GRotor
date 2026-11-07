#!/usr/bin/env python3
"""Ablation: cumulative BIT, DAG scheduling and RT-Core routing speedups."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from _support import ROOT, evaluator_metrics, run_logged, sha256, require_cuda
from _support import local_path, source_manifest, validate_luts
from _support import gpu_monitor, parse_rtcore_timings
from _support import prepare_output_folder
from _ablation_report import VARIANTS, parse_run, table_rows, write_summary


def validate_case_name(name):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',name):
        raise ValueError('Use one simple dataset name without an extension')
    if name.lower().replace('-','_').startswith('tera_cluster'):
        raise ValueError('tera_cluster is excluded from this Ablation benchmark')


def validate_flags(build):
    cache=dict((line.split('=',1)[0].split(':',1)[0],line.split('=',1)[1])
        for line in (build/'CMakeCache.txt').read_text().splitlines() if '=' in line and not line.startswith(('#','//')))
    for key,value in (('GROTOR_BUILD_ROUTING_ABLATION','ON'),('GROTOR_ENABLE_RTCORE','ON'),('CMAKE_BUILD_TYPE','Release')):
        if cache.get(key)!=value:raise ValueError(f'Ablation requires {key}={value}; rerun without --no-build')
    for variant in VARIANTS:
        flags=(build/'CMakeFiles'/('grotor_ablation_'+variant+'.dir')/'flags.make').read_text()
        expected={'GROTOR_REPORT_ABLATION_CONFIG':True,'GROTOR_FULL_COST_REFERENCE':variant=='original',
                  'GROTOR_SERIAL_DAG':variant in ('original','bit'),'GROTOR_ENABLE_RTCORE':variant=='rtcore',
                  'GROTOR_PREFIX_ACCESS_COUNTS':False}
        for key,enabled in expected.items():
            if (('-D'+key+'=1') in flags)!=enabled:raise ValueError('Wrong compile switch: '+variant+' '+key)


def log_hashes(target):
    return {p.name:sha256(p) for v in VARIANTS for p in (target/(v+'.routing.log'),target/(v+'.evaluator.log'))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,help='One dataset name without an extension; excludes tera_cluster')
    parser.add_argument('--data-dir', help='Directory containing matching <case>.cap/.net inputs')
    parser.add_argument('--output-dir',help='Result directory (default: results/ablation/<case>)')
    parser.add_argument('--gpu',default='0');parser.add_argument('--cuda-arch',default='89')
    parser.add_argument('--nvcc',default=os.environ.get('CUDACXX',os.environ.get('NVCC','nvcc')))
    parser.add_argument('--optix-root',default=os.environ.get('OPTIX_ROOT_DIR'))
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--no-build',action='store_true');parser.add_argument('--clean-results',action='store_true')
    parser.add_argument('--plot-only',action='store_true',help='Recreate tables and plots from validated saved logs, without GPU work')
    parser.add_argument('--validate-batches',action='store_true',help='Also validate RT-Core batch lists against CPU; validation is outside measured intervals')
    parser.add_argument("--build-dir",type=Path,help="Reuse or create a custom build directory")
    args=parser.parse_args()
    if args.data_dir is None and not args.plot_only:
        parser.error('--data-dir is required unless --plot-only is used')
    try:validate_case_name(args.case)
    except ValueError as error:parser.error(str(error))
    if not args.gpu.isdigit() or not args.cuda_arch.isdigit() or args.jobs<1:parser.error('Use numeric GPU/architecture and positive jobs')
    folder=local_path(args.output_dir or f'results/ablation/{args.case}')
    if ROOT/'results' not in folder.parents:parser.error('Output must be below repository results/')
    try:import matplotlib
    except ImportError:parser.error('Plotting requires matplotlib: python3 -m pip install matplotlib')
    if args.plot_only:
        if args.clean_results:parser.error('--plot-only cannot delete its inputs with --clean-results')
        result=json.loads((folder/'summary.json').read_text())
        cases=result.get('cases',[])
        if len(cases)!=1 or cases[0]['name']!=args.case or cases[0]['status']!='passed':
            parser.error('Use a completed single-dataset report matching --case')
        case=cases[0]
        target=folder
        if log_hashes(target)!=case['log_sha256']:raise ValueError('Saved routing/evaluator logs changed: '+case['name'])
        for variant in VARIANTS:
            run=case['runs'][variant]
            run['timings']=parse_run((target/(variant+'.routing.log')).read_text(),variant)
            run['metrics']=evaluator_metrics(target/(variant+'.evaluator.log'),run['evaluator_exit_status'])
        table_rows(result)
        result['plot_generation']=dict(script_sha256=sha256(Path(__file__)),reporting_sha256=sha256(ROOT/'script/experiments/_ablation_report.py'),matplotlib=matplotlib.__version__)
        write_summary(folder,result,plots=True);print('Report:',folder/'summary.md');return 0 if result['status']=='passed' else 1
    data=local_path(args.data_dir)
    name=args.case
    for ext in ('cap','net'):
        if not (data/(name+'.'+ext)).is_file():parser.error('Missing input: '+str(data/(name+'.'+ext)))
    try:prepare_output_folder(folder,args.clean_results)
    except ValueError as error:parser.error(str(error))
    build=(args.build_dir or ROOT/'build/ablation'/('sm_'+args.cuda_arch)).resolve()
    result=dict(status='running',configuration=dict(cumulative=True,baseline_batching='cpu',sparsity=1,
        variants=list(VARIANTS),dataset=args.case,gpu=args.gpu,cuda_arch=args.cuda_arch,profiling_counters=False,
        validate_batches=args.validate_batches,sources=source_manifest(),script_sha256=sha256(Path(__file__)),
        reporting_sha256=sha256(ROOT/'script/experiments/_ablation_report.py'),matplotlib=matplotlib.__version__),cases=[])
    try:
        validate_luts()
        if not args.no_build:
            nvcc=shutil.which(args.nvcc)
            if not nvcc:raise RuntimeError('nvcc not found')
            config=['cmake','-S',str(ROOT),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_CUDA_COMPILER='+str(Path(nvcc).resolve()),'-DCMAKE_CUDA_ARCHITECTURES='+args.cuda_arch,
                '-DGROTOR_ENABLE_RTCORE=ON','-DGROTOR_BUILD_ROUTING_ABLATION=ON']
            if args.optix_root:config.append('-DOPTIX_ROOT_DIR='+str(Path(args.optix_root).resolve()))
            compile_cmd=['cmake','--build',str(build),'--target',*('grotor_ablation_'+v for v in VARIANTS),'grotor_evaluator','--parallel',str(args.jobs)]
            result['build_commands']=[config,compile_cmd]
            for command,label in ((config,'configure.log'),(compile_cmd,'build.log')):
                status,_=run_logged(command,folder/label)
                if status:raise RuntimeError('Ablation build failed: '+label)
        validate_flags(build)
        result['configuration']['cmake_cache_sha256']=sha256(build/'CMakeCache.txt')
        result['binaries']={p.name:sha256(p) for p in [*(build/('grotor_ablation_'+v) for v in VARIANTS),(ROOT/'resources/evaluator'),build/'rtcore_batches.ptx',build/'rtcore_cached.ptx']}
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu;os.environ['LC_ALL']='C';require_cuda()
        result['gpu_details']=subprocess.check_output(['nvidia-smi','--id='+args.gpu,'--query-gpu=name,memory.total,driver_version','--format=csv'],text=True)
        name=args.case
        target=folder;inputs=[data/(name+'.'+ext) for ext in ('cap','net')]
        case=dict(name=name,status='running',runs={},inputs={str(p):sha256(p) for p in inputs});result['cases'].append(case)
        write_summary(folder,result)
        with tempfile.TemporaryDirectory(prefix='ablation-evaluation-',dir=build) as temporary:
            output=Path(temporary)/(name+'.out')
            for variant in VARIANTS:
                run=dict(status='running');case['runs'][variant]=run;write_summary(folder,result)
                batch='rtcore' if variant=='rtcore' else 'cpu'
                command=[str(build/('grotor_ablation_'+variant)),'-cap',str(inputs[0]),'-net',str(inputs[1]),'-out',str(output),
                    '--batching',batch,'--sparsity','1','--validate-batches','1' if args.validate_batches and variant=='rtcore' else '0','--profile-stages','1']
                run['router_command']=command
                run['router_exit_status'],run['process_seconds'],run['gpu_memory']=gpu_monitor(
                    ['/usr/bin/time','-v','-o',str(target/(variant+'.host-memory.log')),*command],
                    target/(variant+'.routing.log'),target/(variant+'.gpu-memory.csv'),args.gpu)
                if run['router_exit_status'] or not output.is_file() or not output.stat().st_size:raise RuntimeError('Routing failed: '+name+' '+variant)
                text=(target/(variant+'.routing.log')).read_text();run['timings']=parse_run(text,variant)
                if variant=='rtcore' and args.validate_batches:parse_rtcore_timings(text,True)
                run['router_wall_seconds']=float(re.search(r'^GROTOR_COMPLETE wall_seconds=([\d.]+)$',text,re.M)[1])
                rss=(target/(variant+'.host-memory.log')).read_text()
                run['cpu_peak_mib']=int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',rss)[1])/1024
                evaluate=[str((ROOT/'resources/evaluator')),*map(str,inputs),str(output)];run['evaluator_command']=evaluate
                run['evaluator_exit_status'],run['evaluator_seconds']=run_logged(evaluate,target/(variant+'.evaluator.log'))
                run['metrics']=evaluator_metrics(target/(variant+'.evaluator.log'),run['evaluator_exit_status']);output.unlink()
                if variant!='original':
                    base=case['runs']['original']
                    if run['metrics']!=base['metrics']:raise ValueError('Evaluator costs differ: '+name+' '+variant)
                    if [(t['nets'],t['batches']) for t in run['timings']]!=[(t['nets'],t['batches']) for t in base['timings']]:
                        raise ValueError('Routing workloads differ: '+name+' '+variant)
                run['status']='passed';write_summary(folder,result)
                print('PASS',name,variant,[(t['algorithm'],round(t['total_ms'],3)) for t in run['timings']],flush=True)
        case['log_sha256']=log_hashes(target);case['status']='passed';write_summary(folder,result,plots=True)
        result['status']='passed';return 0
    except BaseException as error:
        result['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed';result['error']=str(error) or type(error).__name__
        if result['cases'] and result['cases'][-1]['status']=='running':
            case=result['cases'][-1];case['status']=result['status']
            for run in case['runs'].values():
                if run['status']=='running':run['status']=result['status']
        print('Ablation failed:',result['error'],file=sys.stderr);return 1
    finally:
        write_summary(folder,result,plots=True);print('Report:',folder/'summary.md',flush=True)


if __name__=='__main__':raise SystemExit(main())
