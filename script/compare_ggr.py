#!/usr/bin/env python3
"""Build and compare native GGR with GRotor's optimized Z-shape pipeline."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from ggr_reporting import parse_native, parse_grotor, write_report
from test_instantgr2 import ROOT, evaluator_metrics, sha256

CASES = ['ariane133_51','ariane133_68','nvdla','bsg_chip','mempool_tile','mempool_group']


def run(command, log, timeout):
    print(shlex.join(map(str,command)),flush=True)
    start=time.perf_counter()
    command=['/usr/bin/time','-f','PROCESS_RESOURCES wall_s=%e max_rss_kib=%M',*map(str,command)]
    with log.open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            status=process.wait(timeout=timeout)
        except BaseException:
            try: os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
    return status,time.perf_counter()-start


def manifest():
    files=[ROOT/'CMakeLists.txt',Path(__file__),ROOT/'script/ggr_reporting.py',ROOT/'script/run_ggr_native.py']
    for folder in ['src','include','third_party','resources','baseline/GGR']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and
            (p.suffix in {'.cu','.cpp','.cuh','.hpp','.h','.dat','.a','.txt'} or p.name=='UPSTREAM_REVISION'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,choices=CASES)
    parser.add_argument('--input-dir',type=Path,default=ROOT.parent/'ISPD24_tmp')
    parser.add_argument('--def-dir',type=Path,default=ROOT.parent/'ISPD24_def/def')
    parser.add_argument('--lef',type=Path,default=ROOT.parent/'ISPD24_def/lef/Nangate.lef')
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results/GGR')
    parser.add_argument('--optix-root',default=os.environ.get('OPTIX_ROOT_DIR'))
    parser.add_argument('--nvcc',default=os.environ.get('CUDACXX','/usr/local/cuda-12.4/bin/nvcc'))
    parser.add_argument('--cuda-arch',default='89')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--repeat',type=int,default=1,help='GRotor evaluator-score determinism runs; outputs reused')
    parser.add_argument('--reuse-native',type=Path,help='Reuse a verified native result from summary.json')
    parser.add_argument('--validate-batches',action='store_true')
    parser.add_argument('--no-build',action='store_true')
    parser.add_argument('--clean-results',action='store_true',help='Replace only this dataset result directory')
    args=parser.parse_args()
    reused = json.loads(args.reuse_native.read_text()) if args.reuse_native else None
    reuse_log = args.reuse_native.parent/'ggr.log' if args.reuse_native else None
    reuse_text = reuse_log.read_text() if reuse_log else None
    if min(args.repeat,args.jobs,args.threads,args.timeout)<1: parser.error('Counts and timeout must be positive')
    # Relative CLI paths always resolve from the caller, never from a vendor directory.
    inputs={name:path.resolve() for name,path in {
        'cap':args.input_dir/(args.case+'.cap'),'net':args.input_dir/(args.case+'.net'),
        'lef':args.lef,'def':args.def_dir/(args.case+'.def')}.items()}
    for path in inputs.values():
        if not path.is_file(): raise FileNotFoundError(path)
    result_root=args.results_dir.resolve();folder=result_root/args.case
    if result_root==ROOT or ROOT in result_root.parents and result_root in (ROOT/'src',ROOT/'baseline',ROOT/'include'):
        raise ValueError('Results must not replace source directories')
    if folder.exists():
        if not args.clean_results: raise FileExistsError(f'{folder}: use --clean-results to replace this case')
        if folder.is_symlink(): raise ValueError('Refusing to remove a symlink result directory')
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    native_build=ROOT/'build/GGR'; gpu_build=ROOT/f'build/GRotor-ggr/sm_{args.cuda_arch}'
    def checked(command,name,timeout=args.timeout):
        status,wall=run(command,folder/name,timeout)
        if status: raise RuntimeError(f'Command failed ({status}): {folder/name}')
        return wall
    if not args.no_build:
        if not args.optix_root: raise ValueError('Set OPTIX_ROOT_DIR or pass --optix-root')
        cuda=Path(shutil.which(args.nvcc) or args.nvcc).resolve()
        checked(['cmake','-S',ROOT/'baseline/GGR','-B',native_build,'-DCMAKE_BUILD_TYPE=Release',
            f'-DPYTHON_EXECUTABLE={sys.executable}',f'-DCMAKE_CUDA_COMPILER={cuda}',
            f'-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}'],'ggr.configure.log')
        checked(['cmake','--build',native_build,'--target','gpugr','flute_cpp','-j',args.jobs],'ggr.build.log')
        checked(['cmake','-S',ROOT,'-B',gpu_build,'-DCMAKE_BUILD_TYPE=Release',
            f'-DCMAKE_CUDA_COMPILER={cuda}',f'-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}',
            '-DGROTOR_ENABLE_RTCORE=ON','-DGROTOR_BUILD_GGR_ROUTER=ON',
            '-DGROTOR_FULL_COST_REFERENCE=OFF',f'-DOPTIX_ROOT_DIR={Path(args.optix_root).resolve()}'],'grotor.configure.log')
        checked(['cmake','--build',gpu_build,'--target','grotor_ggr','grotor_evaluator','-j',args.jobs],'grotor.build.log')
    flags=(gpu_build/'CMakeFiles/grotor_ggr.dir/flags.make').read_text()
    if any('-D'+key in flags for key in ['GROTOR_FULL_COST_REFERENCE','GROTOR_PREFIX_ACCESS_COUNTS','GROTOR_SERIAL_DAG']):
        raise ValueError('Comparison requires incremental cost/BIT and cooperative scheduling without profiling counters')
    if any('-D'+key not in flags for key in ['GROTOR_GGR_ROUTING=1','GROTOR_ENABLE_RTCORE=1']):
        raise ValueError('Wrong GRotor comparison executable configuration')
    record=dict(dataset=args.case,command=sys.argv,inputs={key:dict(path=str(path),sha256=sha256(path)) for key,path in inputs.items()},
        source_sha256=manifest(),runs=[],timing_policy='ggr-native-kernel-sum-vs-grotor-continuous-loop-v1')
    record['binary_sha256']={str(path.relative_to(ROOT)):sha256(path) for path in
        [gpu_build/'grotor_ggr',gpu_build/'grotor_evaluator',*(native_build/'modules').glob('*.so')]}
    if reused:
        if reused['dataset']!=args.case:
            raise ValueError('Reused native result belongs to a different dataset')
        for key in ('lef','def'):
            if reused['inputs'][key]['sha256'] != record['inputs'][key]['sha256']:
                raise ValueError('Reused native input differs: '+key)
        for key, digest in record['binary_sha256'].items():
            if key.startswith('build/GGR/modules/') and reused['binary_sha256'].get(key)!=digest:
                raise ValueError('Reused native module differs: '+key)
        for key, digest in record['source_sha256'].items():
            if (key.startswith('baseline/GGR/') or key=='script/run_ggr_native.py') and reused['source_sha256'].get(key)!=digest:
                raise ValueError('Reused native source differs: '+key)
        if parse_native(reuse_text)!=reused['native']['timing']:
            raise ValueError('Reused native log differs from summary timing')
    guide=folder/'ggr.guide';output=folder/'grotor.out'
    try:
        if reused:
            (folder/'ggr.log').write_text(reuse_text)
            record['native']=reused['native']
            record['native_reuse']=dict(summary=str(args.reuse_native.resolve()),log_sha256=sha256(folder/'ggr.log'))
        else:
            wall=checked([sys.executable,ROOT/'script/run_ggr_native.py','--lef',inputs['lef'],'--def',inputs['def'],
                '--design',args.case,'--guide',guide,'--rrr-iters',0,'--threads',args.threads],'ggr.log')
            if not guide.is_file() or guide.stat().st_size==0: raise RuntimeError('GGR did not write a routing guide')
            record['native']=dict(timing=parse_native((folder/'ggr.log').read_text()),wall_seconds=wall,guide_bytes=guide.stat().st_size,
                quality_note='Native LEF/DEF guides use a different cost model; no CAP/NET evaluator score is assigned.')
            guide.unlink()
        first=None
        for index in range(args.repeat):
            suffix='' if index==0 else f'.repeat{index+1}'
            log=f'grotor{suffix}.log';evaluation=f'grotor{suffix}.evaluate.log'
            wall=checked([gpu_build/'grotor_ggr','-cap',inputs['cap'],'-net',inputs['net'],'-out',output,
                '--batching','rtcore','--sparsity',1,'--validate-batches',int(args.validate_batches)],log)
            if not output.is_file() or output.stat().st_size==0: raise RuntimeError('GRotor did not write routing output')
            output_bytes=output.stat().st_size
            status,_=run([gpu_build/'grotor_evaluator',inputs['cap'],inputs['net'],output],folder/evaluation,args.timeout)
            quality=evaluator_metrics(folder/evaluation,status)
            timing=parse_grotor((folder/log).read_text())
            record['runs'].append(dict(timing=timing,quality=quality,wall_seconds=wall,output_bytes=output_bytes))
            output.unlink()
            if first is not None and quality!=first: raise RuntimeError('Repeated evaluator scores differ')
            first=quality
        write_report(folder,record)
    finally:
        guide.unlink(missing_ok=True);output.unlink(missing_ok=True)
    print((folder/'summary.md').read_text())
    print(f'Results: {folder}')


if __name__=='__main__':
    main()
