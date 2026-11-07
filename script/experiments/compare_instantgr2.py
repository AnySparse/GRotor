#!/usr/bin/env python3
"""Compare fully optimized GRotor with InstantGR2: total, batching and DAG runtimes."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from _datasets import DEFAULT_CASES, LARGE_CASES
from _comparison_report import comparison_rows, write_summary
from _rtcore_metrics import parse_totals
from _support import (ROOT, EVALUATOR, configure_build, evaluator_metrics, gpu_monitor,
    local_path, parse_batch_timings, parse_routing_timings, parse_rtcore_timings,
    prepare_output_folder, require_cuda, run_logged, sha256, source_manifest, validate_luts)


def run_router(args, build, inputs, output, folder, implementation, gpu):
    baseline = implementation == 'baseline'
    binary = build / ('instantgr2_reference' if baseline else 'GRotor')
    command = [str(binary), '-cap', str(inputs[0]), '-net', str(inputs[1]), '-out', str(output),
               '--batching', 'dual' if baseline else 'rtcore', '--sparsity', '1']
    if not baseline:
        command += ['--validate-batches', str(int(args.validate_batches)), '--profile-stages', '1']
    log = folder / (implementation + '.routing.log')
    record = dict(router_command=command)
    if baseline:
        status, seconds = run_logged(command, log, ROOT / 'resources/flute')
    else:
        rss = folder / 'host-memory.log'
        status, seconds, memory = gpu_monitor(['/usr/bin/time','-v','-o',str(rss),*command], log,
                                               folder / 'gpu-memory.csv', gpu)
        record.update(gpu_memory=memory, cpu_peak_mib=int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',rss.read_text())[1])/1024)
    record.update(router_exit_status=status, router_seconds=seconds)
    if status or not output.is_file() or not output.stat().st_size:
        raise RuntimeError('Routing failed: ' + str(log))
    text = log.read_text()
    if not baseline:
        for marker in ('GROTOR_CONFIG batching=rtcore cost=incremental prefix=bit memory=explicit',
                       'DAG_SCHEDULING lshape=warp flt=block'):
            if marker not in text.splitlines():
                raise RuntimeError('Unexpected optimized configuration: ' + marker)
    record['batch_timings'] = parse_batch_timings(text,'dual','cpu') if baseline else parse_rtcore_timings(text,args.validate_batches)
    record['dag_timings'], _ = parse_routing_timings(text,record['batch_timings'])
    if not baseline:
        record['totals'] = parse_totals(text,record['batch_timings'])
    evaluation = [str(EVALUATOR), *map(str,inputs), str(output)]
    elog = folder / (implementation + '.evaluator.log')
    code, seconds = run_logged(evaluation,elog)
    record.update(evaluator_command=evaluation, evaluator_exit_status=code,
                  evaluator_seconds=seconds, metrics=evaluator_metrics(elog,code))
    output.unlink()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES,
                        help='Default: all 12 non-tera paper cases; tera cases must be selected explicitly')
    parser.add_argument('--data-dir', help='Directory containing matching <case>.cap/.net inputs')
    parser.add_argument('--output-dir', default='results/instantgr2-comparison')
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--cuda-arch', default='89')
    parser.add_argument('--nvcc', default=os.environ.get('CUDACXX',os.environ.get('NVCC','nvcc')))
    parser.add_argument('--optix-root', default=os.environ.get('OPTIX_ROOT_DIR'))
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--gpu', type=int, help='Physical GPU index; otherwise preserve CUDA_VISIBLE_DEVICES')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--validate-batches', action='store_true', help='Optional correctness mode; leave off for performance')
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--clean-results', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--report-only', type=Path, help='Regenerate tables/plots from a saved summary without a GPU')
    args = parser.parse_args()
    if args.data_dir is None and not args.report_only:
        parser.error('--data-dir is required unless --report-only is used')
    if (min(args.repeats,args.jobs)<1 or args.warmups<0 or not args.cuda_arch.isdigit()
        or len(set(args.cases))!=len(args.cases) or any(not re.fullmatch(r'[A-Za-z0-9_-]+',c) for c in args.cases)
        or args.gpu is not None and args.gpu<0):
        parser.error('Use positive counts, a numeric architecture and unique case names')
    folder = local_path(args.output_dir)
    if args.report_only:
        source = args.report_only.resolve()
        result = json.loads(source.read_text())
        prepare_output_folder(folder,args.clean_results,(source,))
        write_summary(folder,result,plots=not args.no_plot)
        return 0 if result['status']=='passed' else 1
    data = local_path(args.data_dir)
    inputs = {case:[data/(case+'.'+ext) for ext in ('cap','net')] for case in args.cases}
    for files in inputs.values():
        for p in files:
            if not p.is_file(): parser.error('Missing input: '+str(p))
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['LC_ALL'] = 'C'
    require_cuda()
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
    hardware = subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip()
    if any(c in LARGE_CASES for c in args.cases) and int(hardware.split(',')[1].strip())<80*1024:
        parser.error('tera_cluster / tera_cluster_rank require an RTX Pro 6000-class large-memory RTX GPU; they are excluded by default')
    prepare_output_folder(folder,args.clean_results)
    build = (args.build_dir or ROOT/'build/GRotor'/('sm_'+args.cuda_arch)).resolve()
    result = dict(status='running',configuration=dict(cases=args.cases,repeats=args.repeats,warmups=args.warmups,
        sparsity=1,cost_update='incremental',prefix='bit',dag_lshape='warp',dag_flt='block',
        validate_batches=args.validate_batches,gpu=gpu,cuda_arch=args.cuda_arch),hardware=hardware,cases=[])
    try:
        validate_luts()
        configure_build(args,build,folder,['GRotor','instantgr2_reference','grotor_evaluator'],
                        ['-DGROTOR_ENABLE_RTCORE=ON','-DGROTOR_FULL_COST_REFERENCE=OFF','-DGROTOR_BUILD_INSTANTGR2=ON'])
        result['source_sha256'] = source_manifest()
        result['binary_sha256'] = {str(p):sha256(p) for p in [build/'GRotor',build/'instantgr2_reference',EVALUATOR,build/'rtcore_batches.ptx',build/'rtcore_cached.ptx']}
        for name in args.cases:
            case = dict(name=name,status='running',inputs={str(p):sha256(p) for p in inputs[name]},runs=[])
            result['cases'].append(case)
            cf=folder/name; cf.mkdir()
            try:
                expected=None
                with tempfile.TemporaryDirectory(prefix='routing-',dir=build) as temp:
                    output=Path(temp)/(name+'.out')
                    for index in range(-args.warmups,args.repeats):
                        pair=dict(name=name,warmup=index<0,repeat=index+1)
                        case['runs'].append(pair)
                        rf=cf/(('warmup-'+str(abs(index))) if index<0 else 'run-'+str(index+1));rf.mkdir()
                        order=('baseline','rtcore') if index%2==0 else ('rtcore','baseline')
                        for implementation in order:
                            pair[implementation]=run_router(args,build,inputs[name],output,rf,implementation,gpu)
                        comparison_rows(pair)
                        if expected is None:expected=pair['baseline']['metrics']
                        if pair['baseline']['metrics']!=expected:
                            raise RuntimeError('Evaluator results changed across repeats')
                case['status']='passed'
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
                case.update(status='failed',error=str(error))
            write_summary(folder,result)
        result['status']='passed' if all(c['status']=='passed' for c in result['cases']) else 'failed'
    except BaseException as error:
        result.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        if result['cases'] and result['cases'][-1]['status']=='running':
            result['cases'][-1].update(status=result['status'],error=str(error))
    finally:
        write_summary(folder,result,plots=not args.no_plot)
        print('Report:',folder/'summary.md',flush=True)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
