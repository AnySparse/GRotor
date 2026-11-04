#!/usr/bin/env python3
"""Measure the unchanged legacy NVSHMEM executors on an idle GPU pair."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / 'GRaptor_multi'
sys.path.insert(0, str(ROOT / 'script'))
from test_instantgr2 import evaluator_metrics, run_logged, sha256


def idle(devices):
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    uuids = dict(tuple(v.strip() for v in row.split(',')) for row in inventory.splitlines())
    selected = {uuids.get(d, d) for d in devices}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], text=True)
    busy = [row for row in raw.splitlines() if row.split(',')[0].strip() in selected]
    if busy:
        raise RuntimeError('Selected GPU has another compute process: ' + '; '.join(busy))
    return dict(unix_time=time.time(), devices=devices, conflicting_processes=busy)


def read_batch(rank_root, count, mode):
    ranks = []
    for rank in range(count):
        paths = list(rank_root.glob(f'**/rank.{rank}/stdout'))
        if len(paths) != 1:
            raise RuntimeError(f'Expected one stdout file for rank {rank}: {paths}')
        text = paths[0].read_text()
        # The legacy progress printf omits a newline before the FIRST batch.
        # Match its record even when prefixed by "Begin detour kernels".
        matches = re.findall(r'Batch (\d+), Size: (\d+), ([\d.eE+-]+) ms\b', text)
        if not matches:
            raise RuntimeError(f'No legacy batch records: {paths[0]}')
        records = [dict(batch_index=int(i), nets=int(n), elapsed_ms=float(t)) for i, n, t in matches]
        ids = [r['batch_index'] for r in records]
        expected = list(range(len(records)))
        if mode == 'detour':
            expected.reverse()
        if ids != expected:
            raise RuntimeError(f'Incomplete or unexpected legacy batch order on rank {rank}')
        last = records[-1]
        if not 0 < last['elapsed_ms'] < 1e6:
            raise RuntimeError('Invalid legacy final-batch time')
        ranks.append(dict(rank=rank, batches=len(records), log=str(paths[0]), **last))
    if len({r['batch_index'] for r in ranks}) != 1 or len({r['batches'] for r in ranks}) != 1:
        raise RuntimeError('GPU ranks timed different batches')
    critical = max(ranks, key=lambda r: r['elapsed_ms'])
    return dict(stage='flt' if mode == 'detour' else 'lshape', batch_index=critical['batch_index'],
                batches=critical['batches'], nets=sum(r['nets'] for r in ranks),
                elapsed_ms=critical['elapsed_ms'], critical_rank=critical['rank'], ranks=ranks)


def summary(folder, result):
    (folder / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    rows = []
    for mode in ('detour', 'p2p'):
        passed = [r for r in result['runs'] if r['mode'] == mode and r['status'] == 'passed']
        single = [r['batch']['elapsed_ms'] for r in passed if r['gpus'] == 1]
        for count in (1, 2):
            runs = [r for r in passed if r['gpus'] == count]
            if not runs or not single:
                continue
            times = [r['batch']['elapsed_ms'] for r in runs]
            median = statistics.median(times)
            rows.append(dict(implementation='legacy', stage=runs[0]['batch']['stage'], gpus=count,
                repeats=len(times), nets=runs[0]['batch']['nets'], batches=runs[0]['batch']['batches'],
                elapsed_ms=median, min_ms=min(times), max_ms=max(times), speedup=statistics.median(single)/median))
    with (folder / 'scaling.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, ['implementation','stage','gpus','repeats','nets','batches','elapsed_ms','min_ms','max_ms','speedup'])
        w.writeheader(); w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-dir', default='/data0/cwc/ISPD24')
    parser.add_argument('--devices', default='1,2')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    folder = Path(args.output_dir).resolve()
    if ROOT / 'results' not in folder.parents or args.repeats < 1:
        parser.error('Use a new directory under GRotor/results and positive repeats')
    folder.mkdir(parents=True, exist_ok=False)
    devices = args.devices.split(',')
    if len(devices) != 2 or len(set(devices)) != 2:
        parser.error('Select two distinct idle GPUs')
    source_files = sorted(p for p in (LEGACY / 'src').rglob('*') if p.is_file())
    result = dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        configuration=vars(args), original_sources={str(p.relative_to(LEGACY)): sha256(p) for p in source_files},
        original_binary_sha256=sha256(LEGACY/'run/graptor_multi'), runner_sha256=sha256(Path(__file__)), runs=[])
    try:
        data = Path(args.data_dir)
        inputs = [data/'mempool_cluster.cap', data/'mempool_cluster.net']
        result['inputs'] = {str(p): sha256(p) for p in inputs}
        result['hardware'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True)
        result['topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
        os.environ['LD_LIBRARY_PATH'] = ':'.join([str(Path.home()/'opt/nccl/lib'),str(Path.home()/'opt/nvshmem/lib'),str(Path.home()/'opt/openmpi/lib'),os.environ.get('LD_LIBRARY_PATH','')])
        os.environ['NVSHMEM_SYMMETRIC_SIZE'] = '25G'
        os.environ['NVSHMEM_INFO'] = '1'
        os.environ['NVSHMEM_DEBUG'] = 'WARN'
        os.environ['NVSHMEM_REMOTE_TRANSPORT'] = 'none'
        os.environ.pop('NVSHMEM_DISABLE_NCCL', None)
        result['environment'] = {k:v for k,v in os.environ.items() if k.startswith(('NVSHMEM_', 'NCCL_')) or k=='LD_LIBRARY_PATH'}
        # Preserve both repositories: only an isolated source copy changes its
        # two existing mode switches. No GPU kernel or timer is edited.
        copied = folder/'source-p2p'
        shutil.copytree(LEGACY/'src', copied)
        common = copied/'common/common.h'
        before = common.read_text()
        assert before.count('#define TEST_DETOUR 1') == 1
        assert before.count('// #define USE_P2P 1') == 1
        common.write_text(before.replace('#define TEST_DETOUR 1','// #define TEST_DETOUR 1').replace('// #define USE_P2P 1','#define USE_P2P 1'))
        result['mode_switches'] = ['TEST_DETOUR: enabled -> disabled', 'USE_P2P: disabled -> enabled']
        result['copied_common_sha256'] = sha256(common)
        p2p = folder/'graptor_p2p_sm80'
        build = ['/usr/local/cuda-12.2/bin/nvcc','-ccbin',str(Path.home()/'opt/openmpi/bin/mpicxx'),'-x','cu',str(copied/'main.cu'),'-o',str(p2p),
            '-I'+str(Path.home()/'opt/nccl/include'),'-L'+str(Path.home()/'opt/nccl/lib'),'-lnccl',
            '-I'+str(Path.home()/'opt/nvshmem/include'),'-L'+str(Path.home()/'opt/nvshmem/lib'),'-lnvshmem_host','-lnvshmem_device','-lcuda',
            '-std=c++17','-O3','-arch=sm_80','-rdc=true']
        result['build_command'] = build
        summary(folder,result)
        code, _ = run_logged(build, folder/'p2p-build.log', copied)
        if code:
            raise RuntimeError('Legacy P2P build failed; see p2p-build.log')
        result['p2p_binary_sha256'] = sha256(p2p)
        evaluator = ROOT/'build/GRotor-multi-grid/sm_80/grotor_evaluator'
        result['evaluator_sha256'] = sha256(evaluator)
        for repeat in range(1,args.repeats+1):
            for mode in ('detour','p2p'):
                binary = LEGACY/'run/graptor_multi' if mode=='detour' else p2p
                for count in (1,2):
                    label = f'{mode}-{count}gpu-{repeat}'
                    work = folder/label; work.mkdir()
                    for name in ('POWV9.dat','POST9.dat'):
                        (work/name).symlink_to(LEGACY/'run'/name)
                    output = work/'routing.out'
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices[:count])
                    check = idle(devices[:count])
                    command = [str(Path.home()/'opt/openmpi/bin/mpirun'),'--bind-to','none','-np',str(count),
                        '--output-filename',str(work/'ranks'),str(binary),'-cap',str(inputs[0]),'-net',str(inputs[1]),'-out',str(output)]
                    run = dict(mode=mode,gpus=count,repeat=repeat,status='running',idle_check=check,
                        cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],command=command,started_unix=time.time())
                    result['runs'].append(run); summary(folder,result)
                    code, wall = run_logged(command, work/'routing.log',work)
                    run.update(returncode=code,wall_seconds=wall)
                    if code or not output.is_file() or not output.stat().st_size:
                        raise RuntimeError(f'Legacy run failed: {work}')
                    run['batch'] = read_batch(work/'ranks',count,mode)
                    previous = [r['batch'] for r in result['runs'] if r is not run and r['mode']==mode and r['status']=='passed']
                    if previous and any((b['batch_index'],b['batches'],b['nets']) != (run['batch']['batch_index'],run['batch']['batches'],run['batch']['nets']) for b in previous):
                        raise RuntimeError('Legacy final-batch workload changed across counts/repeats')
                    # The original detour entry does not collect worker routes
                    # after FLT. Evaluate its first pair to expose that limit;
                    # do not require it to match the newer InstantGR2 algorithm.
                    if mode=='detour' and repeat==1:
                        evaluate = [str(evaluator),*map(str,inputs),str(output)]
                        ec, seconds = run_logged(evaluate,work/'evaluator.log')
                        run.update(evaluator_returncode=ec,evaluator_seconds=seconds,evaluator_command=evaluate)
                        try:
                            run.update(metrics=evaluator_metrics(work/'evaluator.log',ec),quality_status='valid')
                        except RuntimeError as error:
                            # A correctness limitation in the historical
                            # experimental program must not hide its timings.
                            run.update(quality_status='invalid',evaluator_error=str(error))
                    output.unlink()
                    run.update(status='passed',finished_unix=time.time())
                    summary(folder,result)
                    print('PASS',label,json.dumps(run['batch']),flush=True)
        if result['original_sources'] != {str(p.relative_to(LEGACY)):sha256(p) for p in source_files}:
            raise RuntimeError('Original legacy sources changed during experiment')
        if sha256(LEGACY/'run/graptor_multi') != result['original_binary_sha256']:
            raise RuntimeError('Original binary changed during experiment')
        result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error)); print('FAILED',error,flush=True)
    finally:
        for output in folder.glob('*/routing.out'):
            output.unlink(missing_ok=True)
        result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        summary(folder,result)
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
