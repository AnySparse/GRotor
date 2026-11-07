#!/usr/bin/env python3
"""Measure 1-8 GPU scaling of the final L-shape batch; stop before FLT."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import threading

from _multigpu_report import parse_run, write_summary
from _support import local_path, source_manifest, validate_luts
from _support import ROOT, evaluator_metrics, run_logged, sha256, require_cuda


class GpuMonitor:
    """Record shared-device load during each router run, outside its CUDA timer."""
    def __init__(self, path):
        self.path = path
        self.stop = threading.Event()

    def sample(self):
        record = dict(unix_time=time.time())
        for key, query in (
            ('gpus', '--query-gpu=index,uuid,utilization.gpu,memory.used,clocks.sm,clocks.mem'),
            ('processes', '--query-compute-apps=gpu_uuid,pid,used_memory')):
            try:
                record[key] = subprocess.check_output(
                    ['nvidia-smi', query, '--format=csv,noheader,nounits'], text=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as error:
                record[key + '_error'] = str(error)
        return record

    def collect(self):
        with self.path.open('x') as stream:
            while True:
                stream.write(json.dumps(self.sample()) + '\n')
                stream.flush()
                if self.stop.wait(5):
                    break

    def __enter__(self):
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, help='Directory containing matching <case>.cap/.net inputs')
    parser.add_argument('--cases', nargs='+', default=['mempool_cluster'])
    parser.add_argument('--counts', nargs='+', type=int, default=list(range(1, 9)))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--devices', help='Override CUDA_VISIBLE_DEVICES; otherwise preserve the existing allocation')
    parser.add_argument('--require-idle', action='store_true', help='Reject devices with existing compute processes before every router launch')
    parser.add_argument('--output-dir', default='results/GRotor-multigpu')
    parser.add_argument('--cuda-arch', default='80')
    parser.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc'))
    parser.add_argument('--nvshmem-root', default=os.environ.get('NVSHMEM_HOME', str(Path.home() / 'opt/nvshmem')))
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--mpi-arg', action='append', default=[], help='Additional launcher argument, repeat as needed; use --mpi-arg=--flag')
    parser.add_argument('--symmetric-size', default='25G')
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--clean-results', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    counts = sorted(set(args.counts))
    if not counts or counts[0] != 1 or counts[-1] > 8 or args.repeats < 1 or args.jobs < 1:
        parser.error('Use positive repeats/jobs and GPU counts from 1 to 8, including 1')
    if not args.cuda_arch.isdigit() or not args.cases or len(set(args.cases)) != len(args.cases) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', c) for c in args.cases):
        parser.error('Use a numeric CUDA architecture and unique simple dataset names')
    if not args.no_plot:
        try:
            import matplotlib
        except ImportError:
            parser.error('Matplotlib is required for plots; select a Python environment with Matplotlib or use --no-plot')
    folder, data = local_path(args.output_dir), local_path(args.data_dir)
    if ROOT / 'results' not in folder.parents or folder.is_symlink():
        parser.error('Select an output directory below this repository\'s results/')
    for case in args.cases:
        for extension in ('cap', 'net'):
            if not (data / f'{case}.{extension}').is_file():
                parser.error(f'Missing input: {data / (case + "." + extension)}')
    if folder.exists():
        if not args.clean_results:
            parser.error('Output exists; use --clean-results to replace it')
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    output = folder / 'routing.out'
    build = ROOT / 'build/GRotor-multi' / ('sm_' + args.cuda_arch)
    result = dict(status='running', configuration=vars(args), command=sys.argv, cases=[])
    result['configuration'].update(counts=counts, stages=['lshape'], grid_mode='spatial',
                                   lshape_mode='legacy', stop_after='lshape')
    try:
        if args.devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
        require_cuda()
        if os.environ.get('GROTOR_LEGACY_VALIDATE_COSTS'):
            raise ValueError('Disable GROTOR_LEGACY_VALIDATE_COSTS for performance runs')
        driver_gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'], text=True)
        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        allocation = devices.split(',') if devices else [line.split(',')[0].strip() for line in driver_gpus.strip().splitlines()]
        if len(set(allocation)) != len(allocation) or len(allocation) < counts[-1]:
            raise ValueError(f'Need {counts[-1]} distinct visible GPUs; allocation is {allocation}')
        result.update(hardware=driver_gpus, topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
                      sources=source_manifest())
        def check_idle(selected):
            if not args.require_idle:
                return
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
            ids = dict(tuple(v.strip() for v in row.split(',')) for row in inventory.strip().splitlines())
            uuids = {ids.get(device, device) for device in selected}
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader,nounits'], text=True)
            busy = [row for row in processes.splitlines() if row.split(',')[0].strip() in uuids]
            result.setdefault('idle_checks', []).append(dict(unix_time=time.time(), devices=selected, conflicting_processes=busy))
            if busy:
                raise RuntimeError('Selected GPU has another compute process: ' + '; '.join(busy))
        result['runner_sha256'] = sha256(Path(__file__))
        result['timing_scope'] = 'my-gr-cuda-event'
        result['reporting_sha256'] = sha256(ROOT / 'script/experiments/_multigpu_report.py')
        validate_luts()
        if not args.no_build:
            nvcc = shutil.which(args.nvcc)
            if not nvcc:
                raise RuntimeError('nvcc is required')
            commands = [
                ['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_CUDA_COMPILER=' + nvcc, '-DCMAKE_CUDA_ARCHITECTURES=' + args.cuda_arch,
                 '-DGROTOR_ENABLE_RTCORE=OFF', '-DGROTOR_FULL_COST_REFERENCE=OFF',
                 '-DGROTOR_BUILD_MULTIGPU=ON', '-DGROTOR_BUILD_INSTANTGR2=OFF', '-DGROTOR_MULTI_GRID=ON',
                 '-DGROTOR_MULTI_LEGACY_LSHAPE=ON',
                 '-DNVSHMEM_ROOT=' + args.nvshmem_root],
                ['cmake', '--build', str(build), '--target', 'GRotor_multi', 'grotor_evaluator', '--parallel', str(args.jobs)]]
            launcher = shutil.which(args.mpirun)
            if launcher:
                mpi_home = Path(launcher).resolve().parent.parent
                if (mpi_home / 'include/mpi.h').is_file():
                    commands[0] += ['-DMPI_CXX_COMPILER=' + str(mpi_home / 'bin/mpicxx'),
                        '-DMPI_CXX_COMPILER_INCLUDE_DIRS=' + str(mpi_home / 'include'),
                        '-DMPI_CXX_HEADER_DIR=' + str(mpi_home / 'include')]
                    if (mpi_home / 'lib/libmpi.so').is_file():
                        commands[0] += ['-DMPI_mpi_LIBRARY=' + str(mpi_home / 'lib/libmpi.so')]
            result['build_commands'] = commands
            for name, command in zip(('configure.log', 'build.log'), commands):
                code, _ = run_logged(command, folder / name)
                if code:
                    raise RuntimeError(f'Build failed: {folder / name}')
        cache = dict(line.split('=', 1) for line in (build / 'CMakeCache.txt').read_text().splitlines()
                     if line and not line.startswith(('#', '//')) and '=' in line)
        result['build_configuration'] = cache
        values = {k.split(':')[0]: v for k, v in cache.items()}
        for key, expected in [('CMAKE_BUILD_TYPE', 'Release'), ('CMAKE_CUDA_ARCHITECTURES', args.cuda_arch),
                              ('GROTOR_ENABLE_RTCORE', 'OFF'), ('GROTOR_FULL_COST_REFERENCE', 'OFF'), ('GROTOR_BUILD_MULTIGPU', 'ON')]:
            if values.get(key) != expected:
                raise ValueError(f'Build requires {key}={expected}')
        if values.get('GROTOR_MULTI_GRID') != 'ON' or values.get('GROTOR_MULTI_LEGACY_LSHAPE') != 'ON':
            raise ValueError('Scaling requires the legacy spatial L-shape executor; rebuild')
        flags = (build / 'CMakeFiles/GRotor_multi.dir/flags.make').read_text()
        for required in ('GROTOR_DENSE_PRESUM', 'GROTOR_MULTI_LEGACY_LSHAPE'):
            if '-D' + required + '=1' not in flags:
                raise ValueError('Missing required compile definition: ' + required)
        if any('-D' + name in flags for name in ('GROTOR_PREFIX_ACCESS_COUNTS', 'GROTOR_SERIAL_DAG', 'GROTOR_FULL_COST_REFERENCE')):
            raise ValueError('Disable profiling/ablation flags for scaling')
        result['compile_flags'] = flags
        binaries = dict(GRotor_multi=build / 'GRotor_multi', grotor_evaluator=(ROOT/'resources/evaluator'))
        result['executable_paths'] = {name: str(path) for name, path in binaries.items()}
        result['binary_sha256'] = {name: sha256(path) for name, path in binaries.items()}
        nvshmem_lib = str(Path(args.nvshmem_root).resolve() / 'lib')
        os.environ['LD_LIBRARY_PATH'] = nvshmem_lib + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['NVSHMEM_SYMMETRIC_SIZE'] = args.symmetric_size
        # The executor explicitly requires NVLink peer mappings; no inter-node
        # transport is used. Avoid probing unrelated IB devices and NIC plugins.
        os.environ['NVSHMEM_REMOTE_TRANSPORT'] = 'none'
        os.environ.pop('NVSHMEM_DISABLE_NCCL', None)
        result['environment'] = {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'NVSHMEM_SYMMETRIC_SIZE',
            'NVSHMEM_REMOTE_TRANSPORT', 'NVSHMEM_DISABLE_NCCL', 'LD_LIBRARY_PATH')}
        write_summary(folder, result, plot=False)
        for name in args.cases:
            cf = folder / name; cf.mkdir()
            inputs = [data / (name + '.' + extension) for extension in ('cap', 'net')]
            case = dict(name=name, status='running', inputs={str(p): sha256(p) for p in inputs}, runs=[])
            result['cases'].append(case)
            route_args = ['-cap', str(inputs[0]), '-net', str(inputs[1]), '-out', str(output), '--batching', 'cpu', '--sparsity', '1']
            route_args += ['--stop-after', 'lshape']
            evaluator_command = [str(binaries['grotor_evaluator']), *map(str, inputs), str(output)]
            validated_outputs = {}
            for count in counts:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(allocation[:count])
                for repeat in range(1, args.repeats + 1):
                    label = f'{count}gpu-{repeat}'
                    command = [args.mpirun, '--bind-to', 'none', '-np', str(count), *args.mpi_arg, str(build / 'GRotor_multi'), *route_args]
                    run = dict(gpus=count, repeat=repeat, status='running', command=command,
                               cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
                    case['runs'].append(run)
                    log = cf / (label + '.routing.log')
                    check_idle(allocation[:count])
                    with GpuMonitor(cf / (label + '.gpu-load.jsonl')):
                        code, wall = run_logged(command, log, ROOT)
                    if code or not output.is_file() or not output.stat().st_size:
                        raise RuntimeError(f'GRotor_multi failed: {log}')
                    text = log.read_text()
                    if 'ROUTING_STOP stage=lshape flt_executed=0' not in text.splitlines():
                        raise ValueError('Router must stop after L-shape, before FLT')
                    if re.search(r'^(?:MULTIGPU_BATCH|DAG_TIMING|TOPOLOGY_TIMING) stage=(?:flt|detour) ', text, re.M):
                        raise ValueError('Unexpected FLT execution in an L-shape-only measurement')
                    policy = 'MULTIGPU_LSHAPE_EXECUTOR implementation=legacy-p2p blocks=1056 threads=256 demand=int2 dirty=bool matrix=full single_path=peer all_batches=distributed'
                    if policy not in text.splitlines():
                        raise ValueError('Legacy execution policy missing; keep the same executor at every GPU count')
                    if not re.search(r'^COST_STORAGE cells=\d+ bit_nodes=0 bit_bytes=0 ', text, re.M):
                        raise ValueError('BIT storage must be disabled on all batches')
                    if 'MULTIGPU_TIMING_POLICY stage=lshape topology=preparation maintenance=after-commit' not in text.splitlines():
                        raise ValueError('L-shape timing order differs from the my-gr policy')
                    run['batches'] = parse_run(text, count, stages=('lshape',))
                    if len(case['runs']) > 1:
                        oracle = case['runs'][0]['batches']
                        if [(b['stage'], b['batch_index'], b['batches'], b['nets']) for b in run['batches']] != [(b['stage'], b['batch_index'], b['batches'], b['nets']) for b in oracle]:
                            raise ValueError('Final-batch workload changed between GPU counts/runs')
                    run.update(wall_seconds=wall, output_sha256=sha256(output))
                    digest = run['output_sha256']
                    evaluation = cf / (label + '.evaluator.log')
                    if digest in validated_outputs:
                        # Identical route bytes need only one independent evaluation.
                        previous = validated_outputs[digest]
                        run.update(metrics=previous['metrics'], evaluator_reused_from=previous['label'],
                                   evaluator_returncode=previous['returncode'], evaluator_seconds=0)
                    else:
                        code, seconds = run_logged(evaluator_command, evaluation)
                        run.update(metrics=evaluator_metrics(evaluation, code), evaluator_seconds=seconds,
                                   evaluator_command=evaluator_command, evaluator_returncode=code)
                        validated_outputs[digest] = dict(metrics=run['metrics'], label=label, returncode=code)
                    output.unlink()
                    if 'lshape_reference' not in case:
                        case['lshape_reference'] = dict(gpus=1, repeat=repeat, metrics=run['metrics'])
                    if run['metrics'] != case['lshape_reference']['metrics']:
                        raise ValueError(f'L-shape output costs differ from 1 GPU: {name} {label}')
                    run['status'] = 'passed'
                    print('PASS', name, label, run['metrics']['total'], flush=True)
                    write_summary(folder, result, plot=False)
            case['status'] = 'passed'
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=str(error))
        print('FAILED:', error, flush=True)
    finally:
        output.unlink(missing_ok=True)
        write_summary(folder, result, plot=not args.no_plot)
    print('Report:', folder / 'summary.md', flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
