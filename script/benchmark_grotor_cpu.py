#!/usr/bin/env python3
"""Build without OptiX and compare repeated CPU-batching GRotor runs with InstantGR2."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from cpu_reporting import comparison_rows, parse_cpu_run, write_summary
from test_instantgr2 import ROOT, BASELINE, evaluator_metrics, require_cuda, run_logged, sha256
from test_grotor_reference import local_path, source_manifest, validate_luts

CASES = ['mempool_tile', 'ariane133_51', 'ariane133_68', 'nvdla', 'bsg_chip', 'mempool_group']
BINARIES = ('GRotor', 'instantgr2_reference', 'grotor_evaluator')


def verify_build(build, architecture):
    cache = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line:
            key, value = line.split('=', 1)
            cache[key.split(':')[0]] = value
    for key, expected in [('GROTOR_ENABLE_RTCORE', 'OFF'), ('GROTOR_FULL_COST_REFERENCE', 'OFF'),
            ('CMAKE_CUDA_ARCHITECTURES', architecture), ('CMAKE_BUILD_TYPE', 'Release')]:
        if cache.get(key) != expected:
            raise ValueError(f'Build requires {key}={expected}; rerun without --no-build')
    flags = (build / 'CMakeFiles/GRotor.dir/flags.make').read_text()
    if any('-D' + name in flags for name in ['GROTOR_ENABLE_RTCORE', 'GROTOR_PREFIX_ACCESS_COUNTS',
            'GROTOR_SERIAL_DAG', 'GROTOR_CUGR2_ROUTING', 'GROTOR_GGR_ROUTING', 'GROTOR_FULL_COST_REFERENCE']):
        raise ValueError('Wrong routing target or profiling configuration in CUDA-only build')
    for binary in BINARIES:
        if not (build / binary).is_file():
            raise FileNotFoundError(build / binary)
    return cache


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', default=CASES)
    parser.add_argument('--data-dir', default='../ISPD24_tmp')
    parser.add_argument('--output-dir', default='results/GRotor-cuda')
    parser.add_argument('--cuda-arch', default='89')
    parser.add_argument('--nvcc', default=os.environ.get('NVCC', 'nvcc'))
    parser.add_argument('--gpu', help='Set CUDA_VISIBLE_DEVICES to one GPU index; otherwise inherit it')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--no-build', action='store_true')
    parser.add_argument('--clean-results', action='store_true')
    args = parser.parse_args(argv)
    if (args.jobs < 1 or args.repeats < 1 or not args.cuda_arch.isdigit()
            or args.gpu is not None and not args.gpu.isdigit()
            or not args.cases or len(set(args.cases)) != len(args.cases)
            or any(not re.fullmatch(r'[A-Za-z0-9_-]+', case) for case in args.cases)):
        parser.error('Use positive jobs/repeats, numeric architecture/GPU and unique simple case names')
    folder, data = local_path(args.output_dir), local_path(args.data_dir)
    if ROOT / 'results' not in folder.parents or folder.is_symlink():
        parser.error('Select a directory below this repository\'s results directory')
    for case in args.cases:
        for ext in ('cap', 'net'):
            if not (data / f'{case}.{ext}').is_file():
                parser.error(f'Missing input: {data / (case + "." + ext)}')
    if folder.exists():
        if not args.clean_results:
            parser.error('Output exists; use --clean-results to replace it')
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    build = ROOT / 'build/GRotor-cuda' / ('sm_' + args.cuda_arch)
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    result = dict(status='running', configuration=dict(batching='cpu', rtcore_compiled=False,
        cuda_arch=args.cuda_arch, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        repeats=args.repeats, sparsity=1, sources=source_manifest(), runner_sha256=sha256(Path(__file__)),
        reporting_sha256=sha256(ROOT / 'script/cpu_reporting.py'),
        presets_sha256=sha256(ROOT / 'CMakePresets.json')), command=sys.argv, cases=[])
    output = folder / 'routing.out'
    try:
        validate_luts()
        if not args.no_build:
            nvcc = shutil.which(args.nvcc)
            if not nvcc:
                raise RuntimeError('nvcc is required; set NVCC or pass --nvcc')
            commands = [
                ['cmake', '-S', str(ROOT), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_CUDA_COMPILER=' + str(Path(nvcc).resolve()),
                 '-DCMAKE_CUDA_ARCHITECTURES=' + args.cuda_arch, '-DGROTOR_ENABLE_RTCORE=OFF',
                 '-DGROTOR_FULL_COST_REFERENCE=OFF', '-DGROTOR_BUILD_ROUTING_ABLATION=OFF',
                 '-DGROTOR_BUILD_CUGR2_ROUTER=OFF', '-DGROTOR_BUILD_GGR_ROUTER=OFF'],
                ['cmake', '--build', str(build), '--target', *BINARIES, '--parallel', str(args.jobs)]]
            result['build_commands'] = commands
            for filename, command in zip(('configure.log', 'build.log'), commands):
                code, _ = run_logged(command, folder / filename)
                if code:
                    raise RuntimeError('Build failed: ' + str(folder / filename))
        cache = verify_build(build, args.cuda_arch)
        result['build_configuration'] = {key: cache.get(key) for key in (
            'CMAKE_CUDA_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_CUDA_ARCHITECTURES',
            'CMAKE_BUILD_TYPE', 'GROTOR_ENABLE_RTCORE', 'GROTOR_FULL_COST_REFERENCE')}
        result['compile_flags'] = (build / 'CMakeFiles/GRotor.dir/flags.make').read_text()
        result['binary_sha256'] = {name: sha256(build / name) for name in BINARIES}
        require_cuda()
        result['hardware'] = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'], text=True).strip()
        for name in args.cases:
            cf = folder / name
            cf.mkdir()
            inputs = [data / (name + '.' + ext) for ext in ('cap', 'net')]
            case = dict(name=name, status='running', inputs={str(p): sha256(p) for p in inputs}, runs=[])
            result['cases'].append(case)
            for optimized, repeat in [(False, 1)] + [(True, n) for n in range(1, args.repeats + 1)]:
                label = 'grotor' + str(repeat) if optimized else 'instantgr2'
                binary = build / ('GRotor' if optimized else 'instantgr2_reference')
                command = [str(binary), '-cap', str(inputs[0]), '-net', str(inputs[1]), '-out', str(output),
                    '--batching', 'cpu', '--sparsity', '1']
                code, wall = run_logged(command, cf / (label + '.routing.log'),
                    ROOT if optimized else BASELINE / 'run')
                if code or not output.is_file() or not output.stat().st_size:
                    raise RuntimeError('Router failed: ' + str(cf / (label + '.routing.log')))
                run = parse_cpu_run((cf / (label + '.routing.log')).read_text(), optimized)
                run.update(router_command=command, wall_seconds=wall, output_bytes=output.stat().st_size)
                command = [str(build / 'grotor_evaluator'), *map(str, inputs), str(output)]
                evaluation = cf / (label + '.evaluator.log')
                code, seconds = run_logged(command, evaluation)
                run.update(metrics=evaluator_metrics(evaluation, code), evaluator_command=command,
                    evaluator_seconds=seconds)
                output.unlink()
                if optimized:
                    case['runs'].append(run)
                    comparison_rows(case)
                else:
                    case['baseline'] = run
                print('PASS', name, label, run['metrics']['total'], flush=True)
                write_summary(folder, result)
            case['status'] = 'passed'
            write_summary(folder, result)
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=str(error))
        print('FAILED:', error, flush=True)
    finally:
        output.unlink(missing_ok=True)
        write_summary(folder, result)
    print('Report:', folder / 'summary.md', flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
