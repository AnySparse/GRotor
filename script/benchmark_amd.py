#!/usr/bin/env python3
"""Benchmark one AMD GPU: CPU/full/thread-depth vs HIPRT/incremental-BIT/warp-block.

Validation is a separate run; only validated cases enter the speedup summary.
Only the shared GRotor pipeline is required; no AMD InstantGR2 target is built.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time

from compare_routing import canonicalize, files_equal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ['ariane133_51', 'ariane133_68', 'ariane_68_rank', 'bsg_chip',
                 'bsg_chip_rank', 'nvdla', 'nvdla_rank', 'mempool_tile',
                 'mempool_tile_rank', 'mempool_group']


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()



def evaluator_metrics(log_path, returncode):
    if returncode not in (0, 1):
        raise RuntimeError(f"Evaluator failed with exit status {returncode}: {log_path}")
    text = log_path.read_text()
    def one(pattern):
        values = re.findall(pattern, text, re.MULTILINE)
        if len(values) != 1:
            raise RuntimeError(f"Missing or ambiguous evaluator result: {pattern}")
        return values[0]
    metrics = {key: one(rf"^{key} cost (\d+\.\d+)$")
               for key in ("wirelength", "via", "overflow", "total")}
    metrics["open_nets"] = int(one(r"^Number of open nets : (\d+)$"))
    metrics["incomplete_nets"] = int(one(r"^Number of incompleted nets : (\d+)$"))
    one(r"^FOR_STAT (.+)$")
    if metrics["open_nets"] or metrics["incomplete_nets"]:
        raise RuntimeError("Evaluator found open or incomplete nets")
    return metrics



def capture(command):
    try:
        p = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return {'command': command, 'status': p.returncode, 'output': p.stdout}
    except OSError as e:
        return {'command': command, 'error': str(e)}


def run(command, log, cwd=ROOT):
    """Sample RSS and device-wide VRAM at 20 ms; retain the full command/status."""
    print(' '.join(map(str, command)), flush=True)
    begin = time.monotonic()
    peak_rss = peak_vram = 0
    vram_files = list(Path('/sys/class/drm').glob('card*/device/mem_info_vram_used'))
    stop = threading.Event()
    with log.open('x') as output:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
        def sample():
            nonlocal peak_rss, peak_vram
            while not stop.is_set():
                try:
                    text = Path(f'/proc/{process.pid}/status').read_text()
                    match = re.search(r'^VmHWM:\s+(\d+) kB', text, re.M)
                    if match: peak_rss = max(peak_rss, int(match[1]) * 1024)
                    peak_vram = max(peak_vram, sum(int(p.read_text()) for p in vram_files))
                except (OSError, ValueError):
                    pass
                stop.wait(.02)
        sampler = threading.Thread(target=sample)
        sampler.start()
        try:
            status = process.wait()
        finally:
            stop.set(); sampler.join()
    record = {'command': list(map(str, command)), 'cwd': str(cwd), 'log': str(log),
              'status': status, 'process_wall_seconds': time.monotonic()-begin,
              'sampled_peak_rss_bytes': peak_rss,
              'sampled_peak_device_vram_bytes': peak_vram,
              'vram_scope': 'all AMD DRM devices, includes other processes; sampled at 20 ms'}
    log.with_suffix('.command.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def parse_timing(text, variant):
    batches = re.findall(r'^BATCH_TIMING backend=(\w+) start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+) used_for_routing=1$', text, re.M)
    dags = re.findall(r'^DAG_TIMING stage=(\w+) batch_index=-1 start_us=(\d+) end_us=(\d+) elapsed_ms=([\d.]+) nets=(\d+) batches=(\d+)$', text, re.M)
    wall = re.findall(r'^GROTOR_COMPLETE wall_seconds=([\d.]+)$', text, re.M)
    if 'SCORE_DIAGNOSTICS enabled=0' not in text:
        raise RuntimeError('Performance runs must disable score diagnostics symmetrically')
    if variant == 'optimized' and len(re.findall(r'^BATCH_SCOPE backend=hiprt scope=generate_batch_ids_call_wall$', text, re.M)) != 2:
        raise RuntimeError('HIPRT binary must report the batching function-call timing scope; rebuild GRotor')
    expected = 'cpu' if variant == 'original' else 'hiprt'
    if len(batches) != 2 or [x[0] for x in batches] != [expected]*2 or [x[0] for x in dags] != ['lshape', 'detour'] or len(wall) != 1:
        raise RuntimeError('Missing or ambiguous two-stage timing/completion records')
    config = 'cost=full prefix=full' if variant == 'original' else 'cost=incremental prefix=bit'
    scheduling = 'lshape=thread flt=global_depth' if variant == 'original' else 'lshape=warp flt=block'
    if config not in text or scheduling not in text:
        raise RuntimeError('Executable configuration differs from declared AMD strategy')
    memory_policy = re.findall(r'^HOST_MEMORY_POLICY saving=([01]) bounded_preparation=([01])$', text, re.M)
    if len(memory_policy) != 1:
        raise RuntimeError('Missing or ambiguous host-memory policy')
    rows = []
    for b, d, stage in zip(batches, dags, ['lshape', 'flt']):
        for item in [b, d]:
            if int(item[2]) < int(item[1]) or abs((int(item[2])-int(item[1]))/1000-float(item[3])) > 1e-5:
                raise RuntimeError('Inconsistent timing timestamps')
        if b[4:] != d[4:] or int(d[1]) < int(b[2]):
            raise RuntimeError('Batch/DAG counts or timing order mismatch')
        rows.append({'stage': stage, 'host_memory_saving': bool(int(memory_policy[0][0])), 'batching_ms': float(b[3]), 'dag_ms': float(d[3]),
                     'algorithm_total_ms': float(b[3])+float(d[3]),
                     'batching_scope': 'cpu_greedy_call_wall' if variant == 'original' else 'generate_batch_ids_call_wall',
                     'dag_scope': 'all_batches_cost_dp_traceback_demand_commit_wall',
                     'router_wall_seconds': float(wall[0]), 'nets': int(b[4]), 'batches': int(b[5])})
    return rows


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(rows, cases, destination):
    summaries = []
    for case in cases:
        for stage in ['lshape', 'flt']:
            group = [r for r in rows if r['case'] == case and r['stage'] == stage]
            original = [r for r in group if r['variant'] == 'original']
            optimized = [r for r in group if r['variant'] == 'optimized']
            if not original or len(original) != len(optimized):
                raise RuntimeError(f'Incomplete repetition set: {case} {stage}')
            a = statistics.median(r['algorithm_total_ms'] for r in original)
            b = statistics.median(r['algorithm_total_ms'] for r in optimized)
            summaries.append({'case': case, 'stage': stage,
                              'original_ms': a, 'optimized_ms': b, 'speedup': a/b,
                              'original_batching_ms': statistics.median(r['batching_ms'] for r in original),
                              'optimized_batching_ms': statistics.median(r['batching_ms'] for r in optimized),
                              'original_dag_ms': statistics.median(r['dag_ms'] for r in original),
                              'optimized_dag_ms': statistics.median(r['dag_ms'] for r in optimized)})
    write_csv(destination/'summary.csv', summaries)
    gm = {s: math.exp(statistics.mean(math.log(r['speedup']) for r in summaries if r['stage'] == s)) for s in ['lshape', 'flt']}
    (destination/'geomean.json').write_text(json.dumps({'cases': cases, 'statistic': 'geomean of per-case median original/optimized', 'speedup': gm}, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build/hip')
    parser.add_argument('--results', type=Path, required=True, help='New directory; existing results are never overwritten')
    parser.add_argument('--cases', nargs='+', default=DEFAULT_CASES)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0: parser.error('Invalid repetition count')
    if len(set(args.cases)) != len(args.cases): parser.error('Duplicate case names')
    data, build, dest = args.data_root.resolve(), args.build_dir.resolve(), args.results.resolve()
    dest.mkdir(parents=True, exist_ok=False)
    environment = {name: capture(command) for name, command in {
        'hip': ['hipcc', '--version'], 'gpu': ['rocminfo'], 'cpu': ['lscpu'],
        'os': ['uname', '-a'], 'smi': ['amd-smi', 'static'],
        'source': ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
        'diff': ['git', '-C', str(ROOT), 'diff'],
        'status': ['git', '-C', str(ROOT), 'status', '--short']}.items()}
    environment['device_environment'] = {k: v for k, v in os.environ.items() if k in ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HSA_OVERRIDE_GFX_VERSION', 'HIPRT_PATH']}
    environment['policy'] = {'timing': 'batching function call wall time + full DAG interval; scene preparation and collection excluded', 'validation': 'separate from performance', 'warmups': args.warmups, 'repeats': args.repeats, 'score_diagnostics': 'disabled symmetrically for GRotor; evaluator and geometry required', 'gpu_frequency': 'system default, no clocks changed', 'case_list': args.cases}
    environment['cmake_cache'] = (build/'CMakeCache.txt').read_text()
    environment['binaries_sha256'] = {n: sha256(build/n) for n in ['GRotor', 'grotor_amd_original', 'grotor_evaluator']}
    # Include untracked source files in reproducibility data.
    environment['source_sha256'] = {str(p.relative_to(ROOT)): sha256(p) for folder in ['src', 'include', 'script'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    for name in ['CMakeLists.txt', 'CMakePresets.json']:
        environment['source_sha256'][name] = sha256(ROOT/name)
    def cache_value(key):
        values = re.findall(rf'^{key}:[^=]+=(.*)$', environment['cmake_cache'], re.M)
        return values[0] if len(values) == 1 else ''
    hiprt_root = Path(cache_value('HIPRT_ROOT'))
    hiprt_library = Path(cache_value('HIPRT_LIBRARY'))
    environment['hiprt'] = capture(['git', '-C', str(hiprt_root), 'rev-parse', 'HEAD'])
    sdk_files = [hiprt_root/'version.txt', hiprt_root/'hiprt/hiprt.h',
                 hiprt_root/'hiprt/hiprt_device.h', hiprt_library]
    environment['hiprt_files_sha256'] = {str(p): sha256(p) for p in sdk_files if p.is_file()}
    (dest/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    measurements, status = [], []
    for case in args.cases:
        folder = dest/case; folder.mkdir()
        case_status = {'case': case, 'status': 'running'}; status.append(case_status)
        try:
            inputs = [data/f'{case}.{ext}' for ext in ['cap', 'net']]
            case_status['inputs_sha256'] = {str(p): sha256(p) for p in inputs}
            reference_metrics = reference_canonical = None
            for variant in ['original', 'optimized']:
                executable = {'original': 'grotor_amd_original', 'optimized': 'GRotor'}[variant]
                output = folder/f'validate-{variant}.out'
                command = [build/executable, '-cap', inputs[0], '-net', inputs[1], '-out', output,
                           '--sparsity', '1', '--batching', 'hiprt' if variant == 'optimized' else 'cpu']
                command += ['--report-score', '0']
                if variant == 'optimized': command += ['--validate-batches', '1']
                log = folder/f'validate-{variant}.log'
                record = run(command, log, ROOT/'resources/flute')
                if record['status']: raise RuntimeError(f'{variant} validation failed: {record["status"]}')
                if variant == 'optimized' and len(re.findall(r'^BATCH_VALIDATION .* exact_lists_equal=1$', log.read_text(), re.M)) != 2:
                    raise RuntimeError('Both HIPRT stages must validate exact batch lists')
                evaluation = folder/f'validate-{variant}.evaluator.log'
                ev = run([build/'grotor_evaluator', *inputs, output], evaluation)
                metrics = evaluator_metrics(evaluation, ev['status'])
                normalized = folder/f'validate-{variant}.normalized.gz'
                canonical = canonicalize(output, normalized, folder)
                if reference_metrics is None:
                    reference_metrics, reference_canonical = metrics, normalized
                elif metrics != reference_metrics or not files_equal(reference_canonical, normalized):
                    raise RuntimeError(f'{variant} differs from original in evaluator or per-net segment multiset')
                (folder/f'validate-{variant}.quality.json').write_text(json.dumps({'metrics': metrics, 'canonical': canonical}, indent=2)+'\n')
            case_status['validation'] = 'PASS: GRotor optimized/original exact HIPRT/CPU batches, evaluator and per-net segment multisets'
            for iteration in range(-args.warmups, args.repeats):
                # Alternate run order to reduce systematic temperature/order bias.
                variants = ['original', 'optimized'] if iteration % 2 == 0 else ['optimized', 'original']
                for variant in variants:
                    prefix = f'{"warmup" if iteration < 0 else "repeat"}-{abs(iteration)}-{variant}'
                    output = folder/f'{prefix}.out'; log = folder/f'{prefix}.log'
                    binary = 'GRotor' if variant == 'optimized' else 'grotor_amd_original'
                    record = run([build/binary, '-cap', inputs[0], '-net', inputs[1], '-out', output,
                                  '--batching', 'hiprt' if variant == 'optimized' else 'cpu', '--profile-stages', '1', '--report-score', '0'], log)
                    if record['status']: raise RuntimeError(f'{prefix} failed: {record["status"]}')
                    parsed = parse_timing(log.read_text(), variant)
                    if 'BATCH_VALIDATION' in log.read_text(): raise RuntimeError('Validation leaked into performance run')
                    normalized = folder/f'{prefix}.normalized.gz'
                    canonicalize(output, normalized, folder)
                    if not files_equal(reference_canonical, normalized): raise RuntimeError(f'{prefix} geometry mismatch')
                    normalized.unlink(); output.unlink()  # Keep validation outputs and every raw log.
                    if iteration >= 0:
                        for row in parsed:
                            measurements.append({'case': case, 'variant': variant, 'repeat': iteration, **row,
                                'validation_enabled': False, 'sampled_peak_rss_bytes': record['sampled_peak_rss_bytes'],
                                'sampled_peak_device_vram_bytes': record['sampled_peak_device_vram_bytes'],
                                'process_wall_seconds': record['process_wall_seconds']})
                        write_csv(dest/'measurements.csv', measurements)
            case_status['metrics'] = reference_metrics
            case_status['status'] = 'PASS'
        except Exception as e:
            case_status['status'] = 'FAIL'; case_status['reason'] = str(e)
            print(f'FAIL {case}: {e}', file=sys.stderr, flush=True)
        (dest/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    if any(s['status'] != 'PASS' for s in status):
        print('Failed cases are retained in status.json; no incomplete geomean is emitted.', file=sys.stderr)
        return 1
    summarize(measurements, args.cases, dest)
    print((dest/'geomean.json').read_text())
    return 0


if __name__ == '__main__':
    sys.exit(main())
