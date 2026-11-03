"""Validate CPU-only build selection, timing scopes and evaluator comparisons."""
import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_grotor_cpu import BINARIES, verify_build
from benchmark_grotor_system import main as system_main
from cpu_reporting import comparison_rows, parse_cpu_run, write_summary


def log_fixture(optimized=True):
    lines = ['GROTOR_CONFIG batching=cpu cost=incremental prefix=bit memory=explicit']
    for i, stage in enumerate(('lshape', 'detour')):
        origin = i * 10000
        lines.append(f'BATCH_TIMING backend=cpu start_us={origin} end_us={origin+1000} elapsed_ms=1.000000 nets=10 batches=3 used_for_routing=1')
        lines.append(f'DAG_TIMING stage={stage} batch_index=-1 start_us={origin+2000} end_us={origin+4000} elapsed_ms=2.000000 nets=10 batches=3')
    lines.append('GROTOR_COMPLETE wall_seconds=0.1')
    return '\n'.join(lines)


def case_fixture():
    metrics = dict(wirelength='10.0000', via='2.0000', overflow='0.0000', total='12.0000',
        open_nets=0, incomplete_nets=0)
    run = dict(**parse_cpu_run(log_fixture(), True), metrics=metrics)
    return dict(name='fixture', status='passed', baseline=copy.deepcopy(run), runs=[run])


class CpuReportTests(unittest.TestCase):
    def test_totals_include_measured_cpu_batching(self):
        rows = comparison_rows(case_fixture())
        self.assertEqual([r['instantgr2_total_ms'] for r in rows], [3, 3])
        self.assertEqual([r['grotor_total_ms'] for r in rows], [3, 3])

    def test_reject_summed_grotor_flt_interval(self):
        with self.assertRaises(ValueError):
            parse_cpu_run(log_fixture().replace('end_us=14000', 'end_us=15000'), True)
        parse_cpu_run(log_fixture().replace('end_us=14000', 'end_us=15000'), False)

    def test_reject_rtcore_or_incomplete_run(self):
        for text in [log_fixture()+'\nRTCORE_STAGE stage=lshape', log_fixture().replace('GROTOR_COMPLETE', 'FAILED')]:
            with self.assertRaises(ValueError):
                parse_cpu_run(text, True)

    def test_reject_quality_and_workload_mismatch(self):
        for mode in ('score', 'workload'):
            case = case_fixture()
            if mode == 'score':
                case['runs'][0]['metrics']['total'] = '13.0000'
            else:
                case['runs'][0]['algorithms'][0]['batches'] += 1
            with self.assertRaises(ValueError):
                comparison_rows(case)

    def test_equal_scores_with_open_nets_are_not_accepted(self):
        case = case_fixture()
        case['baseline']['metrics']['open_nets'] = 1
        case['runs'][0]['metrics']['open_nets'] = 1
        with self.assertRaises(ValueError):
            comparison_rows(case)

    def test_tables_keep_cpu_timings_and_no_backend_column(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_summary(folder, dict(status='passed', cases=[case_fixture()]))
            with (folder/'runtime.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['algorithm'] for r in rows], ['L-shape', 'FLT'])
            self.assertNotIn('stage', rows[0])
            self.assertNotIn('routing_backend', rows[0])
            self.assertFalse(list(folder.glob('*.out*')))

    def test_public_runner_dispatches_cpu_selection(self):
        with patch('benchmark_grotor_cpu.main', return_value=0) as run:
            self.assertEqual(system_main(['--batching', 'cpu', '--cases', 'fixture']), 0)
            run.assert_called_once_with(['--cases', 'fixture'])

    def test_wrong_cached_build_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flags = root/'CMakeFiles/GRotor.dir'
            flags.mkdir(parents=True)
            (flags/'flags.make').write_text('CUDA_DEFINES = -DG_ROTOR_RESOURCE_DIR=resources\n')
            good = 'GROTOR_ENABLE_RTCORE:BOOL=OFF\nGROTOR_FULL_COST_REFERENCE:BOOL=OFF\nCMAKE_CUDA_ARCHITECTURES:STRING=80\nCMAKE_BUILD_TYPE:STRING=Release\n'
            (root/'CMakeCache.txt').write_text(good)
            for name in BINARIES:
                (root/name).touch()
            verify_build(root, '80')
            (root/'CMakeCache.txt').write_text(good.replace('RTCORE:BOOL=OFF', 'RTCORE:BOOL=ON'))
            with self.assertRaises(ValueError):
                verify_build(root, '80')
            (root/'CMakeCache.txt').write_text(good)
            (flags/'flags.make').write_text('CUDA_DEFINES = -DGROTOR_ENABLE_RTCORE=1\n')
            with self.assertRaises(ValueError):
                verify_build(root, '80')


if __name__ == '__main__':
    unittest.main()
