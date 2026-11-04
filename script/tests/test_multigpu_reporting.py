import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multigpu_reporting import parse_run, scaling_rows


def log(gpus=2):
    lines = [f'MULTIGPU_DEVICE rank={p} pe={p} device={p} name=A100' for p in range(gpus)]
    for stage in ('lshape', 'flt'):
        lines += [f'MULTIGPU_PARTITION stage={stage} rank={p} nets=10 nodes=100' for p in range(gpus)]
        lines += [f'MULTIGPU_BATCH stage={stage} gpus={gpus} batch_index=9 batches=10 nets={10*gpus} elapsed_ms=8 preparation_ms=4 snapshot_ms=1 solve_ms=5 commit_ms=2']
    return '\n'.join(lines)


class MultiGpuReportingTests(unittest.TestCase):
    def test_lshape_only_requires_exactly_one_final_interval(self):
        text = '\n'.join(line for line in log().splitlines() if 'stage=flt' not in line)
        batches = parse_run(text, 2, stages=('lshape',))
        self.assertEqual([b['stage'] for b in batches], ['lshape'])
        for invalid in (log(), text.replace('batch_index=9', 'batch_index=8'),
                        text + '\n' + next(line for line in text.splitlines() if line.startswith('MULTIGPU_BATCH '))):
            with self.assertRaises(ValueError):
                parse_run(invalid, 2, stages=('lshape',))

    def test_missing_gpu_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'GPU ranks'):
            parse_run(log().replace('MULTIGPU_DEVICE rank=1', 'HIDDEN rank=1'), 2)

    def test_wrong_batch_or_incomplete_workload_is_rejected(self):
        for text in (log().replace('batch_index=9', 'batch_index=8'),
                     log().replace('rank=1 nets=10', 'rank=1 nets=9')):
            with self.assertRaises(ValueError):
                parse_run(text, 2)

    def test_invalid_timings_are_rejected(self):
        for value in ('nan', 'inf', '0', '-1'):
            with self.assertRaises(ValueError):
                parse_run(log().replace('elapsed_ms=8', 'elapsed_ms=' + value), 2)

    def test_cuda_event_components_and_excluded_times(self):
        text = log().replace('commit_ms=2', 'commit_ms=2 finalization_ms=3 end_to_end_ms=15 critical_rank=1')
        batch = parse_run(text, 2)[0]
        self.assertEqual(batch['elapsed_ms'], 8)
        self.assertEqual(batch['finalization_ms'], 3)
        for bad in (text.replace('critical_rank=1', 'critical_rank=2'),
                    text.replace('commit_ms=2', 'commit_ms=3'),
                    text.replace('finalization_ms=3', 'finalization_ms=nan')):
            with self.assertRaises(ValueError):
                parse_run(bad, 2)

    def test_median_speedup_uses_only_validated_runs(self):
        batches = parse_run(log(), 2)
        runs = []
        for count, times in [(1, [10, 20, 30]), (2, [5, 10, 15])]:
            for elapsed in times:
                data = copy.deepcopy(batches)
                for batch in data:
                    batch['elapsed_ms'] = elapsed
                runs.append(dict(status='passed', gpus=count, batches=data))
        runs.append(dict(status='failed', gpus=2, batches=[dict(stage='lshape', elapsed_ms=.001)]))
        rows = scaling_rows(dict(cases=[dict(name='test', runs=runs)]))
        for row in rows:
            self.assertEqual(row['speedup'], row['gpus'])
            self.assertEqual(row['efficiency'], 1)
            self.assertEqual(row['repeats'], 3)


if __name__ == '__main__':
    unittest.main()
