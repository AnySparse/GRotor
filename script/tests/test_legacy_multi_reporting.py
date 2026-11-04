import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_legacy_multi import read_batch


class LegacyMultiReportingTests(unittest.TestCase):
    def test_prefixed_first_batch_and_maximum_rank_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rank, count, elapsed in ((0, 4, 10), (1, 5, 11)):
                folder = root / '1' / f'rank.{rank}'
                folder.mkdir(parents=True)
                (folder / 'stdout').write_text(
                    '[1.2] Begin detour kernelsBatch 2, Size: 0, 1 ms\n'
                    'Batch 1, Size: 1, 2 ms\n'
                    f'Batch 0, Size: {count}, {elapsed} ms\n')
            batch = read_batch(root, 2, 'detour')
            self.assertEqual(batch['batches'], 3)
            self.assertEqual(batch['nets'], 9)
            self.assertEqual(batch['elapsed_ms'], 11)
            self.assertEqual(batch['critical_rank'], 1)

    def test_missing_middle_batch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / '1' / 'rank.0'
            folder.mkdir(parents=True)
            (folder / 'stdout').write_text('Batch 2, Size: 1, 2 ms\nBatch 0, Size: 4, 10 ms\n')
            with self.assertRaisesRegex(RuntimeError, 'batch order'):
                read_batch(root, 1, 'detour')


if __name__ == '__main__':
    unittest.main()
