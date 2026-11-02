"""Check the experiment's timing boundaries and reported arithmetic without a GPU."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_instantgr2 import parse_batch_timings, parse_routing_timings, write_summary


class BatchTimingOrderTest(unittest.TestCase):
    def setUp(self):
        self.lines = [
            "BATCH_TIMING backend=cuda_core start_us=10000 end_us=11000 elapsed_ms=1.000000 nets=10 batches=3 used_for_routing=0",
            "BATCH_TIMING backend=cpu start_us=12000 end_us=13000 elapsed_ms=1.000000 nets=10 batches=3 used_for_routing=1",
            "BATCH_TIMING backend=cuda_core start_us=20000 end_us=21000 elapsed_ms=1.000000 nets=5 batches=2 used_for_routing=0",
            "BATCH_TIMING backend=cpu start_us=22000 end_us=23000 elapsed_ms=1.000000 nets=5 batches=2 used_for_routing=1",
        ]

    def test_unlabelled_records_are_assigned_by_call_order(self):
        rows = parse_batch_timings("\n".join(self.lines), "dual", "cpu")
        self.assertEqual([r["stage"] for r in rows], ["lshape", "lshape", "detour", "detour"])
        self.assertEqual([r["nets"] for r in rows], [10, 10, 5, 5])

    def test_single_backend_modes(self):
        cpu = "\n".join(self.lines[1::2])
        gpu = "\n".join(self.lines[::2]).replace("used_for_routing=0", "used_for_routing=1")
        for text, mode, backend in ((cpu, "determinism", "cpu"), (gpu, "performance", "gpu")):
            with self.subTest(mode=mode):
                rows = parse_batch_timings(text, mode, backend)
                self.assertEqual([r["stage"] for r in rows], ["lshape", "detour"])

    def test_missing_duplicate_or_reordered_records_are_rejected(self):
        for lines in (self.lines[:-1], self.lines + [self.lines[-1]], list(reversed(self.lines))):
            with self.subTest(lines=lines), self.assertRaises(RuntimeError):
                parse_batch_timings("\n".join(lines), "dual", "cpu")


class RoutingTimingsTest(unittest.TestCase):
    def setUp(self):
        self.batching = [
            {"stage": "lshape", "backend": "cuda_core", "elapsed_ms": 2, "used_for_routing": False},
            {"stage": "lshape", "backend": "cpu", "elapsed_ms": 99, "used_for_routing": True,
             "end_us": 10000, "nets": 10, "batches": 3},
            {"stage": "detour", "backend": "cuda_core", "elapsed_ms": 3, "used_for_routing": False},
            {"stage": "detour", "backend": "cpu", "elapsed_ms": 200, "used_for_routing": True,
             "end_us": 1500000, "nets": 5, "batches": 2},
        ]
        # FLT's enclosing span includes CPU packing; elapsed_ms sums only GPU suffixes.
        self.text = "\n".join([
            "DAG_TIMING stage=lshape batch_index=-1 start_us=1000000 end_us=1004000 elapsed_ms=4.000000 nets=10 batches=3",
            "DAG_TIMING stage=detour batch_index=-1 start_us=2000000 end_us=9007000 elapsed_ms=12.000000 nets=5 batches=2",
        ])

    def test_sum_excludes_cpu_packing_and_cpu_batching(self):
        _, rows = parse_routing_timings(self.text, self.batching)
        self.assertEqual([r["dag_solve_ms"] for r in rows], [4, 12])
        self.assertEqual([r["algorithm_total_ms"] for r in rows], [6, 15])

    def test_cpu_only_does_not_substitute_cpu_batching(self):
        _, rows = parse_routing_timings(self.text, [t for t in self.batching if t["backend"] == "cpu"])
        self.assertTrue(all(r["cuda_batch_ms"] is None and r["algorithm_total_ms"] is None for r in rows))

    def test_incomplete_duplicate_or_mislabeled_batch_is_rejected(self):
        lines = self.text.splitlines()
        for invalid in ("\n".join(lines[:-1]), self.text + "\n" + lines[-1],
                        self.text.replace("stage=detour batch_index=-1", "stage=detour batch_index=0")):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                parse_routing_timings(invalid, self.batching)

    def test_incorrect_duration_or_net_coverage_is_rejected(self):
        for invalid in (self.text.replace("elapsed_ms=12.000000", "elapsed_ms=7008.000000"),
                        self.text.replace("elapsed_ms=4.000000", "elapsed_ms=3.000000"),
                        self.text.replace("nets=5", "nets=4"),
                        self.text.replace("start_us=2000000", "start_us=1000000")):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                parse_routing_timings(invalid, self.batching)

    def test_report_keeps_distinct_algorithm_totals(self):
        _, rows = parse_routing_timings(self.text, self.batching)
        result = {"status": "passed", "mode": "dual", "batching_backend": "cpu", "repeats": 1,
                  "source_commit": "fixture", "cuda_arch": "89", "sparsity": 1, "cases": [
                      {"name": "fixture", "status": "passed", "runs": [
                          {"round": 1, "status": "passed", "algorithm_runtime": rows}]}]}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_summary(folder, result)
            with (folder / "runtime.csv").open() as source:
                exported = list(csv.DictReader(source))
            self.assertEqual([float(r["algorithm_total_ms"]) for r in exported], [6, 15])
            report = (folder / "summary.md").read_text()
            self.assertIn("| fixture | L-shape | 2.000000 | 4.000000 | 6.000000 |", report)
            self.assertIn("| fixture | FLT | 3.000000 | 12.000000 | 15.000000 |", report)


if __name__ == "__main__":
    unittest.main()
