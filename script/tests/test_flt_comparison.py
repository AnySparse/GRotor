"""Reject inconsistent paired FLT timings and plans before reporting speedups."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_flt_batching import parse_comparison

TEXT = """FLT_VARIANT method=rounds scope=coloring_loop_wall elapsed_ms=1 start_us=1000 end_us=2000 nets=3 batches=2
FLT_VARIANT method=cached scope=pipeline_to_coloring_wall elapsed_ms=2 start_us=5000 end_us=7000 nets=3 batches=2
FLT_PLAN_CHECK exact_lists_equal=1 routing_method=cached
"""


class FltComparisonTests(unittest.TestCase):
    def test_native_spans_and_routing_method(self):
        rows = parse_comparison(TEXT, dict(nets=3, batches=2, elapsed_ms=2))
        self.assertEqual([r["elapsed_ms"] for r in rows], [1, 2])
        self.assertEqual([r["method"] for r in rows], ["rounds", "cached"])

    def test_reject_incorrect_timing_plan_or_workload(self):
        for changed in (TEXT.replace("elapsed_ms=1", "elapsed_ms=0.5"),
                        TEXT.replace("exact_lists_equal=1", "exact_lists_equal=0"),
                        TEXT.replace("nets=3", "nets=4", 1),
                        TEXT.replace("scope=coloring_loop_wall", "scope=kernel_sum")):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                parse_comparison(changed, dict(nets=3, batches=2, elapsed_ms=2))

    def test_reject_overlapping_calls_or_wrong_routing_time(self):
        overlap = TEXT.replace("start_us=5000 end_us=7000", "start_us=1500 end_us=3500")
        with self.assertRaises(RuntimeError):
            parse_comparison(overlap, dict(nets=3, batches=2, elapsed_ms=2))
        with self.assertRaises(RuntimeError):
            parse_comparison(TEXT, dict(nets=3, batches=2, elapsed_ms=1))


if __name__ == "__main__":unittest.main()
