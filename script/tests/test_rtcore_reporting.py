"""Require measured RT-Core durations to match their contiguous timestamp spans."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_grotor_rtcore import parse_rtcore_timings, parse_execution_timings
from test_instantgr2 import parse_routing_timings


def log_text(elapsed="2.000000"):
    return f"""BATCH_VALIDATION nets=3 rtcore_batches=2 cpu_batches=2 exact_lists_equal=1
BATCH_BACKEND=rtcore
BATCH_TIMING backend=rt_core start_us=1000 end_us=3000 elapsed_ms={elapsed} nets=3 batches=2 used_for_routing=1
DAG_TIMING stage=lshape batch_index=-1 start_us=4000 end_us=5000 elapsed_ms=1.000000 nets=3 batches=2
BATCH_VALIDATION nets=2 rtcore_batches=1 cpu_batches=1 exact_lists_equal=1
BATCH_BACKEND=rtcore
BATCH_TIMING backend=rt_core start_us=6000 end_us=8000 elapsed_ms=2.000000 nets=2 batches=1 used_for_routing=1
DAG_TIMING stage=detour batch_index=-1 start_us=9000 end_us=12000 elapsed_ms=2.000000 nets=2 batches=1
"""


class RtcoreReportingTest(unittest.TestCase):
    def test_wall_interval_is_used_directly(self):
        batches = parse_rtcore_timings(log_text())
        self.assertEqual([r["elapsed_ms"] for r in batches], [2, 2])
        _, algorithms = parse_routing_timings(log_text(), batches)
        self.assertEqual([r["dag_solve_ms"] for r in algorithms], [1, 2])
        self.assertTrue(all(r["routing_backend"] == "rt_core" for r in algorithms))

    def test_reject_partial_duration_inside_larger_span(self):
        with self.assertRaises(RuntimeError):parse_rtcore_timings(log_text(elapsed="0.750000"))

    def test_failed_or_missing_validation_is_rejected(self):
        for changed in (log_text().replace("exact_lists_equal=1", "exact_lists_equal=0", 1),
                        log_text().replace("BATCH_VALIDATION", "MISSING", 1)):
            with self.assertRaises(RuntimeError):
                parse_rtcore_timings(changed)

    def test_performance_mode_can_skip_validation(self):
        text = "\n".join(line for line in log_text().splitlines() if not line.startswith("BATCH_VALIDATION"))
        self.assertEqual(len(parse_rtcore_timings(text, validate=False)), 2)

    def test_incomplete_timing_or_cpu_fallback_is_rejected(self):
        for changed in (log_text().replace("backend=rt_core", "backend=cpu", 1),
                        log_text().replace("BATCH_BACKEND=rtcore", "BATCH_BACKEND=cpu", 1),
                        log_text(elapsed="3.000000")):
            with self.assertRaises(RuntimeError):
                parse_rtcore_timings(changed)


if __name__ == "__main__":
    unittest.main()
