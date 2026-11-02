"""Check the CUDA-control and RT-Core stage accounting requested by the report."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_rtcore_hybrid import routing_comparison

class SystemRuntimeTests(unittest.TestCase):
    def test_stage_total_uses_cuda_time_with_cpu_plan(self):
        case = dict(name="fixture", baseline=dict(
            batch_timings=[dict(stage=stage, backend=backend, elapsed_ms=ms)
                for stage in ("lshape", "detour")
                for backend, ms in (("cpu", 999), ("cuda_core", 10))],
            dag_timings=[dict(elapsed_ms=20), dict(elapsed_ms=40)]),
            rtcore=dict(totals=[dict(elapsed_ms=2),dict(elapsed_ms=4)],
                dag_timings=[dict(elapsed_ms=3),dict(elapsed_ms=6)]))
        dag, stages = routing_comparison(case)
        self.assertEqual([r["instantgr2_total_ms"] for r in stages], [30,50])
        self.assertEqual([r["grotor_total_ms"] for r in stages], [5,10])
        self.assertEqual([r["speedup"] for r in stages], [6,5])
        self.assertEqual([r["speedup"] for r in dag], [20/3,40/6])
        self.assertEqual([r["algorithm"] for r in stages], ["L-shape","FLT"])

if __name__ == "__main__":unittest.main()
