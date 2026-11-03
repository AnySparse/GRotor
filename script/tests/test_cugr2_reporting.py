"""Check stage coverage, timing scopes and workload-aware CPU/GPU comparison."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cugr2_reporting import STAGES, comparison_rows, parse_cpu, parse_grotor, pending_cpu_tasks, write_report
from compare_cugr2 import relative_cpu_sources


def gpu_fixture():
    lines = []
    for i, stage in enumerate(STAGES):
        start = i * 10000
        lines += [f"ROUTING_STAGE_BEGIN stage={stage} nets=10",
            f"BATCH_TIMING backend=rt_core start_us={start} end_us={start + 1000} elapsed_ms=1.000000 nets=10 batches=2 used_for_routing=1",
            f"DAG_TIMING stage={stage} batch_index=-1 start_us={start + 2000} end_us={start + 4000} elapsed_ms=2.000000 nets=10 batches=2"]
        if stage == "maze":
            lines.append("MAZE_SEARCH elapsed_ms=4.000000 nets=10 threads=8 sparse_interval=10")
            lines.append("MAZE_QUALITY_GATE accepted=0 before_score=10 candidate_score=20")
            lines.append(f"MAZE_PHASE_TIMING start_us={start} end_us={start + 7000} elapsed_ms=7.000000")
        lines.append(f"ROUTING_STAGE_END stage={stage}")
    return "\n".join(lines)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.cpu = "\n".join(f"CUGR_STAGE_RUNTIME stage={stage} solver_ms=14.000000 full_ms=20.000000 nets=20"
                             for stage in STAGES)
        self.result = dict(dataset="fixture", status="passed", configuration=dict(max_net_ratio=10),
            cpu=dict(stages=parse_cpu(self.cpu)), grotor=dict(stages=parse_grotor(gpu_fixture())))

    def test_all_stage_totals_include_only_batching_and_gpu_dag(self):
        rows = comparison_rows(self.result)
        self.assertEqual([r["grotor_total_ms"] for r in rows], [3, 3, 3])
        self.assertAlmostEqual(rows[-1]["speedup"], 14 / 3)
        self.assertAlmostEqual(rows[-1]["per_net_speedup"], 7 / 3)
        self.assertEqual(rows[-1]["grotor_maze_search_ms"], 4)
        self.assertEqual(rows[-1]["grotor_maze_phase_ms"], 7)

    def test_comparison_does_not_use_a_stale_full_phase_total(self):
        self.result["grotor"]["stages"]["maze"]["total_ms"] = 100
        row = comparison_rows(self.result)[2]
        self.assertEqual(row["grotor_total_ms"], 3)
        self.assertAlmostEqual(row["speedup"], 14 / 3)

    def test_cpu_source_reuse_survives_vendor_directory_rename(self):
        previous = {"baseline/vendor-before/src/main.cpp": "main-digest",
                    "baseline/vendor-before/LICENSE": "license-digest"}
        current = {"baseline/cu-gr-2/src/main.cpp": "main-digest",
                   "baseline/cu-gr-2/LICENSE": "license-digest"}
        self.assertEqual(relative_cpu_sources(previous), relative_cpu_sources(current))
        current["baseline/cu-gr-2/src/main.cpp"] = "changed-digest"
        self.assertNotEqual(relative_cpu_sources(previous), relative_cpu_sources(current))

    def test_duplicate_relative_cpu_source_fingerprints_are_rejected(self):
        with self.assertRaises(ValueError):
            relative_cpu_sources({"baseline/vendor-before/src/main.cpp": "old",
                                  "baseline/cu-gr-2/src/main.cpp": "new"})

    def test_partial_cpu_run_retains_only_completed_comparisons(self):
        self.result["cpu"]["stages"] = parse_cpu(self.cpu.splitlines()[0])
        rows = comparison_rows(self.result)
        self.assertIsNotNone(rows[0]["speedup"])
        self.assertIsNone(rows[1]["speedup"])
        self.assertEqual(rows[2]["workload_status"], "incomplete")

    def test_archived_skipped_maze_is_not_a_zero_time_measurement(self):
        text = ("CUGR_RUNTIME_METRICS 2.0 3.0 0.0\nCUGR_FULL_STAGE_METRICS 4.0 5.0 0.0\n"
                "CUGR_TASK_COUNTS 20 10 0\n5 / 20 nets have overflows.\n"
                "stage 3: maze routing skipped\n")
        self.result["cpu"].update(stages=parse_cpu(text), pending_nets=pending_cpu_tasks(text))
        row = comparison_rows(self.result)[2]
        self.assertEqual(row["cugr2_nets"], 5)
        self.assertEqual(row["cpu_task_status"], "planned")
        self.assertIsNone(row["cugr2_ms"])
        self.assertIsNone(row["speedup"])

    def test_different_task_scale_has_no_speedup(self):
        self.result["cpu"]["stages"]["maze"]["nets"] = 101
        row = comparison_rows(self.result)[2]
        self.assertEqual(row["workload_status"], "different-scale")
        self.assertIsNone(row["speedup"])

    def test_one_empty_side_has_no_speedup(self):
        self.result["cpu"]["stages"]["maze"]["nets"] = 0
        row = comparison_rows(self.result)[2]
        self.assertEqual(row["workload_status"], "one-side-empty")
        self.assertIsNone(row["speedup"])

    def test_cpu_duplicate_or_reordered_stages_rejected(self):
        for text in (self.cpu + "\n" + self.cpu.splitlines()[0], "\n".join(reversed(self.cpu.splitlines()))):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_cpu(text)

    def test_missing_maze_search_rejected(self):
        with self.assertRaises(ValueError):
            parse_grotor(gpu_fixture().replace("MAZE_SEARCH", "IGNORED_SEARCH"))

    def test_incomplete_gpu_stage_rejected(self):
        with self.assertRaises(ValueError):
            parse_grotor(gpu_fixture().replace("ROUTING_STAGE_END stage=maze", ""))

    def test_maze_phase_overhead_is_retained_only_as_diagnostics(self):
        rows = parse_grotor(gpu_fixture().replace("end_us=27000 elapsed_ms=7.000000", "end_us=30000 elapsed_ms=10.000000"))
        self.assertEqual(rows["maze"]["total_ms"], 3)
        self.assertEqual(rows["maze"]["maze_phase_ms"], 10)
        self.assertEqual(rows["maze"]["maze_other_ms"], 3)
        self.assertFalse(rows["maze"]["candidate_accepted"])

    def test_noncontiguous_or_mismatched_work_rejected(self):
        for text in (gpu_fixture().replace("elapsed_ms=2.000000", "elapsed_ms=1.000000"),
                     gpu_fixture().replace("nets=10 batches=2", "nets=11 batches=2")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_grotor(text)

    def test_report_writes_component_runtime_and_quality(self):
        self.result["grotor"]["metrics"] = dict(wirelength="1", via="2", overflow="3", total="6",
                                                open_nets=0, incomplete_nets=0)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_report(folder, self.result)
            self.assertIn("Maze | 20 | 10 | 14.000 | 3.000 | 4.67", (folder / "summary.md").read_text())
            self.assertIn("## GRotor Components", (folder / "summary.md").read_text())
            self.assertEqual(len((folder / "runtime.csv").read_text().splitlines()), 4)
            self.assertIn("0.014,0.003,4.666666666666667", (folder / "table3.csv").read_text())
            self.assertEqual(len((folder / "quality.csv").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
