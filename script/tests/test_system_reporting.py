"""Check public component tables, measurement scopes and report-only operation."""
import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_rtcore_hybrid as engine
from benchmark_grotor_system import main
from system_reporting import comparison_rows, write_summary


def fixture():
    metrics = dict(wirelength="10.1250", via="2.0000", overflow="0.0000", total="12.1250", open_nets=0, incomplete_nets=0)
    baseline = dict(batch_timings=[], dag_timings=[], metrics=metrics.copy())
    native = dict(batch_timings=[], dag_timings=[], totals=[], metrics=metrics.copy(), cpu_peak_mib=123,
        gpu_memory=dict(sampled_peak_used_mib=456, sampled_min_free_mib=789))
    for i, stage in enumerate(("lshape", "detour")):
        def timing(ms, **extra):
            return dict(stage=stage, elapsed_ms=ms, start_us=1000, end_us=1000+ms*1000, nets=10, batches=3, **extra)
        baseline["batch_timings"] += [timing(999, backend="cpu", used_for_routing=True), timing(10, backend="cuda_core", used_for_routing=False)]
        baseline["batch_timings"][-1]["batches"] = 9  # GPU plan differs; CPU supplies routing.
        baseline["dag_timings"].append(timing(20*(i+1)))
        native["batch_timings"].append(timing(2*(i+1), backend="rt_core", used_for_routing=True))
        native["dag_timings"].append(timing(3*(i+1)))
        native["totals"].append(dict(method=("rounds","cached")[i], elapsed_ms=2*(i+1)))
    baseline["dag_timings"][1]["end_us"] += 2000000  # Old FLT excludes CPU gaps.
    return dict(status="passed", configuration=dict(sparsity=1,cost_update="incremental",prefix="bit",
        dag_lshape="warp",dag_flt="block",validate_batches=False), cases=[dict(name="fixture",status="passed",baseline=baseline,rtcore=native)])


class SystemReportTests(unittest.TestCase):
    def test_partitioned_tables_preserve_the_broader_scope(self):
        result = fixture()
        for total in result["cases"][0]["rtcore"]["totals"]:
            total.update(method="partitioned", timing_scope="partitioned_batch_wall")
        with tempfile.TemporaryDirectory() as directory:
            write_summary(Path(directory), result)
            with (Path(directory)/"batching-runtime.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["grotor_method"] for r in rows], ["partitioned"] * 2)
            self.assertEqual([r["grotor_timing_scope"] for r in rows], ["partitioned_batch_wall"] * 2)
        result["cases"][0]["rtcore"]["totals"][0]["timing_scope"] = "coloring_loop_wall"
        with self.assertRaises(ValueError):comparison_rows(result["cases"][0])

    def test_prebuilt_scope_is_preserved_for_both_large_stages(self):
        result = fixture()
        for total in result["cases"][0]["rtcore"]["totals"]:
            total.update(method="partitioned", timing_scope="partitioned_pipeline_prebuilt_wall")
        rows = comparison_rows(result["cases"][0])[1]
        self.assertEqual([r["grotor_timing_scope"] for r in rows],
                         ["partitioned_pipeline_prebuilt_wall"] * 2)

    def test_csv_and_markdown_keep_components_and_totals_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);write_summary(folder,fixture())
            with (folder/"runtime.csv").open() as f:rows=list(csv.DictReader(f))
            self.assertEqual([float(r["instantgr2_total_ms"]) for r in rows],[30,50])
            self.assertEqual([float(r["grotor_total_ms"]) for r in rows],[5,10])
            self.assertEqual([float(r["total_speedup"]) for r in rows],[6,5])
            self.assertEqual([float(r["batching_speedup"]) for r in rows],[5,2.5])
            self.assertEqual([float(r["dag_speedup"]) for r in rows],[20/3,40/6])
            self.assertNotIn("stage", rows[0]);self.assertNotIn("routing_backend",rows[0])
            for name in ("stage-runtime.csv","dag-runtime.csv","batching-runtime.csv","quality.csv","memory.csv","summary.json"):
                self.assertTrue((folder/name).is_file())
            text=(folder/"summary.md").read_text()
            self.assertEqual(text.count("### L-shape"),3);self.assertEqual(text.count("### FLT"),3)

    def test_reject_inconsistent_quality_workload_and_native_timestamps(self):
        for change in ("quality","open","workload","timestamp","method","negative","nan","baseline_span"):
            with self.subTest(change=change):
                case=copy.deepcopy(fixture()["cases"][0]);native=case["rtcore"]
                if change=="quality":native["metrics"]["via"]="3.0000"
                if change=="open":native["metrics"]["open_nets"]=1
                if change=="workload":native["batch_timings"][0]["batches"]=4
                if change=="timestamp":native["dag_timings"][1]["end_us"]+=1000
                if change=="method":native["totals"][0]["method"]="cached"
                if change=="negative":case["baseline"]["dag_timings"][0]["elapsed_ms"]=-1
                if change=="nan":case["baseline"]["dag_timings"][0]["elapsed_ms"]=float("nan")
                if change=="baseline_span":case["baseline"]["dag_timings"][1]["elapsed_ms"]=3000
                with self.assertRaises(ValueError):comparison_rows(case)

    def test_zero_time_has_no_infinite_speedup(self):
        case=fixture()["cases"][0]
        case["rtcore"]["dag_timings"][0].update(elapsed_ms=0,end_us=1000)
        rows,_,_,_=comparison_rows(case)
        self.assertIsNone(rows[0]["dag_speedup"])

    def test_incomplete_cases_do_not_generate_passed_rows(self):
        result=fixture();result["status"]="failed";result["cases"][0]["status"]="failed"
        with tempfile.TemporaryDirectory() as directory:
            write_summary(Path(directory),result)
            with (Path(directory)/"runtime.csv").open() as f:self.assertEqual(list(csv.DictReader(f)),[])

    def test_report_only_does_not_build_or_use_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"input.json";source.write_text(json.dumps(fixture()))
            output=root/"results/tables"
            with patch.object(engine,"ROOT",root),patch.object(engine,"require_cuda") as gpu,patch.object(engine,"run_logged") as build,contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--report-only",str(source),"--output-dir",str(output)]),0)
                gpu.assert_not_called();build.assert_not_called()
            self.assertEqual(json.loads(source.read_text()),fixture())
            report=json.loads((output/"summary.json").read_text())
            self.assertEqual(report["report_generation"]["source"],str(source))

    def test_cleanup_is_confined_and_preserves_report_input(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(engine,"ROOT",Path(directory)):
            root=Path(directory);folder=root/"results/tables";folder.mkdir(parents=True)
            source=folder/"summary.json";source.write_text("keep")
            for target,protected in ((root/"results",()),(root/"outside",()),(folder,(source,))):
                with self.assertRaises(ValueError):engine.prepare_output_folder(target,True,protected)
            self.assertEqual(source.read_text(),"keep")
            with self.assertRaises(ValueError):engine.prepare_output_folder(folder)
            engine.prepare_output_folder(folder,True)
            self.assertEqual(list(folder.iterdir()),[])


if __name__=="__main__":unittest.main()
