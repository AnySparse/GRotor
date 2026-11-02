"""Check speedup aggregation and reject incompatible experiment records."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_batching import compare, restore_stage_labels


def experiments():
    metrics = dict(wirelength="1.0000", via="0.0000", overflow="0.0000", total="1.0000",
                   open_nets=0, incomplete_nets=0)
    inputs = {"case.cap": "cap-hash", "case.net": "net-hash"}
    runs = []
    for lshape, flt in ((1, 100), (10, 10), (100, 1)):
        timings = []
        for stage, elapsed in (("lshape", lshape), ("detour", flt)):
            timings += [dict(stage=stage, backend="cpu", elapsed_ms=elapsed, nets=10, batches=1),
                        dict(stage=stage, backend="cuda_core", elapsed_ms=elapsed + 2 if elapsed < 100 else 120,
                             nets=10, batches=2)]
        runs.append(dict(status="passed", metrics=metrics.copy(), batch_timings=timings))
    baseline = dict(status="passed", mode="dual", sparsity=1, batching_backend="cpu", cuda_arch="89", gpu="0",
                    source_sha256={"src/a.cu": "source-hash"},
                    cases=[dict(name="case", status="passed", inputs=inputs.copy(), runs=runs)])
    rt = dict(status="passed", configuration=dict(sparsity=1, cuda_arch="89", gpu="0", validate_batches=True,
              sources={"baseline/InstantGR2/src/a.cu": "source-hash"}), binary_sha256={"GRotor": "binary-hash"},
              oracles={"case": dict(inputs=inputs.copy(), metrics=metrics.copy())},
              runs=[dict(dataset="case", status="passed", matches_instantgr2=True, metrics=metrics.copy(),
                         batch_timings=[dict(stage=s, backend="rt_core", elapsed_ms=t, nets=10, batches=1)
                                        for s, t in (("lshape", 2), ("detour", 3))])])
    return baseline, rt


class BatchingComparisonTest(unittest.TestCase):
    def test_compatibility_transform_restores_only_stage_labels(self):
        source = 'auto batches = generate_batches_rsmt_selected(nets2route, 1000000);\n'
        self.assertEqual(restore_stage_labels("src/Lshape_route.hpp", source),
                         'auto batches = generate_batches_rsmt_selected(nets2route, 1000000, "lshape");\n')
        self.assertIsNone(restore_stage_labels("src/unknown.hpp", source))

    def test_ratio_of_medians_and_sum_within_each_run(self):
        baseline, rt = experiments()
        rows = {r["algorithm"]: r for r in compare(baseline, [rt])}
        self.assertEqual(rows["L-shape"]["speedup_vs_cpu"], 5)
        self.assertEqual(rows["FLT"]["speedup_vs_cpu"], 10 / 3)
        # The median of the sums is 101; summing the stage medians gives 20.
        self.assertEqual(rows["Both"]["cpu_ms"], 101)
        self.assertEqual(rows["Both"]["speedup_vs_cpu"], 101 / 5)

    def test_reject_changed_input_source_cost_or_work_count(self):
        baseline, rt = experiments()
        for kind in ("input", "source", "cost", "work"):
            candidate = copy.deepcopy(rt)
            if kind == "input":
                candidate["oracles"]["case"]["inputs"]["case.cap"] = "changed"
            elif kind == "source":
                candidate["configuration"]["sources"]["baseline/InstantGR2/src/a.cu"] = "changed"
            elif kind == "cost":
                candidate["runs"][0]["metrics"]["total"] = "2.0000"
            else:
                candidate["runs"][0]["batch_timings"][0]["batches"] = 2
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                compare(baseline, [candidate])

    def test_reject_missing_dataset_or_failed_run(self):
        baseline, rt = experiments()
        with self.assertRaises(ValueError):
            compare(baseline, [])
        rt["runs"][0]["status"] = "failed"
        with self.assertRaises(ValueError):
            compare(baseline, [rt])


if __name__ == "__main__":
    unittest.main()
