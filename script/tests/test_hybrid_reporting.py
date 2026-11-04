"""Validate contiguous spans and uniform stage-based dispatch in reports."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtcore_reporting import parse_totals


def records(flt_method="cached", requested="auto", design_nets=5000000):
    lines, batches = [], []
    for i, (stage, method) in enumerate((("lshape", "rounds" if requested == "auto" else requested), ("flt", flt_method))):
        elapsed = 2 if method == "rounds" else 3
        scope = "coloring_loop_wall" if method == "rounds" else "pipeline_to_coloring_wall"
        lines += [f"RTCORE_STAGE stage={stage} method={method} design_nets={design_nets} input_nets=3",
                  f"RTCORE_TIMING method={method} scope={scope} elapsed_ms={elapsed}",
                  f"BATCH_PREPARATION method={method} preparation_ms=5 collection_ms=1 call_ms={elapsed+6}"]
        batches.append(dict(nets=3, elapsed_ms=elapsed, start_us=10000*i, end_us=10000*i+elapsed*1000))
    return "\n".join(lines), batches


class HybridReportingTests(unittest.TestCase):
    def test_partitioned_scope_includes_scene_construction_and_commits(self):
        text, batches = records(flt_method="partitioned", requested="partitioned", design_nets=55315745)
        text = text.replace("scope=pipeline_to_coloring_wall", "scope=partitioned_batch_wall")
        self.assertEqual([r["method"] for r in parse_totals(text, batches)], ["partitioned"] * 2)
        with self.assertRaises(RuntimeError):
            parse_totals(text.replace("scope=partitioned_batch_wall", "scope=coloring_loop_wall", 1), batches)

    def test_precompiled_partitioned_scope_remains_one_interval(self):
        text, batches = records(flt_method="partitioned", requested="partitioned", design_nets=55315745)
        text = text.replace("scope=pipeline_to_coloring_wall", "scope=partitioned_batch_wall")
        text = text.replace("scope=partitioned_batch_wall", "scope=partitioned_pipeline_prebuilt_wall", 1)
        self.assertEqual(parse_totals(text, batches)[0]["timing_scope"], "partitioned_pipeline_prebuilt_wall")
        both = text.replace("scope=partitioned_batch_wall", "scope=partitioned_pipeline_prebuilt_wall")
        self.assertEqual([r["timing_scope"] for r in parse_totals(both, batches)],
                         ["partitioned_pipeline_prebuilt_wall"] * 2)

    def test_contiguous_times_are_forwarded_without_component_sums(self):
        text, batches = records()
        rows = parse_totals(text, batches)
        self.assertEqual([x["method"] for x in rows], ["rounds", "cached"])
        self.assertEqual([x["elapsed_ms"] for x in rows], [2, 3])

    def test_stage_policy_does_not_depend_on_design_size(self):
        for size in (3, 5000000, 16000000):
            text, batches = records(design_nets=size)
            self.assertEqual([r["method"] for r in parse_totals(text, batches)], ["rounds", "cached"])
        text, batches = records(flt_method="rounds")
        with self.assertRaises(RuntimeError):parse_totals(text, batches)


    def test_reject_component_sum_or_mismatched_timestamp_span(self):
        text, batches = records()
        for changed in (text.replace("elapsed_ms=3", "elapsed_ms=4"),
                        text.replace("scope=coloring_loop_wall", "scope=queued_gpu_intervals", 1),
                        text.replace("scope=pipeline_to_coloring_wall", "scope=bvh_plus_coloring", 1)):
            with self.subTest(text=changed), self.assertRaises(RuntimeError):parse_totals(changed, batches)
        batches[1]["elapsed_ms"] = 4
        with self.assertRaises(RuntimeError):parse_totals(text.replace("elapsed_ms=3", "elapsed_ms=4"), batches)

    def test_reject_missing_or_duplicate_interval(self):
        text, batches = records()
        for changed in (text.replace("RTCORE_TIMING", "MISSING", 1), text+"\n"+text):
            with self.assertRaises(RuntimeError):parse_totals(changed, batches)


if __name__ == "__main__":unittest.main()
