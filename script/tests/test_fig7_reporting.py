"""Check full-round paired counts and reject padded or inconsistent ablations."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fig7_reporting import read_counts


def fixture(folder):
    checks=dict(status="passed",same_iteration_count=True,exact_assignment_rounds=True,exact_batch_ids=True,
        exact_cpu_batch_lists=True,publication="round_end",eligibility="frozen",counter="actual_optixTrace_calls",
        conflict_cache=False,input_rays=3,nets=3,iterations=3,without_blocker_total=6,with_blocker_total=5)
    (folder/"checks.json").write_text(json.dumps(checks))
    (folder/"ray-counts.csv").write_text("without_blocker,with_blocker\n3,3\n2,1\n1,1\n")
    (folder/"round-stats.csv").write_text("iteration,without_blocker,with_blocker,unassigned_before,unassigned_after,assigned_this_round\n0,3,3,3,2,1\n1,2,1,2,1,1\n2,1,1,1,0,1\n")
    return checks


class Fig7ReportingTests(unittest.TestCase):
    def test_exact_two_columns_and_all_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);fixture(folder)
            rows,checks=read_counts(folder)
            self.assertEqual(rows,[(3,3),(2,1),(1,1)])
            self.assertEqual(checks["iterations"],len(rows))

    def test_reject_invalid_counts_padding_and_extra_column(self):
        for bad in ("3,3\n2,0\n1,1\n","3,3\n2,3\n1,1\n","3,3\n2,1\n",
                    "3,3\n2,1\n1,1\n0,0\n","3,3\n2.0,1\n1,1\n"):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);fixture(folder)
                (folder/"ray-counts.csv").write_text("without_blocker,with_blocker\n"+bad)
                with self.assertRaises(ValueError):read_counts(folder)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);fixture(folder)
            p=folder/"ray-counts.csv";p.write_text(p.read_text().replace("without_blocker,","iteration,without_blocker,",1))
            with self.assertRaises(ValueError):read_counts(folder)

    def test_reject_asynchronous_or_unvalidated_inputs(self):
        for key,value in (("publication","immediate"),("eligibility","mutable"),("counter","launch_width"),
            ("conflict_cache",True),("exact_assignment_rounds",False),("exact_cpu_batch_lists",False),
            ("iterations",4),("with_blocker_total",9)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);checks=fixture(folder);checks[key]=value
                (folder/"checks.json").write_text(json.dumps(checks))
                with self.assertRaises(ValueError):read_counts(folder)

    def test_reject_incomplete_or_inconsistent_assignment_history(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);fixture(folder)
            p=folder/"round-stats.csv";p.write_text(p.read_text().replace("2,1,1,1,0,1","2,1,1,1,1,0"))
            with self.assertRaises(ValueError):read_counts(folder)


if __name__=="__main__":unittest.main()
