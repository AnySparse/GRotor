"""Reject mismatched batch plans, incorrect units and unequal routing quality."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fig10_reporting import read_variant, load_measurements, export_case, VARIANTS, FIELDS


def fixture(folder):
    metadata=dict(status="completed",stage="lshape",shared_batch_plan=True,same_initial_demand=True,sparsity=1,
        nets=30,batches=2,timer="synchronized_steady_clock",timing_scope="cost_prefix_solve_and_demand_commit",
        without_solver="thread",without_cost="full",without_prefix="full",with_solver="warp",with_cost="incremental",
        with_prefix="bit",routing_outputs_retained=0,evaluator_exit_status=[1,1])
    (folder/"measurement.json").write_text(json.dumps(metadata))
    for name,times in zip(VARIANTS,((1000,4000),(1000,1000))):
        start=0
        with (folder/(name+".csv")).open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["batch","color_id","nets","nodes","plan_hash","start_us","end_us","elapsed_ms","nets_per_second"])
            for i,(nets,us) in enumerate(zip((10,20),times)):
                writer.writerow([i,1-i,nets,nets*2,100+i,start,start+us,us/1000,nets*1e6/us]);start+=us+100
        (folder/(name+".evaluator.log")).write_text("\n".join([
            "wirelength cost 12.0000","via cost 3.0000","overflow cost 2.0000","total cost 17.0000",
            "Number of open nets : 0","Number of incompleted nets : 0","FOR_STAT 12 3 2 17"]))
    return metadata


class Fig10ReportingTests(unittest.TestCase):
    def test_units_same_batches_and_weighted_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);fixture(folder)
            rows,checks=load_measurements(folder)
            self.assertEqual([r["without_nets_per_second"] for r in rows],[10000,5000])
            self.assertEqual([r["with_nets_per_second"] for r in rows],[10000,20000])
            self.assertEqual(checks["without_aggregate_throughput"],6000)
            self.assertEqual(checks["with_aggregate_throughput"],15000)
            self.assertEqual(checks["aggregate_speedup"],2.5)
            self.assertTrue(checks["all_four_evaluator_costs_equal"])

    def test_exactly_two_export_columns_and_every_batch(self):
        with tempfile.TemporaryDirectory() as directory,patch("fig10_reporting.plot_throughput"):
            folder=Path(directory);fixture(folder);export_case("test",folder)
            with (folder/"throughput.csv").open() as stream:rows=list(csv.reader(stream))
            self.assertEqual(rows[0],FIELDS)
            self.assertEqual(rows[1:],[['10000.0','10000.0'],['5000.0','20000.0']])
            with (folder/"batch-runtime.csv").open() as stream:rows=list(csv.DictReader(stream))
            self.assertEqual([int(r["nets"]) for r in rows],[10,20])
            self.assertEqual(len(rows),2)

    def test_reject_different_membership_count_topology_or_batch_order(self):
        for key,value in (("nets",11),("nodes",21),("plan_hash",999),("color_id",0)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);fixture(folder);path=folder/(VARIANTS[1]+".csv")
                with path.open() as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
                rows[0][key]=str(value)
                if key=="nets":rows[0]["nets_per_second"]="11000"
                with path.open("w",newline="") as stream:
                    writer=csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
                with self.assertRaises(ValueError):load_measurements(folder)

    def test_reject_invalid_or_truncated_timing_measurements(self):
        for bad in ("0,1,10,20,100,0,1000,1,10\n",  # nets/ms mistakenly labelled nets/s
                    "0,1,10,20,100,0,0,0,10000\n",
                    "0,1,10,20,100,0,1000,nan,10000\n",
                    "0,1,10,20,100,0,1000,1,inf\n",
                    "1,1,10,20,100,0,1000,1,10000\n",
                    "0,1,10,20,100,0,1000,1,10000\n"):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);fixture(folder);path=folder/(VARIANTS[0]+".csv")
                path.write_text(path.read_text().splitlines()[0]+"\n"+bad)
                with self.assertRaises(ValueError):load_measurements(folder)

    def test_reject_configuration_that_does_not_ablate_requested_optimizations(self):
        for key,value in (("with_cost","full"),("with_prefix","full"),("without_solver","warp"),
                          ("same_initial_demand",False),("shared_batch_plan",False),("nets",31),("batches",3)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);metadata=fixture(folder);metadata[key]=value
                (folder/"measurement.json").write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):load_measurements(folder)

    def test_reject_unequal_costs_or_incomplete_external_evaluation(self):
        for old,new in (("17.0000","17.0001"),("open nets : 0","open nets : 1"),("FOR_STAT","MISSING")):
            with self.subTest(new=new),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);fixture(folder);path=folder/(VARIANTS[1]+".evaluator.log")
                path.write_text(path.read_text().replace(old,new))
                with self.assertRaises((ValueError,RuntimeError)):load_measurements(folder)


if __name__=="__main__":unittest.main()
