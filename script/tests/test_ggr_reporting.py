"""Check single-stage timing validation and native/optimized report semantics."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ggr_reporting import parse_native,parse_grotor,table_rows,write_report

NATIVE='number of batches: 2;  number of nets: 10\nGGR_RUNTIME_METRICS 0.004000 0.008000 0.000000 0.000000\n'
GPU='''BATCH_TIMING backend=rt_core start_us=1000 end_us=2000 elapsed_ms=1.000000 nets=11 batches=3 used_for_routing=1
DAG_TIMING stage=zshape batch_index=-1 start_us=3000 end_us=5000 elapsed_ms=2.000000 nets=11 batches=3
ROUTING_STAGE_END stage=zshape
GROTOR_COMPLETE wall_seconds=0.1
'''

class GgrReportingTests(unittest.TestCase):
    def test_components_and_total(self):
        rows=table_rows('fixture',parse_native(NATIVE),parse_grotor(GPU))
        self.assertEqual([r['ggr_ms'] for r in rows],[4,8,12])
        self.assertEqual([r['grotor_ms'] for r in rows],[1,2,3])
        self.assertEqual([r['speedup'] for r in rows],[4,4,4])

    def test_reject_duplicate_intervals(self):
        with self.assertRaises(ValueError): parse_grotor(GPU+GPU)

    def test_reject_wrong_timestamps(self):
        with self.assertRaises(ValueError): parse_grotor(GPU.replace('end_us=5000','end_us=9000'))

    def test_reject_different_task_counts(self):
        with self.assertRaises(ValueError): parse_grotor(GPU.replace('nets=11 batches=3\nROUTING','nets=12 batches=3\nROUTING'))

    def test_reject_maze_measurements(self):
        with self.assertRaises(ValueError): parse_native(NATIVE.replace('0.000000 0.000000','1.000000 0.000000'))

    def test_reject_incomplete_run(self):
        with self.assertRaises(ValueError): parse_grotor(GPU.replace('GROTOR_COMPLETE','FAILED'))

    def test_workloads_must_have_comparable_scale(self):
        with self.assertRaises(ValueError): table_rows('fixture',parse_native(NATIVE),{**parse_grotor(GPU),'nets':100})

    def test_summary_separates_native_cost_model_and_records_evaluated_scores(self):
        record=dict(dataset='fixture',native=dict(timing=parse_native(NATIVE)),runs=[dict(timing=parse_grotor(GPU),quality=dict(
            wirelength='1.0000',via='2.0000',overflow='3.0000',total='6.0000',open_nets=0,incomplete_nets=0))])
        with tempfile.TemporaryDirectory() as directory:
            write_report(directory,record)
            report=(Path(directory)/'summary.md').read_text()
            self.assertIn('quality scores are not interchangeable',report)
            self.assertIn('one continuous synchronized batch-loop interval',report)
            self.assertTrue((Path(directory)/'runtime.csv').is_file())
            self.assertTrue((Path(directory)/'quality.csv').is_file())

if __name__=='__main__': unittest.main()
