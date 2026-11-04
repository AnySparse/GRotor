"""Reject wrong prefix counter scopes, unequal workloads and missing batches."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fig11_reporting import load_measurements, export_case, RAW_FIELDS, VARIANTS, FIELDS


def fixture(folder):
    metadata=dict(status='completed',stage='lshape',shared_batch_plan=True,same_initial_demand=True,sparsity=1,
        solver='warp',counter='logical_prefix_elements',total_scope='query_reads_plus_update_writes',
        bit_storage='full_shifted_bit_without_dense_cache',first_batch_initialization_included=True,
        routing_outputs_retained=0,batches=2,nets=30,cells=1000,evaluator_exit_status=[1,1,1])
    (folder/'measurement.json').write_text(json.dumps(metadata))
    for mode,name in enumerate(VARIANTS):
        with (folder/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,RAW_FIELDS);writer.writeheader()
            for i in range(2):
                reads=(20,50,30)[mode];writes=1000 if not mode or not i else 5
                writer.writerow(dict(batch=i,color_id=1-i,nets=10*(i+1),nodes=20*(i+1),plan_hash=100+i,
                    query_calls=10,query_reads=reads,prefix_writes=writes,total_accesses=reads+writes,
                    maintenance_reads=100 if mode and i else 0))
        (folder/(name+'.evaluator.log')).write_text('\n'.join([
            'wirelength cost 12.0000','via cost 3.0000','overflow cost 2.0000','total cost 17.0000',
            'Number of open nets : 0','Number of incompleted nets : 0','FOR_STAT 12 3 2 17']))
    return metadata


def mutate(folder,variant,row,key,value):
    path=folder/(VARIANTS[variant]+'.csv')
    with path.open() as stream:rows=list(csv.DictReader(stream))
    rows[row][key]=value
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,RAW_FIELDS);writer.writeheader();writer.writerows(rows)


class Fig11ReportingTests(unittest.TestCase):
    def test_totals_include_initialization_and_exclude_maintenance_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);fixture(folder);rows,checks=load_measurements(folder)
            self.assertEqual([r['bit'] for r in rows],[1050,55])
            self.assertEqual(checks['original_total'],2040)
            self.assertEqual(checks['bit_total'],1105)
            self.assertEqual(checks['bit_pruning_total'],1065)
            self.assertEqual(checks['bit_maintenance_reads'],100)

    def test_three_columns_preserve_every_batch(self):
        with tempfile.TemporaryDirectory() as directory,patch('fig11_reporting.plot_accesses'):
            folder=Path(directory);fixture(folder);export_case('example',folder)
            with (folder/'accesses.csv').open() as stream:rows=list(csv.reader(stream))
            self.assertEqual(rows[0],FIELDS)
            self.assertEqual(rows[1:],[['1020','1050','1030'],['1020','55','35']])

    def test_reject_mismatched_query_or_incremental_update_workloads(self):
        for variant,key,value in ((1,'query_calls',11),(2,'plan_hash',999),(2,'prefix_writes',6),
                                  (2,'maintenance_reads',101),(1,'nets',21)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);fixture(folder);mutate(folder,variant,1,key,value)
                with self.assertRaises(ValueError):load_measurements(folder)

    def test_reject_invalid_counter_definition_or_omitted_initialization(self):
        for variant,row,key,value in ((0,1,'prefix_writes',999),(0,1,'query_reads',21),
                (1,0,'prefix_writes',5),(1,1,'total_accesses',155),(2,1,'query_reads',51),
                (2,1,'query_reads',-1),(0,1,'batch',2)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);fixture(folder);mutate(folder,variant,row,key,value)
                with self.assertRaises(ValueError):load_measurements(folder)

    def test_reject_wrong_method_or_incomplete_coverage(self):
        for key,value in (('counter','dram_transactions'),('bit_storage','dense_cache'),('solver','mixed'),
                          ('first_batch_initialization_included',False),('nets',31),('batches',3)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);metadata=fixture(folder);metadata[key]=value
                (folder/'measurement.json').write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):load_measurements(folder)

    def test_reject_quality_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);fixture(folder);path=folder/'bit-pruning.evaluator.log'
            path.write_text(path.read_text().replace('17.0000','17.0001'))
            with self.assertRaises(ValueError):load_measurements(folder)


if __name__=='__main__':unittest.main()
