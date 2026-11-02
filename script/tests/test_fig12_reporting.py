"""Check cumulative ablations, CPU controls, timer scopes and speedup denominators."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fig12_reporting import VARIANTS, parse_run, table_rows
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('run_fig12',Path(__file__).resolve().parents[1]/'run-fig12.py')
runner=module_from_spec(spec);spec.loader.exec_module(runner)


def fixture():
    metrics=dict(wirelength='1.0000',via='2.0000',overflow='3.0000',total='6.0000',open_nets=0,incomplete_nets=0)
    runs={}
    for variant,dag,batch in zip(VARIANTS,(90,40,15,15),(10,10,10,5)):
        runs[variant]=dict(status='passed',metrics=copy.deepcopy(metrics),timings=[
            dict(algorithm=name,nets=30,batches=3,batching_ms=batch,dag_ms=dag,total_ms=batch+dag) for name in ('L-shape','FLT')])
    return dict(status='passed',cases=[dict(name='example',status='passed',runs=runs)])


def cpu_text(variant):
    cost='full prefix=full' if variant=='original' else 'incremental prefix=bit'
    schedule='thread flt=global_depth' if variant in ('original','bit') else 'warp flt=block'
    return f'''GROTOR_CONFIG batching=cpu cost={cost} memory=explicit
DAG_SCHEDULING lshape={schedule}
ABLATION_CONFIG variant={variant} counters=off batching=cpu
BATCH_BACKEND=cpu
BATCH_TIMING backend=cpu start_us=1000 end_us=11000 elapsed_ms=10 nets=30 batches=3 used_for_routing=1
DAG_TIMING stage=lshape batch_index=-1 start_us=12000 end_us=102000 elapsed_ms=90 nets=30 batches=3
BATCH_BACKEND=cpu
BATCH_TIMING backend=cpu start_us=200000 end_us=210000 elapsed_ms=10 nets=20 batches=2 used_for_routing=1
DAG_TIMING stage=detour batch_index=-1 start_us=220000 end_us=310000 elapsed_ms=90 nets=20 batches=2
'''


class Fig12ReportingTests(unittest.TestCase):
    def test_cumulative_total_and_step_speedups(self):
        rows=table_rows(fixture())
        self.assertEqual([r['speedup'] for r in rows[:4]],[1,2,4,5])
        self.assertEqual([r['incremental_speedup'] for r in rows[:4]],[1,2,2,1.25])
        self.assertEqual([r['dag_speedup'] for r in rows[:4]],[1,2.25,6,6])
        self.assertEqual([r['algorithm'] for r in rows],['L-shape']*4+['FLT']*4)

    def test_reject_quality_or_batch_workload_changes(self):
        for key,value in (('total_ms',99),('nets',31),('batches',4),('dag_ms',float('nan'))):
            with self.subTest(key=key):
                r=fixture();r['cases'][0]['runs']['bit']['timings'][0][key]=value
                with self.assertRaises(ValueError):table_rows(r)
        r=fixture();r['cases'][0]['runs']['rtcore']['metrics']['total']='6.0001'
        with self.assertRaises(ValueError):table_rows(r)

    def test_correct_cpu_configuration_and_contiguous_timers(self):
        for variant in VARIANTS[:3]:
            with self.subTest(variant=variant):
                rows=parse_run(cpu_text(variant),variant)
                self.assertEqual([r['total_ms'] for r in rows],[100,100])

    def test_reject_profile_counters_wrong_scheduler_or_partial_flt_timer(self):
        text=cpu_text('original')
        for changed in (text.replace('counters=off','counters=on'),text.replace('flt=global_depth','flt=block'),
                        text.replace('backend=cpu','backend=cuda_core'),text.replace('elapsed_ms=90 nets=20','elapsed_ms=80 nets=20'),
                        text.replace('batch_index=-1','batch_index=0'),text.replace('cost=full','cost=incremental')):
            with self.subTest(changed=changed):
                with self.assertRaises((ValueError,RuntimeError)):parse_run(changed,'original')

    def test_exclude_tera_cluster_and_invalid_dataset_names(self):
        runner.validate_case_name('mempool_cluster')
        runner.validate_case_name('nvdla')
        for name in ('tera_cluster','tera_cluster_sample','TERA-CLUSTER-rank','../nvdla','nvdla mempool_tile'):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):runner.validate_case_name(name)


if __name__=='__main__':unittest.main()
