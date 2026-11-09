"""Validate paired per-batch L-shape throughput and produce DAG throughput."""
import csv
import json
import math
from pathlib import Path
from _support import evaluator_metrics

FIELDS = ["without_optimizations", "with_optimizations"]
DETAIL_FIELDS = ["batch","color_id","nets","nodes","plan_hash",
    "without_start_us","without_end_us","with_start_us","with_end_us",
    "without_ms","with_ms","without_nets_per_second","with_nets_per_second","speedup"]
VARIANTS = ("without-optimizations", "with-optimizations")


def read_variant(path):
    with path.open() as stream:
        reader=csv.DictReader(stream)
        expected=["batch","color_id","nets","nodes","plan_hash","start_us","end_us","elapsed_ms","nets_per_second"]
        if reader.fieldnames!=expected:raise ValueError("Unexpected DAG throughput timing columns")
        result=[]
        for i,raw in enumerate(reader):
            if set(raw)!=set(expected):raise ValueError("Malformed DAG throughput timing row")
            row={k:int(raw[k]) for k in expected[:7]}
            row.update({k:float(raw[k]) for k in expected[7:]})
            us=row["end_us"]-row["start_us"]
            if row["batch"]!=i or row["nets"]<1 or row["nodes"]<row["nets"] or us<=0:
                raise ValueError("Invalid batch coverage or timer interval")
            if not math.isfinite(row["elapsed_ms"]) or not math.isfinite(row["nets_per_second"]):
                raise ValueError("Nonfinite DAG throughput measurement")
            if abs(row["elapsed_ms"]-us/1000)>1e-8 or not math.isclose(row["nets_per_second"],row["nets"]*1e6/us,rel_tol=1e-12):
                raise ValueError("Throughput must equal measured nets / synchronized batch seconds")
            if result and row["start_us"]<result[-1]["end_us"]:raise ValueError("Overlapping or reordered batch intervals")
            result.append(row)
    if not result:raise ValueError("No measured L-shape batches")
    return result


def load_measurements(folder):
    measured=json.loads((folder/"measurement.json").read_text())
    required={"stage":"lshape","shared_batch_plan":True,"same_initial_demand":True,"sparsity":1,
        "timer":"synchronized_steady_clock","timing_scope":"cost_prefix_solve_and_demand_commit",
        "without_solver":"thread","without_cost":"full","without_prefix":"full",
        "with_solver":"warp","with_cost":"incremental","with_prefix":"bit","routing_outputs_retained":0}
    if measured.get("status")!="completed" or any(measured.get(k)!=v for k,v in required.items()):
        raise ValueError("DAG throughput requires shared-plan thread/full versus warp/BIT/incremental measurements")
    variants=[read_variant(folder/(name+".csv")) for name in VARIANTS]
    old,new=variants
    if len(old)!=len(new) or len(old)!=measured["batches"]:
        raise ValueError("Both variants must measure every identical batch")
    if sum(r["nets"] for r in old)!=measured["nets"]:
        raise ValueError("Measured batches do not cover all routing nets")
    rows=[]
    for i,(a,b) in enumerate(zip(old,new)):
        if any(a[k]!=b[k] for k in ("batch","color_id","nets","nodes","plan_hash")):
            raise ValueError("DAG throughput batch membership or topology differs")
        if a["color_id"]!=len(old)-1-i:
            raise ValueError("Batch order must match reversed-color L-shape execution")
        rows.append(dict(batch=i,color_id=a["color_id"],nets=a["nets"],nodes=a["nodes"],plan_hash=a["plan_hash"],
            without_start_us=a["start_us"],without_end_us=a["end_us"],with_start_us=b["start_us"],with_end_us=b["end_us"],
            without_ms=a["elapsed_ms"],with_ms=b["elapsed_ms"],without_nets_per_second=a["nets_per_second"],
            with_nets_per_second=b["nets_per_second"],speedup=a["elapsed_ms"]/b["elapsed_ms"]))
    codes=measured["evaluator_exit_status"]
    if len(codes)!=2:raise ValueError("Both actual routing outputs must be evaluated")
    metrics=[evaluator_metrics(folder/(name+".evaluator.log"),code) for name,code in zip(VARIANTS,codes)]
    if metrics[0]!=metrics[1]:raise ValueError("L-shape evaluator costs differ between variants")
    checks=dict(status="passed",stage="lshape",shared_batch_plan=True,same_batch_order=True,same_initial_demand=True,
        batches=len(rows),nets=measured["nets"],all_four_evaluator_costs_equal=True,metrics=metrics[0],
        without_total_ms=sum(r["without_ms"] for r in rows),with_total_ms=sum(r["with_ms"] for r in rows),routing_outputs_retained=0)
    checks["without_aggregate_throughput"]=checks["nets"]*1000/checks["without_total_ms"]
    checks["with_aggregate_throughput"]=checks["nets"]*1000/checks["with_total_ms"]
    checks["aggregate_speedup"]=checks["without_total_ms"]/checks["with_total_ms"]
    return rows,checks


def plot_throughput(name,rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context({"font.size":10,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"}):
        fig,axis=plt.subplots(figsize=(4.8,3.6))
        x=[r["batch"] for r in rows];markers=max(1,len(rows)//30)
        axis.plot(x,[r["without_nets_per_second"] for r in rows],color="black",marker="s",markerfacecolor="white",
            markersize=3,markevery=markers,linewidth=1.1,label="Without optimizations")
        axis.plot(x,[r["with_nets_per_second"] for r in rows],color="#1754df",marker="o",markerfacecolor="white",
            markersize=3,markevery=markers,linewidth=1.1,label="Warp + BIT + incremental cost")
        axis.set(title=name,xlabel="Batch",ylabel="Throughput (nets/s)",yscale="log")
        axis.set_xlim((0,len(rows)-1) if len(rows)>1 else (-.5,.5))
        axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
        axis.grid(axis="y",which="major",linestyle="--",linewidth=.6,alpha=.5)
        handles,labels=axis.get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",ncol=2,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.91))
        for suffix in ("pdf","svg","png"):
            fig.savefig(Path(str(output)+"."+suffix),dpi=300,facecolor="white",bbox_inches="tight")
        plt.close(fig)


def export_case(name,folder):
    rows,checks=load_measurements(folder)
    with (folder/"throughput.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(FIELDS)
        writer.writerows((r["without_nets_per_second"],r["with_nets_per_second"]) for r in rows)
    with (folder/"batch-runtime.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,DETAIL_FIELDS);writer.writeheader();writer.writerows(rows)
    lines=[f"# DAG throughput: {name}","","Throughput is in routing nets per second. Row 1 is batch 0 in execution order. Every batch is retained.","",
        "| Without optimizations | With optimizations |","|---:|---:|"]
    lines += [f"| {r['without_nets_per_second']:.9g} | {r['with_nets_per_second']:.9g} |" for r in rows]
    (folder/"throughput.md").write_text("\n".join(lines)+"\n")
    (folder/"checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    plot_throughput(name,rows,folder/"dag-throughput")
    return checks


def write_summary(folder,result):
    (folder/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# DAG throughput: Per-Batch L-shape Throughput","",f"Status: {result['status']}","",
        "Unoptimized: one thread per DAG, full-grid cost refresh and full prefix scans. Optimized: one warp per DAG with incremental costs and BIT prefix maintenance. Both reuse one immutable CPU batch plan, topology buffers and maximum-batch scratch, and start with the same reset demand, routes and timestamps.","",
        "Each batch uses synchronized steady-clock timestamps around cost/prefix refresh, DAG solving/traceback and wire/via demand commits. Throughput = batch routing-net count / elapsed seconds. The first batch includes the first cost/prefix initialization; no initialization samples are removed.","",
        "The two-column throughput.csv is ordered by batch execution index starting at zero. batch-runtime.csv retains net/node counts, common plan fingerprints, timing boundaries, durations and speedups. Every batch and every plotted point is retained. Markers alone are spaced for legibility.","",
        "| Dataset | Batches | Routing nets | Without ms | With ms | Aggregate speedup | Without aggregate nets/s | With aggregate nets/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case in result["cases"]:
        if case["status"]!="passed":continue
        c=case["checks"]
        lines.append(f"| {case['name']} | {c['batches']} | {c['nets']} | {c['without_total_ms']:.3f} | {c['with_total_ms']:.3f} | {c['aggregate_speedup']:.3f}x | {c['without_aggregate_throughput']:.3f} | {c['with_aggregate_throughput']:.3f} |")
    lines += ["","Aggregate throughput = total nets / sum of measured batch seconds. Aggregate speedup uses summed runtimes, not the arithmetic mean of per-batch ratios.","",
        "## Quality","","Each variant writes one actual L-shape-only routing output and runs the external evaluator. All four decimal cost strings and zero open/incomplete nets are required to match. The baseline output is removed before optimized execution; the optimized output is removed after evaluation. No routing outputs or per-batch route snapshots are retained.","",
        "| Dataset | Wirelength | Via | Overflow | Total | Open / incomplete |","|---|---:|---:|---:|---:|---|"]
    for case in result["cases"]:
        if case["status"]!="passed":continue
        m=case["checks"]["metrics"]
        lines.append(f"| {case['name']} | {m['wirelength']} | {m['via']} | {m['overflow']} | {m['total']} | {m['open_nets']} / {m['incomplete_nets']} |")
    lines += ["","## Interpretation and files","",
        "These are L-shape-only evaluator costs, before FLT rerouting. The unoptimized curve uses the original scalar DAG algorithm in the same explicit-memory framework; it is not a separate managed-memory InstantGR2 executable.","",
        "The selected result directory contains throughput.csv/md, batch-runtime.csv, both raw variant CSVs, both evaluator logs, measurement.json, checks.json, worker.log, host-memory.log, gpu-memory.csv and dag-throughput.pdf/svg/png.","",
        "Measurements are single sequential runs with unlocked GPU clocks. Both solver entry points are warmed with inactive launches before timing. Sampling covers the entire worker including external CPU evaluation; GPU memory samples are device-wide every 100 ms. GNU time RSS is a process-tree observation, not the summed family memory footprint."]
    if result.get("plot_generation"):lines += ["","Plots regenerated from saved, validated raw batch timings and evaluator logs; no routing rerun."]
    if result.get("error"):lines += ["","## Error","",result["error"]]
    (folder/"summary.md").write_text("\n".join(lines)+"\n")
