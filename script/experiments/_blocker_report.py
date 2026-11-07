"""Validate and plot the paired, synchronous blocker-ablation ray counts."""
import csv
import json
from pathlib import Path

FIELDS = ["without_blocker", "with_blocker"]


def read_counts(folder):
    checks = json.loads((folder / "checks.json").read_text())
    required = ("same_iteration_count", "exact_assignment_rounds", "exact_batch_ids", "exact_cpu_batch_lists")
    if checks.get("status") != "passed" or any(checks.get(k) is not True for k in required):
        raise ValueError("Blocker requires identical per-net assignment histories and CPU batch lists")
    if checks.get("publication") != "round_end" or checks.get("eligibility") != "frozen" or checks.get("counter") != "actual_optixTrace_calls" or checks.get("conflict_cache") is not False:
        raise ValueError("Blocker requires uncached synchronous tracing with frozen eligibility")
    with (folder / "ray-counts.csv").open() as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Ray table must have exactly two count columns")
        rows = []
        for row in reader:
            try:
                if set(row) != set(FIELDS) or any(not row[k].isdigit() for k in FIELDS):
                    raise ValueError("Counts must be nonnegative integers")
                without, with_blocker = (int(row[k]) for k in FIELDS)
            except (TypeError, AttributeError) as error:
                raise ValueError("Malformed Blocker count row") from error
            if not 0 < with_blocker <= without <= checks["input_rays"]:
                raise ValueError("Invalid actual ray count for a synchronous round")
            rows.append((without, with_blocker))
    if len(rows) != checks["iterations"] or not rows:
        raise ValueError("Count table must contain every iteration from both runs")
    if rows[0] != (checks["input_rays"], checks["input_rays"]):
        raise ValueError("Both variants must launch all input rays in round zero")
    if any(rows[i][0] > rows[i-1][0] for i in range(1,len(rows))):
        raise ValueError("Unassigned-net ray count cannot increase")
    if sum(r[0] for r in rows) != checks["without_blocker_total"] or sum(r[1] for r in rows) != checks["with_blocker_total"]:
        raise ValueError("Trace count totals differ from the worker checks")
    with (folder / "round-stats.csv").open() as stream:
        stats = list(csv.DictReader(stream))
    if len(stats) != len(rows):
        raise ValueError("Missing per-round assignment statistics")
    remaining = checks["nets"]
    for i,(record,counts) in enumerate(zip(stats,rows)):
        before,after,assigned = (int(record[k]) for k in ("unassigned_before","unassigned_after","assigned_this_round"))
        if int(record["iteration"]) != i or tuple(int(record[k]) for k in FIELDS) != counts or before != remaining or not 0 <= after < before or assigned != before-after:
            raise ValueError("Invalid synchronous assignment history")
        remaining = after
    if remaining:
        raise ValueError("Ablation ended before all nets were assigned")
    return rows, checks


def plot_counts(name,rows,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    with plt.rc_context({"font.size":10,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"}):
        fig,axis=plt.subplots(figsize=(4.8,3.6))
        x=list(range(len(rows)));markers=max(1,len(rows)//30)
        axis.plot(x,[r[0] for r in rows],color="black",marker="s",markerfacecolor="white",markersize=3,
            markevery=markers,linewidth=1.15,label="Without blocker")
        axis.plot(x,[r[1] for r in rows],color="#1754df",marker="o",markerfacecolor="white",markersize=3,
            markevery=markers,linewidth=1.15,label="With blocker")
        axis.set(title=name,xlabel="Iteration",ylabel="Rays launched")
        axis.set_xlim((0,len(rows)-1) if len(rows)>1 else (-.5,.5));axis.set_ylim(bottom=0)
        formatter=ScalarFormatter(useMathText=True);formatter.set_powerlimits((-3,4));axis.yaxis.set_major_formatter(formatter)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
        axis.grid(axis="y",linestyle="--",linewidth=.6,alpha=.5)
        handles,labels=axis.get_legend_handles_labels()
        fig.legend(handles,labels,loc="upper center",ncol=2,frameon=False,fontsize=9)
        fig.tight_layout(rect=(0,0,1,.91))
        for suffix in ("pdf","svg","png"):
            fig.savefig(Path(str(output)+"."+suffix),dpi=300,facecolor="white",bbox_inches="tight")
        plt.close(fig)


def export_case(name,folder):
    rows,checks=read_counts(folder)
    lines=[f"# Blocker: {name}","","Two columns of actual optixTrace counts. Row 1 is iteration 0; no iterations are padded or sampled.","",
        "| Without blocker | With blocker |","|---:|---:|"]
    lines += [f"| {a} | {b} |" for a,b in rows]
    (folder/"ray-counts.md").write_text("\n".join(lines)+"\n")
    plot_counts(name,rows,folder/"blocker")
    return checks


def write_summary(folder,result):
    (folder/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Blocker: Blocker Ablation","",f"Status: {result['status']}","",
        "L-shape only, sparsity 1, uncached RT-Core conflict queries. Both variants use the same scene, priority order and synchronous round-end publication. Eligibility is frozen for the whole launch; readiness changes within a launch cannot suppress another eligible ray.","",
        "Counts are incremented immediately before actual optixTrace calls and checked against the frozen launch width. Every net must receive the same batch ID in the same iteration in both variants. Final complete batch lists must equal CPU batching. No iteration padding or conflict-graph caching is used.","",
        "This is a ray-count microbenchmark. Instrumented execution times are not performance results. It stops after L-shape batching and creates no routing .out files or evaluator scores.","",
        "| Dataset | Nets | Input rays | Iterations (both) | Without blocker total | With blocker total | Total ray reduction |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    completed=[]
    for case in result["cases"]:
        if case["status"]!="passed":continue
        c=case["checks"];name=case["name"]
        lines.append(f"| {name} | {c['nets']} | {c['input_rays']} | {c['iterations']} | {c['without_blocker_total']} | {c['with_blocker_total']} | {c['without_blocker_total']/c['with_blocker_total']:.3f}x |")
        completed.append(name)
    lines += ["","## Files and validation","",
        "The selected result directory contains ray-counts.csv (exactly two numeric columns, rows ordered from iteration 0), ray-counts.md (the same table), round-stats.csv, checks.json, worker.log, host-memory.log, gpu-memory.csv and blocker.pdf/svg/png. The last three are exportable publication plots; all iterations are plotted.","",
        "The two variants run independently and sequentially with reset color, blocker and history arrays, reusing one immutable scene and one workspace. The with-blocker run is performed first. Coloring results are committed only by a separate kernel after all trace and assignment work finishes. The control independently recomputes every iteration.","",
        "Raw input, executable, PTX and source hashes, exact build/run commands, and sampled memory data are retained in summary.json. The default application batching path and its performance timers are not used by this ablation."]
    if result.get("plot_generation"):lines += ["","Plots regenerated from saved, validated ray counts; no GPU work rerun."]
    if result.get("error"):lines += ["","## Error","",result["error"]]
    (folder/"summary.md").write_text("\n".join(lines)+"\n")
