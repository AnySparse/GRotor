#!/usr/bin/env python3
"""Figure 11: per-batch prefix query reads plus update writes."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

from test_instantgr2 import ROOT, run_logged, sha256
from test_grotor_reference import local_path, source_manifest, validate_luts
from test_grotor_rtcore import gpu_monitor
from benchmark_rtcore_hybrid import prepare_output_folder
from fig11_reporting import export_case, write_summary, VARIANTS


def measurement_hashes(folder):
    paths=[folder/"measurement.json"]
    paths += [folder/(name+suffix) for name in VARIANTS for suffix in (".csv", ".evaluator.log")]
    return {p.name:sha256(p) for p in paths}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",required=True,help="One dataset name without an extension")
    parser.add_argument("--data-dir",default="../ISPD24_tmp")
    parser.add_argument("--output-dir",help="Result directory (default: results/fig11/<case>)")
    parser.add_argument("--gpu",default="0")
    parser.add_argument("--cuda-arch",default="89")
    parser.add_argument("--nvcc",default=os.environ.get("NVCC","nvcc"))
    parser.add_argument("--jobs",type=int,default=2)
    parser.add_argument("--no-build",action="store_true")
    parser.add_argument("--clean-results",action="store_true")
    parser.add_argument("--plot-only",action="store_true",help="Regenerate plots/tables from validated raw measurements, without GPU work")
    args=parser.parse_args()
    if not args.gpu.isdigit() or not args.cuda_arch.isdigit() or args.jobs<1:
        parser.error("Use numeric GPU/architecture and positive jobs")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",args.case):
        parser.error("Use one simple dataset name without an extension")
    folder=local_path(args.output_dir or f"results/fig11/{args.case}")
    if ROOT/"results" not in folder.parents:parser.error("Output must be below repository results/")
    try:
        import matplotlib
    except ImportError:parser.error("Plotting requires matplotlib: python3 -m pip install matplotlib")
    if args.plot_only:
        if args.clean_results:parser.error("--plot-only cannot delete its input measurements with --clean-results")
        result=json.loads((folder/"summary.json").read_text())
        cases=result.get("cases",[])
        if len(cases)!=1 or cases[0]["name"]!=args.case or cases[0]["status"]!="passed":
            parser.error("Use a completed single-dataset report matching --case")
        case=cases[0]
        if measurement_hashes(folder)!=case["measurement_sha256"]:
            raise ValueError("Saved raw measurements or evaluator logs changed: "+args.case)
        case["checks"]=export_case(args.case,folder)
        result["plot_generation"]=dict(script_sha256=sha256(Path(__file__)),reporting_sha256=sha256(ROOT/"script/fig11_reporting.py"),matplotlib=matplotlib.__version__)
        write_summary(folder,result);print("Report:",folder/"summary.md");return 0
    data=local_path(args.data_dir)
    name=args.case
    for ext in ("cap","net"):
        if not (data/(name+"."+ext)).is_file():parser.error("Missing input: "+str(data/(name+"."+ext)))
    try:prepare_output_folder(folder,args.clean_results)
    except ValueError as error:parser.error(str(error))
    build=ROOT/"build/fig11"/("sm_"+args.cuda_arch)
    sources=source_manifest()
    sources["script/tests/lshape_thread_oracle.cuh"]=sha256(ROOT/"script/tests/lshape_thread_oracle.cuh")
    result=dict(status="running",configuration=dict(stage="lshape",sparsity=1,shared_batch_plan=True,
        profiling="compile_time_prefix_access_only",plot_y_scale="linear",
        dataset=args.case,gpu=args.gpu,cuda_arch=args.cuda_arch,sources=sources,script_sha256=sha256(Path(__file__)),
        reporting_sha256=sha256(ROOT/"script/fig11_reporting.py"),matplotlib=matplotlib.__version__),cases=[])
    try:
        validate_luts()
        if not args.no_build:
            nvcc=shutil.which(args.nvcc)
            if not nvcc:raise RuntimeError("nvcc not found")
            configure=["cmake","-S",str(ROOT),"-B",str(build),"-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_COMPILER="+str(Path(nvcc).resolve()),"-DCMAKE_CUDA_ARCHITECTURES="+args.cuda_arch,
                "-DGROTOR_ENABLE_RTCORE=OFF","-DGROTOR_BUILD_PREFIX_ACCESS_BENCHMARK=ON"]
            commands=[configure,["cmake","--build",str(build),"--target","grotor_prefix_access_benchmark","grotor_evaluator","--parallel",str(args.jobs)]]
            result["build_commands"]=commands
            for command,label in zip(commands,("configure.log","build.log")):
                status,_=run_logged(command,folder/label)
                if status:raise RuntimeError("Fig11 build failed: "+label)
        binary,evaluator=build/"grotor_prefix_access_benchmark",build/"grotor_evaluator"
        result["binaries"]={p.name:sha256(p) for p in (binary,evaluator)}
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu;os.environ["LC_ALL"]="C"
        name=args.case
        case=dict(name=name,status="running");result["cases"].append(case);write_summary(folder,result)
        target=folder
        cap,net=(data/(name+"."+ext) for ext in ("cap","net"))
        case["inputs"]={str(p):sha256(p) for p in (cap,net)}
        with tempfile.TemporaryDirectory(prefix="fig11-evaluation-",dir=build) as temporary:
            output=Path(temporary)/(name+".out")
            command=[str(binary),"--cap",str(cap),"--net",str(net),"--output-dir",str(target),
                "--route-output",str(output),"--evaluator",str(evaluator)]
            case["worker_command"]=command
            case["evaluator_command_per_variant"]=[str(evaluator),str(cap),str(net),str(output)]
            case["exit_status"],case["process_seconds"],case["gpu_memory"]=gpu_monitor(
                ["/usr/bin/time","-v","-o",str(target/"host-memory.log"),*command],target/"worker.log",target/"gpu-memory.csv",args.gpu)
            if case["exit_status"]:raise RuntimeError("Fig11 worker failed: "+str(target/"worker.log"))
            if output.exists():raise RuntimeError("Worker retained a temporary routing output")
        rss=(target/"host-memory.log").read_text()
        case["cpu_peak_mib"]=int(re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",rss)[1])/1024
        case["checks"]=export_case(name,target)
        case["measurement_sha256"]=measurement_hashes(target)
        case["status"]="passed";write_summary(folder,result)
        c=case["checks"]
        print(f"PASS {name}: {c['batches']} matched batches, {c['nets']} nets; aggregate access counts Original/BIT/Pruning {c['original_total']}/{c['bit_total']}/{c['bit_pruning_total']}; evaluator costs identical",flush=True)
        result["status"]="passed";return 0
    except BaseException as error:
        result["status"]="interrupted" if isinstance(error,KeyboardInterrupt) else "failed"
        result["error"]=str(error) or type(error).__name__
        if result["cases"] and result["cases"][-1]["status"]=="running":result["cases"][-1]["status"]=result["status"]
        print("Fig11 failed:",result["error"],file=sys.stderr);return 1
    finally:
        write_summary(folder,result);print("Report:",folder/"summary.md",flush=True)


if __name__=="__main__":raise SystemExit(main())
