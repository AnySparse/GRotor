#!/usr/bin/env python3
"""Run native GGR Z-shape routing on one LEF/DEF dataset; retain its .guide."""
import argparse
import sys

from _common import (ROOT, add_case_arguments, cli, execute_case, gpu_environment,
                     lef_def_inputs, output_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser, lef_def=True)
    parser.add_argument("--gpu", type=int, help="Physical GPU index; otherwise preserve CUDA_VISIBLE_DEVICES")
    parser.add_argument("--rrr-iters", type=int, default=0, help="Maze reroute iterations; 0 selects Z-shape only")
    args = parser.parse_args()
    if args.rrr_iters < 0 or args.gpu is not None and args.gpu < 0:
        parser.error("--gpu and --rrr-iters must be nonnegative")
    build = (args.build_dir or ROOT / "build/GGR").resolve()
    modules = build / "modules"
    for name in ("flute_cpp", "gpugr", "io_parser"):
        if not any(modules.glob(name + "*.so")):
            parser.error(f"Missing {name} extension in {modules}; build GGR with this Python environment")
    lef, def_file = lef_def_inputs(args)
    folder = output_folder(args, "ggr")
    output = folder / (args.case + ".guide")
    command = [sys.executable, ROOT / "script/_ggr.py", "--module-dir", modules, "--lef", lef,
               "--def", def_file, "--design", args.case, "--guide", output,
               "--threads", str(args.threads), "--rrr-iters", str(args.rrr_iters)]
    return execute_case(folder, command, output, env=gpu_environment(args.gpu),
                        metadata=dict(router="GGR", lef=str(lef), def_file=str(def_file)))


if __name__ == "__main__":
    raise SystemExit(cli(main))
