#!/usr/bin/env python3
"""Run one InstantGR2 dataset and evaluate its retained .out solution."""
import argparse

from _common import (ROOT, add_case_arguments, cli, execute_case, gpu_environment,
                     output_folder, positive_int, require_file, routing_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser)
    parser.add_argument("--cuda-arch", type=positive_int, default=89, help="Architecture used for the default build directory")
    parser.add_argument("--batching", choices=("cpu", "gpu", "dual"), default="cpu")
    parser.add_argument("--gpu", type=int, help="Physical GPU index; otherwise preserve CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    build = (args.build_dir or ROOT / "build/InstantGR2" / f"sm_{args.cuda_arch}").resolve()
    binary = require_file(build / "instantgr2_reference", executable=True)
    evaluator = require_file(args.evaluator, executable=True)
    inputs = routing_inputs(args)
    folder = output_folder(args, "instantgr2")
    output = folder / (args.case + ".out")
    command = [binary, "-cap", inputs[0], "-net", inputs[1], "-out", output,
               "--batching", args.batching, "--sparsity", "1"]
    # The baseline resolves its FLUTE tables relative to the working directory.
    return execute_case(folder, command, output, cwd=ROOT / "resources/flute", inputs=inputs,
                        evaluator=evaluator, env=gpu_environment(args.gpu),
                        metadata=dict(router="InstantGR2", batching=args.batching))


if __name__ == "__main__":
    raise SystemExit(cli(main))
