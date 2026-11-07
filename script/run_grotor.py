#!/usr/bin/env python3
"""Run one GRotor dataset and evaluate its retained .out solution."""
import argparse
import os

from _common import (ROOT, add_case_arguments, cli, execute_case, gpu_environment,
                     output_folder, positive_int, require_file, require_program, routing_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser)
    parser.add_argument("--variant", choices=("rtx", "cuda", "multi", "hip", "hiprt"), default="rtx")
    parser.add_argument("--cuda-arch", type=positive_int, default=89,
                        help="Select the default NVIDIA build directory (default: 89)")
    parser.add_argument("--batching", choices=("cpu", "rtcore", "hiprt"), help="Override the variant's default")
    parser.add_argument("--validate-batches", action="store_true", help="Compare ray-tracing batches with CPU")
    parser.add_argument("--gpu", type=int, help="Select one physical NVIDIA GPU; otherwise preserve CUDA_VISIBLE_DEVICES")
    parser.add_argument("--gpus", type=positive_int, default=2, help="MPI rank/GPU count for --variant multi")
    parser.add_argument("--devices", help="Visible NVIDIA devices for --variant multi, e.g. 0,1,2,3")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--mpi-arg", action="append", default=[], help="Repeat as --mpi-arg=--launcher-option")
    parser.add_argument("--symmetric-size", help="NVSHMEM heap per GPU; preserves the environment or defaults to 25G")
    parser.add_argument("--stop-after", choices=("lshape", "routing"), default="routing")
    args = parser.parse_args()
    batching = args.batching or dict(rtx="rtcore", cuda="cpu", multi="cpu", hip="cpu", hiprt="hiprt")[args.variant]
    allowed = dict(rtx=("cpu", "rtcore"), cuda=("cpu",), multi=("cpu",), hip=("cpu",), hiprt=("cpu", "hiprt"))
    if batching not in allowed[args.variant]:
        parser.error("Batching backend does not match the selected variant")
    if args.validate_batches and batching == "cpu":
        parser.error("--validate-batches requires rtcore or hiprt batching")
    if args.gpu is not None and (args.gpu < 0 or args.variant in ("multi", "hip", "hiprt")):
        parser.error("--gpu selects a nonnegative NVIDIA single-GPU index; multi uses --devices")
    if args.variant != "multi" and (args.devices or args.symmetric_size or args.mpi_arg):
        parser.error("--devices, --symmetric-size and --mpi-arg require --variant multi")
    if args.variant == "multi" and args.gpus > 8:
        parser.error("The multi-GPU executor supports at most 8 GPUs")
    builds = dict(rtx=f"GRotor/sm_{args.cuda_arch}", cuda=f"GRotor/sm_{args.cuda_arch}",
                  multi=f"GRotor-multi/sm_{args.cuda_arch}", hip="hip-cpu", hiprt="hip")
    build = (args.build_dir or ROOT / "build" / builds[args.variant]).resolve()
    binary = require_file(build / ("GRotor_multi" if args.variant == "multi" else "GRotor"), executable=True)
    evaluator = require_file(args.evaluator, executable=True)
    inputs = routing_inputs(args)
    folder = output_folder(args, "grotor-" + args.variant)
    output = folder / (args.case + ".out")
    command = [binary, "-cap", inputs[0], "-net", inputs[1], "-out", output,
               "--batching", batching, "--sparsity", "1", "--validate-batches", str(int(args.validate_batches)),
               "--stop-after", args.stop_after]
    env = gpu_environment(args.gpu)
    if args.variant == "multi":
        if args.devices:
            devices = args.devices.split(",")
            if any(not d.strip() or d != d.strip() for d in devices) or len(devices) != len(set(devices)) or len(devices) < args.gpus:
                parser.error("--devices must list distinct devices and cover all requested GPU ranks")
            env["CUDA_VISIBLE_DEVICES"] = args.devices
        env["NVSHMEM_SYMMETRIC_SIZE"] = args.symmetric_size or os.environ.get("NVSHMEM_SYMMETRIC_SIZE", "25G")
        env["NVSHMEM_REMOTE_TRANSPORT"] = "none"
        command = [require_program(args.mpirun), *args.mpi_arg, "-np", str(args.gpus), *command]
    return execute_case(folder, command, output, inputs=inputs, evaluator=evaluator, env=env,
                        metadata=dict(router="GRotor", variant=args.variant, batching=batching,
                                      stop_after=args.stop_after))


if __name__ == "__main__":
    raise SystemExit(cli(main))
