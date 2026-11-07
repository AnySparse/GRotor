#!/usr/bin/env python3
"""Run standalone GGR on one LEF/DEF design."""

import argparse
import os
import sys
from pathlib import Path

XPLACE_ROOT = Path(__file__).resolve().parents[1] / "baseline/GGR"
MODULE_DIR = Path(__file__).resolve().parents[1] / "build/GGR/modules"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lef", required=True)
    parser.add_argument("--def", dest="def_file", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--guide", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rrr-iters", type=int, default=0)
    parser.add_argument("--route-x-size", type=int, default=0)
    parser.add_argument("--route-y-size", type=int, default=0)
    parser.add_argument("--csrn-scale", type=int, default=0)
    parser.add_argument("--module-dir", type=Path, default=MODULE_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.module_dir.resolve()))
    import torch
    import flute_cpp
    import gpugr
    import io_parser

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GGR requires an NVIDIA GPU")

    torch.cuda.set_device(args.gpu)
    flute_cpp.read_lut(
        str(XPLACE_ROOT / "thirdparty/flute/POWV9.dat"),
        str(XPLACE_ROOT / "thirdparty/flute/POST9.dat"),
    )

    params = {
        "benchmark": "ispd24",
        "lef": os.path.abspath(args.lef),
        "def": os.path.abspath(args.def_file),
        "design_name": args.design,
    }
    params.update(
        {
            "verbose_parser_log": True,
            "global_log_level": 2,
            "lite_mode": True,
            "random_place": False,
            "num_threads": args.threads,
        }
    )
    rawdb, gpdb = io_parser.start(params)

    gpugr.load_gr_params(
        {
            "device_id": args.gpu,
            "route_xSize": args.route_x_size,
            "route_ySize": args.route_y_size,
            "rrrIters": args.rrr_iters,
            "csrn_scale": args.csrn_scale,
            "route_guide": os.path.abspath(args.guide),
        }
    )
    grdb = gpugr.create_grdatabase(rawdb, gpdb)
    routeforce = gpugr.create_routeforce(grdb)
    routeforce.run_ggr()
    torch.cuda.synchronize(args.gpu)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"GGR_RUN_ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
