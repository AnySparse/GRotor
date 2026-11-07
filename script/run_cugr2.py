#!/usr/bin/env python3
"""Run native CUGR2 on one LEF/DEF dataset and retain its .guide output."""
import argparse

from _common import (ROOT, add_case_arguments, cli, execute_case, lef_def_inputs,
                     output_folder, require_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser, lef_def=True)
    args = parser.parse_args()
    build = (args.build_dir or ROOT / "build/CUGR2").resolve()
    binary = require_file(build / "route", executable=True)
    lef, def_file = lef_def_inputs(args)
    folder = output_folder(args, "cugr2")
    output = folder / (args.case + ".guide")
    command = [binary, "-lef", lef, "-def", def_file, "-threads", str(args.threads), "-output", output]
    return execute_case(folder, command, output, cwd=ROOT / "baseline/cu-gr-2/flute",
                        metadata=dict(router="CUGR2", lef=str(lef), def_file=str(def_file)))


if __name__ == "__main__":
    raise SystemExit(cli(main))
