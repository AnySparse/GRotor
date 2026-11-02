#!/usr/bin/env python3
"""Build, route and evaluate GRotor vs InstantGR2; export total, batching and DAG comparisons."""
from pathlib import Path
from benchmark_rtcore_hybrid import main as run_benchmark
from system_reporting import write_summary


def main(argv=None):
    output = None

    def report(folder, result):
        nonlocal output
        write_summary(folder, result)
        output = folder

    status = run_benchmark(argv, report_writer=report, default_output_dir="results/GRotor-system",
        description=__doc__, entry_script=Path(__file__))
    if output:
        print(f"Report: {output / 'summary.md'}", flush=True)
        if status == 0:
            print((output / "summary.md").read_text(), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
