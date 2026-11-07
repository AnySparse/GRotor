#!/usr/bin/env python3
"""Evaluate an existing CAP/NET routing solution without running a router."""
import argparse
import json

from _common import (add_case_arguments, cli, evaluate, output_folder, require_connected,
                     require_file, routing_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser, build=False)
    parser.add_argument("--out", required=True, help="Routing solution to evaluate")
    args = parser.parse_args()
    inputs = routing_inputs(args)
    solution = require_file(args.out)
    evaluator = require_file(args.evaluator, executable=True)
    folder = output_folder(args, "evaluate")
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", output=str(solution))
    try:
        record["evaluation"] = evaluate(inputs, solution, evaluator, folder / "evaluation.log")
        require_connected(record["evaluation"]["metrics"])
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error=str(error) or type(error).__name__)
        raise
    finally:
        (folder / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Results: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli(main))
