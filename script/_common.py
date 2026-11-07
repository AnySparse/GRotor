"""Shared input, logging and evaluation support for single-dataset runs."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "resources/evaluator"


def case_name(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError("Use a dataset name without a path or extension")
    return value


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def add_case_arguments(parser, *, lef_def=False, build=True):
    parser.add_argument("--case", required=True, type=case_name, help="One dataset name, without extension")
    if build:
        parser.add_argument("--build-dir", type=Path, help="Directory containing the compiled router")
    parser.add_argument("--output-dir", type=Path, help="New directory for the solution, logs and summary")
    if lef_def:
        parser.add_argument("--lef", type=Path, required=True, help="Technology LEF file")
        parser.add_argument("--def-dir", type=Path, required=True, help="Directory containing <case>.def files")
        parser.add_argument("--threads", type=positive_int, default=8)
    else:
        parser.add_argument("--data-dir", type=Path, required=True,
                            help="Directory containing matching <case>.cap and <case>.net files")
        parser.add_argument("--evaluator", type=Path, default=EVALUATOR)


def require_file(path, *, executable=False):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"File is not executable: {path}")
    return path


def require_program(value):
    path = shutil.which(value)
    if path is None:
        raise FileNotFoundError(f"Program not found: {value}")
    return path


def routing_inputs(args):
    return [require_file(args.data_dir / (args.case + "." + ext)) for ext in ("cap", "net")]


def lef_def_inputs(args):
    name = "cluster" if args.case == "mempool_cluster" else args.case
    return [require_file(args.lef), require_file(args.def_dir / (name + ".def"))]


def output_folder(args, label):
    return (args.output_dir or ROOT / "results/run" / label / args.case).resolve()


def gpu_environment(gpu=None):
    env = dict(os.environ, LC_ALL="C")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def run_logged(command, log, *, cwd=ROOT, env=None):
    command = list(map(str, command))
    print(shlex.join(command), flush=True)
    started = time.monotonic()
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   start_new_session=True)
        try:
            for line in process.stdout:
                stream.write(line)
                print(line, end="", flush=True)
            status = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            process.stdout.close()
    return status, time.monotonic() - started


def evaluation_metrics(text, status):
    # Accept the upstream evaluator's historical success code as well.
    if status not in (0, 1):
        raise RuntimeError(f"Evaluator exited with status {status}")

    def one(pattern):
        values = re.findall(pattern, text, re.MULTILINE)
        if len(values) != 1:
            raise RuntimeError("Evaluator did not produce a complete, unambiguous score")
        return values[0]

    metrics = {name: one(rf"^{name} cost (\d+\.\d+)$")
               for name in ("wirelength", "via", "overflow", "total")}
    metrics["open_nets"] = int(one(r"^Number of open nets : (\d+)$"))
    metrics["incomplete_nets"] = int(one(r"^Number of incompleted nets : (\d+)$"))
    one(r"^FOR_STAT (.+)$")
    return metrics


def evaluate(inputs, output, evaluator, log):
    command = [evaluator, *inputs, output]
    status, seconds = run_logged(command, log, env=dict(os.environ, LC_ALL="C"))
    return dict(command=list(map(str, command)), exit_status=status, wall_seconds=seconds,
                metrics=evaluation_metrics(log.read_text(), status))


def require_connected(metrics):
    if metrics["open_nets"] or metrics["incomplete_nets"]:
        raise RuntimeError("Evaluator found open or incomplete nets; see evaluation.log")


def execute_case(folder, command, output, *, cwd=ROOT, env=None, inputs=None,
                 evaluator=EVALUATOR, metadata=None):
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(status="running", output=str(output), router_command=list(map(str, command)),
                  working_directory=str(cwd), **(metadata or {}))

    def save():
        (folder / "summary.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    try:
        status, seconds = run_logged(command, folder / "routing.log", cwd=cwd, env=env)
        record.update(router_exit_status=status, router_wall_seconds=seconds)
        if status:
            raise RuntimeError(f"Router exited with status {status}; see {folder / 'routing.log'}")
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("Router did not write a nonempty solution")
        if inputs is not None:
            record["evaluation"] = evaluate(inputs, output, evaluator, folder / "evaluation.log")
            require_connected(record["evaluation"]["metrics"])
        record["status"] = "passed"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = str(error) or type(error).__name__
        raise
    finally:
        save()
        print(f"Results: {folder}", flush=True)
    return 0


def cli(main):
    try:
        return main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
