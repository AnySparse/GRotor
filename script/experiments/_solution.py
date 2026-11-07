"""Exact per-net segment-multiset validation for AMD experiments."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
SEGMENT = re.compile(rb"-?[0-9]+(?: -?[0-9]+){5}")

def open_solution(path):
    path = Path(path)
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")

def files_equal(left, right):
    """Compare complete decompressed contents, not just hashes."""
    with open_solution(left) as a, open_solution(right) as b:
        while True:
            x, y = a.read(1024**2), b.read(1024**2)
            if x != y:
                return False
            if not x:
                return True

def read_nets(path):
    with open_solution(path) as source:
        while True:
            name = source.readline()
            if not name:
                return
            name = name.strip()
            if not name:
                continue
            if source.readline().strip() != b"(":
                raise ValueError(f"Missing opening parenthesis after {name!r}")
            segments = []
            for line in source:
                if line.strip() == b")":
                    break
                segment = b" ".join(line.split())
                if not SEGMENT.fullmatch(segment):
                    raise ValueError(f"Invalid segment in {name!r}")
                segments.append(segment)
            else:
                raise ValueError(f"Unclosed net {name!r}")
            yield name, segments

def canonicalize(source, destination, scratch):
    """Sort exact per-net records using bounded RAM and compressed disk scratch."""
    destination, scratch = Path(destination), Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    nets = segments = 0
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="normalize-", dir=scratch) as temporary:
        temporary = Path(temporary)
        unsorted = temporary / "records.gz"
        with gzip.open(unsorted, "wb", compresslevel=1) as output:
            for name, routes in read_nets(source):
                record = [name.decode("utf-8", errors="surrogateescape"),
                          [r.decode("ascii") for r in sorted(routes)]]
                output.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n")
                nets += 1
                segments += len(routes)
        if not nets:
            raise ValueError("Empty routing solution")
        # GNU sort passes -d when decompressing its temporary runs.
        compressor = temporary / "compress.sh"
        compressor.write_text('#!/bin/sh\nexec gzip -1 "$@"\n')
        compressor.chmod(0o700)
        unpack = subprocess.Popen(["gzip", "-dc", str(unsorted)], stdout=subprocess.PIPE)
        sorter = subprocess.Popen(
            ["sort", "-S", "128M", "--parallel=1", "-T", str(temporary),
             f"--compress-program={compressor}"], stdin=unpack.stdout,
            stdout=subprocess.PIPE, env=dict(os.environ, LC_ALL="C"))
        unpack.stdout.close()
        try:
            with destination.open("xb") as file:
                with gzip.GzipFile(filename="", mode="wb", compresslevel=1,
                                   fileobj=file, mtime=0) as output:
                    for block in iter(lambda: sorter.stdout.read(1024**2), b""):
                        digest.update(block)
                        output.write(block)
            sort_status, unpack_status = sorter.wait(), unpack.wait()
            if sort_status or unpack_status:
                raise RuntimeError(f"Canonical sort failed: sort={sort_status}, gzip={unpack_status}")
        finally:
            sorter.stdout.close()
            for process in (sorter, unpack):
                if process.poll() is None:
                    process.terminate()
                process.wait()
    return {"nets": nets, "segments": segments, "normalized_sha256": digest.hexdigest()}
