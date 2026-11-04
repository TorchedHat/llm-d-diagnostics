"""Optional, source-checked vLLM 0.31.0 instrumentation for NIXL rank timings."""

import argparse
import dataclasses
import hashlib
import http.server
import importlib.util
import json
import os
import pathlib
import time
import urllib.parse

ROOT = pathlib.Path(os.environ.get("NIXL_TRACE_DIR", "/nixl-trace"))
CLOCK_DOMAIN = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
STARTED: dict[int, int] = {}
FILES: dict[tuple[int, int], int] = {}
LAYOUTS: set[tuple[int, int]] = set()
PATCHES = {
    "base_worker.py": (
        "348be8df3f089e774ecbc7289c40662b5b6930c8a39399a2c81d238e34acab30",
        "                        res = self.nixl_wrapper.get_xfer_telemetry(handle)\n",
        1,
    ),
    "pull_worker.py": (
        "276d3643a4ed6df1ede83c48281caf580720140173f5193930f8bd004ad24725",
        "self.nixl_wrapper.transfer(handle)",
        2,
    ),
}


def start(handle):
    key = int(handle._handle)
    if key in STARTED:
        raise RuntimeError("NIXL trace handle reused before completion")
    STARTED[key] = time.monotonic_ns()


def finish(rank, request_id, handle, telemetry, worker=None):
    observed = time.monotonic_ns()
    started = STARTED.pop(int(handle._handle))
    row = {
        "rank": rank,
        "request_id": request_id,
        "clock_domain": CLOCK_DOMAIN,
        "posted_monotonic_ns": started,
        "observed_done_monotonic_ns": observed,
        "nixl_start_us": telemetry.startTime,
        "nixl_xfer_us": telemetry.xferDuration,
        "nixl_post_us": telemetry.postDuration,
        "bytes": telemetry.totalBytes,
        "descriptors": telemetry.descCount,
    }
    key = os.getpid(), rank
    if worker is not None and key not in LAYOUTS:
        layout = {
            "rank": rank,
            "block_size": worker.block_size,
            "physical_blocks_per_logical": worker._physical_blocks_per_logical_kv_block,
            "has_mamba": worker._has_mamba,
            "mamba_state_sizes": worker._mamba_ssm_size,
            "layer_specs": {
                name: {"type": type(spec).__name__, **dataclasses.asdict(spec)}
                for name, spec in worker._layer_specs.items()
            },
        }
        (ROOT / f"layout-rank-{rank}-{os.getpid()}.json").write_text(
            json.dumps(layout, default=str) + "\n"
        )
        LAYOUTS.add(key)
    if key not in FILES:
        FILES[key] = os.open(
            ROOT / f"rank-{rank}-{key[0]}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644
        )
    data = (json.dumps(row, allow_nan=False) + "\n").encode()
    if os.write(FILES[key], data) != len(data):
        raise RuntimeError("incomplete NIXL rank trace write")


def patched_sources(root):
    """Check and compile every source before modifying any installed file."""
    results = []
    for name, (expected, needle, count) in PATCHES.items():
        path = root / name
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError(
                f"unsupported vLLM source: {path}; review instrumentation for this version"
            )
        source = original.decode()
        if source.count(needle) != count:
            raise ValueError("unexpected instrumentation site count: " + name)
        if name == "base_worker.py":
            updated = source.replace(
                needle,
                needle
                + "                        __import__('nixl_trace').finish(self.tp_rank, req_id, handle, res, self)\n",
            )
        else:
            lines = []
            for line in source.splitlines(keepends=True):
                if needle in line:
                    indent = line[: len(line) - len(line.lstrip())]
                    line = line.replace(
                        needle, "__import__('nixl_trace').start(handle)\n" + indent + needle
                    )
                lines.append(line)
            updated = "".join(lines)
        compile(updated, str(path), "exec")
        results.append((path, original, updated.encode()))
    return results


def instrument(command):
    if not command:
        raise ValueError("instrument requires a server command after --")
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("vLLM package not found")
    root = (
        pathlib.Path(next(iter(spec.submodule_search_locations)))
        / "distributed/kv_transfer/kv_connector/v1/nixl"
    )
    sources = patched_sources(root)
    ROOT.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        for path, original, updated in sources:
            temporary = path.with_name(path.name + ".nixl-trace-tmp")
            temporary.write_bytes(updated)
            temporary.chmod(path.stat().st_mode)
            temporary.replace(path)
            written.append((path, original))
            print(
                json.dumps(
                    {
                        "instrumented": path.name,
                        "original_sha256": hashlib.sha256(original).hexdigest(),
                        "updated_sha256": hashlib.sha256(updated).hexdigest(),
                    }
                ),
                flush=True,
            )
    except OSError:
        for path, original in written:
            path.write_bytes(original)
        raise
    module_dir = str(pathlib.Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = module_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.execvp(command[0], command)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        query = urllib.parse.urlparse(self.path)
        if query.path != "/snapshot":
            self.send_error(404)
            return
        try:
            offsets = urllib.parse.parse_qs(query.query)
            rows, counts = [], {}
            for path in sorted(ROOT.glob("rank-*.jsonl")):
                # A read may overlap an append; expose only complete records.
                data = path.read_bytes()
                lines = data[: data.rfind(b"\n") + 1].splitlines()
                counts[path.name] = len(lines)
                offset = int(offsets.get(path.name, ["0"])[0])
                if offset < 0 or offset > len(lines):
                    raise ValueError("trace reset or invalid offset")
                rows.extend(json.loads(line) for line in lines[offset:])
            if set(offsets) - set(counts):
                raise ValueError("trace file disappeared")
            payload = json.dumps({"counts": counts, "rows": rows}, allow_nan=False).encode()
        except (ValueError, OSError) as exc:
            self.send_error(409, str(exc))
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    sub = commands.add_parser("instrument")
    sub.add_argument("command", nargs=argparse.REMAINDER)
    sub = commands.add_parser("serve")
    sub.add_argument("--host", default="127.0.0.1")
    sub.add_argument("--port", type=int, default=9109)
    args = parser.parse_args()
    if args.mode == "instrument":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        try:
            instrument(command)
        except (ValueError, OSError) as exc:
            parser.exit(1, f"nixl_trace: {exc}\n")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        http.server.ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
