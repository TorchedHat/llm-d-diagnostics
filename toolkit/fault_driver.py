#!/usr/bin/env python3
"""
Fault Driver — in-pod load generator and probe for fault tolerance experiments.

Runs INSIDE the test-client pod. Provides two modes:

1. **probe** — Send requests at a fixed interval, record success/failure
   with millisecond-precision timing. Used by exp4 to measure detection
   and recovery windows.

2. **load** — Sustained QPS with Poisson arrivals and streaming ITL
   measurement. Used by exp4k/4l for load-bearing fault tests.

The key advantage over `oc exec` per request: this runs in-cluster with
real timing. No subprocess overhead per probe. Resolution is limited only
by request latency, not by the oc API round-trip.

Usage (from exp4_fault.py via oc exec):
    # Probe mode: send requests every 200ms, output JSONL
    oc exec test-client -- python3 fault_driver.py probe \\
        --url http://vllm-decode-svc:8000/v1/completions \\
        --prefill-host vllm-prefill-svc.ns.svc.cluster.local:8000 \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --interval 0.2 --duration 120

    # Load mode: sustained 4 QPS with streaming, output CSV
    oc exec test-client -- python3 fault_driver.py load \\
        --url http://vllm-decode-svc:8000/v1/completions \\
        --prefill-host vllm-prefill-svc.ns.svc.cluster.local:8000 \\
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --qps 4 --duration 60 --output /scripts/toolkit/data/load-results.csv

Output:
    probe mode: JSONL to stdout, one line per probe
    load mode:  CSV to --output file + summary to stdout
"""

import csv
import http.client
import json
import os
import random
import signal
import ssl
import threading
import time
from urllib.parse import urlparse


def http_error(status, raw):
    if status < 400:
        return ""
    detail = raw.decode("utf-8", errors="replace")[:500]
    return f"HTTP {status}: {detail}".rstrip(": ")


def send_request(host, port, path, payload, headers, use_tls, timeout=10):
    """Send a single non-streaming request. Returns (ttft_ms, total_ms, status, error)."""
    try:
        start = time.monotonic()
        if use_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=timeout)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", path, body=payload, headers=headers)
        resp = conn.getresponse()
        ttft = time.monotonic() - start
        raw = resp.read()
        total = time.monotonic() - start
        status = resp.status
        conn.close()
        return round(ttft * 1000, 1), round(total * 1000, 1), status, http_error(status, raw)
    except Exception as e:
        elapsed = time.monotonic() - start
        return 0, round(elapsed * 1000, 1), 0, str(e)


def send_streaming(host, port, path, payload, headers, use_tls, timeout=30):
    """Send a streaming request.

    Returns (ttft_ms, total_ms, itl_mean_ms, itl_p99_ms, token_count,
             status, error, received_text).

    On exception, returns partial results if any tokens were received
    before the connection died. This is critical for detecting mid-transfer
    corruption: tokens > 0 with status == 0 means data arrived before failure.
    """
    token_times = []
    received_text = []
    try:
        start = time.monotonic()
        if use_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=timeout)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("POST", path, body=payload, headers=headers)
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            raw = resp.read()
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            conn.close()
            return (elapsed_ms, elapsed_ms, 0, 0, 0, status,
                    http_error(status, raw), "")
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        token_times.append(time.monotonic() - start)
                        received_text.append(choices[0]["text"])
                except (json.JSONDecodeError, ValueError):
                    pass
        total = time.monotonic() - start
        conn.close()

        ttft = token_times[0] * 1000 if token_times else total * 1000
        itl_mean, itl_p99 = compute_itl(token_times)
        return (round(ttft, 2), round(total * 1000, 2), itl_mean, itl_p99,
                len(token_times), status, "", "".join(received_text))
    except Exception as e:
        elapsed = time.monotonic() - start
        ttft = token_times[0] * 1000 if token_times else 0
        itl_mean, itl_p99 = compute_itl(token_times) if len(token_times) >= 2 else (0, 0)
        return (round(ttft, 2), round(elapsed * 1000, 2),
                itl_mean, itl_p99, len(token_times), 0, str(e),
                "".join(received_text))


def compute_itl(token_times):
    """Compute inter-token latency statistics."""
    if len(token_times) < 2:
        return 0.0, 0.0
    gaps = [(token_times[i + 1] - token_times[i]) * 1000
            for i in range(len(token_times) - 1)]
    gaps.sort()
    n = len(gaps)
    mean = sum(gaps) / n
    idx = (n - 1) * 0.99
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    frac = idx - lo
    p99 = gaps[lo] + frac * (gaps[hi] - gaps[lo])
    return round(mean, 2), round(p99, 2)


def parse_url(url):
    """Parse URL into (host, port, path, use_tls)."""
    p = urlparse(url)
    return p.hostname, p.port, p.path or "/", p.scheme == "https"


def request_headers(args):
    """Build headers for the selected manual or EPP route."""
    headers = {"Content-Type": "application/json"}
    if args.route_mode == "manual":
        if not args.prefill_host:
            raise ValueError("--prefill-host is required with --route-mode manual")
        headers["x-prefiller-host-port"] = args.prefill_host
    return headers


def run_probe(args):
    """Probe mode: send requests at fixed intervals, output JSONL to stdout."""
    host, port, path, use_tls = parse_url(args.url)
    payload = json.dumps({
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
    })
    headers = request_headers(args)

    stop = threading.Event()

    def handle_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    deadline = time.time() + args.duration if args.duration > 0 else float("inf")
    seq = 0

    while not stop.is_set() and time.time() < deadline:
        seq += 1
        epoch_ms = int(time.time() * 1000)
        ttft, total, status, error = send_request(
            host, port, path, payload, headers, use_tls, timeout=args.timeout)

        line = json.dumps({
            "seq": seq,
            "epoch_ms": epoch_ms,
            "ttft_ms": ttft,
            "total_ms": total,
            "status": status,
            "error": error,
        })
        print(line, flush=True)

        # Sleep for the remainder of the interval (subtract request time)
        elapsed = total / 1000.0
        sleep_time = max(0, args.interval - elapsed)
        if sleep_time > 0:
            stop.wait(sleep_time)


def run_load(args):
    """Load mode: sustained QPS with Poisson arrivals and streaming measurement."""
    host, port, path, use_tls = parse_url(args.url)
    headers = request_headers(args)

    stop = threading.Event()
    results = []
    results_lock = threading.Lock()

    def handle_signal(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    start_time = time.time()
    seq_counter = [0]
    seq_lock = threading.Lock()

    # Centralized Poisson arrival scheduler: one thread generates arrival
    # times at the target QPS, worker threads pick up and execute requests.
    # This ensures aggregate arrival rate = args.qps regardless of worker count.
    work_queue = []
    work_queue_lock = threading.Lock()
    work_ready = threading.Event()

    def scheduler():
        """Generate Poisson arrivals at the target QPS."""
        while not stop.is_set():
            sleep_time = random.expovariate(args.qps)
            if stop.wait(sleep_time):
                break
            with work_queue_lock:
                work_queue.append(True)
            work_ready.set()

    def worker():
        while not stop.is_set():
            # Wait for work from the scheduler
            work_ready.wait(timeout=0.5)
            got_work = False
            with work_queue_lock:
                if work_queue:
                    work_queue.pop()
                    got_work = True
                if not work_queue:
                    work_ready.clear()
            if not got_work:
                continue

            with seq_lock:
                seq_counter[0] += 1
                seq = seq_counter[0]

            epoch_ms = int(time.time() * 1000)
            t_since_start = time.time() - start_time

            payload = json.dumps({
                "model": args.model,
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "stream": True,
            })

            ttft, total, itl_mean, itl_p99, tokens, status, error, text = send_streaming(
                host, port, path, payload, headers, use_tls, timeout=args.timeout)

            row = {
                "seq": seq,
                "epoch_ms": epoch_ms,
                "elapsed_s": round(t_since_start, 2),
                "ttft_ms": ttft,
                "total_ms": total,
                "itl_mean_ms": itl_mean,
                "itl_p99_ms": itl_p99,
                "token_count": tokens,
                "status": status,
                "error": error,
                "received_text": text,
            }
            with results_lock:
                results.append(row)

    # Start scheduler thread + worker pool
    sched_thread = threading.Thread(target=scheduler, daemon=True)
    sched_thread.start()

    n_workers = min(max(int(args.qps * 4), 4), 32)
    threads = []
    for _ in range(n_workers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    # Wait for duration
    deadline = time.time() + args.duration
    while time.time() < deadline and not stop.is_set():
        stop.wait(1)
    stop.set()

    sched_thread.join(timeout=5)
    for t in threads:
        t.join(timeout=5)

    # Write results
    fields = ["seq", "epoch_ms", "elapsed_s", "ttft_ms", "total_ms",
              "itl_mean_ms", "itl_p99_ms", "token_count", "status", "error",
              "received_text"]

    output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(results, key=lambda r: r["seq"]):
            writer.writerow(row)

    # Summary to stdout
    ok = sum(1 for r in results if r["status"] == 200)
    total_reqs = len(results)
    summary = {
        "total_requests": total_reqs,
        "ok": ok,
        "failed": total_reqs - ok,
        "duration_s": round(time.time() - start_time, 1),
        "output": output_path,
    }
    print(json.dumps(summary), flush=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="In-pod fault driver")
    parser.add_argument("--url", required=True)
    parser.add_argument("--route-mode", choices=("manual", "epp"), default="manual")
    parser.add_argument("--prefill-host")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=10)

    sub = parser.add_subparsers(dest="mode", required=True)

    probe_p = sub.add_parser("probe")
    probe_p.add_argument("--interval", type=float, default=0.2)
    probe_p.add_argument("--duration", type=float, default=120)

    load_p = sub.add_parser("load")
    load_p.add_argument("--qps", type=float, default=4)
    load_p.add_argument("--duration", type=float, default=60)
    load_p.add_argument("--output", default="data/load-results.csv")

    args = parser.parse_args()
    if args.route_mode == "manual" and not args.prefill_host:
        parser.error("--prefill-host is required with --route-mode manual")

    if args.mode == "probe":
        run_probe(args)
    elif args.mode == "load":
        run_load(args)


if __name__ == "__main__":
    main()
