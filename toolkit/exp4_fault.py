#!/usr/bin/env python3
"""
Experiment 4: Fault Tolerance

Runs from OUTSIDE the cluster. Uses `oc exec` to send requests via test-client
and `oc` commands to kill/scale pods.

Sub-experiments:
    4a. Decode pod failure (consumer dies)
    4b. Prefill pod failure (producer dies)
    4d. Graceful degradation (3 GPU -> 2 GPU -> 3 GPU)
    4e. Failure under load (prefill dies while requests in flight)
    4f. Network partition (block NIXL between prefill and decode)
    4g. Slow network (tc netem latency + packet loss on prefill egress)
    4h. UCX keepalive characterization (kill-to-detection timing)
    4i. Mid-transfer failure (prefill dies during active KV transfer)
    4j. Container restart vs pod restart (same IP vs new IP)
    4k. Concurrent load during failure (sustained QPS with kill)
    4l. Rolling update under load (zero-downtime validation)
    4m. Progressive degradation (packet loss sweep 0-100%, gray failure)

Usage:
    python3 toolkit/exp4_fault.py                    # all experiments
    python3 toolkit/exp4_fault.py 4a 4h              # just 4a and 4h
    python3 toolkit/exp4_fault.py 4h --skip-control  # just 4h, skip control
    python3 toolkit/exp4_fault.py --stop-on-failure   # abort on first crash
    NS=my-ns python3 toolkit/exp4_fault.py 4k 4l     # custom namespace

Additional env vars:
    TEST_CLIENT       Pod name for test client (default: test-client)
    PREFILL_DEPLOY    Prefill deployment name (default: vllm-prefill)
    DECODE_DEPLOY     Decode deployment name (default: vllm-decode)
    PREFILL_SELECTOR  Label selector for prefill pods (default: app=vllm-prefill)
    DECODE_SELECTOR   Label selector for decode pods (default: app=vllm-decode)
    WORKLOAD_TYPE     K8s workload type: deployment or statefulset (default: deployment)
    VLLM_CONTAINER    vLLM container name (default: vllm)
    SIDECAR_CONTAINER Sidecar container name (default: routing-sidecar)

    NETPOLICY_FILE    Network partition manifest for 4f
                      (default: manifests/testing/exp4f-networkpolicy-partition.yaml)
    NETEM_DELAY_MS    Added latency for 4g slow-network test (default: 100)
    NETEM_LOSS_PCT    Packet loss percentage for 4g (default: 10)
    KEEPALIVE_RUNS    Number of UCX keepalive measurement runs for 4h (default: 3)
    KILL_DELAYS_MS    Comma-separated kill delays for 4i mid-transfer (default: 700,800,900,1000)
                      These are wall-clock delays from request start. The kill
                      runs in parallel with `oc exec` (~580ms overhead) + prefill
                      compute (~70ms for 1024 tokens). So a 700ms delay fires
                      ~50ms into the actual KV transfer window.
    PARTITION_DURATIONS Comma-separated partition durations in seconds for 4f (default: 5,30,120)
    LOAD_QPS          Target QPS for 4k/4l sustained load (default: 4)
    LOAD_DURATION     Duration in seconds for 4k load test (default: 60)
    KILL_AT_S         Seconds before prefill kill in 4k (default: 10)
    ROLLOUT_DURATION  Duration in seconds for 4l rollout test (default: 90)

Outputs:
    data/exp4-results.csv   Timing and status data for all sub-experiments.
    data/exp4-logs/         Pod logs captured at key moments during each
                            fault scenario (pre-kill, post-recovery, during
                            partition). Shows NIXL reconnection, ZMQ discovery,
                            and sidecar error handling sequences.
"""

import argparse
import calendar
import csv as csv_mod
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    DATA_DIR,
    DISAGG_D1_URL,
    DISAGG_D2_URL,
    EPP_URL,
    MODEL,
    NS,
    PREFILL_HOST,
    ROUTING_MODE,
    env,
    progress,
    send_disagg,
    write_run_info,
)
from schemas import Exp4Row, TypedCSVWriter

TEST_CLIENT = env("TEST_CLIENT", "test-client")
VLLM_CONTAINER = env("VLLM_CONTAINER", "vllm")

DECODE1_URL = EPP_URL if ROUTING_MODE == "epp" else DISAGG_D1_URL
DECODE2_URL = EPP_URL if ROUTING_MODE == "epp" else DISAGG_D2_URL

WORKLOAD_TYPE = env("WORKLOAD_TYPE", "deployment")  # or "statefulset"
PREFILL_DEPLOY = env("PREFILL_DEPLOY", "vllm-prefill")
DECODE1_DEPLOY = env("DECODE1_DEPLOY", env("DECODE_DEPLOY", "vllm-decode"))
DECODE2_DEPLOY = env("DECODE2_DEPLOY", env("DECODE_DEPLOY", "vllm-decode"))
PREFILL_SELECTOR = env("PREFILL_SELECTOR", "app=vllm-prefill")
DECODE1_SELECTOR = env("DECODE1_SELECTOR", env("DECODE_SELECTOR", "app=vllm-decode"))
NETPOLICY_FILE = env("NETPOLICY_FILE", "manifests/testing/exp4f-networkpolicy-partition.yaml")
SIDECAR_CONTAINER = env("SIDECAR_CONTAINER", "routing-sidecar")
NETEM_DELAY_MS = int(env("NETEM_DELAY_MS", "100"))
NETEM_LOSS_PCT = int(env("NETEM_LOSS_PCT", "10"))
LOAD_QPS = int(env("LOAD_QPS", "4"))
BASELINE_N = int(env("BASELINE_N", "30"))
METRICS_INTERVAL = float(env("METRICS_INTERVAL", "2"))
KEEPALIVE_RUNS = int(env("KEEPALIVE_RUNS", "3"))
KILL_DELAYS_MS = [int(d) for d in env("KILL_DELAYS_MS", "700,800,900,1000").split(",")]
KILL_REPEATS = int(env("KILL_REPEATS", "1"))
PARTITION_DURATIONS = [int(d) for d in env("PARTITION_DURATIONS", "5,30,120").split(",")]
PARTITION_ISOLATED = env("PARTITION_ISOLATED", "false").lower() in ("true", "1", "yes")
LOAD_DURATION = int(env("LOAD_DURATION", "60"))
KILL_AT_S = int(env("KILL_AT_S", "10"))
ROLLOUT_DURATION = int(env("ROLLOUT_DURATION", "90"))

# Hardcoded experiment parameters
LOAD_DURATION_4E = 30
KILL_AT_4E = 5
ROLLOUT_AT_S = 10
# Prompt for 4i mid-transfer kill tests. Larger prompts extend the KV transfer
# window, making it easier to land a kill during transfer.
#
# Sizing rationale (from exp4 metrics on a 3xT4 cluster):
#   - Observed: 352KB per transfer for 3-token prompt at 103 MB/s
#   - Empirical: ~120KB/token (includes NIXL framing, 5.7x theoretical KV size)
#   - 1024 tokens: 12-123 MB transfer, 115-1194ms window (two scenarios)
#   - 4096 tokens: 492 MB, ~5s transfer — exceeds socket timeout, breaks baseline
#
# Kill timing (kill runs in parallel with request via separate thread):
#   - oc exec overhead: ~580ms (request hasn't reached vLLM yet)
#   - Prefill compute: ~68ms for 1024 tokens at ~15k tokens/s
#   - Kill delay must be > 648ms to land during actual KV transfer
#   - Default delays 700-1000ms → 52-352ms into transfer window
MID_TRANSFER_PROMPT_TOKENS = int(env("MID_TRANSFER_PROMPT_TOKENS", "1024"))
MID_TRANSFER_PROMPT = " ".join(["word"] * MID_TRANSFER_PROMPT_TOKENS)



# ── Type Definitions ─────────────────────────────────────────────────────────

#: Single baseline measurement: (ttft_ms, total_ms, status_str, error_str).
BaselineSample = namedtuple("BaselineSample", ["ttft", "total", "status", "error"])


@dataclass(frozen=True)
class Calibration:
    """Immutable result of instrument calibration (Phase 0a)."""
    oc_exec_mean_ms: float
    oc_exec_max_ms: float
    clock_skew_mean_ms: float
    clock_skew_uncertainty_ms: float
    probe_interval_max_ms: float  # 0.0 if insufficient data


@dataclass(frozen=True)
class Baseline:
    """Immutable baseline distribution from run_baseline()."""
    n: int
    warmup: int
    ok: int
    fail: int
    ttft_mean: float
    ttft_stddev: float
    ttft_ci95: tuple[float, float]
    ttft_p50: float
    ttft_p99: float
    ttft_min: float
    ttft_max: float
    total_mean: float
    total_p50: float
    total_p99: float
    stationary: bool
    stationarity_z: float
    raw: tuple[BaselineSample, ...]  # immutable sequence of per-request measurements


@dataclass
class ExperimentContext:
    """Shared state passed to each experiment function.

    Calibration and baselines are frozen (computed once, never modified).
    The writer is the only mutable component (append-only CSV output).
    The findings dict accumulates key results for cross-experiment summary.
    """
    writer: TypedCSVWriter
    log_dir: str
    calibration: Calibration
    baseline_d1: Baseline
    baseline_d2: Baseline
    findings: dict[str, Any] = field(default_factory=dict)

    def epoch_ms(self) -> int:
        return int(time.time() * 1000)

    def finding(self, key, value):
        """Record a key finding for cross-experiment synthesis."""
        self.findings[key] = value

    def record(self, sub, phase, ttft_ms, total_ms, status, note, error="",
               **kwargs):
        _record(self.writer, sub, phase, ttft_ms, total_ms, status, note,
                error=error, **kwargs)


def _record(writer, sub, phase, ttft_ms, total_ms, status, note, error="",
            itl_mean_ms="", itl_p99_ms="", token_count="",
            detect_epoch_ms="", recover_epoch_ms="",
            probes_to_detect="", probes_to_recover=""):
    """Write a single row to the experiment CSV.

    Used by ExperimentContext.record() and by pre-context functions
    (run_calibration, run_baselines) that operate before the context exists.
    """
    writer.write({
        "experiment": "exp4", "sub": sub, "phase": phase,
        "timestamp": timestamp(), "epoch_ms": int(time.time() * 1000),
        "ttft_ms": ttft_ms, "total_ms": total_ms,
        "itl_mean_ms": itl_mean_ms, "itl_p99_ms": itl_p99_ms,
        "token_count": token_count,
        "status_code": status, "note": note, "error": error,
        "detect_epoch_ms": detect_epoch_ms,
        "recover_epoch_ms": recover_epoch_ms,
        "probes_to_detect": probes_to_detect,
        "probes_to_recover": probes_to_recover,
    })
    progress(f"    [{sub}/{phase}] code={status} ttft={ttft_ms}ms "
             f"total={total_ms}ms {note}")


def oc(*args):
    """Run an oc command and return stdout. Warns on non-zero exit."""
    cmd = ["oc", *args]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        progress(f"  WARNING: oc {' '.join(args[:3])} exited {result.returncode}: "
                 f"{result.stderr.strip()[:200]}")
    return result.stdout.strip(), result.stderr.strip()


REMOTE_DIR = "/scripts/toolkit"


def fault_driver_route_args():
    """Return the route options shared by in-pod fault-driver commands."""
    args = ["--route-mode", ROUTING_MODE]
    if ROUTING_MODE == "manual":
        args.extend(["--prefill-host", PREFILL_HOST])
    elif not EPP_URL:
        raise ValueError("ROUTING_MODE=epp requires EPP_URL")
    return args


def send_via_test_client(url, prompt="Hello world", max_tokens=10,
                         curl_timeout=10, subprocess_timeout=30):
    """Send a single request via the in-pod fault driver.

    Returns parsed timing: (ttft_ms, total_ms, status_code, error).
    Uses fault_driver.py probe mode with duration=0 (single shot).
    """
    # duration=0.001 (1ms): the probe loop checks deadline at the *top* of each
    # iteration, so the first iteration always executes. The request itself takes
    # ~100-2000ms, after which the deadline has passed and the loop exits.
    # interval=0: no sleep between requests (irrelevant for single-shot).
    cmd = [
        "oc", "exec", TEST_CLIENT, "-n", NS, "--",
        "python3", f"{REMOTE_DIR}/fault_driver.py",
        "--url", url,
        *fault_driver_route_args(),
        "--model", MODEL,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--timeout", str(curl_timeout),
        "probe", "--interval", "0", "--duration", "0.001",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=subprocess_timeout)
        output = result.stdout.strip()
        if not output:
            return 0, 0, "no_output", result.stderr.strip()[:200]

        # Parse first JSONL line
        data = json.loads(output.splitlines()[0])
        ttft = data.get("ttft_ms", 0)
        total = data.get("total_ms", 0)
        status = str(data.get("status", 0))
        error = data.get("error", "")
        return ttft, total, status, error

    except subprocess.TimeoutExpired:
        return 0, 0, "timeout", "oc exec timed out"
    except (json.JSONDecodeError, ValueError) as e:
        return 0, 0, "parse_error", str(e)
    except Exception as e:
        return 0, 0, "error", str(e)


def send_streaming_via_test_client(url, prompt="Hello world", max_tokens=20,
                                    subprocess_timeout=60, socket_timeout=8):
    """Send a single streaming request via the in-pod fault driver.

    Uses fault_driver.py load mode with qps=1 and duration that exceeds the
    socket timeout (fires one request quickly, waits for it to complete or
    timeout, then exits). The socket timeout must be shorter than
    duration + join_timeout so the worker thread completes and writes
    its CSV row even if the connection dies mid-stream.

    Args:
        socket_timeout: Per-request socket timeout in seconds. Default 8s is
            fine for small prompts. For large prompts (1024+ tokens), use 30s
            to allow time for KV transfer.

    Returns:
        (ttft_ms, total_ms, itl_mean_ms, itl_p99_ms, token_count,
         status_code, error, received_text)
    """
    # Duration must exceed socket_timeout so the worker finishes before exit
    duration = socket_timeout + 5
    output_path = f"{REMOTE_DIR}/data/_streaming_single.csv"
    cmd = [
        "oc", "exec", TEST_CLIENT, "-n", NS, "--",
        "python3", f"{REMOTE_DIR}/fault_driver.py",
        "--url", url,
        *fault_driver_route_args(),
        "--model", MODEL,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--timeout", str(socket_timeout),
        "load", "--qps", "1", "--duration", str(duration),
        "--output", output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=subprocess_timeout)
        # Read the CSV from the pod
        csv_out, _ = oc("exec", TEST_CLIENT, "-n", NS, "--",
                         "cat", output_path)
        if not csv_out:
            return 0, 0, 0, 0, 0, "no_output", result.stderr.strip()[:200], ""

        reader = csv_mod.DictReader(io.StringIO(csv_out))
        row = next(reader, None)
        if not row:
            return 0, 0, 0, 0, 0, "empty_csv", "", ""

        return (
            float(row.get("ttft_ms", 0)),
            float(row.get("total_ms", 0)),
            float(row.get("itl_mean_ms", 0)),
            float(row.get("itl_p99_ms", 0)),
            int(row.get("token_count", 0)),
            str(row.get("status", 0)),
            row.get("error", ""),
            row.get("received_text", ""),
        )

    except subprocess.TimeoutExpired:
        return 0, 0, 0, 0, 0, "timeout", "oc exec timed out", ""
    except Exception as e:
        return 0, 0, 0, 0, 0, "error", str(e), ""


def start_probe(url, interval=0.2, duration=120):
    """Start the fault driver in probe mode as a background subprocess.

    Returns a Popen object whose stdout yields JSONL lines in real-time.
    Each line: {"seq":N, "epoch_ms":T, "ttft_ms":F, "total_ms":F, "status":N, "error":""}

    The probe runs in-pod with real timing — ~200ms resolution, not limited
    by oc exec round-trip.
    """
    cmd = [
        "oc", "exec", TEST_CLIENT, "-n", NS, "--",
        "python3", f"{REMOTE_DIR}/fault_driver.py",
        "--url", url,
        *fault_driver_route_args(),
        "--model", MODEL,
        "--timeout", "5",
        "probe", "--interval", str(interval), "--duration", str(duration),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    return proc


def start_local_probe(url, interval=0.05, stop_event=None):
    """Probe from local machine at high frequency. No clock skew.

    Uses client.send_disagg() to send requests directly to the decode
    service URL from outside the cluster. Both kill and probe timestamps
    use the local clock, so detection gap has zero clock-skew uncertainty.

    Resolution is limited by probe interval (default 50ms) + network RTT.
    Requires the decode service to be reachable from this machine
    (via route or port-forward).

    Returns (thread, results_list). Results are dicts with:
        seq, epoch_ms, ttft_ms, total_ms, status, error
    """
    results = []

    def _probe():
        seq = 0
        while not stop_event.is_set():
            seq += 1
            epoch_ms = int(time.time() * 1000)
            try:
                r = send_disagg(url, prompt="Hello", max_tokens=1)
                results.append({
                    "seq": seq, "epoch_ms": epoch_ms,
                    "ttft_ms": r.ttft_ms, "total_ms": r.total_ms,
                    "status": r.status, "error": r.error,
                })
                elapsed = r.total_ms / 1000.0
            except Exception as e:
                results.append({
                    "seq": seq, "epoch_ms": epoch_ms,
                    "ttft_ms": 0, "total_ms": 0,
                    "status": 0, "error": str(e),
                })
                elapsed = 0
            wait = max(0, interval - elapsed)
            if wait > 0:
                stop_event.wait(wait)

    t = threading.Thread(target=_probe, daemon=True)
    t.start()
    return t, results


def read_probe_until_fail(proc, timeout=120):
    """Read JSONL from a running probe until a non-200 status appears.

    Returns: (first_fail_epoch_ms, probes_sent, error_msg, all_probes)
    where all_probes is a list of parsed JSONL dicts.
    """
    probes = []
    deadline = time.time() + timeout
    for line in proc.stdout:
        if time.time() > deadline:
            break
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            probes.append(data)
            if data.get("status") != 200:
                return (data["epoch_ms"], len(probes),
                        data.get("error", "") or f"status={data['status']}",
                        probes)
        except (json.JSONDecodeError, ValueError):
            continue
    return 0, len(probes), "timeout: no failure detected", probes


def read_probe_until_recover(proc, timeout=120):
    """Read JSONL from a running probe until a 200 status appears.

    Returns: (recover_epoch_ms, probes_sent, recovery_latency_ms, all_probes)
    """
    probes = []
    deadline = time.time() + timeout
    for line in proc.stdout:
        if time.time() > deadline:
            break
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            probes.append(data)
            if data.get("status") == 200:
                return (data["epoch_ms"], len(probes),
                        data.get("total_ms", 0), probes)
        except (json.JSONDecodeError, ValueError):
            continue
    return 0, len(probes), 0, probes


def stop_probe(proc):
    """Stop a running probe subprocess."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def start_metrics_collector(sample_interval=2):
    """Start metrics collector in the test-client pod (background).

    Returns a Popen object. Metrics are written to data/metrics-timeseries.csv
    inside the pod.
    """
    metric_env = []
    for name in ("METRICS_ENDPOINTS", "EPP_URL", "EPP_METRICS_URL",
                 "EPP_RELEASE_NAME", "EPP_SERVICE_NAME"):
        value = os.environ.get(name)
        if value:
            metric_env.append(f"{name}={value}")
    cmd = [
        "oc", "exec", TEST_CLIENT, "-n", NS, "--",
        "env", f"NS={NS}",
        f"DATA_DIR={REMOTE_DIR}/data",
        f"SAMPLE_INTERVAL={sample_interval}",
        *metric_env,
        "python3", f"{REMOTE_DIR}/metrics_collector.py",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    time.sleep(1)  # let it start
    return proc


def stop_metrics_collector(proc):
    """Stop the metrics collector and return."""
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def start_load(url, qps=4, duration=60, output_name="load-results.csv"):
    """Start sustained load in the test-client pod (background).

    Returns a Popen object. Load results are written to the output CSV
    inside the pod.
    """
    output_path = f"{REMOTE_DIR}/data/{output_name}"
    cmd = [
        "oc", "exec", TEST_CLIENT, "-n", NS, "--",
        "python3", f"{REMOTE_DIR}/fault_driver.py",
        "--url", url,
        *fault_driver_route_args(),
        "--model", MODEL,
        "--max-tokens", "20",
        "--timeout", "15",
        "load", "--qps", str(qps), "--duration", str(duration),
        "--output", output_path,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    return proc, output_path


def collect_load_results(output_path):
    """Pull load results CSV from the pod. Returns list of dicts."""
    csv_out, _ = oc("exec", TEST_CLIENT, "-n", NS, "--", "cat", output_path)
    if not csv_out:
        return []
    reader = csv_mod.DictReader(io.StringIO(csv_out))
    return list(reader)


# Log patterns for structured parsing (reuses categories from analyze.py)
_LOG_CATEGORIES = {
    "nixl_error": [
        "nixl", "kv_transfer", "kv transfer", "pull_async", "push_async",
    ],
    "nixl_reconnect": [
        "reconnect", "retry", "backoff", "re-establish",
        "connection restored", "recovered", "handshake", "register_agent",
    ],
    "zmq_discovery": [
        "zmq", "discovery", "new peer", "peer lost", "endpoint",
        "service_discovery",
    ],
    "ucx_error": [
        "ucx", "ucp", "uct", "ECONNRESET", "ECONNREFUSED", "ETIMEDOUT",
        "broken pipe", "connection reset", "connection refused", "timed out",
    ],
    "sidecar_error": [
        "prefill.*fail", "prefill.*error", "fallback", "no backend",
        "upstream.*error", r"\b502\b", r"\b503\b", r"\b504\b",
    ],
}

# Timestamp patterns for vLLM/Python log lines

_TS_PATTERNS = [
    # Python logging: 2024-03-15 10:30:45,123
    (re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),(\d{3})'), "%Y-%m-%d %H:%M:%S"),
    # ISO 8601: 2024-03-15T10:30:45.123456
    (re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.?\d*'), "%Y-%m-%dT%H:%M:%S"),
]


def parse_log_timestamps(filepath):
    """Extract timestamped events from a log file.

    Returns list of (epoch_ms, category, line) tuples for lines matching
    any category in _LOG_CATEGORIES.
    """
    results = []
    try:
        with open(filepath) as f:
            for line in f:
                line_lower = line.lower()
                matched_cat = None
                for cat, patterns in _LOG_CATEGORIES.items():
                    for pat in patterns:
                        if re.search(pat, line_lower):
                            matched_cat = cat
                            break
                    if matched_cat:
                        break
                if not matched_cat:
                    continue

                # Extract timestamp
                epoch_ms = 0
                for ts_re, ts_fmt in _TS_PATTERNS:
                    m = ts_re.search(line)
                    if m:
                        try:
                            t = time.strptime(m.group(1), ts_fmt)
                            epoch_s = calendar.timegm(t)
                            # Add milliseconds if available
                            if len(m.groups()) > 1:
                                epoch_ms = epoch_s * 1000 + int(m.group(2))
                            else:
                                epoch_ms = epoch_s * 1000
                        except (ValueError, OverflowError):
                            pass
                        break

                results.append((epoch_ms, matched_cat, line.strip()))
    except OSError:
        pass
    return results


def _parse_iso_epoch(ts_str):
    """Convert ISO 8601 timestamp like '2026-04-04T01:02:19Z' to epoch_ms."""
    try:
        # Strip trailing Z and parse
        clean = ts_str.rstrip("Z")
        t = time.strptime(clean, "%Y-%m-%dT%H:%M:%S")
        return calendar.timegm(t) * 1000
    except (ValueError, OverflowError):
        return 0


def parse_sidecar_fallbacks(log_content):
    """Parse sidecar logs for 'fallback to decode' events.

    The routing sidecar logs structured JSON with msg="fallback to decode"
    when it falls back to local inference because prefill is unreachable.
    This is the only observable signal that distinguishes disaggregated
    responses from fallback responses.

    Returns list of (epoch_ms, request_id) tuples.
    """
    fallbacks = []
    for line in log_content.splitlines():
        try:
            entry = json.loads(line.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if entry.get("msg") == "fallback to decode":
            ts = entry.get("ts", "")
            rid = entry.get("request_id", "")
            epoch_ms = _parse_iso_epoch(ts)
            fallbacks.append((epoch_ms, rid))
    return fallbacks


def classify_response(ttft_ms, status, error, baseline_ttft,
                      fallback_observed=False):
    """Classify which path a request took through the system.

    Every experiment needs to answer the same question: did this request
    use the disaggregated path (prefill→NIXL→decode), the fallback path
    (local decode), or did it fail? This function provides a single,
    consistent classification instead of ad-hoc thresholds in each
    experiment.

    Args:
        ttft_ms: Time-to-first-token in milliseconds.
        status: HTTP status code as string.
        error: Error string (empty if no error).
        baseline_ttft: Baseline TTFT for the disaggregated path.
        fallback_observed: Whether sidecar log confirms fallback for this
            request window. This is the definitive signal when available.

    Returns one of:
        "disagg"              — normal disaggregated path (prefill + NIXL + decode)
        "fallback"            — sidecar fell back to local decode (log-confirmed)
        "fallback_suspected"  — likely fallback based on elevated TTFT (heuristic)
        "error"               — request returned an error
        "timeout"             — request timed out with no response

    The distinction between "fallback" and "fallback_suspected" matters:
    "fallback" is a definitive classification from sidecar logs;
    "fallback_suspected" is a conjecture from TTFT that could also be
    slow NIXL reconnection. Downstream code should not conflate these.
    Use `is_fallback(classification)` to test for either.
    """
    if status != "200" or ttft_ms <= 0:
        if error and ("timeout" in error.lower() or "timed out" in error.lower()):
            return "timeout"
        return "error"

    # Definitive signal: sidecar logs confirm fallback
    if fallback_observed:
        return "fallback"

    # Heuristic: fallback responses have elevated TTFT because the decode
    # pod must do prefill locally (which it normally doesn't). A request
    # at >3x baseline likely took the fallback path. But this is weaker
    # than the log-based signal — slow NIXL reconnection can also produce
    # elevated TTFT without fallback.
    if baseline_ttft > 0 and ttft_ms > baseline_ttft * 3:
        return "fallback_suspected"

    return "disagg"


def is_fallback(classification):
    """Test if a classification indicates fallback (definitive or suspected)."""
    return classification in ("fallback", "fallback_suspected")


def wait_for_ready(workload_name, timeout_s=600, min_ready=1):
    """Wait for a statefulset/deployment to have min_ready replicas."""
    sys.stderr.write(f"  Waiting for {workload_name} to be ready")
    sys.stderr.flush()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out, _ = oc("get", WORKLOAD_TYPE, workload_name, "-n", NS,
                     "-o", "jsonpath={.status.readyReplicas}")
        try:
            ready = int(out.strip())
        except (ValueError, TypeError):
            ready = 0
        if ready >= min_ready:
            progress(" ready")
            return
        sys.stderr.write(".")
        sys.stderr.flush()
        time.sleep(5)
    progress(f" TIMEOUT after {timeout_s}s")
    raise TimeoutError(f"{workload_name} not ready after {timeout_s}s")


def get_pod_name(selector):
    """Get the name of the first pod matching a label selector."""
    out, _ = oc("get", "pods", "-n", NS, "-l", selector,
                "-o", "jsonpath={.items[0].metadata.name}")
    return out.strip()


def timestamp():
    return time.strftime("%H:%M:%S")


def collect_cluster_info():
    """Gather cluster-level metadata via oc. Returns dict."""
    info = {}

    # oc version (client + server)
    out, _ = oc("version", "--output=json")
    if out:
        try:
            v = json.loads(out)
            info["oc_client"] = v.get("clientVersion", {}).get("gitVersion", "")
            info["server"] = v.get("serverVersion", {}).get("gitVersion", "")
        except (json.JSONDecodeError, ValueError):
            pass

    # Pod images (vLLM and sidecar versions)
    for selector, label in [(PREFILL_SELECTOR, "prefill"),
                            (DECODE1_SELECTOR, "decode1")]:
        out, _ = oc("get", "pods", "-n", NS, "-l", selector,
                     "-o", "jsonpath={.items[0].spec.containers[*].image}")
        if out:
            info[f"{label}_images"] = out.strip().split()

    # GPU info from node labels
    out, _ = oc("get", "nodes", "-o",
                "jsonpath={range .items[*]}{.metadata.name}={.metadata.labels.nvidia\\.com/gpu\\.product} ")
    if out:
        gpu_nodes = {}
        for item in out.strip().split():
            if "=" in item:
                node, gpu = item.split("=", 1)
                if gpu:
                    gpu_nodes[node] = gpu
        if gpu_nodes:
            info["gpu_nodes"] = gpu_nodes

    # vLLM/NIXL/NCCL environment variables from running pods.
    # These are the knobs that control fault tolerance and communication behavior.
    _INTERESTING_PREFIXES = (
        "VLLM_", "NIXL_", "NCCL_", "CUDA_", "TORCH_",
        "RDMA_", "UCX_", "GLOO_",
    )
    for selector, label in [(PREFILL_SELECTOR, "prefill"),
                            (DECODE1_SELECTOR, "decode1")]:
        pod = get_pod_name(selector)
        if not pod:
            continue
        out, _ = oc("exec", pod, "-c", VLLM_CONTAINER, "-n", NS,
                     "--", "env")
        if out:
            env_vars = {}
            for line in out.splitlines():
                if "=" in line:
                    k, _, v = line.partition("=")
                    if k.startswith(_INTERESTING_PREFIXES):
                        env_vars[k] = v
            if env_vars:
                info[f"{label}_env"] = env_vars

    return info


def capture_logs(pod, container, label, log_dir):
    """Capture pod logs to a file. Returns the filepath written.

    Args:
        pod: Pod name (or empty string to skip).
        container: Container name within the pod.
        label: Descriptive label for the filename (e.g., '4a-decode1-pre').
        log_dir: Directory to write log files.
    """
    if not pod:
        return None
    filepath = os.path.join(log_dir, f"{label}-{container}.log")
    out, _err = oc("logs", pod, "-c", container, "-n", NS)
    if out:
        with open(filepath, "w") as f:
            f.write(out)
        return filepath
    return None


def capture_pod_logs(selector, containers, label, log_dir):
    """Capture logs from all containers of the first pod matching selector.

    Errors are logged but never propagate — log capture must not crash
    the experiment.
    """
    try:
        pod = get_pod_name(selector)
        if not pod:
            progress(f"  (no pod found for {selector}, skipping log capture)")
            return
        files = []
        for c in containers:
            path = capture_logs(pod, c, label, log_dir)
            if path:
                files.append(os.path.basename(path))
        if files:
            progress(f"  Captured: {', '.join(files)}")
    except Exception as e:
        progress(f"  WARNING: log capture failed for {label}: {e}")


def snapshot_transport_state(selector, container, label, log_dir):
    """Capture TCP connection state from a pod for transport-layer observability.

    Runs `ss -tnp` inside the pod to capture all TCP connections with their
    state (ESTAB, TIME-WAIT, CLOSE-WAIT, etc.) and the owning process.

    This is the interior view that request-level probes cannot provide:
    - How many connections exist between pods
    - Are stale connections accumulating (C8 cumulative stress)
    - Are connections in unexpected states after fault recovery

    Returns dict of {state: count} or None on failure.
    """
    try:
        pod = get_pod_name(selector)
        if not pod:
            return None
        out, _err = oc("exec", pod, "-c", container, "-n", NS,
                       "--", "ss", "-tnp")
        if not out:
            return None

        # Save raw output
        filepath = os.path.join(log_dir, f"{label}-tcp-state.txt")
        with open(filepath, "w") as f:
            f.write(out)

        # Parse state counts
        state_counts = {}
        for line in out.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) >= 1:
                state = parts[0]
                state_counts[state] = state_counts.get(state, 0) + 1

        total = sum(state_counts.values())
        summary = " ".join(f"{s}={c}" for s, c in sorted(state_counts.items()))
        progress(f"  TCP state [{label}]: {total} connections — {summary}")
        return state_counts
    except Exception as e:
        progress(f"  WARNING: TCP state capture failed for {label}: {e}")
        return None


def diff_transport_state(before, after, label):
    """Compare two transport state snapshots and report changes.

    Returns a summary string describing what changed.
    """
    if not before or not after:
        return "n/a (missing snapshot)"

    changes = []
    all_states = set(list(before.keys()) + list(after.keys()))
    for state in sorted(all_states):
        b = before.get(state, 0)
        a = after.get(state, 0)
        if a != b:
            changes.append(f"{state}: {b}→{a}")

    if not changes:
        return "no change"

    summary = ", ".join(changes)
    progress(f"  TCP state delta [{label}]: {summary}")
    return summary


def copy_scripts_to_pod():
    """Copy toolkit scripts to the test-client pod."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    progress("  Copying toolkit scripts to test-client pod...")
    oc("exec", TEST_CLIENT, "-n", NS, "--",
       "mkdir", "-p", REMOTE_DIR, f"{REMOTE_DIR}/data")
    # Copy the scripts we need
    for script in ["fault_driver.py", "metrics_collector.py", "client.py"]:
        src = os.path.join(script_dir, script)
        if os.path.exists(src):
            subprocess.run(["oc", "cp", src,
                            f"{NS}/{TEST_CLIENT}:{REMOTE_DIR}/{script}"],
                           capture_output=True, timeout=30)
    progress("  Scripts copied")


def collect_metrics_csv():
    """Pull the metrics timeseries CSV from the pod. Returns local path."""
    remote_path = f"{REMOTE_DIR}/data/metrics-timeseries.csv"
    local_path = os.path.join(DATA_DIR, "exp4-metrics.csv")
    subprocess.run(["oc", "cp",
                    f"{NS}/{TEST_CLIENT}:{remote_path}", local_path],
                   capture_output=True, timeout=30)
    if os.path.exists(local_path):
        progress(f"  Metrics saved: {local_path}")
    return local_path


# ── Fermi Method: Baseline & Predictions ─────────────────────────────────────


def percentile(values, pct):
    """Compute percentile from a pre-sorted list using linear interpolation."""
    if not values:
        return 0
    k = (len(values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    frac = k - lo
    return values[lo] + frac * (values[hi] - values[lo])


def _t95(df):
    """Two-tailed t critical value at 95% confidence for given df."""
    # Tabulated values; linear interpolation between entries.
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980,
    }
    if df >= 120:
        return 1.96
    keys = sorted(table)
    if df <= keys[0]:
        return table[keys[0]]
    if df in table:
        return table[df]
    # Linear interpolation
    for i in range(len(keys) - 1):
        if keys[i] < df < keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            frac = (df - lo) / (hi - lo)
            return table[lo] + frac * (table[hi] - table[lo])
    return 1.96


def sample_stats(values):
    """Compute sample statistics: mean, stddev (Bessel-corrected), 95% CI.

    Uses sample variance (N-1 denominator) and t critical values for CI.
    """
    n = len(values)
    if n == 0:
        return 0, 0, (0, 0)
    mean = sum(values) / n
    if n < 2:
        return mean, 0, (mean, mean)
    # Bessel-corrected sample variance (N-1)
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    stddev = math.sqrt(var)
    # 95% CI: mean ± t * (stddev / sqrt(n))
    t = _t95(n - 1)
    margin = t * stddev / math.sqrt(n)
    ci = (round(mean - margin, 1), round(mean + margin, 1))
    return round(mean, 1), round(stddev, 1), ci


def stationarity_check(values):
    """Split-half stationarity test.

    Splits the sequence in half, compares means. If the two halves have
    significantly different means (>2x the pooled standard error), the
    process is non-stationary — the baseline is drifting.

    Returns (is_stationary, first_half_mean, second_half_mean, z_score).
    z_score is 0.0 for degenerate cases (too few samples or zero variance)
    where the test is inconclusive — do not interpret as a real z-score.
    """
    n = len(values)
    if n < 6:
        return True, 0, 0, 0.0  # too few samples to test
    mid = n // 2
    first = values[:mid]
    second = values[mid:]
    m1 = sum(first) / len(first)
    m2 = sum(second) / len(second)
    # Pooled standard error (Bessel-corrected)
    v1 = sum((x - m1) ** 2 for x in first) / max(len(first) - 1, 1)
    v2 = sum((x - m2) ** 2 for x in second) / max(len(second) - 1, 1)
    se = math.sqrt(v1 / len(first) + v2 / len(second))
    if se == 0:
        return True, round(m1, 1), round(m2, 1), 0.0  # zero variance
    # Two-sample z-test (approximate)
    z = abs(m1 - m2) / se
    # z > 2.0 ≈ p < 0.05
    return z <= 2.0, round(m1, 1), round(m2, 1), round(z, 2)


WARMUP_N = 5  # discard first N requests (cold path: DNS, conn pool, JIT)


def run_baseline(url, n=30, warmup=None, label="baseline"):
    """Send warmup + n requests, compute distribution statistics.

    The first `warmup` requests are discarded (cold-path bias).
    Returns a frozen Baseline dataclass with distribution statistics
    and the raw measurement tuples.
    """
    if warmup is None:
        warmup = WARMUP_N

    # Warmup phase — discard these
    for _ in range(warmup):
        send_via_test_client(url)

    # Measurement phase
    results = []
    for _i in range(n):
        t, tot, code, err = send_via_test_client(url)
        results.append(BaselineSample(t, tot, code, err))

    ok_ttfts = [r.ttft for r in results if r.status == "200" and r.ttft > 0]
    ok_totals = [r.total for r in results if r.status == "200" and r.total > 0]
    ok_count = len(ok_ttfts)
    fail_count = n - ok_count

    if not ok_ttfts:
        return Baseline(
            n=n, warmup=warmup, ok=0, fail=fail_count,
            ttft_mean=0, ttft_stddev=0, ttft_ci95=(0, 0),
            ttft_p50=0, ttft_p99=0, ttft_min=0, ttft_max=0,
            total_mean=0, total_p50=0, total_p99=0,
            stationary=True, stationarity_z=0,
            raw=tuple(results))

    ttft_mean, ttft_stddev, ttft_ci = sample_stats(ok_ttfts)
    total_mean, _, _ = sample_stats(ok_totals)
    ok_ttfts_sorted = sorted(ok_ttfts)
    ok_totals_sorted = sorted(ok_totals)

    # Stationarity: is the system drifting during baseline?
    is_stat, _half1, _half2, stat_z = stationarity_check(ok_ttfts)

    return Baseline(
        n=n, warmup=warmup, ok=ok_count, fail=fail_count,
        ttft_mean=ttft_mean, ttft_stddev=ttft_stddev, ttft_ci95=ttft_ci,
        ttft_p50=round(percentile(ok_ttfts_sorted, 50), 1),
        ttft_p99=round(percentile(ok_ttfts_sorted, 99), 1),
        ttft_min=round(ok_ttfts_sorted[0], 1),
        ttft_max=round(ok_ttfts_sorted[-1], 1),
        total_mean=total_mean,
        total_p50=round(percentile(ok_totals_sorted, 50), 1),
        total_p99=round(percentile(ok_totals_sorted, 99), 1),
        stationary=is_stat, stationarity_z=stat_z,
        raw=tuple(results))


def print_baseline(bl, label=""):
    """Print baseline statistics with confidence intervals."""
    prefix = f"  [{label}] " if label else "  "
    progress(f"{prefix}n={bl.n} (after {bl.warmup} warmup) "
             f"ok={bl.ok} fail={bl.fail}")
    progress(f"{prefix}TTFT: mean={bl.ttft_mean}ms "
             f"95%CI=[{bl.ttft_ci95[0]}, {bl.ttft_ci95[1]}]ms "
             f"stddev={bl.ttft_stddev}ms")
    progress(f"{prefix}      p50={bl.ttft_p50}ms p99={bl.ttft_p99}ms "
             f"range=[{bl.ttft_min}, {bl.ttft_max}]ms")
    progress(f"{prefix}Total: mean={bl.total_mean}ms p50={bl.total_p50}ms "
             f"p99={bl.total_p99}ms")
    if not bl.stationary:
        progress(f"{prefix}WARNING: Non-stationary (z={bl.stationarity_z}). "
                 f"System may be drifting during measurement.")
    else:
        progress(f"{prefix}Stationarity: OK (z={bl.stationarity_z})")


def verify_steady_state(url, baseline, n=5, label="gate"):
    """Verify system matches baseline using the baseline's 95% CI.

    Sends n requests, checks that mean TTFT falls within
    [0, CI_upper + 2*stddev]. This uses the baseline's own statistical
    properties rather than arbitrary thresholds.

    Returns (ok, gate_stats).
    """
    gate = run_baseline(url, n=n, warmup=0, label=label)
    if gate.ok == 0:
        progress(f"  [{label}] FAIL: 0/{n} requests succeeded")
        return False, gate

    # Threshold: upper bound of baseline 95% CI + 2 stddev headroom.
    # This accounts for both sampling uncertainty and natural variance.
    # Degenerate case: if baseline had all failures (ci_upper=0, mean=0, stddev=0),
    # threshold = 150ms — any working request passes, which is correct: recovery
    # from a broken baseline is always "improvement."
    ci_upper = baseline.ttft_ci95[1]
    headroom = 2 * max(baseline.ttft_stddev, 20)
    threshold = ci_upper + headroom if ci_upper > 0 else (
        baseline.ttft_mean + 3 * max(baseline.ttft_stddev, 50))

    if gate.ttft_mean <= threshold:
        # Check for bimodality: if max >> min, the system may be oscillating
        # between NIXL path and sidecar fallback. The mean looks fine but
        # the system is not in a single steady state.
        # Check for bimodality: look for a large gap in sorted TTFT values.
        # A ratio check (max > 3*min) is too aggressive for n=5 with
        # log-normal latency distributions. Instead, check if the largest
        # gap between consecutive sorted values exceeds 50% of the range.
        ok_ttfts = sorted(r.ttft for r in gate.raw if r.status == "200" and r.ttft > 0)
        if len(ok_ttfts) >= 3:
            data_range = ok_ttfts[-1] - ok_ttfts[0]
            if data_range > 0:
                gaps = [ok_ttfts[i+1] - ok_ttfts[i] for i in range(len(ok_ttfts)-1)]
                max_gap = max(gaps)
                if max_gap > 0.5 * data_range and data_range > baseline.ttft_mean:
                    progress(f"  [{label}] WARN: possible bimodality — "
                             f"gap={max_gap:.0f}ms in range "
                             f"[{ok_ttfts[0]:.0f}, {ok_ttfts[-1]:.0f}]ms. "
                             f"Re-checking with more probes...")
                    gate2 = run_baseline(url, n=10, warmup=0,
                                         label=f"{label}-bimodal")
                    ok2 = sorted(r.ttft for r in gate2.raw
                                 if r.status == "200" and r.ttft > 0)
                    if len(ok2) >= 3:
                        range2 = ok2[-1] - ok2[0]
                        gaps2 = [ok2[i+1] - ok2[i] for i in range(len(ok2)-1)]
                        max_gap2 = max(gaps2) if gaps2 else 0
                        if (range2 > 0 and max_gap2 > 0.5 * range2
                                and range2 > baseline.ttft_mean):
                            progress(f"  [{label}] WARN: bimodality confirmed "
                                     f"(gap={max_gap2:.0f}ms in "
                                     f"[{ok2[0]:.0f}, {ok2[-1]:.0f}]ms)")
                            return False, gate2
                    progress(f"  [{label}] OK: bimodality not confirmed "
                             f"on re-check")
                    return True, gate2

        progress(f"  [{label}] OK: mean={gate.ttft_mean}ms "
                 f"<= {round(threshold, 1)}ms "
                 f"(CI_upper={ci_upper}ms + {round(headroom, 1)}ms headroom)")
        return True, gate
    else:
        progress(f"  [{label}] WARN: mean={gate.ttft_mean}ms "
                 f"> {round(threshold, 1)}ms — system may not be at steady state")
        return False, gate


def wait_for_steady_state(url, baseline, timeout=120, interval=10, label="settle"):
    """Wait until system returns to baseline performance. Returns True if settled."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, _ = verify_steady_state(url, baseline, label=label)
        if ok:
            return True
        remaining = int(deadline - time.time())
        progress(f"  [{label}] Retrying in {interval}s ({remaining}s remaining)...")
        time.sleep(interval)
    progress(f"  [{label}] TIMEOUT: system did not return to baseline after {timeout}s")
    return False


# ── Prediction Ledger ─────────────────────────────────────────────────────────
# Tracks all predictions and results for the summary table at the end.

_prediction_ledger: list[dict[str, str]] = []


def predict(experiment, prediction, reasoning):
    """Print a Fermi-style prediction before running an experiment."""
    progress(f"  PREDICTION: {prediction}")
    progress(f"  REASONING:  {reasoning}")
    _prediction_ledger.append({
        "experiment": experiment,
        "prediction": prediction,
        "measured": None,
        "verdict": None,
    })


def evaluate(experiment, predicted, measured, unit="ms"):
    """Compare measured result to prediction. Flag surprises.

    Uses log-symmetric bounds: |log2(ratio)| < 1.6 (~3x either direction)
    treats "3x faster" and "3x slower" as equally surprising.
    """
    if predicted == 0:
        progress(f"  RESULT: measured={measured}{unit} (no prediction to compare)")
        return
    if measured < 0:
        progress(f"  RESULT: measured={measured}{unit} (negative — likely clock skew artifact)")
        return
    ratio = measured / predicted
    log_ratio = math.log2(ratio) if ratio > 0 else float("inf")
    if abs(log_ratio) <= 1.6:  # ~3x in either direction
        verdict = "CONSISTENT"
    elif log_ratio < -1.6:
        verdict = "SURPRISE (much faster than predicted)"
    else:
        verdict = "SURPRISE (much slower than predicted)"
    progress(f"  RESULT: predicted={predicted}{unit} measured={measured}{unit} "
             f"ratio={ratio:.1f}x — {verdict}")

    # Update ledger
    for entry in reversed(_prediction_ledger):
        if entry["experiment"] == experiment and entry["measured"] is None:
            entry["measured"] = f"{measured}{unit}"
            entry["verdict"] = verdict
            break


def print_prediction_summary():
    """Print final summary table of all predictions vs measurements."""
    progress("")
    progress("=== Prediction vs Measurement Summary ===")
    progress(f"  {'Exp':<6} {'Verdict':<12} {'Measured':<20} Prediction")
    progress(f"  {'---':<6} {'-------':<12} {'--------':<20} ----------")
    surprises = 0
    for entry in _prediction_ledger:
        v = entry["verdict"] or "NOT MEASURED"
        m = entry["measured"] or "-"
        p = entry["prediction"][:60]
        exp = entry["experiment"]
        progress(f"  {exp:<6} {v:<12} {m:<20} {p}")
        if v and "SURPRISE" in v:
            surprises += 1
    progress("")
    total = len(_prediction_ledger)
    measured = sum(1 for e in _prediction_ledger if e["verdict"])
    progress(f"  {total} predictions, {measured} measured, {surprises} surprises")
    if surprises > 0:
        progress("  Surprises are the interesting findings — investigate these first.")





def run_calibration(writer: TypedCSVWriter) -> Calibration:
    """Phase 0a: Instrument Calibration.

    Fermi rule: know the resolution and bias of your apparatus
    before trusting its readings.
    Returns a frozen Calibration dataclass.
    """
    progress("=== Phase 0a: Instrument Calibration ===")
    progress("")

    # 1. Measure oc exec overhead (the cost of each send_via_test_client call)
    progress("  Measuring oc exec overhead (5 no-op round-trips)...")
    oc_exec_times = []
    for _ in range(5):
        t0 = time.monotonic()
        oc("exec", TEST_CLIENT, "-n", NS, "--", "echo", "ping")
        oc_exec_times.append((time.monotonic() - t0) * 1000)
    oc_mean = sum(oc_exec_times) / len(oc_exec_times)
    oc_min = min(oc_exec_times)
    oc_max = max(oc_exec_times)
    progress(f"  oc exec overhead: mean={oc_mean:.0f}ms min={oc_min:.0f}ms max={oc_max:.0f}ms")
    progress("  (This is the floor for send_via_test_client round-trip, but does NOT")
    progress("   affect in-pod TTFT/total measurements — those use pod-local timing)")
    _record(writer, "calibration", "oc-exec-overhead", oc_mean, oc_max, "n/a",
            f"mean={oc_mean:.0f}ms min={oc_min:.0f}ms max={oc_max:.0f}ms")

    # 2. Measure clock skew between local machine and pod
    #    This is CRITICAL: 4h detection gap = pod_epoch - local_epoch.
    #    If clocks differ, the gap is meaningless.
    progress("")
    progress("  Measuring clock skew (local vs pod)...")
    skew_samples = []
    for _ in range(5):
        local_before = int(time.time() * 1000)
        pod_time_str, _ = oc("exec", TEST_CLIENT, "-n", NS, "--",
                              "python3", "-c",
                              "import time; print(int(time.time() * 1000))")
        local_after = int(time.time() * 1000)
        try:
            pod_time = int(pod_time_str.strip())
            # Best estimate: pod time vs midpoint of local before/after
            local_mid = (local_before + local_after) // 2
            skew = pod_time - local_mid
            oc_rtt = local_after - local_before
            skew_samples.append((skew, oc_rtt))
        except (ValueError, TypeError):
            pass

    if skew_samples:
        skews = [s[0] for s in skew_samples]
        rtts = [s[1] for s in skew_samples]
        skew_mean = sum(skews) / len(skews)
        skew_min = min(skews)
        skew_max = max(skews)
        rtt_mean = sum(rtts) / len(rtts)
        progress(f"  Clock skew (pod - local): mean={skew_mean:.0f}ms "
                 f"range=[{skew_min}, {skew_max}]ms")
        progress(f"  oc exec RTT: mean={rtt_mean:.0f}ms (limits skew precision)")
        skew_uncertainty = max(abs(skew_max - skew_mean), abs(skew_mean - skew_min))
        if skew_uncertainty > 500:
            progress(f"  WARNING: Clock skew uncertainty ({skew_uncertainty:.0f}ms) exceeds 500ms.")
            progress(f"  Detection gap measurements (4h) will have ±{skew_uncertainty:.0f}ms error.")
        else:
            progress(f"  Clock skew uncertainty: ±{skew_uncertainty:.0f}ms (acceptable)")
        _record(writer, "calibration", "clock-skew", skew_mean, skew_uncertainty, "n/a",
                f"skew={skew_mean:.0f}ms ±{skew_uncertainty:.0f}ms rtt={rtt_mean:.0f}ms")
    else:
        skew_mean = 0.0
        skew_uncertainty = 0.0
        progress("  WARNING: Could not measure clock skew")
        _record(writer, "calibration", "clock-skew", 0, 0, "n/a", "measurement failed")

    # 3. Measure probe interval jitter
    #    Run a 5s probe, check actual intervals vs requested 200ms
    progress("")
    progress("  Measuring probe interval jitter (5s at 200ms)...")
    jitter_probe = start_probe(DECODE1_URL, interval=0.2, duration=5)
    try:
        jitter_probe.wait(timeout=10)
    except subprocess.TimeoutExpired:
        stop_probe(jitter_probe)
    jitter_stdout = jitter_probe.stdout.read() if jitter_probe.stdout else ""
    probe_epochs = []
    for line in jitter_stdout.splitlines():
        try:
            d = json.loads(line.strip())
            probe_epochs.append(d.get("epoch_ms", 0))
        except (json.JSONDecodeError, ValueError):
            pass

    if len(probe_epochs) >= 3:
        intervals = [probe_epochs[i+1] - probe_epochs[i]
                     for i in range(len(probe_epochs) - 1)]
        int_mean = sum(intervals) / len(intervals)
        int_min = min(intervals)
        int_max = max(intervals)
        int_stddev = math.sqrt(sum((x - int_mean)**2 for x in intervals) / max(len(intervals) - 1, 1))
        progress(f"  Probe intervals: mean={int_mean:.0f}ms min={int_min}ms max={int_max}ms "
                 f"stddev={int_stddev:.0f}ms (requested=200ms)")
        progress(f"  Probe count: {len(probe_epochs)} in 5s "
                 f"(expected ~{5000//200}, actual rate = {len(probe_epochs)/5:.1f}/s)")
        if int_max > 500:
            progress(f"  WARNING: Max interval {int_max}ms >> 200ms requested. "
                     f"Detection timing has at least ±{int_max}ms resolution.")
        _record(writer, "calibration", "probe-jitter", int_mean, int_max, "n/a",
                f"mean={int_mean:.0f}ms max={int_max}ms stddev={int_stddev:.0f}ms "
                f"n={len(probe_epochs)}")
        probe_interval_max = float(int_max)
    else:
        progress(f"  WARNING: Only {len(probe_epochs)} probes in 5s — probe may be too slow")
        _record(writer, "calibration", "probe-jitter", 0, 0, "n/a",
                f"insufficient data ({len(probe_epochs)} probes)")
        probe_interval_max = 0.0

    # Total detection gap uncertainty: clock skew + probe interval are independent
    # error sources. The detection gap is at most off by their sum.
    detection_uncertainty = skew_uncertainty + probe_interval_max
    progress("")
    progress("  Calibration complete. Systematic error budget:")
    progress(f"    Clock skew: ±{skew_uncertainty:.0f}ms (affects detection gap measurements)")
    progress(f"    Probe resolution: ±{probe_interval_max if probe_interval_max else '?'}ms "
             f"(limits timing precision)")
    progress(f"    oc exec overhead: ~{oc_mean:.0f}ms "
             f"(affects baseline measurement cadence, not TTFT accuracy)")
    progress(f"    Detection gap uncertainty: ±{detection_uncertainty:.0f}ms "
             f"(skew + probe interval, worst case)")
    if detection_uncertainty > 2000:
        progress(f"    WARNING: Detection gap uncertainty >{detection_uncertainty:.0f}ms — "
                 f"sub-second keepalive detection (4h) cannot be resolved")
    progress("")

    return Calibration(
        oc_exec_mean_ms=oc_mean,
        oc_exec_max_ms=oc_max,
        clock_skew_mean_ms=skew_mean,
        clock_skew_uncertainty_ms=skew_uncertainty,
        probe_interval_max_ms=probe_interval_max,
    )


def run_baselines(writer: TypedCSVWriter) -> tuple:
    """Phase 0b: Baseline Characterization.

    Fermi rule: know what "normal" looks like before you perturb.
    Returns (baseline_d1, baseline_d2) as frozen Baseline dataclasses.
    """
    progress("=== Phase 0b: Baseline Characterization ===")
    progress("  Establishing steady-state distribution (30 requests)...")
    progress("")

    baseline = run_baseline(DECODE1_URL, n=BASELINE_N, label="d1-baseline")
    print_baseline(baseline, "D1")
    for r in baseline.raw:
        _record(writer, "baseline", "d1", r.ttft, r.total, r.status, "baseline request", r.error)

    baseline_d2 = run_baseline(DECODE2_URL, n=BASELINE_N, label="d2-baseline")
    print_baseline(baseline_d2, "D2")
    for r in baseline_d2.raw:
        _record(writer, "baseline", "d2", r.ttft, r.total, r.status, "baseline request", r.error)

    progress("")
    progress("  Reference frame established:")
    progress(f"    D1: TTFT={baseline.ttft_mean}ms "
             f"95%CI=[{baseline.ttft_ci95[0]}, {baseline.ttft_ci95[1]}]ms "
             f"(p99={baseline.ttft_p99}ms)")
    progress(f"    D2: TTFT={baseline_d2.ttft_mean}ms "
             f"95%CI=[{baseline_d2.ttft_ci95[0]}, {baseline_d2.ttft_ci95[1]}]ms "
             f"(p99={baseline_d2.ttft_p99}ms)")
    if not baseline.stationary:
        progress("  WARNING: D1 baseline is non-stationary — system was drifting")
    if not baseline_d2.stationary:
        progress("  WARNING: D2 baseline is non-stationary — system was drifting")
    progress("  All subsequent measurements are deltas from this baseline.")
    progress("")

    return (baseline, baseline_d2)


def run_control(ctx: ExperimentContext) -> None:
    """Phase 0c: Control Run (null experiment).

    Fermi rule: run the measurement without the perturbation.
    Runs 20s probe + 15s load with no faults to isolate measurement artifacts.
    """
    progress("=== Phase 0c: Control Run (no fault injection) ===")
    progress("  Running 20s probe + 15s load with no faults.")
    progress("  This isolates measurement artifacts from fault effects.")
    progress("")

    # Probe control: 20s of probing, nothing should fail
    control_probe = start_probe(DECODE1_URL, interval=0.2, duration=20)
    try:
        control_probe.wait(timeout=25)
    except subprocess.TimeoutExpired:
        stop_probe(control_probe)
    control_stdout = control_probe.stdout.read() if control_probe.stdout else ""
    control_ok = 0
    control_fail = 0
    for line in control_stdout.splitlines():
        try:
            d = json.loads(line.strip())
            status = "200" if d.get("status") == 200 else str(d.get("status", 0))
            ctx.record("control", "probe", d.get("ttft_ms", 0), d.get("total_ms", 0),
                       status, "control probe", d.get("error", ""))
            if d.get("status") == 200:
                control_ok += 1
            else:
                control_fail += 1
        except (json.JSONDecodeError, ValueError):
            pass
    progress(f"  Control probe: {control_ok} OK, {control_fail} failed")
    if control_fail > 0:
        progress(f"  WARNING: {control_fail} failures with NO fault injection — "
                 f"measurement artifact or unstable system")

    # Load control: 15s of load, nothing should fail
    control_load_proc, _control_load_path = start_load(
        DECODE1_URL, qps=LOAD_QPS, duration=15, output_name="control-load.csv")
    try:
        control_load_proc.wait(timeout=25)
    except subprocess.TimeoutExpired:
        control_load_proc.terminate()
    control_load_summary = control_load_proc.stdout.read().strip()
    if control_load_summary:
        try:
            cs = json.loads(control_load_summary)
            progress(f"  Control load: {cs.get('total_requests', 0)} requests, "
                     f"{cs.get('ok', 0)} OK, {cs.get('failed', 0)} failed")
            if cs.get("failed", 0) > 0:
                progress("  WARNING: failures under load with NO fault — "
                         "system may be at capacity")
        except json.JSONDecodeError:
            pass
    progress("")


def run_exp_4a(ctx: ExperimentContext) -> None:
    """4a: Decode pod failure."""
    progress("=== 4a: Decode pod failure ===")
    predict("4a",
            "Decode-2 unaffected. Decode-1 fails immediately, recovers in 60-120s (GPU init).",
            "Decode pods are independent — killing one shouldn't affect the other. "
            "Recovery = new pod schedule + GPU init + NIXL handshake.")
    progress("")

    progress("  Pre-flight: verify both decode pods healthy")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4a", "pre-d1", t, tot, code, "decode-1 before kill", err)
    t, tot, code, err = send_via_test_client(DECODE2_URL)
    ctx.record("4a", "pre-d2", t, tot, code, "decode-2 before kill", err)

    # Capture pre-kill logs from decode-1 (vllm + sidecar)
    capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER, SIDECAR_CONTAINER],
                     "4a-decode1-pre", ctx.log_dir)

    progress("")
    progress("  Killing decode-1 pod...")
    decode1_pod = get_pod_name(DECODE1_SELECTOR)
    out, _ = oc("delete", "pod", decode1_pod, "-n", NS,
                "--grace-period=0", "--force")
    ctx.record("4a", "kill", 0, 0, "n/a", f"killed {decode1_pod}")
    progress(f"  {out}")
    progress("")

    progress("  Immediate request to decode-2 (should still work):")
    t, tot, code, err = send_via_test_client(DECODE2_URL)
    ctx.record("4a", "during-d2", t, tot, code, "decode-2 while decode-1 dead", err)

    progress("  Immediate request to decode-1 (should fail or timeout):")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4a", "during-d1", t, tot, code, "decode-1 just killed", err)

    progress("")
    progress("  Waiting for decode-1 replacement...")
    recovery_start = time.time()
    wait_for_ready(DECODE1_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Recovery time: {recovery_time}s")

    progress("  Post-recovery request to decode-1:")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4a", "post-d1", t, tot, code,
               f"decode-1 after recovery ({recovery_time}s)", err)

    # Check new engine_id
    new_decode1 = get_pod_name(DECODE1_SELECTOR)
    progress(f"  Old pod: {decode1_pod} -> New pod: {new_decode1}")
    engine_line = "(engine_id not found in logs)"
    out, _ = oc("logs", new_decode1, "-c", VLLM_CONTAINER, "-n", NS)
    for line in out.splitlines():
        if "engine_id:" in line:
            engine_line = line
    progress(f"  {engine_line}")

    # Capture post-recovery logs — shows NIXL handshake re-establishment
    capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER, SIDECAR_CONTAINER],
                     "4a-decode1-post", ctx.log_dir)

    # Verify warm-up: send a second request to see if cold start clears
    progress("  Second request (should be at steady-state):")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4a", "post-d1-warm", t, tot, code,
               "decode-1 second request after recovery", err)
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4b")


def run_exp_4b(ctx: ExperimentContext) -> None:
    """4b: Prefill pod failure."""
    progress("=== 4b: Prefill pod failure ===")
    if ROUTING_MODE == "epp":
        predict("4b",
                "Requests may fail during endpoint discovery, then use decode-only "
                "serving until a replacement prefill endpoint is available.",
                "EPP can schedule decode without prefill. Successful responses alone "
                "do not establish that KV transfer has resumed.")
    else:
        predict("4b",
                "Both decode pods fail immediately (prefill unavailable for KV transfer). "
                f"Recovery in 60-120s. Post-recovery TTFT within 2x of baseline ({ctx.baseline_d1.ttft_mean}ms).",
                "Prefill is shared — its death affects all decode pods. Recovery = "
                "new prefill pod + GPU init + ZMQ re-discovery by decode sidecars.")
    progress("")

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4b", "pre", t, tot, code, "before prefill kill", err)

    # Capture pre-kill logs from prefill and both decode sidecars
    capture_pod_logs(PREFILL_SELECTOR, [VLLM_CONTAINER],
                     "4b-prefill-pre", ctx.log_dir)
    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER],
                     "4b-decode1-pre", ctx.log_dir)

    progress("")
    progress("  Killing prefill pod...")
    prefill_pod = get_pod_name(PREFILL_SELECTOR)
    out, _ = oc("delete", "pod", prefill_pod, "-n", NS,
                "--grace-period=0", "--force")
    ctx.record("4b", "kill", 0, 0, "n/a", f"killed {prefill_pod}")
    progress(f"  {out}")
    progress("")

    route_label = "EPP route" if ROUTING_MODE == "epp" else "decode-1 sidecar"
    progress(f"  Immediate request through {route_label} (prefill down):")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4b", "during", t, tot, code, "prefill dead", err)

    progress("")
    progress("  Waiting for prefill replacement...")
    recovery_start = time.time()
    wait_for_ready(PREFILL_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Recovery time: {recovery_time}s")

    new_prefill = get_pod_name(PREFILL_SELECTOR)
    progress(f"  Old pod: {prefill_pod} -> New pod: {new_prefill}")
    out, _ = oc("get", "pod", new_prefill, "-n", NS,
                "-o", "jsonpath={.status.podIP}")
    progress(f"  New prefill IP: {out}")

    progress("  Post-recovery requests:")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4b", "post-1", t, tot, code, "first request after prefill recovery", err)
    t, tot, code, err = send_via_test_client(DECODE2_URL)
    ctx.record("4b", "post-2", t, tot, code, "decode-2 after prefill recovery", err)

    # Capture post-recovery logs — shows ZMQ discovery of new prefill
    capture_pod_logs(PREFILL_SELECTOR, [VLLM_CONTAINER],
                     "4b-prefill-post", ctx.log_dir)
    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER, VLLM_CONTAINER],
                     "4b-decode1-post", ctx.log_dir)

    # Parse sidecar logs for fallback events (decode served locally)
    sidecar_log = os.path.join(ctx.log_dir, "4b-decode1-post-routing-sidecar.log")
    try:
        with open(sidecar_log) as f:
            fallbacks = parse_sidecar_fallbacks(f.read())
        ctx.record("4b", "fallback-count", 0, 0, "n/a",
                   f"sidecar fallback events: {len(fallbacks)}")
        if fallbacks:
            progress(f"  Sidecar fallbacks detected: {len(fallbacks)} "
                     f"(confirms decode served locally while prefill was down)")
    except OSError:
        pass

    # Second requests to check warm-up
    progress("  Second requests (steady-state check):")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4b", "post-1-warm", t, tot, code, "decode-1 second request", err)
    t, tot, code, err = send_via_test_client(DECODE2_URL)
    ctx.record("4b", "post-2-warm", t, tot, code, "decode-2 second request", err)
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4d")


def run_exp_4d(ctx: ExperimentContext) -> None:
    """4d: Graceful degradation (3 GPU -> 2 GPU -> 3 GPU)."""
    progress("=== 4d: Graceful degradation (3 GPU -> 2 GPU -> 3 GPU) ===")
    predict("4d",
            f"Decode-1 TTFT unchanged (~{ctx.baseline_d1.ttft_mean}ms) with decode-2 down. "
            f"Scale-up recovery 60-120s. No request failures.",
            "Decode-1 and decode-2 are independent workers behind separate services. "
            "Removing decode-2 just reduces capacity, not per-request latency.")
    progress("")

    progress(f"  Scaling {DECODE2_DEPLOY} to 0...")
    oc("scale", WORKLOAD_TYPE, DECODE2_DEPLOY, "--replicas=0", "-n", NS)
    time.sleep(5)

    progress("  Sending 5 requests through decode-1 (should all work):")
    for i in range(1, 6):
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        ctx.record("4d", f"2gpu-{i}", t, tot, code,
                   f"decode-1 only ({DECODE2_DEPLOY} scaled down)", err)

    progress("")
    progress(f"  Scaling {DECODE2_DEPLOY} back to 1...")
    oc("scale", WORKLOAD_TYPE, DECODE2_DEPLOY, "--replicas=1", "-n", NS)
    recovery_start = time.time()
    wait_for_ready(DECODE2_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Scale-up time: {recovery_time}s")

    progress("  Request to restored decode-2:")
    t, tot, code, err = send_via_test_client(DECODE2_URL)
    ctx.record("4d", "restored", t, tot, code,
               f"decode-2 after scale-up ({recovery_time}s)", err)

    # Second request to check warm-up
    t, tot, code, err = send_via_test_client(DECODE2_URL)
    ctx.record("4d", "restored-warm", t, tot, code,
               "decode-2 second request after scale-up", err)
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4e")


def run_exp_4e(ctx: ExperimentContext) -> None:
    """4e: Failure under load."""
    progress(f"=== 4e: Failure under load ({LOAD_QPS} QPS, kill at T+{KILL_AT_4E}s) ===")
    predict("4e",
            f"Requests in-flight at kill time fail. Requests before kill succeed. "
            f"~{KILL_AT_4E * LOAD_QPS} OK requests before kill, then failures until recovery.",
            "In-pod load driver sends real concurrent requests. Kill happens mid-stream. "
            "Requests that already completed prefill should succeed; those mid-transfer fail.")
    progress("")

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4e", "pre", t, tot, code, "before load test", err)

    # Start in-pod load driver (replaces old ThreadPoolExecutor + oc exec approach)
    progress(f"  Starting in-pod load driver ({LOAD_QPS} QPS, {LOAD_DURATION_4E}s)...")
    load_proc_4e, load_path_4e = start_load(
        DECODE1_URL, qps=LOAD_QPS, duration=LOAD_DURATION_4E,
        output_name="4e-load-results.csv")

    # Kill prefill after delay
    progress(f"  Waiting {KILL_AT_4E}s before killing prefill...")
    time.sleep(KILL_AT_4E)

    prefill_pod = get_pod_name(PREFILL_SELECTOR)
    kill_epoch_4e = ctx.epoch_ms()
    oc("delete", "pod", prefill_pod, "-n", NS,
       "--grace-period=0", "--force")
    ctx.record("4e", "kill", 0, 0, "n/a",
               f"killed {prefill_pod}", "", detect_epoch_ms=kill_epoch_4e)
    progress(f"  [T+{KILL_AT_4E}s] Killed prefill: {prefill_pod}")

    # Wait for load to finish
    progress("  Waiting for load driver to complete...")
    try:
        load_proc_4e.wait(timeout=LOAD_DURATION_4E + 30)
    except subprocess.TimeoutExpired:
        load_proc_4e.terminate()

    load_summary_4e = load_proc_4e.stdout.read().strip()
    if load_summary_4e:
        try:
            summary = json.loads(load_summary_4e)
            progress(f"  Load results: {summary.get('total_requests', 0)} requests, "
                     f"{summary.get('ok', 0)} OK, {summary.get('failed', 0)} failed")
        except json.JSONDecodeError:
            pass

    # Capture sidecar logs
    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER],
                     "4e-decode1-during", ctx.log_dir)

    # Wait for prefill recovery
    progress("")
    progress("  Waiting for prefill recovery...")
    recovery_start = time.time()
    wait_for_ready(PREFILL_DEPLOY)
    recovery_time = int(time.time() - recovery_start)
    progress(f"  Recovery time: {recovery_time}s")

    # Pull per-request data and partition by kill epoch.
    # Note: r_epoch is pod clock, kill_epoch_4e is local clock. The clock skew
    # (measured in calibration) makes this boundary approximate — requests within
    # ~skew_uncertainty of the kill time may be misclassified. This is acceptable
    # for the aggregate count; individual requests near the boundary are ambiguous.
    load_results_4e = collect_load_results(load_path_4e)
    pre_kill_ok = 0
    for r in load_results_4e:
        r_epoch = int(r.get("epoch_ms", 0))
        during_fault = 1 if kill_epoch_4e and r_epoch > kill_epoch_4e else 0
        if not during_fault and str(r.get("status", 0)) == "200":
            pre_kill_ok += 1
        ctx.record("4e", f"req-{r.get('seq', 0)}",
                   float(r.get("ttft_ms", 0)), float(r.get("total_ms", 0)),
                   str(r.get("status", 0)),
                   f"during_fault={during_fault} elapsed={r.get('elapsed_s', 0)}s",
                   r.get("error", ""),
                   itl_mean_ms=r.get("itl_mean_ms", ""),
                   itl_p99_ms=r.get("itl_p99_ms", ""),
                   token_count=r.get("token_count", ""))

    evaluate("4e", KILL_AT_4E * LOAD_QPS, pre_kill_ok, " OK requests before kill")

    progress("  Post-recovery request:")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4e", "post", t, tot, code,
               f"after prefill recovery ({recovery_time}s)", err)
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4f")


def run_exp_4f(ctx: ExperimentContext) -> None:
    """4f: Network partition with duration sweep.

    Tests partition durations of 5s, 30s, 120s (configurable via
    PARTITION_DURATIONS) to find the UCX reconnection timeout threshold.
    For each duration: apply partition, probe during, remove, probe recovery.
    """
    mode_label = "ISOLATED" if PARTITION_ISOLATED else "sequential"
    progress(f"=== 4f: Network partition (duration sweep, {mode_label}) ===")
    predict("4f",
            f"Short partitions ({PARTITION_DURATIONS[0]}s) may self-heal. "
            f"Longer partitions trigger sticky UCX failure requiring pod restart. "
            f"The threshold reveals UCX's internal reconnection timeout.",
            "UCX/TCP connections don't fail on dropped packets — they retry for "
            "tcp_retries2 * RTO (~13-30s). NetworkPolicy drops packets silently. "
            "After removal, UCX may or may not reconnect depending on whether "
            "internal state was torn down during the partition.")
    progress("")

    sweep_results = []  # collect (nominal_s, actual_s, recovered, recovery_s, mechanism)

    for dur_i, duration_s in enumerate(PARTITION_DURATIONS):
        progress(f"  --- Partition duration: {duration_s}s ---")

        # In isolated mode, restart the decode pod before each test to prevent
        # cumulative stress from prior partitions confounding results (C8).
        if PARTITION_ISOLATED and dur_i > 0:
            progress("  ISOLATED: restarting decode-1 for clean state...")
            decode1_pod = get_pod_name(DECODE1_SELECTOR)
            oc("delete", "pod", decode1_pod, "-n", NS,
               "--grace-period=0", "--force")
            wait_for_ready(DECODE1_DEPLOY)
            time.sleep(10)
            # Wait for NIXL reconnection
            progress("  ISOLATED: waiting for NIXL re-establishment...")
            time.sleep(15)
            ctx.record("4f", f"isolated-restart-{duration_s}s", 0, 0, "n/a",
                       "pod restarted for isolated partition test")

        progress("  Pre-flight: verify disagg works")
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        ctx.record("4f", f"pre-{duration_s}s", t, tot, code,
                   f"before {duration_s}s partition", err)
        if code != "200":
            progress(f"  SKIP: system not healthy (code={code})")
            continue

        # Transport state before fault — the interior view
        tcp_before = snapshot_transport_state(
            DECODE1_SELECTOR, VLLM_CONTAINER,
            f"4f-pre-{duration_s}s", ctx.log_dir)

        progress(f"  Applying network partition: {NETPOLICY_FILE}")
        apply_epoch = ctx.epoch_ms()
        _out, err_str = oc("apply", "-f", NETPOLICY_FILE, "-n", NS)
        ctx.record("4f", f"partition-on-{duration_s}s", 0, 0, "n/a",
                   f"partition applied for {duration_s}s",
                   detect_epoch_ms=apply_epoch)
        partition_epoch = apply_epoch  # will be updated to verified time if possible
        if err_str:
            progress(f"  stderr: {err_str}")

        try:
            # Verify partition is felt by the application rather than assuming
            # a fixed delay. This measures when the sidecar's request to prefill
            # first fails — which depends on both NetworkPolicy propagation AND
            # whether the sidecar's existing TCP connection has timed out.
            # For short-lived connections this approximates policy propagation;
            # for persistent connections it may lag by up to tcp_retries2 timeout.
            progress("  Verifying partition onset...")
            partition_verified = False
            for vi in range(6):  # up to 6s to detect
                time.sleep(1)
                t, tot, code, err = send_via_test_client(
                    DECODE1_URL, curl_timeout=2, subprocess_timeout=5)
                if code != "200":
                    partition_epoch = ctx.epoch_ms()  # update to verified time
                    propagation_ms = partition_epoch - apply_epoch
                    partition_verified = True
                    progress(f"  Partition verified after {vi + 1}s "
                             f"(propagation ~{propagation_ms}ms, code={code})")
                    ctx.record("4f", f"partition-verified-{duration_s}s", 0, 0,
                               "n/a",
                               f"propagation_delay={propagation_ms}ms",
                               detect_epoch_ms=partition_epoch)
                    break
            if not partition_verified:
                progress("  WARNING: partition not detected after 6s — "
                         "NetworkPolicy may not have propagated")

            # Probe during partition — use short timeout (3s) to minimize
            # probe inflation. Previously 10s default caused actual partition
            # durations to be 1.4-2.5x the nominal value.
            probe_timeout = 3
            n_probes = max(1, (duration_s - 3) // 5)
            for pi in range(n_probes):
                t, tot, code, err = send_via_test_client(
                    DECODE1_URL, curl_timeout=probe_timeout,
                    subprocess_timeout=probe_timeout + 5)
                elapsed = (ctx.epoch_ms() - partition_epoch) / 1000
                ctx.record("4f", f"during-{duration_s}s-{pi+1}", t, tot, code,
                           f"partition active ({elapsed:.0f}s elapsed)", err)
                progress(f"  [{elapsed:.0f}s] code={code} ttft={t}ms err={err[:60] if err else 'none'}")
                if pi < n_probes - 1:
                    time.sleep(5)

            # Wait for remaining duration
            elapsed_s = (ctx.epoch_ms() - partition_epoch) / 1000
            remaining = max(0, duration_s - elapsed_s)
            if remaining > 0:
                progress(f"  Waiting {remaining:.0f}s for partition to complete...")
                time.sleep(remaining)

        finally:
            progress(f"  Removing network partition after {duration_s}s...")
            remove_epoch = ctx.epoch_ms()
            actual_duration_s = (remove_epoch - partition_epoch) / 1000
            _out, err_str = oc("delete", "-f", NETPOLICY_FILE, "-n", NS)
            progress(f"  Actual partition duration: {actual_duration_s:.1f}s "
                     f"(nominal {duration_s}s)")
            ctx.record("4f", f"partition-off-{duration_s}s", 0, 0, "n/a",
                       f"partition removed nominal={duration_s}s "
                       f"actual={actual_duration_s:.1f}s",
                       recover_epoch_ms=remove_epoch)

        # Probe recovery at 1s, 5s, 15s, 30s after removal.
        # Each probe classifies the response mode (disagg vs fallback vs error)
        # so we distinguish liveness (system responds) from correctness
        # (system restored to disaggregated operation).
        recovery_delays = [1, 5, 15, 30]
        recovered = False  # liveness: got a 200
        recovered_disagg = False  # correctness: 200 via disagg path
        first_recovery_s = None
        first_recovery_ttft = None
        first_disagg_s = None
        last_fail_s = 0.0  # tracks last failing probe time for uncertainty window
        baseline_ttft = ctx.baseline_d1.ttft_mean
        for delay in recovery_delays:
            wait_since_remove = max(0, delay - (ctx.epoch_ms() - remove_epoch) / 1000)
            if wait_since_remove > 0:
                time.sleep(wait_since_remove)

            t, tot, code, err = send_via_test_client(DECODE1_URL)
            elapsed_from_remove = (ctx.epoch_ms() - remove_epoch) / 1000

            # Classify: liveness vs mode
            if code == "200" and t > 0:
                # Use TTFT to infer mode (sidecar logs not yet captured)
                mode = classify_response(t, code, err, baseline_ttft)
                status_label = f"responding ({mode})"
            else:
                mode = "error"
                status_label = "still_failing"

            ctx.record("4f", f"post-{duration_s}s-{delay}s", t, tot, code,
                       f"recovery probe +{elapsed_from_remove:.0f}s "
                       f"status={status_label}", err)
            progress(f"  [+{elapsed_from_remove:.0f}s] code={code} ttft={t}ms "
                     f"status={status_label}")
            if code == "200" and t > 0:
                if not recovered:
                    first_recovery_s = elapsed_from_remove
                    first_recovery_ttft = t
                    # Uncertainty: recovery happened between last_fail_s and now
                    recovery_lower_s = last_fail_s
                recovered = True
                if mode == "disagg" and not recovered_disagg:
                    first_disagg_s = elapsed_from_remove
                    recovered_disagg = True
            else:
                last_fail_s = elapsed_from_remove

        # Also probe decode-2 (should be unaffected)
        t, tot, code, err = send_via_test_client(DECODE2_URL)
        ctx.record("4f", f"post-d2-{duration_s}s", t, tot, code,
                   f"decode-2 after {duration_s}s partition", err)

        # Transport state after fault — diff reveals what changed inside
        tcp_after = snapshot_transport_state(
            DECODE1_SELECTOR, VLLM_CONTAINER,
            f"4f-post-{duration_s}s", ctx.log_dir)
        tcp_delta = diff_transport_state(tcp_before, tcp_after,
                                         f"4f-{duration_s}s")
        ctx.record("4f", f"tcp-delta-{duration_s}s", 0, 0, "n/a",
                   f"transport state change: {tcp_delta}")

        # Capture logs for this duration
        capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER, VLLM_CONTAINER],
                         f"4f-decode1-{duration_s}s", ctx.log_dir)
        capture_pod_logs(PREFILL_SELECTOR, [VLLM_CONTAINER],
                         f"4f-prefill-{duration_s}s", ctx.log_dir)

        # Parse sidecar logs for fallback events during this partition
        sidecar_log = os.path.join(ctx.log_dir,
                                   f"4f-decode1-{duration_s}s-{SIDECAR_CONTAINER}.log")
        fallback_count = 0
        if os.path.exists(sidecar_log):
            with open(sidecar_log) as f:
                fallbacks = parse_sidecar_fallbacks(f.read())
            # Count fallbacks within the partition window
            fallback_during = [fb for fb in fallbacks
                               if apply_epoch <= fb[0] <= remove_epoch]
            fallback_after = [fb for fb in fallbacks if fb[0] > remove_epoch]
            fallback_count = len(fallback_during)
            progress(f"  Sidecar fallbacks: {len(fallback_during)} during partition, "
                     f"{len(fallback_after)} after removal, "
                     f"{len(fallbacks)} total in log")
            ctx.record("4f", f"fallbacks-{duration_s}s", 0, 0, "n/a",
                       f"during={len(fallback_during)} after={len(fallback_after)} "
                       f"total={len(fallbacks)}")

        # Classify recovery mechanism using both TTFT and sidecar fallback data.
        # The definitive signal (sidecar fallback count) MUST take priority
        # over timing heuristics. A fast recovery with fallback events is
        # "fallback" (the sidecar served locally), not "transparent" (disagg
        # path survived the fault). Confusing these would claim resilience
        # the disagg path doesn't actually have.
        if recovered and first_recovery_s is not None and first_recovery_ttft is not None:
            if fallback_count > 0:
                # Sidecar logs confirm fallback — definitive, regardless of speed
                mechanism = "fallback"
            elif first_recovery_s < 3 and first_recovery_ttft < baseline_ttft * 2:
                # Fast recovery with no fallback events: disagg path survived
                mechanism = "transparent"
            elif first_recovery_ttft > baseline_ttft * 3:
                # No sidecar fallback logged but TTFT is elevated — may be
                # slow NIXL reconnection, not fallback. Label conservatively.
                mechanism = "slow_recovery"
            else:
                mechanism = "extended"
        else:
            mechanism = "sticky"

        sweep_results.append((duration_s, actual_duration_s, recovered,
                              first_recovery_s, mechanism,
                              recovery_lower_s if recovered else None))

        # Summary for this duration
        if recovered:
            disagg_note = ""
            if recovered_disagg and first_disagg_s is not None:
                if first_disagg_s == first_recovery_s:
                    disagg_note = " [disagg restored immediately]"
                else:
                    disagg_note = (f" [liveness at +{first_recovery_s:.0f}s, "
                                   f"disagg restored at +{first_disagg_s:.0f}s]")
            elif not recovered_disagg:
                disagg_note = " [liveness only — disagg not restored within probe window]"
            progress(f"  {duration_s}s partition: RECOVERED ({mechanism}) "
                     f"at +{first_recovery_s:.0f}s "
                     f"(actual: {recovery_lower_s:.0f}-{first_recovery_s:.0f}s), "
                     f"ttft={first_recovery_ttft}ms{disagg_note}")
        else:
            progress(f"  {duration_s}s partition: STICKY FAILURE — needs pod restart")
            # Restart decode to recover for next iteration
            progress("  Restarting decode-1 to recover...")
            decode1_pod = get_pod_name(DECODE1_SELECTOR)
            oc("delete", "pod", decode1_pod, "-n", NS,
               "--grace-period=0", "--force")
            wait_for_ready(DECODE1_DEPLOY)
            time.sleep(10)

        # Verify disaggregated mode is restored before next iteration.
        # After a sticky failure + pod restart, the new pod must re-establish
        # NIXL connections. If we don't verify, the next partition test starts
        # from a degraded (local-only) baseline.
        progress("  Verifying disaggregated mode restored...")
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        if code == "200" and t > ctx.baseline_d1.ttft_mean * 0.5:
            progress(f"  Disagg confirmed: ttft={t}ms (baseline={ctx.baseline_d1.ttft_mean:.0f}ms)")
        elif code == "200":
            progress(f"  WARNING: ttft={t}ms is low — may be local fallback "
                     f"(baseline={ctx.baseline_d1.ttft_mean:.0f}ms)")
            # Wait and retry — NIXL handshake may still be in progress
            time.sleep(15)
            t2, _, code2, _ = send_via_test_client(DECODE1_URL)
            progress(f"  Retry: ttft={t2}ms code={code2}")
        else:
            progress(f"  WARNING: health check failed (code={code})")

        ctx.record("4f", f"verify-{duration_s}s", t, tot, code,
                   f"disagg verification after {duration_s}s partition", err)
        progress("")

    # Summary table
    if sweep_results:
        progress("  === 4f Summary ===")
        progress(f"  {'Nominal':>8s}  {'Actual':>8s}  {'Recovered':>9s}  "
                 f"{'Recovery Window':>17s}  {'Mechanism'}")
        for nom, act, _rec, delay, mech, lower in sweep_results:  # type: ignore[assignment]
            if delay is not None and lower is not None:
                delay_str = f"{lower:.0f}-{delay:.0f}s"
            elif delay is not None:
                delay_str = f"{delay:.0f}s"
            else:
                delay_str = "n/a"
            progress(f"  {nom:>7d}s  {act:>7.1f}s  {'yes':>9s if rec else 'NO':>9s}  "
                     f"{delay_str:>17s}  {mech}")
        mode_label = "ISOLATED" if PARTITION_ISOLATED else "sequential"
        progress(f"  Mode: {mode_label}")
        ctx.finding("4f_sweep", sweep_results)
        ctx.finding("4f_mode", mode_label)
        progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4g")


def run_exp_4g(ctx: ExperimentContext) -> None:
    """4g: Slow network (tc netem)."""
    progress(f"=== 4g: Slow network (delay={NETEM_DELAY_MS}ms, loss={NETEM_LOSS_PCT}%) ===")
    predicted_ttft_increase = NETEM_DELAY_MS  # 1 RTT = 1x delay
    predict("4g",
            f"TTFT increases by ~{predicted_ttft_increase}-{predicted_ttft_increase * 3}ms "
            f"(1-3 RTTs in KV transfer). {NETEM_LOSS_PCT}% loss causes ~{NETEM_LOSS_PCT}% retry overhead.",
            f"If NIXL KV transfer is 1 RTT, TTFT += {NETEM_DELAY_MS}ms. If multiple "
            f"RTTs (handshake + transfer + ack), TTFT += N*{NETEM_DELAY_MS}ms. "
            f"The multiplier tells us the RTT count in the transfer protocol.")
    progress("")

    # Check if tc is available in the prefill pod
    tc_check, _ = oc("exec", get_pod_name(PREFILL_SELECTOR), "-c", VLLM_CONTAINER,
                      "-n", NS, "--", "which", "tc")
    if not tc_check.strip():
        progress("  SKIPPED: tc (iproute2) not available in prefill container.")
        progress("  To enable 4g, install iproute2 in the vLLM image or add a")
        progress("  privileged debug sidecar with NET_ADMIN capability.")
        ctx.record("4g", "skip", 0, 0, "n/a", "tc not available in container")
    else:
        progress("  Pre-flight: verify disagg works")
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        ctx.record("4g", "pre", t, tot, code, "before network degradation", err)

        prefill_pod = get_pod_name(PREFILL_SELECTOR)

        # Apply netem: add latency + packet loss to the prefill pod's egress.
        # This degrades NIXL KV transfers without completely blocking them —
        # a harder failure mode than a clean partition.
        progress(f"  Applying: tc netem delay {NETEM_DELAY_MS}ms loss {NETEM_LOSS_PCT}%")
        _out, err_str = oc("exec", prefill_pod, "-c", VLLM_CONTAINER, "-n", NS,
                          "--", "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
                          "delay", f"{NETEM_DELAY_MS}ms", "20ms", "distribution", "normal",
                          "loss", f"{NETEM_LOSS_PCT}%")
        ctx.record("4g", "netem-on", 0, 0, "n/a",
                   f"delay={NETEM_DELAY_MS}ms loss={NETEM_LOSS_PCT}%")
        if err_str:
            progress(f"  stderr: {err_str}")

        try:
            time.sleep(2)

            # Send requests under degraded network — measures how NIXL handles
            # slow/lossy connections vs clean failures
            progress("  Requests during network degradation:")
            degraded_ttfts = []
            for i in range(1, 6):
                t, tot, code, err = send_via_test_client(DECODE1_URL, curl_timeout=30,
                                                          subprocess_timeout=60)
                ctx.record("4g", f"during-{i}", t, tot, code,
                           f"request {i}/5 under degraded network", err)
                if code == "200" and t > 0:
                    degraded_ttfts.append(t)

            if degraded_ttfts:
                avg_degraded = sum(degraded_ttfts) / len(degraded_ttfts)
                ttft_increase = avg_degraded - ctx.baseline_d1.ttft_mean
                progress(f"  Degraded TTFT: avg={avg_degraded:.0f}ms "
                         f"(+{ttft_increase:.0f}ms vs baseline)")
                evaluate("4g", predicted_ttft_increase, ttft_increase,
                         f"ms TTFT increase (implies ~{ttft_increase/NETEM_DELAY_MS:.1f} RTTs)")

            progress("  Request to decode-2 (also affected by prefill degradation):")
            t, tot, code, err = send_via_test_client(DECODE2_URL, curl_timeout=30,
                                                      subprocess_timeout=60)
            ctx.record("4g", "during-d2", t, tot, code,
                       "decode-2 under degraded network", err)

            # Capture logs during degradation
            capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER],
                             "4g-decode1-during", ctx.log_dir)

        finally:
            # Always remove netem rules
            progress("")
            progress("  Removing network degradation...")
            oc("exec", prefill_pod, "-c", VLLM_CONTAINER, "-n", NS,
               "--", "tc", "qdisc", "del", "dev", "eth0", "root")
            ctx.record("4g", "netem-off", 0, 0, "n/a", "network degradation removed")

        time.sleep(2)

        progress("  Post-recovery requests:")
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        ctx.record("4g", "post-1", t, tot, code, "after removing degradation", err)
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        ctx.record("4g", "post-2", t, tot, code, "steady-state after degradation", err)

        capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER, VLLM_CONTAINER],
                         "4g-decode1-post", ctx.log_dir)
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4h")


def run_exp_4h(ctx: ExperimentContext) -> None:
    """4h: UCX keepalive characterization."""
    progress("=== 4h: UCX keepalive characterization ===")
    predict("4h",
            "Kill-to-detection gap of ~5-10s (UCX TCP keepalive default). "
            "If much shorter (<1s), NIXL has active health checks. "
            "If much longer (>30s), keepalive may be disabled.",
            "UCX over TCP relies on OS-level TCP keepalive for dead-peer detection. "
            "Default keepalive_time is 7200s (2h), but UCX typically overrides to ~5s. "
            "NIXL may add its own heartbeat on top. Multiple runs reveal variance.")
    progress("")

    skew_mean = ctx.calibration.clock_skew_mean_ms
    skew_uncertainty = ctx.calibration.clock_skew_uncertainty_ms
    probe_max = ctx.calibration.probe_interval_max_ms
    # Total uncertainty for in-pod probe detection gap = clock skew + probe interval
    total_uncertainty = skew_uncertainty + probe_max

    detection_gaps = []  # collect across runs for summary (method A: in-pod)
    detection_gaps_local = []  # method B: local probe (zero skew)
    detection_gaps_log = []  # method C: log-based (zero skew)

    # Check if decode service is reachable locally for method B
    local_probe_available = False
    try:
        r = send_disagg(DECODE1_URL, prompt="Hello", max_tokens=1)
        if r.status == 200:
            local_probe_available = True
            progress("  Local probe available (direct access to decode service)")
        else:
            progress(f"  Local probe unavailable (status={r.status}), "
                     f"using in-pod probe only")
    except Exception as e:
        progress(f"  Local probe unavailable ({e}), using in-pod probe only")

    for run_i in range(1, KEEPALIVE_RUNS + 1):
        progress(f"  --- Run {run_i}/{KEEPALIVE_RUNS} ---")

        progress("  Pre-flight: verify disagg works")
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        ctx.record("4h", f"pre-{run_i}", t, tot, code, f"baseline run {run_i}", err)

        if code != "200":
            progress("  SKIP: baseline failed, cannot measure keepalive")
            continue

        # Capture pre-kill decode logs to diff later
        capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER],
                         f"4h-decode1-pre-{run_i}", ctx.log_dir)

        # Start probe FIRST — establish steady-state before kill
        progress("  Starting in-pod probe (200ms intervals)...")
        probe_proc = start_probe(DECODE1_URL, interval=0.2, duration=180)

        # Start local probe if available (50ms interval, zero clock skew)
        local_stop = threading.Event()
        local_thread, local_results = None, []
        if local_probe_available:
            progress("  Starting local probe (50ms intervals, zero clock skew)...")
            local_thread, local_results = start_local_probe(
                DECODE1_URL, interval=0.05, stop_event=local_stop)

        # Let baseline probes establish (verify system is working)
        progress("  Establishing baseline probes (2s)...")
        time.sleep(2)

        # Get kill timestamp in BOTH clock domains
        prefill_pod = get_pod_name(PREFILL_SELECTOR)
        decode1_pod = get_pod_name(DECODE1_SELECTOR)

        # Pod-clock timestamp for log-based detection (method C, zero skew)
        pod_time_str, _ = oc("exec", decode1_pod, "-c", VLLM_CONTAINER,
                              "-n", NS, "--", "date", "+%s%3N")
        kill_epoch_pod = 0
        try:
            kill_epoch_pod = int(pod_time_str.strip())
        except (ValueError, AttributeError):
            progress("  WARNING: could not get pod-clock timestamp")

        # NOW kill — single kill, properly timed
        progress(f"  Killing prefill {prefill_pod} (SIGKILL, no graceful shutdown)...")
        kill_epoch = ctx.epoch_ms()  # local clock (method A)
        oc("delete", "pod", prefill_pod, "-n", NS,
           "--grace-period=0", "--force")
        ctx.record("4h", f"kill-{run_i}", 0, 0, "n/a",
                   f"killed {prefill_pod}", "",
                   detect_epoch_ms=kill_epoch)

        # Read probe output until failure is detected
        progress("  Waiting for failure detection...")
        fail_epoch, fail_probes, fail_err, fail_data = read_probe_until_fail(
            probe_proc, timeout=120)

        if fail_epoch:
            # Method A: in-pod probe (cross-clock, skew-corrected)
            raw_gap_ms = fail_epoch - kill_epoch
            detect_gap_ms = raw_gap_ms - int(skew_mean)
            detection_gaps.append(detect_gap_ms)
            progress(f"  [Method A: in-pod] Failure after {fail_probes} probes "
                     f"({detect_gap_ms}ms ±{total_uncertainty:.0f}ms, "
                     f"raw={raw_gap_ms}ms, skew={int(skew_mean)}ms): "
                     f"{fail_err}")
        else:
            detect_gap_ms = 0
            progress(f"  [Method A: in-pod] No failure detected after "
                     f"{fail_probes} probes (timeout)")

        ctx.record("4h", f"detect-A-{run_i}", 0, 0, "n/a",
                   f"method=in-pod gap={detect_gap_ms}ms "
                   f"±{total_uncertainty:.0f}ms probes={fail_probes}",
                   fail_err,
                   detect_epoch_ms=fail_epoch, probes_to_detect=fail_probes)

        # Method B: local probe (zero clock skew)
        # Always stop local probe before processing results
        local_stop.set()
        if local_thread:
            local_thread.join(timeout=5)
        if local_probe_available and local_results:
            # Find first failure in local results after kill
            local_fails = [r for r in local_results
                           if r["epoch_ms"] > kill_epoch and r["status"] != 200]
            if local_fails:
                local_gap = local_fails[0]["epoch_ms"] - kill_epoch
                # Count successful probes between kill and first failure
                local_ok_after_kill = sum(
                    1 for r in local_results
                    if kill_epoch < r["epoch_ms"] < local_fails[0]["epoch_ms"]
                    and r["status"] == 200)
                detection_gaps_local.append(local_gap)
                progress(f"  [Method B: local] Failure at +{local_gap}ms "
                         f"(zero skew, {local_ok_after_kill} OK probes "
                         f"between kill and detection)")
            else:
                local_gap = 0
                progress(f"  [Method B: local] No failure detected in "
                         f"{len(local_results)} probes")
            ctx.record("4h", f"detect-B-{run_i}", 0, 0, "n/a",
                       f"method=local gap={local_gap}ms "
                       f"probes={len(local_results)} (zero skew)",
                       "")

        # Record individual probe data points for fine-grained analysis
        for p in fail_data:
            ctx.record("4h", f"probe-{run_i}", p.get("ttft_ms", 0),
                       p.get("total_ms", 0), str(p.get("status", 0)),
                       f"probe seq={p.get('seq', 0)}", p.get("error", ""),
                       detect_epoch_ms=p.get("epoch_ms", 0))

        # Capture post-kill decode logs — UCX/NIXL error entries
        capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER],
                         f"4h-decode1-post-{run_i}", ctx.log_dir)

        # Method C: log-based detection (pod-clock kill timestamp, zero skew)
        log_path = os.path.join(ctx.log_dir, f"4h-decode1-post-{run_i}-vllm.log")
        log_events = parse_log_timestamps(log_path)
        ucx_events = [(ts, cat, line) for ts, cat, line in log_events
                      if cat in ("ucx_error", "nixl_error")]
        if ucx_events:
            first_ts = ucx_events[0][0]
            if first_ts and kill_epoch_pod:
                # Both timestamps are in pod's clock domain — zero skew
                log_gap = first_ts - kill_epoch_pod
                detection_gaps_log.append(log_gap)
                progress(f"  [Method C: logs] First UCX/NIXL event: +{log_gap}ms "
                         f"(zero skew, pod-clock kill timestamp)")
            elif first_ts and kill_epoch:
                # Fallback to cross-clock if pod timestamp unavailable
                log_gap = first_ts - kill_epoch - int(skew_mean)
                progress(f"  [Method C: logs] First UCX/NIXL event: {log_gap}ms "
                         f"(skew-corrected, ±{skew_uncertainty:.0f}ms)")
            else:
                log_gap = 0
            ctx.record("4h", f"detect-C-{run_i}", 0, 0, "n/a",
                       f"method=logs gap={log_gap}ms "
                       f"{'zero_skew' if kill_epoch_pod else 'skew_corrected'}",
                       ucx_events[0][2][:200],
                       detect_epoch_ms=first_ts)

        # Wait for prefill recovery before probing for recovery
        progress("  Waiting for prefill recovery...")
        wait_for_ready(PREFILL_DEPLOY)

        # Continue reading probe for recovery (same probe process)
        progress("  Waiting for recovery detection...")
        recover_epoch, recover_probes, recover_lat, _recover_data = \
            read_probe_until_recover(probe_proc, timeout=120)

        stop_probe(probe_proc)

        if recover_epoch:
            # Apply same clock skew correction as detection gap:
            # recover_epoch is pod clock, kill_epoch is local clock
            recover_gap = recover_epoch - kill_epoch - int(skew_mean)
            progress(f"  Recovered after {recover_probes} probes "
                     f"({recover_gap}ms from kill, skew-corrected), "
                     f"latency={recover_lat}ms")
        else:
            recover_gap = 0
            progress(f"  Recovery not confirmed after {recover_probes} probes")

        ctx.record("4h", f"recover-{run_i}", recover_lat, recover_lat, "200",
                   f"recovery gap={recover_gap}ms probes={recover_probes}",
                   "",
                   recover_epoch_ms=recover_epoch, probes_to_recover=recover_probes)

        # Settle before next run
        wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label=f"4h-settle-{run_i}")

    # Cross-run summary — report all detection methods
    progress(f"  4h SUMMARY: {KEEPALIVE_RUNS} runs, 3 detection methods:")

    # Per-method uncertainty (ms). These are used for agreement checking.
    method_uncertainty = {
        "A": total_uncertainty,  # clock skew + probe interval
        "B": 50.0,              # probe interval only
        "C": 10.0,              # log flush latency only
    }

    def _summarize_gaps(name, method_key, gaps):
        if not gaps:
            progress(f"    {name}: no data")
            return None
        avg = sum(gaps) / len(gaps)
        mn, mx = min(gaps), max(gaps)
        unc = method_uncertainty[method_key]
        progress(f"    {name}: avg={avg:.0f}ms min={mn}ms max={mx}ms "
                 f"n={len(gaps)} ±{unc:.0f}ms")
        return avg

    avg_a = _summarize_gaps("Method A (in-pod, skew-corrected)",
                            "A", detection_gaps)
    avg_b = _summarize_gaps("Method B (local, zero skew)",
                            "B", detection_gaps_local)
    avg_c = _summarize_gaps("Method C (logs, zero skew)",
                            "C", detection_gaps_log)

    # Cross-method agreement check — compare using uncertainty intervals.
    # Two methods agree if their intervals [avg - unc, avg + unc] overlap.
    # A fixed 200ms threshold would let methods with very different precision
    # appear to agree when they actually don't (e.g., A at 300±400ms vs
    # C at 500±10ms look close but C's precision shows the gap is real).
    available = [(n, v, method_uncertainty[n])
                 for n, v in [("A", avg_a), ("B", avg_b), ("C", avg_c)]
                 if v is not None]
    if len(available) >= 2:
        all_agree = True
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                n1, v1, u1 = available[i]
                n2, v2, u2 = available[j]
                # Intervals overlap iff the gap between centers < sum of radii
                gap = abs(v1 - v2)
                margin = u1 + u2
                if gap > margin:
                    all_agree = False
                    progress(f"  WARNING: Methods {n1} and {n2} disagree: "
                             f"|{v1:.0f} - {v2:.0f}| = {gap:.0f}ms > "
                             f"±{u1:.0f} + ±{u2:.0f} = {margin:.0f}ms")
        if all_agree:
            values = [v for _, v, _ in available]
            spread = max(values) - min(values)
            progress(f"  Methods agree (intervals overlap, spread={spread:.0f}ms)")

    # Use best available method for finding (prefer zero-skew methods)
    best_avg = avg_b if avg_b is not None else (avg_c if avg_c is not None else avg_a)
    if best_avg is not None:
        if best_avg < 1000:
            progress("  FINDING: Detection <1s — NIXL likely has active health checks, "
                     "not relying on TCP keepalive alone")
        elif best_avg < 10000:
            progress(f"  FINDING: Detection ~{best_avg/1000:.0f}s — consistent with "
                     f"UCX TCP keepalive override")
        else:
            progress(f"  FINDING: Detection >{best_avg/1000:.0f}s — keepalive may be "
                     f"disabled or set very high")
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4i")


def run_exp_4i(ctx: ExperimentContext) -> None:
    """4i: Mid-transfer failure."""
    progress("=== 4i: Mid-transfer failure ===")

    progress(f"  Prompt size: {MID_TRANSFER_PROMPT_TOKENS} tokens "
             f"(~{MID_TRANSFER_PROMPT_TOKENS * 3 // 1024}KB input)")
    progress("  Pre-flight: measure baseline transfer time with large prompt")
    ttft, total, itl_m, itl_p, tcount, code, err, text = send_streaming_via_test_client(
        DECODE1_URL, prompt=MID_TRANSFER_PROMPT, max_tokens=20, socket_timeout=30)
    ctx.record("4i", "baseline", ttft, total, code,
               f"baseline large prompt ttft={ttft}ms tokens={MID_TRANSFER_PROMPT_TOKENS}",
               err, itl_mean_ms=itl_m, itl_p99_ms=itl_p, token_count=tcount)
    progress(f"  Baseline: ttft={ttft}ms total={total}ms tokens={tcount}")

    # Prediction uses measured baseline
    baseline_ttft = ttft if ttft > 0 else ctx.baseline_d1.ttft_mean
    oc_overhead = ctx.calibration.oc_exec_mean_ms
    # Derive prefill speed from baseline data: small-prompt TTFT is mostly
    # transfer + overhead; large-prompt TTFT adds prefill compute for extra tokens.
    small_ttft = ctx.baseline_d1.ttft_mean  # ~3 token prompt
    extra_tokens = max(MID_TRANSFER_PROMPT_TOKENS - 3, 1)
    prefill_compute_est = max(baseline_ttft - small_ttft, extra_tokens / 15.0)  # ms
    transfer_start_est = oc_overhead + prefill_compute_est
    progress(f"  Baseline TTFT (in-pod, includes prefill+transfer): {baseline_ttft:.0f}ms")
    progress(f"  oc exec overhead: {oc_overhead:.0f}ms")
    progress(f"  Prefill compute estimate: {prefill_compute_est:.0f}ms")
    progress(f"  Transfer starts ~{transfer_start_est:.0f}ms after kill timer begins")
    progress(f"  Kill delays: {KILL_DELAYS_MS} → "
             f"land {[f'{d - transfer_start_est:.0f}ms' for d in KILL_DELAYS_MS]} "
             f"into transfer")
    predict("4i",
            f"Kill delays {KILL_DELAYS_MS}ms fire during request lifecycle. "
            f"oc exec adds ~{oc_overhead:.0f}ms before request reaches vLLM. "
            f"Effective kill offset into transfer: delay - {transfer_start_est:.0f}ms. "
            f"Mid-transfer kills: expect timeout/error, not corruption.",
            f"Kill thread and request thread start simultaneously. The request "
            f"takes ~{oc_overhead:.0f}ms (oc exec) + ~{prefill_compute_est:.0f}ms "
            f"(prefill) before KV transfer begins. Kill delays > "
            f"{transfer_start_est:.0f}ms land during active transfer.")

    total_trials = len(KILL_DELAYS_MS) * KILL_REPEATS
    if KILL_REPEATS > 1:
        progress(f"  Repeat mode: {KILL_REPEATS} repeats × {len(KILL_DELAYS_MS)} delays "
                 f"= {total_trials} total trials")

    outcome_counts = {"success": 0, "partial": 0, "clean_error": 0,
                      "empty": 0, "error": 0}
    corruption_count = 0
    trial_num = 0

    # Classify each delay relative to the oc_exec + prefill overhead
    for repeat_i in range(KILL_REPEATS):
      if KILL_REPEATS > 1:
          progress(f"  --- Repeat {repeat_i + 1}/{KILL_REPEATS} ---")
      for delay_ms in KILL_DELAYS_MS:
        effective_offset = delay_ms - transfer_start_est
        if effective_offset < 0:
            timing = "before transfer (kills during oc_exec/prefill)"
        elif effective_offset < baseline_ttft * 0.5:
            timing = "early in transfer"
        elif effective_offset < baseline_ttft:
            timing = "mid-to-late transfer"
        else:
            timing = "likely after transfer"
        trial_num += 1
        trial_label = (f"  --- Kill delay: {delay_ms}ms "
                       f"(effective: +{effective_offset:.0f}ms into transfer, {timing})")
        if KILL_REPEATS > 1:
            trial_label += f" [trial {trial_num}/{total_trials}]"
        progress(trial_label + " ---")

        # Verify system is healthy
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        if code != "200":
            progress(f"  SKIP: system not healthy (code={code})")
            ctx.record("4i", f"skip-{delay_ms}", t, tot, code,
                       "skipped, system unhealthy", err)
            continue

        prefill_pod = get_pod_name(PREFILL_SELECTOR)
        kill_done = threading.Event()
        kill_time = [0]

        def _kill_after_delay(delay_s, pod,
                              _kill_time=kill_time, _kill_done=kill_done):
            time.sleep(delay_s)
            _kill_time[0] = int(time.time() * 1000)
            oc("delete", "pod", pod, "-n", NS,
               "--grace-period=0", "--force")
            _kill_done.set()

        killer = threading.Thread(target=_kill_after_delay,
                                  args=(delay_ms / 1000.0, prefill_pod))
        killer.start()

        # Send streaming request (will overlap with kill)
        ttft, total, itl_m, itl_p, tcount, code, err, text = send_streaming_via_test_client(
            DECODE1_URL, prompt=MID_TRANSFER_PROMPT, max_tokens=20,
            subprocess_timeout=60, socket_timeout=30)

        killer.join()

        # Classify the outcome for data integrity analysis.
        # The conditions must partition the input space completely:
        #   tcount > 0: success (200) or partial (tokens before death)
        #   tcount == 0: clean_error (with error msg) or empty (no error msg)
        if code == "200" and tcount > 0:
            outcome = "success"
        elif tcount > 0:
            # Got tokens but non-200 (including code "0" = connection died)
            # — possible corruption signal. This is the most interesting
            # case for data integrity: partial KV transfer may produce
            # garbled output.
            outcome = "partial"
        elif err:
            outcome = "clean_error"
        else:
            outcome = "empty"

        # Check received text for corruption (non-printable = corrupted KV)
        corruption_flag = ""
        if text and tcount > 0:
            non_printable = sum(1 for c in text if not c.isprintable() and c not in '\n\r\t')
            if non_printable > 0:
                corruption_flag = f"CORRUPTION: {non_printable} non-printable chars in {len(text)} chars"
                progress(f"  WARNING: {corruption_flag}")
                corruption_count += 1

        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        ctx.record("4i", f"during-{delay_ms}ms-r{repeat_i}", ttft, total, code,
                   f"kill_delay={delay_ms}ms effective_offset={effective_offset:.0f}ms "
                   f"outcome={outcome} timing={timing} "
                   f"pod={prefill_pod} tokens={tcount} "
                   f"prompt_tokens={MID_TRANSFER_PROMPT_TOKENS}"
                   f"{' ' + corruption_flag if corruption_flag else ''}",
                   err, itl_mean_ms=itl_m, itl_p99_ms=itl_p, token_count=tcount,
                   detect_epoch_ms=kill_time[0])
        progress(f"  Outcome: {outcome} code={code} ttft={ttft}ms "
                 f"total={total}ms tokens={tcount} "
                 f"text={repr(text[:80]) if text else 'none'} "
                 f"err={err[:100] if err else 'none'}")

        # Capture decode logs for NIXL transfer state
        capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER],
                         f"4i-decode1-{delay_ms}ms", ctx.log_dir)

        # Recover
        progress("  Waiting for prefill recovery...")
        wait_for_ready(PREFILL_DEPLOY)
        time.sleep(5)

    # 4i Summary
    progress("  === 4i Summary ===")
    progress(f"  Total trials: {trial_num}")
    for outcome_type in ("success", "partial", "clean_error", "empty", "error"):
        count = outcome_counts.get(outcome_type, 0)
        if count > 0:
            progress(f"    {outcome_type}: {count}")
    progress(f"  Corruption detected: {corruption_count}")
    upper_bound = None
    if trial_num > 0 and corruption_count == 0:
        # Rule of three: 95% confidence upper bound on corruption rate
        upper_bound = 3.0 / trial_num * 100
        progress(f"  95% upper bound on corruption rate: {upper_bound:.1f}% "
                 f"(need ~300 trials for <1%)")
    ctx.finding("4i_trials", trial_num)
    ctx.finding("4i_outcomes", dict(outcome_counts))
    ctx.finding("4i_corruptions", corruption_count)
    ctx.finding("4i_upper_bound", upper_bound)
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4j")


def run_exp_4j(ctx: ExperimentContext) -> None:
    """4j: Container restart vs pod restart."""
    progress("=== 4j: Container restart vs pod restart ===")
    predict("4j",
            "Container restart (same IP): recovery ~same as pod restart. "
            "Pod restart (new IP): recovery ~same as container restart. "
            "If they differ significantly, ZMQ discovery is caching pod IPs.",
            "ZMQ discovery uses service DNS, not pod IP directly. If this is true, "
            "IP changes shouldn't matter — DNS resolves to the new pod either way. "
            "A significant difference reveals IP-level caching in the discovery layer.")
    progress("")

    # --- Container restart (kill PID 1 in vllm container) ---
    progress("  --- Container restart (kill PID 1) ---")
    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4j", "pre-container", t, tot, code, "before container kill", err)

    decode1_pod = get_pod_name(DECODE1_SELECTOR)
    pre_ip, _ = oc("get", "pod", decode1_pod, "-n", NS,
                    "-o", "jsonpath={.status.podIP}")
    pre_restarts, _ = oc("get", "pod", decode1_pod, "-n", NS,
                          "-o", "jsonpath={.status.containerStatuses[0].restartCount}")
    progress(f"  Pod: {decode1_pod}, IP: {pre_ip}, restarts: {pre_restarts}")

    progress("  Killing vllm container (PID 1)...")
    container_kill_epoch = ctx.epoch_ms()
    oc("exec", decode1_pod, "-c", VLLM_CONTAINER, "-n", NS,
       "--", "kill", "1")
    ctx.record("4j", "container-kill", 0, 0, "n/a",
               f"killed PID 1 in {VLLM_CONTAINER}, pod={decode1_pod} ip={pre_ip}",
               "", detect_epoch_ms=container_kill_epoch)

    # Wait for container restart (not pod replacement)
    progress("  Waiting for container restart...")
    container_start = time.time()
    deadline = time.time() + 600
    while time.time() < deadline:
        restarts, _ = oc("get", "pod", decode1_pod, "-n", NS,
                          "-o", "jsonpath={.status.containerStatuses[0].restartCount}")
        if restarts and restarts != pre_restarts:
            progress(f"  Container restarted (restarts: {pre_restarts} -> {restarts})")
            break
        time.sleep(2)
    else:
        progress("  WARNING: container did not restart within timeout")

    # Wait for readiness
    wait_for_ready(DECODE1_DEPLOY)
    container_recovery_s = int(time.time() - container_start)

    post_ip, _ = oc("get", "pod", decode1_pod, "-n", NS,
                     "-o", "jsonpath={.status.podIP}")
    ip_changed = "yes" if post_ip != pre_ip else "no"
    progress(f"  Post-restart IP: {post_ip} (changed: {ip_changed})")
    progress(f"  Container recovery time: {container_recovery_s}s")

    # Start probe for continuous monitoring through the recovery
    probe_proc = start_probe(DECODE1_URL, interval=0.5, duration=300)
    progress("  Probing for functional recovery...")

    recover_epoch, recover_probes, recover_lat, _ = read_probe_until_recover(
        probe_proc, timeout=120)
    stop_probe(probe_proc)

    ctx.record("4j", "container-recover", recover_lat, recover_lat, "200",
               f"container restart recovery={container_recovery_s}s ip_changed={ip_changed} ip={post_ip}",
               "",
               detect_epoch_ms=container_kill_epoch,
               recover_epoch_ms=recover_epoch,
               probes_to_recover=recover_probes)

    # Capture post-restart logs
    capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER, SIDECAR_CONTAINER],
                     "4j-decode1-container-post", ctx.log_dir)

    # Steady-state check
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4j", "container-steady", t, tot, code,
               "steady-state after container restart", err)

    time.sleep(5)

    # --- Pod restart (delete pod, new pod with new IP) ---
    progress("")
    progress("  --- Pod restart (delete pod) ---")
    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4j", "pre-pod", t, tot, code, "before pod kill", err)

    decode1_pod = get_pod_name(DECODE1_SELECTOR)
    pre_ip, _ = oc("get", "pod", decode1_pod, "-n", NS,
                    "-o", "jsonpath={.status.podIP}")
    progress(f"  Pod: {decode1_pod}, IP: {pre_ip}")

    # Start probe before kill for continuous monitoring
    probe_proc = start_probe(DECODE1_URL, interval=0.5, duration=300)

    progress("  Deleting pod...")
    pod_kill_start = time.time()
    pod_kill_epoch = ctx.epoch_ms()
    oc("delete", "pod", decode1_pod, "-n", NS,
       "--grace-period=0", "--force")
    ctx.record("4j", "pod-kill", 0, 0, "n/a",
               f"deleted {decode1_pod} ip={pre_ip}",
               "", detect_epoch_ms=pod_kill_epoch)

    progress("  Waiting for replacement pod...")
    wait_for_ready(DECODE1_DEPLOY)
    pod_recovery_s = int(time.time() - pod_kill_start)

    new_pod = get_pod_name(DECODE1_SELECTOR)
    post_ip, _ = oc("get", "pod", new_pod, "-n", NS,
                     "-o", "jsonpath={.status.podIP}")
    ip_changed = "yes" if post_ip != pre_ip else "no"
    progress(f"  New pod: {new_pod}, IP: {post_ip} (changed: {ip_changed})")

    # Read probe for recovery
    recover_epoch, recover_probes, recover_lat, _ = read_probe_until_recover(
        probe_proc, timeout=120)
    stop_probe(probe_proc)

    ctx.record("4j", "pod-recover", recover_lat, recover_lat, "200",
               f"pod restart recovery={pod_recovery_s}s ip_changed={ip_changed} ip={post_ip}",
               "",
               detect_epoch_ms=pod_kill_epoch,
               recover_epoch_ms=recover_epoch,
               probes_to_recover=recover_probes)

    capture_pod_logs(DECODE1_SELECTOR, [VLLM_CONTAINER, SIDECAR_CONTAINER],
                     "4j-decode1-pod-post", ctx.log_dir)

    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4j", "pod-steady", t, tot, code,
               "steady-state after pod restart", err)
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4k")


def run_exp_4k(ctx: ExperimentContext) -> None:
    """4k: Concurrent load during failure."""
    progress("=== 4k: Concurrent load during failure ===")
    predict("4k",
            f"~{KILL_AT_S * LOAD_QPS} requests succeed before kill. "
            f"Failure window lasts until prefill recovers (60-120s). "
            f"Requests during failure: error, not hang.",
            "Under load, in-flight requests that need prefill will fail. "
            "Requests that already completed KV transfer should finish normally. "
            "The key question: do failures cascade (decode pod crashes) or stay contained?")
    progress("")

    progress(f"  Config: QPS={LOAD_QPS}, duration={LOAD_DURATION}s, "
             f"kill prefill at T+{KILL_AT_S}s")

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4k", "pre", t, tot, code, "before load test", err)

    # Start in-pod load driver (runs entirely inside the cluster)
    progress(f"  Starting in-pod load driver ({LOAD_QPS} QPS, {LOAD_DURATION}s)...")
    load_proc, load_csv_path = start_load(
        DECODE1_URL, qps=LOAD_QPS, duration=LOAD_DURATION,
        output_name="4k-load-results.csv")

    # Kill prefill after delay
    progress(f"  Waiting {KILL_AT_S}s before killing prefill...")
    time.sleep(KILL_AT_S)

    prefill_pod = get_pod_name(PREFILL_SELECTOR)
    kill_epoch_4k = ctx.epoch_ms()
    oc("delete", "pod", prefill_pod, "-n", NS,
       "--grace-period=0", "--force")
    ctx.record("4k", "kill", 0, 0, "n/a",
               f"killed {prefill_pod}", "", detect_epoch_ms=kill_epoch_4k)
    progress(f"  [T+{KILL_AT_S}s] Killed prefill: {prefill_pod}")

    # Wait for load to finish
    progress("  Waiting for load driver to complete...")
    try:
        load_proc.wait(timeout=LOAD_DURATION + 30)
    except subprocess.TimeoutExpired:
        load_proc.terminate()

    # Read summary from stdout
    load_summary = load_proc.stdout.read().strip()
    if load_summary:
        try:
            summary = json.loads(load_summary)
            progress(f"  Load results: {summary.get('total_requests', 0)} requests, "
                     f"{summary.get('ok', 0)} OK, {summary.get('failed', 0)} failed")
        except json.JSONDecodeError:
            progress(f"  Load output: {load_summary[:200]}")

    # Wait for prefill recovery
    progress("  Waiting for prefill recovery...")
    wait_for_ready(PREFILL_DEPLOY)

    # Pull load results CSV from pod and partition by kill epoch.
    # Same cross-clock-domain caveat as 4e: r_epoch is pod clock, kill_epoch_4k
    # is local clock. Boundary accuracy limited by calibrated clock skew.
    load_results = collect_load_results(load_csv_path)
    pre_kill_ok = 0
    during_fault_ok = 0
    during_fault_fail = 0
    for r in load_results:
        r_epoch = int(r.get("epoch_ms", 0))
        during_fault = 1 if kill_epoch_4k and r_epoch > kill_epoch_4k else 0
        is_ok = str(r.get("status", 0)) == "200"
        if not during_fault and is_ok:
            pre_kill_ok += 1
        elif during_fault:
            if is_ok:
                during_fault_ok += 1
            else:
                during_fault_fail += 1
        ctx.record("4k", f"req-{r.get('seq', 0)}",
                   float(r.get("ttft_ms", 0)), float(r.get("total_ms", 0)),
                   str(r.get("status", 0)),
                   f"during_fault={during_fault} elapsed={r.get('elapsed_s', 0)}s",
                   r.get("error", ""),
                   itl_mean_ms=r.get("itl_mean_ms", ""),
                   itl_p99_ms=r.get("itl_p99_ms", ""),
                   token_count=r.get("token_count", ""))

    progress(f"  Partitioned: {pre_kill_ok} OK before kill, "
             f"{during_fault_ok} OK / {during_fault_fail} failed during fault")
    evaluate("4k", KILL_AT_S * LOAD_QPS, pre_kill_ok, " OK requests before kill")

    # Recovery curve
    progress("  Recovery curve:")
    for i in range(1, 6):
        t, tot, code, err = send_via_test_client(DECODE1_URL)
        ctx.record("4k", f"post-{i}", t, tot, code, f"recovery probe {i}", err)
        time.sleep(1)

    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER],
                     "4k-decode1-post", ctx.log_dir)

    # Parse sidecar logs for fallback events — quantifies how many
    # "successful" requests during fault were actually served locally
    sidecar_log = os.path.join(ctx.log_dir, "4k-decode1-post-routing-sidecar.log")
    try:
        with open(sidecar_log) as f:
            fallbacks = parse_sidecar_fallbacks(f.read())
        pre_kill_fb = sum(1 for ts, _ in fallbacks if ts and ts < kill_epoch_4k)
        during_fb = sum(1 for ts, _ in fallbacks if ts and ts >= kill_epoch_4k)
        ctx.record("4k", "fallback-count", 0, 0, "n/a",
                   f"sidecar fallbacks: {len(fallbacks)} total "
                   f"({pre_kill_fb} pre-kill, {during_fb} during fault)")
        if fallbacks:
            progress(f"  Sidecar fallbacks: {len(fallbacks)} total "
                     f"({pre_kill_fb} pre-kill, {during_fb} during fault)")
            progress(f"  Of {during_fault_ok} OK requests during fault, "
                     f"{during_fb} were fallback-to-local")
    except OSError:
        pass

    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="pre-4l")


def run_exp_4l(ctx: ExperimentContext) -> None:
    """4l: Rolling update under load."""
    progress("=== 4l: Rolling update under load ===")
    predict("4l",
            "Zero failures if readiness probes + ZMQ discovery work correctly. "
            "If failures occur, they reveal stale ZMQ registrations routing to terminating pods.",
            "Rolling update: old pod gets SIGTERM, new pod starts, passes readiness probe, "
            "then old pod terminates. If the sidecar routes requests to the old pod during "
            "its termination grace period, those requests fail. preStop hooks can mitigate.")
    progress("")

    progress(f"  Config: QPS={LOAD_QPS}, duration={ROLLOUT_DURATION}s, "
             f"trigger rollout at T+{ROLLOUT_AT_S}s")

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4l", "pre", t, tot, code, "before rolling update", err)

    # Start in-pod load driver
    progress(f"  Starting in-pod load driver ({LOAD_QPS} QPS, {ROLLOUT_DURATION}s)...")
    rollout_load_proc, rollout_csv_path = start_load(
        DECODE1_URL, qps=LOAD_QPS, duration=ROLLOUT_DURATION,
        output_name="4l-load-results.csv")

    # Trigger rollout after delay
    progress(f"  Waiting {ROLLOUT_AT_S}s before triggering rollout...")
    time.sleep(ROLLOUT_AT_S)

    rollout_epoch = ctx.epoch_ms()
    ts = str(int(time.time()))
    oc("set", "env", f"{WORKLOAD_TYPE}/{DECODE1_DEPLOY}",
       f"FORCE_RESTART={ts}", "-n", NS)
    ctx.record("4l", "rollout", 0, 0, "n/a",
               f"rollout triggered (FORCE_RESTART={ts})",
               "", detect_epoch_ms=rollout_epoch)
    progress(f"  [T+{ROLLOUT_AT_S}s] Triggered rolling update")

    # Wait for load to finish
    progress("  Waiting for load driver to complete...")
    try:
        rollout_load_proc.wait(timeout=ROLLOUT_DURATION + 30)
    except subprocess.TimeoutExpired:
        rollout_load_proc.terminate()

    # Read summary
    rollout_summary = rollout_load_proc.stdout.read().strip()
    if rollout_summary:
        try:
            summary = json.loads(rollout_summary)
            ok = summary.get("ok", 0)
            failed = summary.get("failed", 0)
            total_reqs = summary.get("total_requests", 0)
            progress(f"  Load results: {total_reqs} requests, {ok} OK, {failed} failed")
            if failed == 0:
                progress("  Zero-downtime rolling update: YES")
            else:
                progress(f"  Zero-downtime rolling update: NO ({failed} failures)")
        except json.JSONDecodeError:
            pass

    # Wait for rollout to complete
    progress("  Waiting for rollout to complete...")
    wait_for_ready(DECODE1_DEPLOY)

    # Pull load results
    rollout_results = collect_load_results(rollout_csv_path)
    for r in rollout_results:
        r_epoch = int(r.get("epoch_ms", 0))
        phase = "during" if rollout_epoch and r_epoch > rollout_epoch else "pre"
        ctx.record("4l", f"req-{r.get('seq', 0)}",
                   float(r.get("ttft_ms", 0)), float(r.get("total_ms", 0)),
                   str(r.get("status", 0)),
                   f"rollout_phase={phase} elapsed={r.get('elapsed_s', 0)}s",
                   r.get("error", ""),
                   itl_mean_ms=r.get("itl_mean_ms", ""),
                   itl_p99_ms=r.get("itl_p99_ms", ""),
                   token_count=r.get("token_count", ""))

    capture_pod_logs(DECODE1_SELECTOR, [SIDECAR_CONTAINER, VLLM_CONTAINER],
                     "4l-decode1-post", ctx.log_dir)

    # Note: we intentionally leave the FORCE_RESTART env var in place.
    # Removing it with "oc set env ... FORCE_RESTART-" would trigger a
    # second unintended rollout. The env var is inert (vLLM ignores it)
    # and will be cleaned up on the next deployment update.
    progress("  Note: FORCE_RESTART env var left in place (removing would trigger another rollout)")
    progress("")


def run_exp_4m(ctx: ExperimentContext) -> None:
    """4m: Progressive degradation sweep.

    Sweep packet loss from 0% to 100% in configurable steps using tc netem on
    the prefill pod's egress. At each level, send N requests and measure:
    - Success rate (did the request complete?)
    - TTFT (how much slower is the KV transfer under loss?)
    - Fallback rate (did the sidecar fall back to local decode?)

    This tests *gray failure* — the regime between "working" and "broken"
    that no existing experiment covers. 4g tests a single point; 4f tests
    binary partition. 4m finds the exact loss threshold where disagg
    degrades, where the sidecar falls back, and where requests start failing.
    """
    loss_steps = [int(x) for x in env("DEGRAD_LOSS_STEPS", "0,5,10,20,30,50,70,90,100").split(",")]
    probes_per_step = int(env("DEGRAD_PROBES", "5"))

    progress("=== 4m: Progressive degradation sweep ===")
    progress(f"  Loss steps: {loss_steps}%")
    progress(f"  Probes per step: {probes_per_step}")
    predict("4m",
            "Low loss (<10%): TTFT increases linearly due to TCP retransmissions. "
            "Medium loss (10-50%): NIXL transfers slow dramatically, some fail. "
            "High loss (>50%): sidecar fallback dominates. "
            "100%: equivalent to network partition.",
            "TCP retransmits on packet loss with exponential backoff. NIXL uses "
            "a single large transfer — even one lost segment stalls the entire "
            "KV payload. The sidecar falls back on any non-4xx error, so the "
            "transition should be: slower NIXL → NIXL timeout → fallback.")
    progress("")

    # Check tc availability
    prefill_pod = get_pod_name(PREFILL_SELECTOR)
    tc_check, _ = oc("exec", prefill_pod, "-c", VLLM_CONTAINER,
                      "-n", NS, "--", "which", "tc")
    if not tc_check.strip():
        progress("  SKIPPED: tc (iproute2) not available in prefill container.")
        ctx.record("4m", "skip", 0, 0, "n/a", "tc not available")
        progress("")
        return

    progress("  Pre-flight: verify disagg works")
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4m", "pre", t, tot, code, "before degradation sweep", err)
    if code != "200":
        progress(f"  SKIP: system not healthy (code={code})")
        return

    baseline_ttft = ctx.baseline_d1.ttft_mean

    sweep_results = []  # (loss%, success_rate, mean_ttft, fallback_rate)

    for step_i, loss_pct in enumerate(loss_steps):
        progress(f"  --- Packet loss: {loss_pct}% ---")

        if loss_pct > 0:
            # Apply netem (replace if exists)
            oc("exec", prefill_pod, "-c", VLLM_CONTAINER, "-n", NS,
               "--", "tc", "qdisc", "replace", "dev", "eth0", "root", "netem",
               "loss", f"{loss_pct}%")
            time.sleep(1)  # let netem settle

            # Verify: read back what netem is actually doing
            tc_stats, _ = oc("exec", prefill_pod, "-c", VLLM_CONTAINER,
                              "-n", NS, "--", "tc", "-s", "qdisc", "show",
                              "dev", "eth0")
            if tc_stats:
                # Parse "Sent X bytes Y pkt (dropped Z, ...)"
                m = re.search(r'Sent (\d+) bytes (\d+) pkt \(dropped (\d+)',
                              tc_stats)
                if m:
                    _sent, pkts, dropped = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    actual_loss = dropped / max(pkts, 1) * 100 if pkts > 0 else 0
                    progress(f"  netem verified: {pkts} pkts sent, "
                             f"{dropped} dropped ({actual_loss:.1f}% actual "
                             f"vs {loss_pct}% requested)")
        elif step_i > 0:
            # Remove netem from prior step to measure clean 0% baseline.
            # Skip on first iteration (no qdisc to remove).
            oc("exec", prefill_pod, "-c", VLLM_CONTAINER, "-n", NS,
               "--", "tc", "qdisc", "del", "dev", "eth0", "root")
            time.sleep(1)

        # Snapshot sidecar log line count before probes for fallback diffing
        full_log, _ = oc("logs", get_pod_name(DECODE1_SELECTOR),
                          "-c", SIDECAR_CONTAINER, "-n", NS)
        log_lines_before = len(full_log.splitlines()) if full_log else 0

        ok_count = 0
        ttfts = []
        for pi in range(probes_per_step):
            t, tot, code, err = send_via_test_client(
                DECODE1_URL, curl_timeout=15, subprocess_timeout=30)
            ctx.record("4m", f"loss-{loss_pct}pct-{pi+1}", t, tot, code,
                       f"loss={loss_pct}% probe={pi+1}/{probes_per_step}", err)
            if code == "200" and t > 0:
                ok_count += 1
                ttfts.append(t)

        # Check sidecar for fallbacks during this step
        full_log_after, _ = oc("logs", get_pod_name(DECODE1_SELECTOR),
                                "-c", SIDECAR_CONTAINER, "-n", NS)
        new_lines = full_log_after.splitlines()[log_lines_before:] if full_log_after else []
        new_log_text = "\n".join(new_lines)
        fallbacks = parse_sidecar_fallbacks(new_log_text)
        fallback_rate = len(fallbacks) / max(probes_per_step, 1)

        # Read netem stats after probes to get actual loss during measurement
        actual_loss_pct = None
        if loss_pct > 0:
            tc_stats_post, _ = oc("exec", prefill_pod, "-c", VLLM_CONTAINER,
                                   "-n", NS, "--", "tc", "-s", "qdisc", "show",
                                   "dev", "eth0")
            if tc_stats_post:
                m = re.search(r'Sent (\d+) bytes (\d+) pkt \(dropped (\d+)',
                              tc_stats_post)
                if m:
                    pkts, dropped = int(m.group(2)), int(m.group(3))
                    actual_loss_pct = dropped / max(pkts, 1) * 100 if pkts > 0 else 0

        success_rate = ok_count / probes_per_step
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        ttft_ratio = mean_ttft / baseline_ttft if baseline_ttft > 0 else 0

        sweep_results.append((loss_pct, success_rate, mean_ttft, fallback_rate,
                              actual_loss_pct))

        progress(f"  {loss_pct}%: success={ok_count}/{probes_per_step} "
                 f"({success_rate:.0%}) ttft={mean_ttft:.0f}ms "
                 f"({ttft_ratio:.1f}x baseline) "
                 f"fallbacks={len(fallbacks)}")

    # Remove netem
    progress("  Removing network degradation...")
    oc("exec", prefill_pod, "-c", VLLM_CONTAINER, "-n", NS,
       "--", "tc", "qdisc", "del", "dev", "eth0", "root")
    time.sleep(2)

    # Post-recovery check
    t, tot, code, err = send_via_test_client(DECODE1_URL)
    ctx.record("4m", "post", t, tot, code, "after degradation sweep", err)

    # Summary table
    progress("")
    progress("  === 4m Summary: Progressive Degradation ===")
    progress(f"  {'Loss%':>6s}  {'Actual':>7s}  {'Success':>8s}  {'TTFT':>8s}  "
             f"{'vs BL':>6s}  {'Fallback':>8s}  {'Regime'}")
    for loss_pct, sr, ttft, fbr, actual in sweep_results:
        ttft_str = f"{ttft:.0f}ms" if ttft > 0 else "n/a"
        ratio = ttft / baseline_ttft if ttft > 0 and baseline_ttft > 0 else 0
        ratio_str = f"{ratio:.1f}x" if ratio > 0 else "n/a"
        actual_str = f"{actual:.0f}%" if actual is not None else "n/a"

        # Classify regime using the unified classifier
        if sr >= 0.8 and fbr < 0.2:
            regime = "normal"
        elif sr >= 0.8 and fbr >= 0.2:
            regime = "fallback"
        elif sr >= 0.4:
            regime = "degraded"
        else:
            regime = "broken"

        progress(f"  {loss_pct:>5d}%  {actual_str:>7s}  {sr:>7.0%}  {ttft_str:>8s}  "
                 f"{ratio_str:>6s}  {fbr:>7.0%}  {regime}")

    # Find thresholds — actual threshold is between previous step and this step
    loss_steps = [r[0] for r in sweep_results]
    degrad_threshold = None
    fallback_threshold = None
    failure_threshold = None
    degrad_prev = 0
    fallback_prev = 0
    failure_prev = 0
    for loss_pct, sr, ttft, fbr, _actual in sweep_results:
        if degrad_threshold is None and ttft > baseline_ttft * 2 and ttft > 0:
            degrad_threshold = loss_pct
            degrad_prev = loss_steps[max(0, loss_steps.index(loss_pct) - 1)]
        if fallback_threshold is None and fbr > 0.5:
            fallback_threshold = loss_pct
            fallback_prev = loss_steps[max(0, loss_steps.index(loss_pct) - 1)]
        if failure_threshold is None and sr < 0.5:
            failure_threshold = loss_pct
            failure_prev = loss_steps[max(0, loss_steps.index(loss_pct) - 1)]

    progress("")
    if degrad_threshold is not None:
        progress(f"  Degradation threshold (>2x baseline TTFT): "
                 f"{degrad_prev}-{degrad_threshold}% loss")
    if fallback_threshold is not None:
        progress(f"  Fallback threshold (>50% requests fall back): "
                 f"{fallback_prev}-{fallback_threshold}% loss")
    if failure_threshold is not None:
        progress(f"  Failure threshold (<50% success): "
                 f"{failure_prev}-{failure_threshold}% loss")
    progress("")

    # ── Steady-state gate ──────────────────────────────
    wait_for_steady_state(DECODE1_URL, ctx.baseline_d1, label="post-4m")


def finalize(ctx: ExperimentContext, metrics_proc) -> None:
    """Stop metrics, collect CSV, log analysis summary, prediction summary."""
    # ── Finalize: stop metrics, collect data ────────────
    progress("=== Stopping metrics collector ===")
    stop_metrics_collector(metrics_proc)
    collect_metrics_csv()

    # ── Log analysis summary ────────────────────────────
    progress("")
    progress("=== Log analysis summary ===")
    log_files = [f for f in os.listdir(ctx.log_dir) if f.endswith(".log")]
    total_events = 0
    for lf in sorted(log_files):
        events = parse_log_timestamps(os.path.join(ctx.log_dir, lf))
        if events:
            cats: dict[str, int] = {}
            for _, cat, _ in events:
                cats[cat] = cats.get(cat, 0) + 1
            progress(f"  {lf}: {len(events)} events — "
                     f"{', '.join(f'{c}={n}' for c, n in sorted(cats.items()))}")
            total_events += len(events)
    progress(f"  Total: {total_events} categorized events across {len(log_files)} log files")
    progress("")

    # ── Cross-Experiment Synthesis ──────────────────────────
    if ctx.findings:
        progress("=== Cross-Experiment Synthesis ===")
        progress("")

        # Partition recovery (4f)
        sweep = ctx.findings.get("4f_sweep")
        if sweep:
            progress("  PARTITION RECOVERY:")
            recovered_any = any(r[2] for r in sweep)
            sticky_any = any(not r[2] for r in sweep)
            if recovered_any:
                # Tuples: (nom, act, recovered, delay, mech, lower_bound)
                recoveries = [(r[0], r[1], r[3], r[5]) for r in sweep
                              if r[2] and r[3] is not None]
                if recoveries:
                    fastest = min(recoveries, key=lambda x: x[2])
                    slowest = max(recoveries, key=lambda x: x[2])
                    f_lo = f"{fastest[3]:.0f}-" if fastest[3] is not None else ""
                    s_lo = f"{slowest[3]:.0f}-" if slowest[3] is not None else ""
                    progress(f"    Fastest recovery: +{f_lo}{fastest[2]:.0f}s "
                             f"(after {fastest[1]:.0f}s actual partition)")
                    progress(f"    Slowest recovery: +{s_lo}{slowest[2]:.0f}s "
                             f"(after {slowest[1]:.0f}s actual partition)")
            if sticky_any:
                sticky = [r for r in sweep if not r[2]]
                progress(f"    Sticky failures: {len(sticky)} "
                         f"(at {', '.join(f'{r[1]:.0f}s' for r in sticky)})")
            mode = ctx.findings.get("4f_mode", "unknown")
            progress(f"    Mode: {mode}")
            progress("")

        # Data integrity (4i)
        trials = ctx.findings.get("4i_trials")
        if trials:
            corruptions = ctx.findings.get("4i_corruptions", 0)
            ub = ctx.findings.get("4i_upper_bound")
            outcomes = ctx.findings.get("4i_outcomes", {})
            progress("  DATA INTEGRITY:")
            progress(f"    Trials: {trials}, Corruptions: {corruptions}")
            if ub is not None:
                progress(f"    95% upper bound: {ub:.1f}%")
            partials = outcomes.get("partial", 0)
            if partials > 0:
                progress(f"    WARNING: {partials} partial results "
                         f"(tokens received before failure)")
            progress("")

        # Cross-reference: consistency checks
        progress("  CONSISTENCY CHECKS:")
        issues = []

        # Check: if 4f shows recovery, 4i should show non-empty results
        if sweep and trials:
            if trials > 0 and all(v == 0 for k, v in ctx.findings.get("4i_outcomes", {}).items()
                                  if k != "empty"):
                issues.append("4i produced only empty results — "
                              "timeout chain may still be misconfigured")

        # Check: 4f and 4i both observe sidecar fallback
        if sweep:
            fallback_in_4f = any(r[4] == "fallback" for r in sweep)
            if fallback_in_4f:
                progress("    4f: sidecar fallback observed during partition recovery")

        if not issues:
            progress("    No contradictions detected across experiments")
        else:
            for issue in issues:
                progress(f"    ISSUE: {issue}")

        progress("")

    # ── Prediction vs Measurement Ledger ──────────────────
    print_prediction_summary()

    ctx.writer.close()
    outfile = os.path.join(DATA_DIR, "exp4-results.csv")
    progress("")
    progress(f"=== Experiment 4 Complete === ({outfile})")
    progress(f"  Results: {outfile}")
    progress(f"  Metrics: {os.path.join(DATA_DIR, 'exp4-metrics.csv')}")
    progress(f"  Logs: {ctx.log_dir}")


# ── Experiment Registry ──────────────────────────────────────────────────────
# Maps sub-experiment names to their functions. Used by main() for selection.
# Must be defined after the functions exist.

EXPERIMENTS = {
    "4a": run_exp_4a,  "4b": run_exp_4b,  "4d": run_exp_4d,
    "4e": run_exp_4e,  "4f": run_exp_4f,  "4g": run_exp_4g,
    "4h": run_exp_4h,  "4i": run_exp_4i,  "4j": run_exp_4j,
    "4k": run_exp_4k,  "4l": run_exp_4l,  "4m": run_exp_4m,
}


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Fault Tolerance for llm-d disaggregated inference")
    parser.add_argument(
        "experiments", nargs="*", default=["all"],
        metavar="SUB_EXP",
        help=f"Sub-experiments to run (choices: {', '.join(EXPERIMENTS)}, all). "
             f"Default: all")
    parser.add_argument(
        "--skip-control", action="store_true",
        help="Skip the Phase 0c control run")
    parser.add_argument(
        "--stop-on-failure", action="store_true",
        help="Stop on first experiment failure (default: log and continue)")
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt (for automation)")
    args = parser.parse_args()

    # Resolve selection
    if "all" in args.experiments:
        selected = list(EXPERIMENTS.keys())
    else:
        unknown = [e for e in args.experiments if e not in EXPERIMENTS]
        if unknown:
            parser.error(f"Unknown experiments: {', '.join(unknown)}. "
                         f"Choices: {', '.join(EXPERIMENTS)}, all")
        selected = args.experiments

    os.makedirs(DATA_DIR, exist_ok=True)
    log_dir = os.path.join(DATA_DIR, "exp4-logs")
    os.makedirs(log_dir, exist_ok=True)
    outfile = os.path.join(DATA_DIR, "exp4-results.csv")

    # Copy scripts to pod (needed for fault_driver and metrics_collector)
    copy_scripts_to_pod()

    # Collect cluster metadata (exp4 runs locally with oc access)
    progress("  Collecting cluster metadata...")
    cluster_info = collect_cluster_info()
    write_run_info("exp4", {"cluster": cluster_info})

    writer = TypedCSVWriter(outfile, Exp4Row)

    # Start metrics collector (scrapes vLLM /metrics every 2s)
    metrics_proc = None
    progress(f"  Starting metrics collector (interval={METRICS_INTERVAL}s)...")
    metrics_proc = start_metrics_collector(sample_interval=METRICS_INTERVAL)
    progress(f"  Metrics collector PID: {metrics_proc.pid}")

    progress("")
    progress("=== Experiment 4: Fault Tolerance (Fermi Method) ===")
    progress(f"  Model: {MODEL}")
    progress(f"  Namespace: {NS}")
    progress(f"  Selected: {', '.join(selected)}")
    progress(f"  Output: {outfile}")
    progress(f"  Logs: {log_dir}")
    progress(f"  Metrics: continuous (interval={METRICS_INTERVAL}s)")
    progress("")
    progress("  WARNING: This experiment KILLS PODS in the target namespace.")
    progress("  It will delete pods, scale deployments, and apply network policies.")
    progress("")

    if not args.yes:
        try:
            answer = input("  Type YES to continue, or Ctrl-C to abort: ")
        except (EOFError, KeyboardInterrupt):
            progress("\n  Aborted.")
            sys.exit(1)
        if answer.strip() != "YES":
            progress("  Aborted.")
            sys.exit(1)
        progress("")

    try:
        # Phase 0a: Instrument Calibration (always — required for ExperimentContext)
        calibration = run_calibration(writer)

        # Phase 0b: Baseline Characterization (always — required for ExperimentContext)
        baseline_d1, baseline_d2 = run_baselines(writer)

        # Build experiment context (immutable calibration + baselines, shared writer)
        ctx = ExperimentContext(
            writer=writer,
            log_dir=log_dir,
            calibration=calibration,
            baseline_d1=baseline_d1,
            baseline_d2=baseline_d2,
        )

        # Phase 0c: Control Run (skippable)
        if not args.skip_control:
            run_control(ctx)
        else:
            progress("=== Phase 0c: Control Run — SKIPPED ===")
            progress("")

        # Experiment pipeline
        failed = []
        for name in selected:
            try:
                EXPERIMENTS[name](ctx)
            except Exception as e:
                progress("")
                progress(f"  FAILED: {name} — {e}")
                traceback.print_exc(file=sys.stderr)
                ctx.record(name, "error", 0, 0, "CRASH", str(e)[:200])
                failed.append(name)
                if args.stop_on_failure:
                    progress("  --stop-on-failure: aborting remaining experiments")
                    break

        # Finalize (always — metrics and prediction summary are valuable on partial runs)
        finalize(ctx, metrics_proc)

        if failed:
            progress(f"  FAILED: {', '.join(failed)}")
            progress(f"  Completed: {len(selected) - len(failed)}/{len(selected)}")

    except Exception as e:
        progress("")
        progress(f"  FATAL: {e}")
        traceback.print_exc(file=sys.stderr)
    finally:
        # Ensure metrics collector is stopped even if calibration/baseline crashes.
        # finalize() calls stop_metrics_collector() on the happy path; this catches
        # the case where we never reach finalize().
        if metrics_proc is not None:
            stop_metrics_collector(metrics_proc)
        writer.close()


if __name__ == "__main__":
    main()
