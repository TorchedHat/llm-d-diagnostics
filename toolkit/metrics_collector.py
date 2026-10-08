#!/usr/bin/env python3
"""
Metrics Collector — continuous Prometheus scraper for vLLM pods.

Runs inside the test-client pod. Scrapes /metrics from all vLLM endpoints
at a configurable interval. Writes time-aligned CSV for correlation with
experiment data.

Usage:
    # Start collecting (runs until killed or duration expires)
    python3 metrics_collector.py

    # Collect for 60 seconds
    COLLECT_DURATION=60 python3 metrics_collector.py

    # Custom endpoints and interval
    METRICS_ENDPOINTS="prefill=http://vllm-prefill-svc:8000,decode=http://vllm-decode-direct-svc:8001" \
    SAMPLE_INTERVAL=1 \
    python3 metrics_collector.py

Environment variables:
    METRICS_ENDPOINTS   Comma-separated name=url pairs (auto-detected if unset)
    SAMPLE_INTERVAL     Seconds between scrapes (default: 2)
    COLLECT_DURATION    Max collection time in seconds (default: unlimited)
    DATA_DIR            Output directory (default: data)
    NS                  Namespace for auto-detection (default: from env)

Output:
    data/metrics-timeseries.csv   Per-scrape gauge/counter values
    data/metrics-snapshots/       Raw Prometheus text snapshots (one per scrape)
"""

import csv
import http.client
import json
import os
import re
import signal
import ssl
import sys
import threading
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))

# ── Configuration ────────────────────────────────────────────────────────────

def env(key, default=""):
    return os.environ.get(key, default)

SAMPLE_INTERVAL = float(env("SAMPLE_INTERVAL", "2"))
COLLECT_DURATION = float(env("COLLECT_DURATION", "0"))  # 0 = unlimited
DATA_DIR = env("DATA_DIR", "data")

# Metrics we care about — everything else is ignored.
# Gauges are sampled directly; counters are recorded as-is (deltas computed in analysis).
METRICS_OF_INTEREST = {
    # NIXL transfer telemetry
    "vllm:nixl_xfer_time_seconds": "histogram",
    "vllm:nixl_bytes_transferred": "histogram",
    "vllm:nixl_num_descriptors": "histogram",
    "vllm:nixl_num_failed_transfers_total": "counter",
    "vllm:nixl_num_failed_notifications_total": "counter",
    "vllm:nixl_num_kv_expired_reqs_total": "counter",
    "vllm:nixl_post_time_seconds": "histogram",

    # Engine state
    "vllm:kv_cache_usage_perc": "gauge",
    "vllm:num_requests_running": "gauge",
    "vllm:num_requests_waiting": "gauge",
    "vllm:num_preemptions_total": "counter",
    "vllm:engine_sleep_state": "gauge",

    # Request-level (server-side ground truth)
    "vllm:time_to_first_token_seconds": "histogram",
    "vllm:inter_token_latency_seconds": "histogram",
    "vllm:e2e_request_latency_seconds": "histogram",
    "vllm:request_prefill_time_seconds": "histogram",
    "vllm:request_decode_time_seconds": "histogram",
    "vllm:request_queue_time_seconds": "histogram",

    # Throughput
    "vllm:prompt_tokens_total": "counter",
    "vllm:generation_tokens_total": "counter",
    "vllm:request_success_total": "counter",

    # Compute
    "vllm:estimated_flops_per_gpu_total": "counter",

    # Process-level
    "process_cpu_seconds_total": "counter",
    "process_resident_memory_bytes": "gauge",
}

# CSV fields for the timeseries output
CSV_FIELDS = [
    "timestamp", "epoch_ms", "endpoint", "scrape_ms",
    # NIXL
    "nixl_xfer_time_sum", "nixl_xfer_time_count",
    "nixl_bytes_sum", "nixl_bytes_count",
    "nixl_failed_transfers", "nixl_failed_notifications",
    "nixl_kv_expired",
    # Engine state
    "kv_cache_usage_pct", "requests_running", "requests_waiting",
    "preemptions",
    # Request latency (histogram sums and counts for computing means)
    "ttft_sum", "ttft_count",
    "itl_sum", "itl_count",
    "e2e_sum", "e2e_count",
    "prefill_time_sum", "prefill_time_count",
    "decode_time_sum", "decode_time_count",
    "queue_time_sum", "queue_time_count",
    # Throughput
    "prompt_tokens", "generation_tokens", "request_success",
    # Compute
    "flops",
    # Process
    "cpu_seconds", "rss_bytes",
    "epp_scheduler_e2e_sum_seconds", "epp_scheduler_e2e_count",
]


def parse_endpoints():
    """Parse METRICS_ENDPOINTS env var or auto-detect from service names."""
    raw = env("METRICS_ENDPOINTS")
    if raw:
        endpoints = {}
        for pair in raw.split(","):
            name, url = pair.strip().split("=", 1)
            endpoints[name.strip()] = url.strip()
        if env("EPP_URL") and "epp" not in endpoints:
            release = env("EPP_RELEASE_NAME", "llm-d-epp")
            service = env("EPP_SERVICE_NAME", f"{release}-epp")
            endpoints["epp"] = env("EPP_METRICS_URL", f"http://{service}:9090")
        return endpoints

    # Auto-detect: use short service names (works within the same namespace).
    # The prefill Service is headless, so DNS returns pod IPs and clients use
    # the pod port directly. Inference-sim listens on 8100; vLLM listens on 8000.
    endpoints = {
        "prefill": f"http://vllm-prefill-svc:{8100 if env('SIM') == '1' else 8000}",
        "decode": "http://vllm-decode-direct-svc:8001",
    }
    if env("EPP_URL"):
        release = env("EPP_RELEASE_NAME", "llm-d-epp")
        service = env("EPP_SERVICE_NAME", f"{release}-epp")
        endpoints["epp"] = env("EPP_METRICS_URL", f"http://{service}:9090")
    return endpoints


def scrape_metrics(url, timeout=5):
    """Scrape /metrics from a vLLM or EPP endpoint. Returns raw text or None."""
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        if parsed.scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout,
                                               context=ssl.create_default_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        path = parsed.path.rstrip("/")
        if not path.endswith("/metrics"):
            path += "/metrics"
        conn.request("GET", path)
        resp = conn.getresponse()
        if resp.status == 200:
            body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            return body
        conn.close()
    except Exception:
        pass
    return None


def parse_prometheus_text(text):
    """Parse Prometheus exposition format into a dict of metric_name -> value(s).

    For gauges/counters: returns the scalar value.
    For histograms: returns {_sum, _count, _bucket} values.
    Ignores _created metrics and lines starting with #.
    """
    metrics = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Parse: metric_name{labels} value [timestamp]
        # or:   metric_name value [timestamp]
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?([^}]*)\}?\s+(\S+)', line)
        if not match:
            continue
        name = match.group(1)
        value_str = match.group(3)

        # Skip _created metrics (not useful for us)
        if name.endswith("_created"):
            continue

        try:
            value = float(value_str)
        except ValueError:
            continue

        # Store with full name (including _sum, _count, _bucket suffixes).
        # Last value wins for counters/gauges (usually only one line).
        # For histograms, _sum and _count each appear once; _bucket lines
        # have different label sets but share the metric name prefix —
        # we only use _sum/_count in extract_row(), so overwriting is fine.
        metrics[name] = value

    return metrics


def extract_row(endpoint_name, metrics):
    """Extract a CSV row from parsed Prometheus metrics."""
    def g(name):
        """Get metric value, return 0 if missing."""
        return metrics.get(name, 0)

    return {
        "endpoint": endpoint_name,
        # NIXL
        "nixl_xfer_time_sum": g("vllm:nixl_xfer_time_seconds_sum"),
        "nixl_xfer_time_count": g("vllm:nixl_xfer_time_seconds_count"),
        "nixl_bytes_sum": g("vllm:nixl_bytes_transferred_sum"),
        "nixl_bytes_count": g("vllm:nixl_bytes_transferred_count"),
        "nixl_failed_transfers": g("vllm:nixl_num_failed_transfers_total"),
        "nixl_failed_notifications": g("vllm:nixl_num_failed_notifications_total"),
        "nixl_kv_expired": g("vllm:nixl_num_kv_expired_reqs_total"),
        # Engine state
        "kv_cache_usage_pct": round(g("vllm:kv_cache_usage_perc") * 100, 2),
        "requests_running": int(g("vllm:num_requests_running")),
        "requests_waiting": int(g("vllm:num_requests_waiting")),
        "preemptions": g("vllm:num_preemptions_total"),
        # Request latency
        "ttft_sum": g("vllm:time_to_first_token_seconds_sum"),
        "ttft_count": g("vllm:time_to_first_token_seconds_count"),
        "itl_sum": g("vllm:inter_token_latency_seconds_sum"),
        "itl_count": g("vllm:inter_token_latency_seconds_count"),
        "e2e_sum": g("vllm:e2e_request_latency_seconds_sum"),
        "e2e_count": g("vllm:e2e_request_latency_seconds_count"),
        "prefill_time_sum": g("vllm:request_prefill_time_seconds_sum"),
        "prefill_time_count": g("vllm:request_prefill_time_seconds_count"),
        "decode_time_sum": g("vllm:request_decode_time_seconds_sum"),
        "decode_time_count": g("vllm:request_decode_time_seconds_count"),
        "queue_time_sum": g("vllm:request_queue_time_seconds_sum"),
        "queue_time_count": g("vllm:request_queue_time_seconds_count"),
        # Throughput
        "prompt_tokens": g("vllm:prompt_tokens_total"),
        "generation_tokens": g("vllm:generation_tokens_total"),
        "request_success": g("vllm:request_success_total"),
        # Compute
        "flops": g("vllm:estimated_flops_per_gpu_total"),
        # Process
        "cpu_seconds": g("process_cpu_seconds_total"),
        "rss_bytes": g("process_resident_memory_bytes"),
        "epp_scheduler_e2e_sum_seconds": g(
            "llm_d_epp_scheduler_e2e_duration_seconds_sum"),
        "epp_scheduler_e2e_count": g(
            "llm_d_epp_scheduler_e2e_duration_seconds_count"),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    snapshot_dir = os.path.join(DATA_DIR, "metrics-snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    endpoints = parse_endpoints()
    outfile = os.path.join(DATA_DIR, "metrics-timeseries.csv")

    print("Metrics collector starting", file=sys.stderr)
    print(f"  Endpoints: {json.dumps(endpoints)}", file=sys.stderr)
    print(f"  Interval: {SAMPLE_INTERVAL}s", file=sys.stderr)
    print(f"  Duration: {'unlimited' if not COLLECT_DURATION else f'{COLLECT_DURATION}s'}", file=sys.stderr)
    print(f"  Output: {outfile}", file=sys.stderr)
    print(f"  Snapshots: {snapshot_dir}", file=sys.stderr)

    # Open CSV
    f = open(outfile, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    writer.writeheader()

    stop = threading.Event()

    def handle_signal(signum, frame):
        print(f"\nSignal {signum} received, stopping...", file=sys.stderr)
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    start_time = time.time()
    sample_num = 0

    while not stop.is_set():
        if COLLECT_DURATION and (time.time() - start_time) >= COLLECT_DURATION:
            break

        sample_num += 1
        ts = time.strftime("%H:%M:%S")
        epoch = int(time.time() * 1000)

        for name, url in endpoints.items():
            scrape_start = time.monotonic()
            raw = scrape_metrics(url)
            scrape_ms = round((time.monotonic() - scrape_start) * 1000, 1)

            if raw is None:
                # Endpoint unreachable — write a row with zeros and a note
                row = {field: 0 for field in CSV_FIELDS}
                row["timestamp"] = ts
                row["epoch_ms"] = epoch
                row["endpoint"] = name
                row["scrape_ms"] = -1  # sentinel for failed scrape
                writer.writerow(row)
                continue

            # Save raw snapshot (one file per endpoint per sample)
            snap_path = os.path.join(snapshot_dir,
                                     f"{sample_num:06d}-{name}.txt")
            with open(snap_path, "w") as sf:
                sf.write(raw)

            metrics = parse_prometheus_text(raw)
            row = extract_row(name, metrics)
            row["timestamp"] = ts
            row["epoch_ms"] = epoch
            row["scrape_ms"] = scrape_ms
            writer.writerow(row)

        f.flush()

        elapsed = time.time() - start_time
        if sample_num % 10 == 0:
            print(f"  [{ts}] sample {sample_num} ({elapsed:.0f}s elapsed)",
                  file=sys.stderr)

        stop.wait(SAMPLE_INTERVAL)

    f.close()
    print(f"\nCollected {sample_num} samples to {outfile}", file=sys.stderr)


if __name__ == "__main__":
    main()
