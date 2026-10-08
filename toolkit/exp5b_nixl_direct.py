#!/usr/bin/env python3
"""
Experiment 5b: Direct NIXL Transfer Physics

Measures NIXL transfer time and bytes by scraping decode-pod prometheus
metrics before and after each D-config request.  The delta gives us the
actual transfer time and bytes per request — direct measurement, immune
to prefix cache artifacts that contaminate D-C TTFT subtraction.

The key regression:
    nixl_xfer_time_delta_ms = protocol_ms + nixl_bytes_delta / (eff_bw * 1e6)

Gives us protocol_ms (intercept) and eff_bw (slope) from actual measured
bytes and times — no KV calculation needed.

Usage:
    python3 toolkit/exp5b_nixl_direct.py
    SWEEP_LENGTHS=10,100,1000 RUNS=30 python3 toolkit/exp5b_nixl_direct.py

Env vars:
    SWEEP_LENGTHS   Comma-separated prompt token targets (default: 10,50,100,250,500,1000)
    RUNS            Runs per length (default: 30)
    NS, MODEL, DATA_DIR, MAX_TOKENS — see client.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from metrics_collector import parse_prometheus_text, scrape_metrics

from client import (
    DATA_DIR,
    MAX_TOKENS,
    PinnedConnection,
    build_prompt,
    decode_pod_url_by_ip,
    discover_pod_ips,
    dot,
    env,
    prefill_pod_host_port,
    print_config,
    progress,
    write_run_info,
)
from schemas import Exp5bRow, TypedCSVWriter

SWEEP_LENGTHS = [int(x) for x in env("SWEEP_LENGTHS", "10,50,100,250,500,1000").split(",")]
SWEEP_RUNS = int(env("RUNS", "30"))

NIXL_TIME_SUM = "vllm:nixl_xfer_time_seconds_sum"
NIXL_TIME_COUNT = "vllm:nixl_xfer_time_seconds_count"
NIXL_BYTES_SUM = "vllm:nixl_bytes_transferred_sum"


def _scrape_nixl(decode_ip):
    """Scrape NIXL metrics from decode pod. Returns parsed dict or None."""
    url = f"http://{decode_ip}:8001"
    raw = scrape_metrics(url)
    if raw is None:
        return None
    return parse_prometheus_text(raw)


def _nixl_deltas(before, after):
    """Compute NIXL metric deltas. Returns (bytes, time_ms, transfers) or None."""
    if before is None or after is None:
        return None

    time_before = before.get(NIXL_TIME_SUM, 0)
    time_after = after.get(NIXL_TIME_SUM, 0)
    bytes_before = before.get(NIXL_BYTES_SUM, 0)
    bytes_after = after.get(NIXL_BYTES_SUM, 0)
    count_before = before.get(NIXL_TIME_COUNT, 0)
    count_after = after.get(NIXL_TIME_COUNT, 0)

    # Counter reset detection
    if count_after < count_before:
        return None

    return (
        bytes_after - bytes_before,
        (time_after - time_before) * 1000,  # seconds → ms
        int(count_after - count_before),
    )


def _preflight(conn_d, decode_ip):
    """Verify NIXL metrics exist and update per-transfer. Aborts if not."""
    progress("  Pre-flight: checking NIXL metrics...")

    before = _scrape_nixl(decode_ip)
    if before is None:
        progress("  FATAL: cannot scrape metrics from decode pod")
        sys.exit(1)

    has_time = NIXL_TIME_SUM in before
    has_bytes = NIXL_BYTES_SUM in before
    has_count = NIXL_TIME_COUNT in before

    if not (has_time and has_bytes and has_count):
        missing = []
        if not has_time:
            missing.append(NIXL_TIME_SUM)
        if not has_bytes:
            missing.append(NIXL_BYTES_SUM)
        if not has_count:
            missing.append(NIXL_TIME_COUNT)
        progress(f"  FATAL: NIXL metrics not found: {', '.join(missing)}")
        progress("  Check vLLM version and NIXL instrumentation.")
        sys.exit(1)

    prompt = build_prompt(50, cache_bust=("preflight", 0))
    r = conn_d.send(prompt, MAX_TOKENS)
    if r.status != 200:
        progress(f"  FATAL: pre-flight request failed (status={r.status}, error={r.error})")
        sys.exit(1)

    after = _scrape_nixl(decode_ip)
    deltas = _nixl_deltas(before, after)
    if deltas is None:
        progress("  FATAL: metrics scrape failed after pre-flight request")
        sys.exit(1)

    bytes_d, time_d, count_d = deltas

    progress(f"    nixl_bytes_delta:    {bytes_d:,.0f}")
    progress(f"    nixl_time_delta_ms:  {time_d:.2f}")
    progress(f"    nixl_transfers:      {count_d}")

    if count_d == 0:
        progress("  FATAL: NIXL transfer count did not increase after D-config request.")
        progress("  Metrics exist but are not updating — NIXL may not be active.")
        sys.exit(1)

    if bytes_d <= 0:
        progress("  WARNING: nixl_bytes_delta <= 0 — unexpected")

    if count_d != 1:
        progress(f"  NOTE: expected 1 transfer, got {count_d} — NIXL may chunk transfers")

    progress("  Pre-flight PASSED")
    return True


def main():
    outfile = os.path.join(DATA_DIR, "exp5b-results.csv")
    write_run_info("exp5b", {"sweep_lengths": SWEEP_LENGTHS, "runs": SWEEP_RUNS})
    writer = TypedCSVWriter(outfile, Exp5bRow)

    progress("=== Experiment 5b: Direct NIXL Transfer Physics ===")
    print_config()

    # ── Discover pods ────────────────────────────────────────────────────
    progress("  Discovering pods...")
    prefill_pods = discover_pod_ips("app=vllm-prefill")
    decode_pods = discover_pod_ips("app=vllm-decode")

    if not prefill_pods:
        progress("  ERROR: no prefill pods found (label: app=vllm-prefill)")
        sys.exit(1)
    if not decode_pods:
        progress("  ERROR: no decode pods found (label: app=vllm-decode)")
        sys.exit(1)

    prefill_name, prefill_ip = prefill_pods[0]
    decode_name, decode_ip = decode_pods[0]

    progress(f"  Prefill pod: {prefill_name} ({prefill_ip})")
    progress(f"  Decode pod:  {decode_name} ({decode_ip})")

    # ── D-config connection (sidecar → prefill → NIXL → decode) ──────
    conn_d = PinnedConnection(
        decode_pod_url_by_ip(decode_ip),
        pod_name=decode_name,
        extra_headers={"x-prefiller-host-port": prefill_pod_host_port(prefill_ip)},
    )

    progress(f"  D-config: {conn_d}")
    progress(f"  Metrics:  http://{decode_ip}:8001/metrics")
    progress(f"  Sweep lengths: {SWEEP_LENGTHS}")
    progress(f"  Runs per length: {SWEEP_RUNS}")
    progress(f"  Output: {outfile}")
    progress("")

    # ── Pre-flight check ─────────────────────────────────────────────────
    _preflight(conn_d, decode_ip)
    progress("")

    try:
        for ptokens in SWEEP_LENGTHS:
            progress(f"--- Prompt target: {ptokens} tokens ---")

            # Warmup at this sequence length
            warmup_prompt = build_prompt(ptokens)
            conn_d.warmup(warmup_prompt, MAX_TOKENS, n=1)

            progress(f"  Measuring {SWEEP_RUNS} runs: ", end="")
            for run in range(1, SWEEP_RUNS + 1):
                prompt = build_prompt(ptokens, cache_bust=(ptokens, run))

                before = _scrape_nixl(decode_ip)
                r = conn_d.send(prompt, MAX_TOKENS)
                after = _scrape_nixl(decode_ip)

                deltas = _nixl_deltas(before, after)

                if deltas is None:
                    nixl_bytes = ""
                    nixl_time = ""
                    nixl_transfers = ""
                    progress("X", end="")
                else:
                    bytes_d, time_d, count_d = deltas
                    nixl_bytes = str(int(bytes_d))
                    nixl_time = f"{time_d:.3f}"
                    nixl_transfers = str(count_d)

                    if count_d != 1:
                        progress(f"[{count_d}]", end="")
                    else:
                        dot()

                writer.write({
                    "experiment": "exp5b",
                    "pod": conn_d.pod_name,
                    "prompt_tokens_target": str(ptokens),
                    "prompt_tokens_actual": str(r.prompt_tokens),
                    "run": str(run),
                    "ttft_ms": str(r.ttft_ms),
                    "total_ms": str(r.total_ms),
                    "status_code": str(r.status),
                    "error": r.error,
                    "nixl_bytes_delta": nixl_bytes,
                    "nixl_xfer_time_delta_ms": nixl_time,
                    "nixl_transfers_delta": nixl_transfers,
                })

            progress(" done")
            progress("")

    finally:
        conn_d.close()

    writer.close()
    progress(f"=== Experiment 5b Complete === ({outfile})")


if __name__ == "__main__":
    main()
