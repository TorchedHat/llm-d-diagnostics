#!/usr/bin/env python3
"""
Experiment 6: Saturation Profiling

Gradually increases QPS to find where each deployment topology saturates.
Uses rate-controlled (open-loop) injection: requests are scheduled at fixed
wall-clock times regardless of whether previous requests have completed.

This is different from exp2 (which uses concurrent batches) because it
measures latency under sustained load at a controlled rate, revealing the
knee in the latency curve where the system transitions from healthy to
saturated.

Key metric: the QPS at which p99 latency exceeds SLO_MULT × baseline p50.
The gap between monolithic and disaggregated saturation points is the
scaling dividend — the quantified benefit of disaggregation.

Usage:
    python3 toolkit/exp6_saturation.py
    QPS_LEVELS=1,2,4,8,16 DURATION_S=60 python3 toolkit/exp6_saturation.py

Additional env vars:
    QPS_LEVELS    Comma-separated target QPS values (default: 1,2,4,8,12,16,24,32)
    DURATION_S    Measurement window per QPS level in seconds (default: 30)
    PROMPT_TOKENS Prompt size in tokens (default: 50)
    SLO_MULT      p99 > SLO_MULT * baseline_p50 = saturated (default: 2.0)
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    BASELINE_URL,
    DATA_DIR,
    DISAGG_D1_URL,
    DISAGG_D2_URL,
    EPP_URL,
    MAX_TOKENS,
    PREFILL_HOST,
    WARMUP,
    build_prompt,
    dot,
    env,
    epp_enabled,
    print_config,
    progress,
    send_request,
    write_run_info,
)
from schemas import ConfigThroughput, Exp6Row, TypedCSVWriter

QPS_LEVELS = [float(x) for x in env("QPS_LEVELS", "1,2,4,8,12,16,24,32").split(",")]
DURATION_S = int(env("DURATION_S", "30"))
PROMPT_TOKENS = int(env("PROMPT_TOKENS", "50"))
SLO_MULT = float(env("SLO_MULT", "2.0"))

PROMPT = build_prompt(PROMPT_TOKENS)
DISAGG_HEADERS = {"x-prefiller-host-port": PREFILL_HOST}

CONFIGS = [
    (ConfigThroughput.BASELINE,  "direct to monolithic vLLM"),
    (ConfigThroughput.DISAGG_1D, "sidecar -> prefill -> NIXL -> 1 decode"),
    (ConfigThroughput.DISAGG_2D, "sidecar -> prefill -> NIXL -> 2 decodes (round-robin)"),
]
if epp_enabled():
    CONFIGS.append((ConfigThroughput.DISAGG_EPP, "EPP Gateway -> scheduled prefill + decode"))


def pick_url(config_name, seq):
    """Select the URL for a given config and sequence number."""
    if config_name == ConfigThroughput.BASELINE:
        return BASELINE_URL
    elif config_name == ConfigThroughput.DISAGG_EPP:
        return EPP_URL
    elif config_name == ConfigThroughput.DISAGG_1D:
        return DISAGG_D1_URL
    else:  # DISAGG-2D: round-robin
        return DISAGG_D1_URL if seq % 2 == 1 else DISAGG_D2_URL


def pick_headers(config_name):
    """Select headers for a given config."""
    if config_name in {ConfigThroughput.BASELINE, ConfigThroughput.DISAGG_EPP}:
        return None
    return DISAGG_HEADERS


def rate_controlled_run(config_name, qps, duration_s, writer):
    """Sustain `qps` requests/sec for `duration_s` seconds.

    Uses a slot-based scheduler: request i departs at time start + i/qps.
    Each request runs in its own thread. The max_workers cap provides
    backpressure — if the server can't keep up, threads block and
    depart_delay_ms grows, revealing saturation.

    Returns list of RequestResult objects for analysis.
    """
    interval = 1.0 / qps
    total_requests = int(qps * duration_s)
    max_workers = min(max(int(qps * 4), 16), 128)
    headers = pick_headers(config_name)

    results = []
    results_lock = threading.Lock()

    def do_request(scheduled_time, seq):
        # Wait until our scheduled departure time
        now = time.monotonic()
        if scheduled_time > now:
            time.sleep(scheduled_time - now)

        actual_depart = time.monotonic()
        url = pick_url(config_name, seq)
        r = send_request(url, PROMPT, MAX_TOKENS, extra_headers=headers)

        depart_delay = round((actual_depart - scheduled_time) * 1000, 2)

        writer.write({
            "experiment": "exp6",
            "config": config_name,
            "qps_target": qps,
            "seq": seq,
            "depart_delay_ms": depart_delay,
            "ttft_ms": r.ttft_ms,
            "total_ms": r.total_ms,
            "status_code": r.status,
            "completion_tokens": r.completion_tokens,
            "error": r.error,
            "pod": "service-lb",
        })
        dot()

        with results_lock:
            results.append(r)

        return r

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i in range(total_requests):
            scheduled = start + i * interval
            # Stagger submission: sleep in the main thread to avoid
            # queueing all tasks at once (avoids memory spike at high QPS)
            now = time.monotonic()
            if scheduled - now > 0.1:
                time.sleep(max(0, scheduled - now - 0.05))
            futures.append(pool.submit(do_request, scheduled, i + 1))

        # Wait for all requests to complete
        for f in futures:
            f.result()

    wall_s = time.monotonic() - start
    actual_qps = total_requests / wall_s if wall_s > 0 else 0
    return results, actual_qps


def main():
    outfile = os.path.join(DATA_DIR, "exp6-results.csv")
    write_run_info("exp6", {"qps_levels": QPS_LEVELS, "duration_s": DURATION_S,
                            "prompt_tokens": PROMPT_TOKENS, "slo_mult": SLO_MULT})
    writer = TypedCSVWriter(outfile, Exp6Row)

    progress("=== Experiment 6: Saturation Profiling ===")
    print_config()
    progress(f"  QPS levels: {QPS_LEVELS}")
    progress(f"  Duration per level: {DURATION_S}s")
    progress(f"  SLO threshold: p99 > {SLO_MULT}x baseline p50")
    progress(f"  Output: {outfile}")
    progress("")

    saturated = set()
    # Warm up all configs first
    for config_name, _desc in CONFIGS:
        for i in range(WARMUP):
            url = pick_url(config_name, i + 1)
            send_request(url, PROMPT, MAX_TOKENS, extra_headers=pick_headers(config_name))

    for i, qps in enumerate(QPS_LEVELS):
        progress(f"--- QPS: {qps} ---")

        # Cooldown between QPS levels (not between configs at same QPS)
        if i > 0:
            time.sleep(5)

        for config_name, _desc in CONFIGS:
            if config_name in saturated:
                progress(f"  {config_name}: SKIPPED (saturated)")
                continue

            progress(f"  {config_name}: ", end="")

            results, actual_qps = rate_controlled_run(config_name, qps, DURATION_S, writer)

            ok = sum(1 for r in results if r.status == 200)
            fail_rate = 1.0 - (ok / len(results)) if results else 0
            throughput_ratio = actual_qps / qps if qps > 0 else 1.0
            progress(f" {ok}/{len(results)} OK (actual {actual_qps:.1f} req/s)")

            if fail_rate > 0.10 or throughput_ratio < 0.50:
                progress(f"    SATURATED (fail={fail_rate:.0%}, throughput={throughput_ratio:.0%})")
                saturated.add(config_name)

        if len(saturated) == len(CONFIGS):
            progress("  All configs saturated, stopping")
            break

    progress("")

    writer.close()
    progress("")
    progress(f"=== Experiment 6 Complete === ({outfile})")


if __name__ == "__main__":
    main()
