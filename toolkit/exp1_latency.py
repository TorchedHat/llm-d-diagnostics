#!/usr/bin/env python3
"""
Experiment 1: Single-Request Latency

Measures per-request overhead of disaggregation across prompt lengths.
Sequential requests, no concurrency. Uses PinnedConnection for pod pinning
and interleaved config ordering to eliminate temporal bias.

Usage (in-pod):
    python3 /scripts/toolkit/exp1_latency.py

Override defaults:
    MODEL="meta-llama/..." RUNS=10 python3 exp1_latency.py

Additional env vars:
    PROMPT_LENGTHS   Comma-separated token targets (default: 10,50,100,500,1000)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    DATA_DIR,
    EPP_URL,
    MAX_TOKENS,
    RUNS,
    PinnedConnection,
    build_prompt,
    decode_pod_url_by_ip,
    discover_pod_ips,
    dot,
    env,
    epp_enabled,
    prefill_pod_host_port,
    prefill_pod_url_by_ip,
    print_config,
    progress,
    write_run_info,
)
from schemas import ConfigExp1, Exp1Row, TypedCSVWriter

PROMPT_LENGTHS = [int(x) for x in env("PROMPT_LENGTHS", "10,50,100,500,1000").split(",")]



def main():
    outfile = os.path.join(DATA_DIR, "exp1-results.csv")
    write_run_info("exp1", {"prompt_lengths": PROMPT_LENGTHS})
    writer = TypedCSVWriter(outfile, Exp1Row)

    progress("=== Experiment 1: Single-Request Latency ===")
    print_config()
    progress(f"  Prompt lengths: {PROMPT_LENGTHS}")
    progress(f"  Output: {outfile}")
    progress("")

    # ── Discover pods and create pinned connections ──────────────────────────
    progress("Discovering pods...")
    prefill_pods = discover_pod_ips("app=vllm-prefill")
    decode_pods = discover_pod_ips("app=vllm-decode")

    if not prefill_pods:
        progress("ERROR: No prefill pods found")
        sys.exit(1)
    if not decode_pods:
        progress("ERROR: No decode pods found")
        sys.exit(1)

    prefill_name, prefill_ip = prefill_pods[0]
    decode1_name, decode1_ip = decode_pods[0]

    progress(f"  Prefill: {prefill_name} ({prefill_ip})")
    progress(f"  Decode1: {decode1_name} ({decode1_ip})")

    prefill_host_port = prefill_pod_host_port(prefill_ip)

    conns = [
        (ConfigExp1.BASELINE, PinnedConnection(
            prefill_pod_url_by_ip(prefill_ip),
            pod_name=prefill_name,
        )),
        (ConfigExp1.DISAGG_D1, PinnedConnection(
            decode_pod_url_by_ip(decode1_ip),
            pod_name=decode1_name,
            extra_headers={"x-prefiller-host-port": prefill_host_port},
        )),
    ]

    if epp_enabled():
        conns.append((ConfigExp1.DISAGG_EPP, PinnedConnection(
            EPP_URL, pod_name="epp-gateway")))

    if len(decode_pods) >= 2:
        decode2_name, decode2_ip = decode_pods[1]
        progress(f"  Decode2: {decode2_name} ({decode2_ip})")
        conns.append((ConfigExp1.DISAGG_D2, PinnedConnection(
            decode_pod_url_by_ip(decode2_ip),
            pod_name=decode2_name,
            extra_headers={"x-prefiller-host-port": prefill_host_port},
        )))
    else:
        progress("  Decode2: (skipped, only 1 decode pod found)")

    progress("")
    progress("Pinned connections:")
    for config_name, conn in conns:
        progress(f"  {config_name}: {conn}")
    progress("")

    # ── Run experiment (interleaved) ────────────────────────────────────────
    try:
        for ptokens in PROMPT_LENGTHS:
            prompt = build_prompt(ptokens)
            progress(f"--- Prompt target: {ptokens} tokens ---")

            # Warmup all configs for this prompt length
            progress("  Warmup: ", end="")
            for _config_name, conn in conns:
                conn.warmup(prompt, MAX_TOKENS)
                dot()
            progress(" done")

            # Measured runs — interleaved across configs
            progress("  Runs:   ", end="")
            for run in range(1, RUNS + 1):
                for config_name, conn in conns:
                    r = conn.send(prompt, MAX_TOKENS)

                    writer.write({
                        "experiment": "exp1",
                        "config": config_name,
                        "run": run,
                        "prompt_tokens_target": ptokens,
                        "max_tokens": MAX_TOKENS,
                        "ttft_ms": r.ttft_ms,
                        "total_ms": r.total_ms,
                        "status_code": r.status,
                        "prompt_tokens_actual": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "pod": conn.pod_name,
                        "error": r.error,
                    })
                    dot()
            progress(" done")
            progress("")

    finally:
        for _, conn in conns:
            conn.close()

    writer.close()
    progress(f"=== Experiment 1 Complete === ({outfile})")


if __name__ == "__main__":
    main()
