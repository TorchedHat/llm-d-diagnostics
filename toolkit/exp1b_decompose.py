#!/usr/bin/env python3
"""
Experiment 1b: Latency Decomposition

Isolates each component of the disaggregated request path:
    A. Baseline:        client -> prefill vLLM (pod port 8000; SIM uses 8100)
    B. Direct decode:   client -> decode vLLM (8001), no sidecar
    C. Sidecar-only:    client -> sidecar (8000) -> decode vLLM (8001), no disagg
    D. Disaggregated:   client -> sidecar (8000) -> prefill -> NIXL -> decode

Derived (paired difference per run):
    T_sidecar    = C - B
    T_prefill_rt = D - C  (includes NIXL)
    T_overhead   = D - A

Methodology:
    Configs are INTERLEAVED within each run (round-robin: A,B,C,D per run)
    rather than run sequentially (all A, then all B, ...). This ensures
    each run's paired differences cancel time-dependent confounds (GPU
    thermal drift, OS scheduling, background load variations).

Usage: python3 /scripts/toolkit/exp1b_decompose.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    DATA_DIR,
    EPP_URL,
    MAX_TOKENS,
    PinnedConnection,
    build_prompt,
    decode_direct_url_by_ip,
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
from schemas import ConfigDecompose, Exp1bRow, TypedCSVWriter

DECOMPOSE_RUNS = int(env("RUNS", "30"))

DECOMPOSE_PROMPT = int(env("DECOMPOSE_PROMPT", "50"))
WARMUP_PROMPT = build_prompt(DECOMPOSE_PROMPT)



def main():
    outfile = os.path.join(DATA_DIR, "exp1b-results.csv")
    write_run_info("exp1b", {"decompose_prompt_tokens": DECOMPOSE_PROMPT,
                             "runs": DECOMPOSE_RUNS})
    writer = TypedCSVWriter(outfile, Exp1bRow)

    progress("=== Experiment 1b: Latency Decomposition ===")
    print_config()
    progress(f"  Runs per config: {DECOMPOSE_RUNS}")
    progress("  Interleaved: A,B,C,D per run (paired differences)")
    progress(f"  Output: {outfile}")
    progress("")

    # Discover pods
    progress("  Discovering pods...")
    prefill_pods = discover_pod_ips("app=vllm-prefill")
    decode_pods = discover_pod_ips("app=vllm-decode")

    if not prefill_pods:
        progress("  ERROR: No prefill pods found")
        sys.exit(1)
    if not decode_pods:
        progress("  ERROR: No decode pods found")
        sys.exit(1)

    prefill_pod_name, prefill_ip = prefill_pods[0]
    decode_pod_name, decode_ip = decode_pods[0]

    progress(f"  Pinned prefill pod: {prefill_pod_name} ({prefill_ip})")
    progress(f"  Pinned decode pod:  {decode_pod_name} ({decode_ip})")
    progress("")

    # Create pinned connections
    conn_a = PinnedConnection(
        prefill_pod_url_by_ip(prefill_ip), pod_name=prefill_pod_name)
    conn_b = PinnedConnection(
        decode_direct_url_by_ip(decode_ip), pod_name=decode_pod_name)
    conn_c = PinnedConnection(
        decode_pod_url_by_ip(decode_ip), pod_name=decode_pod_name)
    conn_d = PinnedConnection(
        decode_pod_url_by_ip(decode_ip), pod_name=decode_pod_name,
        extra_headers={"x-prefiller-host-port": prefill_pod_host_port(prefill_ip)})

    conn_epp = PinnedConnection(EPP_URL, pod_name="epp-gateway") if epp_enabled() else None

    CONFIGS = [
        (ConfigDecompose.A_PREFILL_DIRECT, conn_a,
         f"client -> {conn_a.url} (baseline)"),
        (ConfigDecompose.B_DECODE_DIRECT,  conn_b,
         "client -> decode:8001 (bypass sidecar)"),
        (ConfigDecompose.C_SIDECAR_ONLY,   conn_c,
         "client -> sidecar:8000 -> decode:8001 (no disagg)"),
        (ConfigDecompose.D_DISAGGREGATED,  conn_d,
         "client -> sidecar -> prefill -> NIXL -> decode"),
    ]
    if conn_epp:
        CONFIGS.append((ConfigDecompose.E_EPP_GATEWAY, conn_epp,
                        "client -> Gateway API -> EPP -> prefill + decode"))

    try:
        # Warm-up: each config gets WARMUP requests
        for config_name, conn, _desc in CONFIGS:
            progress(f"  Warming up {config_name}...")
            conn.warmup(WARMUP_PROMPT, unique=True)

        # Interleaved runs: cycle through all configs per run.
        # This makes run N's (A,B,C,D) measurements temporally adjacent,
        # so paired differences (C-B, D-C, D-A) cancel time-varying noise.
        progress(f"  Running {DECOMPOSE_RUNS} interleaved rounds...")
        for run in range(1, DECOMPOSE_RUNS + 1):
            for config_name, conn, _desc in CONFIGS:
                prompt = build_prompt(
                    DECOMPOSE_PROMPT, cache_bust=("exp1b", str(config_name), run))
                r = conn.send(prompt, MAX_TOKENS)

                writer.write({
                    "experiment": "exp1b",
                    "config": config_name,
                    "run": run,
                    "pod": conn.pod_name,
                    "ttft_ms": r.ttft_ms,
                    "total_ms": r.total_ms,
                    "status_code": r.status,
                    "completion_tokens": r.completion_tokens,
                    "error": r.error,
                })
            dot()

        progress(" done")
    finally:
        conn_a.close()
        conn_b.close()
        conn_c.close()
        conn_d.close()
        if conn_epp:
            conn_epp.close()

    writer.close()
    progress(f"=== Experiment 1b Complete === ({outfile})")


if __name__ == "__main__":
    main()
