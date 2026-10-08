#!/usr/bin/env python3
"""
Experiment 5: Sequence Length Sweep

Runs the exp1b 4-config decomposition at multiple prompt lengths to find the
crossover point where KV cache transfer time dominates TTFT.

KV cache size grows linearly with sequence length. At short sequences the
transfer tax is negligible; at some crossover point, transfer time exceeds
prefill compute time. Finding that crossover is the most important
characterization of a disaggregated deployment.

Configs (identical to exp1b):
    A. Baseline:        client -> prefill vLLM (pod port 8000; SIM uses 8100)
    B. Direct decode:   client -> decode vLLM (8001), no sidecar
    C. Sidecar-only:    client -> sidecar (8000) -> decode vLLM (8001), no disagg
    D. Disaggregated:   client -> sidecar (8000) -> prefill -> NIXL -> decode

Derived at each sequence length:
    T_sidecar    = C - B
    T_transfer   = D - C  (includes NIXL KV transfer)
    T_overhead   = D - A

Crossover: the prompt length where T_transfer > median(A).

Pod pinning: each config gets a PinnedConnection created once at startup.
Connections persist across all prompt lengths, so TLS handshake cost is
amortized and every measurement hits the same pod (no service LB jitter).

Usage:
    python3 toolkit/exp5_seqlen_sweep.py
    SWEEP_LENGTHS=10,100,1000,4096 python3 toolkit/exp5_seqlen_sweep.py

Additional env vars:
    SWEEP_LENGTHS   Comma-separated prompt token targets (default: 10,50,100,250,500,1000,2000,4096)
    RUNS            Runs per config per length (default: 30)
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
from schemas import ConfigDecompose, Exp5Row, TypedCSVWriter

SWEEP_LENGTHS = [int(x) for x in env("SWEEP_LENGTHS", "10,50,100,250,500,1000,2000,4096").split(",")]
SWEEP_RUNS = int(env("RUNS", "30"))



def main():
    outfile = os.path.join(DATA_DIR, "exp5-results.csv")
    write_run_info("exp5", {"sweep_lengths": SWEEP_LENGTHS, "runs": SWEEP_RUNS})
    writer = TypedCSVWriter(outfile, Exp5Row)

    progress("=== Experiment 5: Sequence Length Sweep ===")
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

    # ── Create pinned connections (persist across all prompt lengths) ──
    conn_a = PinnedConnection(
        prefill_pod_url_by_ip(prefill_ip),
        pod_name=prefill_name,
    )
    conn_b = PinnedConnection(
        decode_direct_url_by_ip(decode_ip),
        pod_name=decode_name,
    )
    conn_c = PinnedConnection(
        decode_pod_url_by_ip(decode_ip),
        pod_name=decode_name,
    )
    conn_d = PinnedConnection(
        decode_pod_url_by_ip(decode_ip),
        pod_name=decode_name,
        extra_headers={"x-prefiller-host-port": prefill_pod_host_port(prefill_ip)},
    )
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

    progress("")
    for name, conn, desc in CONFIGS:
        progress(f"  {name}: {desc}")
        progress(f"    {conn}")
    progress("")
    progress(f"  Sweep lengths: {SWEEP_LENGTHS}")
    progress(f"  Runs per config per length: {SWEEP_RUNS}")
    progress(f"  Output: {outfile}")
    progress("")

    # Track configs that hit the context window limit (all 400s at a length).
    # Once a config hits 400 at length N, skip it for all longer lengths.
    configs_maxed = set()

    try:
        for ptokens in SWEEP_LENGTHS:
            warmup_prompt = build_prompt(ptokens)
            progress(f"--- Prompt target: {ptokens} tokens ---")

            active_configs = [(n, c, d) for n, c, d in CONFIGS
                              if n not in configs_maxed]
            if not active_configs:
                progress("  All configs maxed out, stopping sweep")
                break

            # Warm-up all active configs at this prompt length
            for _config_name, conn, _desc in active_configs:
                conn.warmup(warmup_prompt, MAX_TOKENS)

            # Interleaved runs: cycle through all active configs per run.
            # This ensures paired differences (C-B, D-C) cancel time-varying
            # noise (GPU thermal drift, OS scheduling).
            progress(f"  Interleaved ({', '.join(n for n, _, _ in active_configs)}): ",
                     end="")
            statuses_by_config = {n: [] for n, _, _ in active_configs}

            for run in range(1, SWEEP_RUNS + 1):
                prompt = build_prompt(ptokens, cache_bust=(ptokens, run))
                for config_name, conn, _desc in active_configs:
                    r = conn.send(prompt, MAX_TOKENS)

                    writer.write({
                        "experiment": "exp5",
                        "config": config_name,
                        "pod": conn.pod_name,
                        "prompt_tokens_target": ptokens,
                        "run": run,
                        "ttft_ms": r.ttft_ms,
                        "total_ms": r.total_ms,
                        "status_code": r.status,
                        "prompt_tokens_actual": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "error": r.error,
                    })
                    statuses_by_config[config_name].append(r.status)
                dot()

            progress(" done")

            # Check for context window limit per config
            for config_name, statuses in statuses_by_config.items():
                if statuses and all(s == 400 for s in statuses):
                    progress(f"  Skipping {config_name} at {ptokens}+ tokens: "
                             "prompt exceeds model context window")
                    configs_maxed.add(config_name)

            progress("")

    finally:
        conn_a.close()
        conn_b.close()
        conn_c.close()
        conn_d.close()
        if conn_epp:
            conn_epp.close()

    writer.close()
    progress("")
    progress(f"=== Experiment 5 Complete === ({outfile})")


if __name__ == "__main__":
    main()
