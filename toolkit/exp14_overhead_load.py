#!/usr/bin/env python3
"""
Experiment 14: Overhead Decomposition Under Concurrent Load

Runs the 4-config decomposition (same as exp5) but under concurrent load
to measure how disaggregation overhead scales with concurrency.

At c=1 (exp5), disagg often wins on latency via pipelining. Under concurrent
load, coordination overhead may grow, explaining throughput loss. This
experiment isolates where the overhead lives.

Configs (pinned to specific pods via PinnedConnection):
    A. Baseline:        client -> prefill vLLM (pod port 8000; SIM uses 8100)
    B. Direct decode:   client -> decode vLLM (8001), no sidecar
    C. Sidecar-only:    client -> sidecar (8000) -> decode vLLM (8001)
    D. Disaggregated:   client -> sidecar -> prefill -> NIXL -> decode

Decomposition (B is the clean non-disagg baseline on the decode pod):
    C - B = sidecar + NIXL overhead (auto-routing)
    D - B = sidecar + NIXL overhead (explicit routing)
    D - C = routing delta (should be small)
    A     = informational (different pod, different GPU)

Uses a pool of PinnedConnections per config (one per concurrent worker)
since http.client is not thread-safe.

Usage: python3 toolkit/exp14_overhead_load.py

Env vars:
    SWEEP_LENGTHS   Comma-separated prompt token targets (default: 50,100,250,500,1000)
    CONCURRENCY     Concurrent requests per batch (default: 8)
    MAX_TOKENS      Output tokens per request (default: 20)
    TOTAL_REQUESTS  Requests per config per length (default: 24)
    RUNS            Alias for TOTAL_REQUESTS
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    DATA_DIR,
    EPP_URL,
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
from client import (
    MAX_TOKENS as DEFAULT_MAX_TOKENS,
)
from schemas import ConfigDecompose, Exp14Row, TypedCSVWriter

SWEEP_LENGTHS = [int(x) for x in env("SWEEP_LENGTHS", "50,100,250,500,1000").split(",")]
CONCURRENCY = int(env("CONCURRENCY", "8"))
MAX_TOKENS = int(env("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
TOTAL_REQUESTS = int(env("TOTAL_REQUESTS", env("RUNS", "24")))


def send_one_pinned(worker_idx, conn_pool, prompt, max_tokens):
    conn = conn_pool[worker_idx]
    return conn.send(prompt, max_tokens)


def main():
    outfile = os.path.join(DATA_DIR, "exp14-results.csv")
    write_run_info("exp14", {
        "sweep_lengths": SWEEP_LENGTHS,
        "concurrency": CONCURRENCY,
        "max_tokens": MAX_TOKENS,
        "total_requests": TOTAL_REQUESTS,
    })
    writer = TypedCSVWriter(outfile, Exp14Row)

    progress("=== Experiment 14: Overhead Decomposition Under Load ===")
    print_config()

    progress("  Discovering pods...")
    prefill_pods = discover_pod_ips("app=vllm-prefill")
    decode_pods = discover_pod_ips("app=vllm-decode")

    if not prefill_pods:
        progress("  ERROR: no prefill pods found")
        sys.exit(1)
    if not decode_pods:
        progress("  ERROR: no decode pods found")
        sys.exit(1)

    prefill_name, prefill_ip = prefill_pods[0]
    decode_name, decode_ip = decode_pods[0]

    progress(f"  Prefill pod: {prefill_name} ({prefill_ip})")
    progress(f"  Decode pod:  {decode_name} ({decode_ip})")

    pool_a = [PinnedConnection(prefill_pod_url_by_ip(prefill_ip),
                               pod_name=prefill_name)
              for _ in range(CONCURRENCY)]
    pool_b = [PinnedConnection(decode_direct_url_by_ip(decode_ip),
                               pod_name=decode_name)
              for _ in range(CONCURRENCY)]
    pool_c = [PinnedConnection(decode_pod_url_by_ip(decode_ip),
                               pod_name=decode_name)
              for _ in range(CONCURRENCY)]
    pool_d = [PinnedConnection(decode_pod_url_by_ip(decode_ip),
                               pod_name=decode_name,
                               extra_headers={"x-prefiller-host-port": prefill_pod_host_port(prefill_ip)})
              for _ in range(CONCURRENCY)]
    pool_epp = ([PinnedConnection(EPP_URL, pod_name="epp-gateway")
                 for _ in range(CONCURRENCY)] if epp_enabled() else [])

    CONFIGS = [
        (ConfigDecompose.A_PREFILL_DIRECT, pool_a, prefill_name),
        (ConfigDecompose.B_DECODE_DIRECT,  pool_b, decode_name),
        (ConfigDecompose.C_SIDECAR_ONLY,   pool_c, decode_name),
        (ConfigDecompose.D_DISAGGREGATED,  pool_d, decode_name),
    ]
    if pool_epp:
        CONFIGS.append((ConfigDecompose.E_EPP_GATEWAY, pool_epp, "epp-gateway"))

    progress("")
    progress(f"  Concurrency: {CONCURRENCY} (pool of {CONCURRENCY} connections per config)")
    progress(f"  Sweep lengths: {SWEEP_LENGTHS}")
    progress(f"  Requests per config per length: {TOTAL_REQUESTS}")
    progress(f"  Output: {outfile}")
    progress("")

    configs_maxed = set()

    try:
        for ptokens in SWEEP_LENGTHS:
            progress(f"--- Prompt: {ptokens} tokens ---")

            active_configs = [(n, p, pn) for n, p, pn in CONFIGS
                              if n not in configs_maxed]
            if not active_configs:
                progress("  All configs maxed out, stopping")
                break

            warmup_prompt = build_prompt(ptokens)
            for _config_name, conn_pool, _pod_name in active_configs:
                for conn in conn_pool:
                    conn.warmup(warmup_prompt, MAX_TOKENS)

            for config_name, conn_pool, pod_name in active_configs:
                progress(f"  {config_name}: ", end="")

                wall_start = time.monotonic()
                run = 0
                statuses = []

                while run < TOTAL_REQUESTS:
                    batch_size = min(CONCURRENCY, TOTAL_REQUESTS - run)
                    futures = []

                    with ThreadPoolExecutor(max_workers=batch_size) as pool:
                        for i in range(batch_size):
                            run += 1
                            run_num = run
                            prompt = build_prompt(ptokens,
                                                  cache_bust=(ptokens, str(config_name), run_num))
                            worker_idx = i % CONCURRENCY
                            f = pool.submit(send_one_pinned, worker_idx,
                                            conn_pool, prompt, MAX_TOKENS)
                            futures.append((f, run_num))

                        for future, run_num in futures:
                            r = future.result()
                            writer.write({
                                "experiment": "exp14",
                                "config": config_name,
                                "pod": pod_name,
                                "prompt_tokens_target": ptokens,
                                "concurrency": CONCURRENCY,
                                "run": run_num,
                                "ttft_ms": r.ttft_ms,
                                "total_ms": r.total_ms,
                                "status_code": r.status,
                                "prompt_tokens_actual": r.prompt_tokens,
                                "completion_tokens": r.completion_tokens,
                                "error": r.error,
                            })
                            statuses.append(r.status)

                    dot()

                wall_s = time.monotonic() - wall_start
                throughput = TOTAL_REQUESTS / wall_s
                progress(f" {wall_s:.1f}s | {throughput:.1f} req/s")

                if statuses and all(s == 400 for s in statuses):
                    progress(f"  Skipping {config_name} at {ptokens}+ tokens: "
                             "exceeds context window")
                    configs_maxed.add(config_name)

            progress("")

    finally:
        for _config_name, conn_pool, _pod_name in CONFIGS:
            for conn in conn_pool:
                conn.close()

    writer.close()
    progress(f"=== Experiment 14 Complete === ({outfile})")


if __name__ == "__main__":
    main()
