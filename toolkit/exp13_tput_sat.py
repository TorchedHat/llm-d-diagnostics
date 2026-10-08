#!/usr/bin/env python3
"""
Experiment 13: Saturation Ceiling

Pushes concurrency beyond normal operating range to find the maximum
sustainable throughput per GPU. Early-stops when error rates spike or
throughput plateaus (indicating client or server saturation).

Unlike exp6 (open-loop QPS injection), this uses closed-loop concurrency
to find the throughput ceiling under back-to-back load.

Configs:
    BASELINE:  all requests to prefill vLLM (non-disaggregated)
    DISAGG-1D: all requests through decode sidecar (1 decode target)
    DISAGG-2D: round-robin across decode sidecar (2 decode targets)

Usage: python3 toolkit/exp13_tput_sat.py

Env vars:
    CONCURRENCY_LEVELS   Comma-separated concurrency levels (default: 16,32,64,128)
    PROMPT_TOKENS        Prompt length in tokens (default: 100)
    MAX_TOKENS           Output tokens per request (default: 20)
    TOTAL_REQUESTS       Requests per config per concurrency level (default: 64)
    CONFIGS              Comma-separated configs to run (default: all)
                         Options: BASELINE, DISAGG-1D, DISAGG-2D
"""

import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    BASELINE_URL,
    DATA_DIR,
    DISAGG_D1_URL,
    DISAGG_D2_URL,
    EPP_URL,
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
from schemas import ConfigThroughput, Exp11Row, TypedCSVWriter

CONCURRENCY_LEVELS = [int(x) for x in env("CONCURRENCY_LEVELS", "16,32,64,128").split(",")]
PROMPT_TOKENS = int(env("PROMPT_TOKENS", "100"))
MAX_TOKENS = int(env("MAX_TOKENS", "20"))
TOTAL_REQUESTS = int(env("TOTAL_REQUESTS", "64"))

_CONFIG_MAP = {
    "BASELINE": ConfigThroughput.BASELINE,
    "DISAGG-1D": ConfigThroughput.DISAGG_1D,
    "DISAGG-2D": ConfigThroughput.DISAGG_2D,
    "EPP": ConfigThroughput.DISAGG_EPP,
}
_ALL_CONFIGS = [ConfigThroughput.BASELINE, ConfigThroughput.DISAGG_1D, ConfigThroughput.DISAGG_2D]
if epp_enabled():
    _ALL_CONFIGS.append(ConfigThroughput.DISAGG_EPP)
CONFIGS = ([_CONFIG_MAP[x.strip()] for x in env("CONFIGS", "").split(",") if x.strip()]
           or _ALL_CONFIGS)
if ConfigThroughput.DISAGG_EPP in CONFIGS and not epp_enabled():
    raise SystemExit("CONFIGS includes EPP but EPP_URL is not configured")

DISAGG_HEADERS = {"x-prefiller-host-port": PREFILL_HOST}


def send_one(url, headers, prompt, max_tokens, tag):
    r = send_request(url, prompt, max_tokens, extra_headers=headers)
    return r, tag


def main():
    outfile = os.path.join(DATA_DIR, "exp13-results.csv")
    write_run_info("exp13", {
        "concurrency_levels": CONCURRENCY_LEVELS,
        "prompt_tokens": PROMPT_TOKENS,
        "max_tokens": MAX_TOKENS,
        "total_requests": TOTAL_REQUESTS,
    })
    writer = TypedCSVWriter(outfile, Exp11Row)

    progress("=== Experiment 13: Saturation Ceiling ===")
    print_config()
    progress(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    progress(f"  Prompt tokens: {PROMPT_TOKENS}")
    progress(f"  Max tokens: {MAX_TOKENS}")
    progress(f"  Requests per config: {TOTAL_REQUESTS}")
    progress(f"  Output: {outfile}")
    progress("")

    saturated = set()
    prev_throughput = {}

    for concurrency in CONCURRENCY_LEVELS:
        progress(f"--- Concurrency: {concurrency} ---")

        config_order = list(CONFIGS)
        random.shuffle(config_order)

        for config_name in config_order:
            if config_name in saturated:
                progress(f"  {config_name}: SKIP (saturated)")
                continue

            progress(f"  {config_name}: ", end="")

            warmup_prompt = build_prompt(PROMPT_TOKENS)
            for i in range(WARMUP):
                if config_name == ConfigThroughput.BASELINE:
                    send_request(BASELINE_URL, warmup_prompt, MAX_TOKENS)
                elif config_name == ConfigThroughput.DISAGG_EPP:
                    send_request(EPP_URL, warmup_prompt, MAX_TOKENS)
                elif config_name == ConfigThroughput.DISAGG_2D and i % 2 == 1:
                    send_request(DISAGG_D2_URL, warmup_prompt, MAX_TOKENS,
                                 extra_headers=DISAGG_HEADERS)
                else:
                    send_request(DISAGG_D1_URL, warmup_prompt, MAX_TOKENS,
                                 extra_headers=DISAGG_HEADERS)

            wall_start = time.monotonic()
            run = 0
            errors = 0

            while run < TOTAL_REQUESTS:
                batch_size = min(concurrency, TOTAL_REQUESTS - run)
                futures = []

                with ThreadPoolExecutor(max_workers=batch_size) as pool:
                    for _i in range(batch_size):
                        run += 1
                        run_num = run
                        prompt = build_prompt(PROMPT_TOKENS,
                                              cache_bust=(concurrency, str(config_name), run_num))

                        if config_name == ConfigThroughput.BASELINE:
                            f = pool.submit(send_one, BASELINE_URL, None,
                                            prompt, MAX_TOKENS, "d1")
                        elif config_name == ConfigThroughput.DISAGG_1D:
                            f = pool.submit(send_one, DISAGG_D1_URL,
                                            DISAGG_HEADERS, prompt, MAX_TOKENS, "d1")
                        elif config_name == ConfigThroughput.DISAGG_EPP:
                            f = pool.submit(send_one, EPP_URL, None,
                                            prompt, MAX_TOKENS, "epp")
                        else:
                            if run % 2 == 1:
                                f = pool.submit(send_one, DISAGG_D1_URL,
                                                DISAGG_HEADERS, prompt, MAX_TOKENS, "d1")
                            else:
                                f = pool.submit(send_one, DISAGG_D2_URL,
                                                DISAGG_HEADERS, prompt, MAX_TOKENS, "d2")
                        futures.append((f, run_num))

                    for future, run_num in futures:
                        r, tag = future.result()
                        writer.write({
                            "experiment": "exp13",
                            "config": config_name,
                            "prompt_tokens_target": PROMPT_TOKENS,
                            "max_tokens": MAX_TOKENS,
                            "concurrency": concurrency,
                            "run": run_num,
                            "ttft_ms": r.ttft_ms,
                            "total_ms": r.total_ms,
                            "status_code": r.status,
                            "prompt_tokens_actual": r.prompt_tokens,
                            "completion_tokens": r.completion_tokens,
                            "target": tag,
                            "error": r.error,
                        })
                        if r.status != 200:
                            errors += 1

                dot()

            wall_s = time.monotonic() - wall_start
            goodput = (TOTAL_REQUESTS - errors) / wall_s
            error_rate = errors / TOTAL_REQUESTS

            progress(f" {wall_s:.1f}s | {goodput:.1f} req/s | "
                     f"errors: {errors}/{TOTAL_REQUESTS} ({error_rate:.0%})")

            if error_rate > 0.10:
                progress(f"    SATURATED: error rate {error_rate:.0%} > 10%")
                saturated.add(config_name)
            elif config_name in prev_throughput:
                prev = prev_throughput[config_name]
                if goodput < prev * 0.9:
                    progress(f"    WARNING: goodput dropped from {prev:.1f} to {goodput:.1f} req/s")

            prev_throughput[config_name] = goodput

        progress("")

        if len(saturated) >= len(CONFIGS):
            progress("  All configs saturated, stopping")
            break

    writer.close()
    progress(f"=== Experiment 13 Complete === ({outfile})")


if __name__ == "__main__":
    main()
