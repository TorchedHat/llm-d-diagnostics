#!/usr/bin/env python3
"""
Experiment 12: Throughput vs Output Length

Measures how disaggregated inference throughput changes with output length.
As output length grows, decode time dominates and the fixed disagg overhead
becomes a smaller fraction of total request time.

Sweeps max_tokens at fixed prompt length and concurrency. Each concurrent
request uses a unique prompt (cache-busted).

Configs:
    BASELINE:  all requests to prefill vLLM (non-disaggregated)
    DISAGG-1D: all requests through decode sidecar (1 decode target)
    DISAGG-2D: round-robin across decode sidecar (2 decode targets)

Usage: python3 toolkit/exp12_tput_outlen.py

Env vars:
    OUTPUT_LENGTHS   Comma-separated output token targets (default: 20,50,100,200)
    PROMPT_TOKENS    Prompt length in tokens (default: 500)
    CONCURRENCY      Concurrent requests per batch (default: 8)
    TOTAL_REQUESTS   Requests per config per output length (default: 24)
    CONFIGS          Comma-separated configs to run (default: all)
                     Options: BASELINE, DISAGG-1D, DISAGG-2D
"""

import os
import random
import statistics
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

OUTPUT_LENGTHS = [int(x) for x in env("OUTPUT_LENGTHS", "20,50,100,200").split(",")]
PROMPT_TOKENS = int(env("PROMPT_TOKENS", "500"))
CONCURRENCY = int(env("CONCURRENCY", "8"))
TOTAL_REQUESTS = int(env("TOTAL_REQUESTS", "24"))

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
    outfile = os.path.join(DATA_DIR, "exp12-results.csv")
    write_run_info("exp12", {
        "output_lengths": OUTPUT_LENGTHS,
        "prompt_tokens": PROMPT_TOKENS,
        "concurrency": CONCURRENCY,
        "total_requests": TOTAL_REQUESTS,
    })
    writer = TypedCSVWriter(outfile, Exp11Row)

    progress("=== Experiment 12: Throughput vs Output Length ===")
    print_config()
    progress(f"  Output lengths: {OUTPUT_LENGTHS}")
    progress(f"  Prompt tokens: {PROMPT_TOKENS}")
    progress(f"  Concurrency: {CONCURRENCY}")
    progress(f"  Requests per config: {TOTAL_REQUESTS}")
    progress(f"  Output: {outfile}")
    progress("")

    for max_tokens in OUTPUT_LENGTHS:
        progress(f"--- Max tokens: {max_tokens} ---")

        warmup_prompt = build_prompt(PROMPT_TOKENS)

        config_order = list(CONFIGS)
        random.shuffle(config_order)

        for config_name in config_order:
            progress(f"  {config_name}: ", end="")

            for i in range(WARMUP):
                if config_name == ConfigThroughput.BASELINE:
                    send_request(BASELINE_URL, warmup_prompt, max_tokens)
                elif config_name == ConfigThroughput.DISAGG_EPP:
                    send_request(EPP_URL, warmup_prompt, max_tokens)
                elif config_name == ConfigThroughput.DISAGG_2D and i % 2 == 1:
                    send_request(DISAGG_D2_URL, warmup_prompt, max_tokens,
                                 extra_headers=DISAGG_HEADERS)
                else:
                    send_request(DISAGG_D1_URL, warmup_prompt, max_tokens,
                                 extra_headers=DISAGG_HEADERS)

            wall_start = time.monotonic()
            run = 0
            completion_counts = []

            while run < TOTAL_REQUESTS:
                batch_size = min(CONCURRENCY, TOTAL_REQUESTS - run)
                futures = []

                with ThreadPoolExecutor(max_workers=batch_size) as pool:
                    for _i in range(batch_size):
                        run += 1
                        run_num = run
                        prompt = build_prompt(PROMPT_TOKENS,
                                              cache_bust=(max_tokens, str(config_name), run_num))

                        if config_name == ConfigThroughput.BASELINE:
                            f = pool.submit(send_one, BASELINE_URL, None,
                                            prompt, max_tokens, "d1")
                        elif config_name == ConfigThroughput.DISAGG_1D:
                            f = pool.submit(send_one, DISAGG_D1_URL,
                                            DISAGG_HEADERS, prompt, max_tokens, "d1")
                        elif config_name == ConfigThroughput.DISAGG_EPP:
                            f = pool.submit(send_one, EPP_URL, None,
                                            prompt, max_tokens, "epp")
                        else:
                            if run % 2 == 1:
                                f = pool.submit(send_one, DISAGG_D1_URL,
                                                DISAGG_HEADERS, prompt, max_tokens, "d1")
                            else:
                                f = pool.submit(send_one, DISAGG_D2_URL,
                                                DISAGG_HEADERS, prompt, max_tokens, "d2")
                        futures.append((f, run_num))

                    for future, run_num in futures:
                        r, tag = future.result()
                        writer.write({
                            "experiment": "exp12",
                            "config": config_name,
                            "prompt_tokens_target": PROMPT_TOKENS,
                            "max_tokens": max_tokens,
                            "concurrency": CONCURRENCY,
                            "run": run_num,
                            "ttft_ms": r.ttft_ms,
                            "total_ms": r.total_ms,
                            "status_code": r.status,
                            "prompt_tokens_actual": r.prompt_tokens,
                            "completion_tokens": r.completion_tokens,
                            "target": tag,
                            "error": r.error,
                        })
                        if r.status == 200 and r.completion_tokens:
                            completion_counts.append(int(r.completion_tokens))

                dot()

            wall_s = time.monotonic() - wall_start
            throughput = TOTAL_REQUESTS / wall_s
            progress(f" {wall_s:.1f}s | {throughput:.1f} req/s")

            if completion_counts:
                median_tokens = statistics.median(completion_counts)
                if median_tokens < 0.8 * max_tokens:
                    progress(f"    WARNING: median completion_tokens={median_tokens:.0f} "
                             f"< 80% of target {max_tokens} (model may be hitting EOS early)")

        progress("")

    writer.close()
    progress(f"=== Experiment 12 Complete === ({outfile})")


if __name__ == "__main__":
    main()
