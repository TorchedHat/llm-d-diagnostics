#!/usr/bin/env python3
"""
Experiment 2: Throughput Under Load

Measures how throughput scales with concurrency. Uses ThreadPoolExecutor
for precise wall-clock timing of concurrent batches.

Configs:
    BASELINE:  all requests to prefill vLLM (1 GPU)
    DISAGG-1D: all requests through decode-1 sidecar (2 GPUs)
    DISAGG-2D: round-robin decode-1 and decode-2 (3 GPUs)

Usage: python3 /scripts/toolkit/exp2_throughput.py

Additional env vars:
    CONCURRENCY_LEVELS   Comma-separated (default: 1,2,4,8,16)
    TOTAL_REQUESTS       Requests per config per concurrency (default: 20)
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
from schemas import ConfigThroughput, Exp2Row, TypedCSVWriter

TOTAL_REQUESTS = int(env("TOTAL_REQUESTS", "20"))
MAX_TOKENS = int(env("MAX_TOKENS", "30"))
CONCURRENCY_LEVELS = [int(x) for x in env("CONCURRENCY_LEVELS", "1,2,4,8,16").split(",")]

PROMPT_TOKENS = int(env("PROMPT_TOKENS", "50"))

DISAGG_HEADERS = {"x-prefiller-host-port": PREFILL_HOST}



def send_one(url, headers, tag, key):
    """Worker function for thread pool; *key* makes the prompt unique."""
    prompt = build_prompt(PROMPT_TOKENS, cache_bust=key)
    r = send_request(url, prompt, MAX_TOKENS, extra_headers=headers)
    return r, tag


def main():
    outfile = os.path.join(DATA_DIR, "exp2-results.csv")
    write_run_info("exp2", {"concurrency_levels": CONCURRENCY_LEVELS,
                            "total_requests": TOTAL_REQUESTS,
                            "prompt_tokens": PROMPT_TOKENS})
    writer = TypedCSVWriter(outfile, Exp2Row)

    progress("=== Experiment 2: Throughput Under Load ===")
    print_config()
    progress(f"  Concurrency levels: {CONCURRENCY_LEVELS}")
    progress(f"  Requests per config: {TOTAL_REQUESTS}")
    progress(f"  Output: {outfile}")
    progress("")

    for concurrency in CONCURRENCY_LEVELS:
        progress(f"--- Concurrency: {concurrency} ---")

        config_order = [ConfigThroughput.BASELINE, ConfigThroughput.DISAGG_1D, ConfigThroughput.DISAGG_2D]
        if epp_enabled():
            config_order.append(ConfigThroughput.DISAGG_EPP)
        random.shuffle(config_order)
        for config_name in config_order:
            progress(f"  {config_name}: ", end="")

            # Warm-up (sequential, exercise all endpoints)
            for i in range(WARMUP):
                prompt = build_prompt(PROMPT_TOKENS,
                                      cache_bust=("exp2-warmup", concurrency, str(config_name), i))
                if config_name == ConfigThroughput.BASELINE:
                    send_request(BASELINE_URL, prompt, MAX_TOKENS)
                elif config_name == ConfigThroughput.DISAGG_2D and i % 2 == 1:
                    send_request(DISAGG_D2_URL, prompt, MAX_TOKENS,
                                 extra_headers=DISAGG_HEADERS)
                elif config_name == ConfigThroughput.DISAGG_EPP:
                    send_request(EPP_URL, prompt, MAX_TOKENS)
                else:
                    send_request(DISAGG_D1_URL, prompt, MAX_TOKENS,
                                 extra_headers=DISAGG_HEADERS)

            # Measured runs in batches of `concurrency`
            wall_start = time.monotonic()
            run = 0

            while run < TOTAL_REQUESTS:
                batch_size = min(concurrency, TOTAL_REQUESTS - run)
                futures = []

                with ThreadPoolExecutor(max_workers=batch_size) as pool:
                    for _i in range(batch_size):
                        run += 1
                        run_num = run  # capture current value for this future
                        key = ("exp2", concurrency, str(config_name), run_num)
                        if config_name == ConfigThroughput.BASELINE:
                            f = pool.submit(send_one, BASELINE_URL, None, "d1", key)
                        elif config_name == ConfigThroughput.DISAGG_1D:
                            f = pool.submit(send_one, DISAGG_D1_URL,
                                            DISAGG_HEADERS, "d1", key)
                        elif config_name == ConfigThroughput.DISAGG_EPP:
                            f = pool.submit(send_one, EPP_URL, None, "epp", key)
                        else:  # DISAGG-2D: round-robin
                            if run % 2 == 1:
                                f = pool.submit(send_one, DISAGG_D1_URL,
                                                DISAGG_HEADERS, "d1", key)
                            else:
                                f = pool.submit(send_one, DISAGG_D2_URL,
                                                DISAGG_HEADERS, "d2", key)
                        futures.append((f, run_num))

                    for future, run_num in futures:
                        r, tag = future.result()
                        writer.write({
                            "experiment": "exp2",
                            "config": config_name,
                            "run": run_num,
                            "concurrency": concurrency,
                            "pod": "service-lb",
                            "ttft_ms": r.ttft_ms,
                            "total_ms": r.total_ms,
                            "status_code": r.status,
                            "target": tag,
                            "error": r.error,
                        })

                dot()

            wall_s = time.monotonic() - wall_start
            throughput = TOTAL_REQUESTS / wall_s

            progress(f" {wall_s:.1f}s | {throughput:.1f} req/s")

        progress("")

    writer.close()
    progress(f"=== Experiment 2 Complete === ({outfile})")


if __name__ == "__main__":
    main()
