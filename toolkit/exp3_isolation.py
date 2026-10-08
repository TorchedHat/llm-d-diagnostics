#!/usr/bin/env python3
"""
Experiment 3: Prefill Isolation (Head-of-Line Blocking)

Tests whether disaggregation protects light requests from heavy prefills.
Mix: 1 heavy (1000-token prompt) + 5 light (10-token prompt) launched simultaneously.

Uses streaming requests for true TTFT (time to first token) and ITL
(inter-token latency) measurement via SSE.

Configs:
    BASELINE:  all 6 requests to prefill vLLM (1 GPU)
    DISAGG-2D: through sidecars, round-robin decode-1/decode-2

Usage: python3 /scripts/toolkit/exp3_isolation.py

Additional env vars:
    TRIALS          Number of trial repetitions (default: 10)
    HEAVY_MAX       Max tokens for heavy request (default: 50)
    LIGHT_MAX       Max tokens for light requests (default: 20)
"""

import os
import sys
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
    send_streaming,
    write_run_info,
)
from schemas import ConfigIsolation, Exp3Row, TypedCSVWriter, Weight

TRIALS = int(env("TRIALS", "10"))
HEAVY_MAX = int(env("HEAVY_MAX", "50"))
LIGHT_MAX = int(env("LIGHT_MAX", "20"))

HEAVY_PROMPT_TOKENS = int(env("HEAVY_PROMPT_TOKENS", "1000"))
LIGHT_PROMPT_TOKENS = int(env("LIGHT_PROMPT_TOKENS", "10"))
HEAVY_PROMPT = build_prompt(HEAVY_PROMPT_TOKENS)
LIGHT_PROMPT = build_prompt(LIGHT_PROMPT_TOKENS)

DISAGG_HEADERS = {"x-prefiller-host-port": PREFILL_HOST}



def send_one(url, headers, prompt, max_tokens):
    """Worker function for thread pool. Uses streaming for true TTFT/ITL."""
    return send_streaming(url, prompt, max_tokens, extra_headers=headers)


def main():
    outfile = os.path.join(DATA_DIR, "exp3-results.csv")
    write_run_info("exp3", {"trials": TRIALS,
                            "heavy_prompt_tokens": HEAVY_PROMPT_TOKENS,
                            "light_prompt_tokens": LIGHT_PROMPT_TOKENS,
                            "heavy_max_tokens": HEAVY_MAX,
                            "light_max_tokens": LIGHT_MAX})
    writer = TypedCSVWriter(outfile, Exp3Row)

    progress("=== Experiment 3: Prefill Isolation ===")
    print_config()
    progress(f"  Mix: 1 heavy ({HEAVY_MAX} max) + 5 light ({LIGHT_MAX} max)")
    progress(f"  Trials: {TRIALS}")
    progress(f"  Output: {outfile}")
    progress("")

    configs = {
        ConfigIsolation.BASELINE: [
            (BASELINE_URL, None, HEAVY_PROMPT, HEAVY_MAX, Weight.HEAVY, 0),
            (BASELINE_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 1),
            (BASELINE_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 2),
            (BASELINE_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 3),
            (BASELINE_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 4),
            (BASELINE_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 5),
        ],
        ConfigIsolation.DISAGG_2D: [
            (DISAGG_D1_URL, DISAGG_HEADERS, HEAVY_PROMPT, HEAVY_MAX, Weight.HEAVY, 0),
            (DISAGG_D1_URL, DISAGG_HEADERS, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 1),
            (DISAGG_D2_URL, DISAGG_HEADERS, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 2),
            (DISAGG_D1_URL, DISAGG_HEADERS, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 3),
            (DISAGG_D2_URL, DISAGG_HEADERS, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 4),
            (DISAGG_D1_URL, DISAGG_HEADERS, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 5),
        ],
    }
    if epp_enabled():
        configs[ConfigIsolation.DISAGG_EPP] = [
            (EPP_URL, None, HEAVY_PROMPT, HEAVY_MAX, Weight.HEAVY, 0),
            (EPP_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 1),
            (EPP_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 2),
            (EPP_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 3),
            (EPP_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 4),
            (EPP_URL, None, LIGHT_PROMPT, LIGHT_MAX, Weight.LIGHT, 5),
        ]

    for config_name, request_specs in configs.items():
        progress(f"  Config: {config_name}")

        # Warm-up (use a light request spec)
        light_spec = next(s for s in request_specs if s[4] == Weight.LIGHT)
        for _ in range(WARMUP):
            send_request(light_spec[0], LIGHT_PROMPT, LIGHT_MAX,
                         extra_headers=light_spec[1])

        for trial in range(1, TRIALS + 1):
            # Launch all 6 requests simultaneously
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures = []
                for url, headers, prompt, max_tok, weight, idx in request_specs:
                    f = pool.submit(send_one, url, headers, prompt, max_tok)
                    futures.append((f, weight, idx))

                for future, weight, idx in futures:
                    r = future.result()

                    # ITL from per-token timestamps (ct-1 gaps)
                    tt = r.token_times
                    if len(tt) >= 2:
                        gaps = [tt[i+1] - tt[i] for i in range(len(tt) - 1)]
                        itl = round(sum(gaps) / len(gaps) * 1000, 2)
                    else:
                        itl = 0

                    writer.write({
                        "experiment": "exp3",
                        "config": config_name,
                        "trial": trial,
                        "weight": weight,
                        "idx": idx,
                        "pod": "service-lb",
                        "ttft_ms": r.ttft_ms,
                        "total_ms": r.total_ms,
                        "status_code": r.status,
                        "completion_tokens": r.completion_tokens,
                        "itl_ms": itl,
                        "error": r.error,
                    })

            dot()

        progress(" done")

    writer.close()
    progress(f"=== Experiment 3 Complete === ({outfile})")


if __name__ == "__main__":
    main()
