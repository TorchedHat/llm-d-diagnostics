#!/usr/bin/env python3
"""
Experiment 12: Throughput vs Output Length

Measures how TTFT, inter-token latency (ITL), total latency and throughput
change with output length. As output length grows, decode time dominates
and the fixed disagg overhead becomes a smaller fraction of total request
time. Output length is also the cheapest knob for moving a workload between
prefill-heavy and decode-heavy without redeploying.

Sweeps max_tokens, optionally across several prompt lengths and concurrency
levels. Each request uses a unique prompt (cache-busted per run).

Measurement:
    STREAMING=1 (default): requests stream, so ttft_ms is the true time to
        the first token and itl_mean_ms is measured per request.
        STREAMING=0 reproduces the older non-streaming runs, where ttft_ms
        is close to total_ms.
    IGNORE_EOS=1 (default): the server generates exactly max_tokens, so the
        output length is the one requested.
    LOAD_MODE=batch (default): synchronized batches of CONCURRENCY requests,
        as in earlier runs. LOAD_MODE=steady keeps CONCURRENCY requests in
        flight (each finished request is replaced at once); analysis then
        drops each cell's first and last CONCURRENCY requests.
    The request count is raised so the labelled concurrency is reached:
    whole batches in batch mode, at least 3x CONCURRENCY in steady mode.

Configs:
    BASELINE:  all requests to prefill vLLM (non-disaggregated)
    DISAGG-1D: all requests through decode sidecar (1 decode target)
    DISAGG-2D: round-robin across decode sidecar (2 decode targets)
    EPP:       through the EPP Gateway (when EPP_URL is set)

Note: BASELINE is one vLLM instance; the DISAGG configs use 2-3. Compare
per-request latency here, not throughput per GPU.

Usage: python3 toolkit/exp12_tput_outlen.py

Env vars:
    OUTPUT_LENGTHS   Comma-separated output token targets (default: 20,50,100,200)
    PROMPT_TOKENS    Prompt length(s) in tokens, comma-separated (default: 500)
    CONCURRENCY      Concurrency level(s), comma-separated (default: 8)
    TOTAL_REQUESTS   Requests per cell, before the minimum above (default: 24)
    STREAMING        1 = streaming requests (default), 0 = non-streaming
    IGNORE_EOS       1 = generate exactly max_tokens (default), 0 = allow early stop
    LOAD_MODE        batch (default) or steady
    STREAM_TIMEOUT   Socket timeout per read, seconds (default: 120)
    CONFIGS          Comma-separated configs to run (default: all)
                     Options: BASELINE, DISAGG-1D, DISAGG-2D, EPP
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    DATA_DIR,
    RUN_NONCE,
    WARMUP,
    build_prompt,
    dot,
    env,
    epp_enabled,
    load_mode_from_env,
    print_config,
    progress,
    route_throughput,
    run_load,
    send_request,
    send_streaming,
    write_run_info,
)
from schemas import ConfigThroughput, Exp12Row, TypedCSVWriter

OUTPUT_LENGTHS = [int(x) for x in env("OUTPUT_LENGTHS", "20,50,100,200").split(",")]
PROMPT_LENGTHS = [int(x) for x in env("PROMPT_TOKENS", "500").split(",")]
CONCURRENCY_LEVELS = [int(x) for x in env("CONCURRENCY", "8").split(",")]
TOTAL_REQUESTS = int(env("TOTAL_REQUESTS", "24"))
STREAMING = env("STREAMING", "1") == "1"
IGNORE_EOS = env("IGNORE_EOS", "1") == "1"
LOAD_MODE = load_mode_from_env("batch")
STREAM_TIMEOUT = int(env("STREAM_TIMEOUT", "120"))

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


def mean_itl_ms(r):
    """Mean inter-token latency of one streamed request, or 0 if undefined.

    Uses the server's token count when reported, since streamed chunks can
    undercount tokens that decode to empty text.
    """
    if not r.token_times:
        return 0.0
    tokens = r.usage_completion_tokens or r.completion_tokens
    if tokens > 1 and r.ttft_ms > 0 and r.total_ms > r.ttft_ms:
        return (r.total_ms - r.ttft_ms) / (tokens - 1)
    return 0.0


def send(url, headers, prompt, max_tokens):
    if STREAMING:
        return send_streaming(url, prompt, max_tokens, extra_headers=headers,
                              timeout=STREAM_TIMEOUT, ignore_eos=IGNORE_EOS,
                              include_usage=True)
    return send_request(url, prompt, max_tokens, extra_headers=headers,
                        ignore_eos=IGNORE_EOS)


def main():
    outfile = os.path.join(DATA_DIR, "exp12-results.csv")
    write_run_info("exp12", {
        "output_lengths": OUTPUT_LENGTHS,
        "prompt_tokens": PROMPT_LENGTHS,
        "concurrency": CONCURRENCY_LEVELS,
        "total_requests": TOTAL_REQUESTS,
        "streaming": STREAMING,
        "ignore_eos": IGNORE_EOS,
        "load_mode": LOAD_MODE,
        "stream_timeout": STREAM_TIMEOUT,
    })
    writer = TypedCSVWriter(outfile, Exp12Row)

    progress("=== Experiment 12: Throughput vs Output Length ===")
    print_config()
    progress(f"  Output lengths: {OUTPUT_LENGTHS}")
    progress(f"  Prompt tokens:  {PROMPT_LENGTHS}")
    progress(f"  Concurrency:    {CONCURRENCY_LEVELS}")
    progress(f"  Requests/cell:  {TOTAL_REQUESTS} (raised to reach each concurrency)")
    progress(f"  Streaming: {STREAMING}  ignore_eos: {IGNORE_EOS}  load mode: {LOAD_MODE}")
    progress(f"  Output: {outfile}")
    progress("")

    for concurrency in CONCURRENCY_LEVELS:
        for ptokens in PROMPT_LENGTHS:
            for max_tokens in OUTPUT_LENGTHS:
                progress(f"--- concurrency={concurrency} prompt={ptokens} "
                         f"max_tokens={max_tokens} ---")
                warmup_prompt = build_prompt(ptokens)
                config_order = list(CONFIGS)
                random.shuffle(config_order)

                for config_name in config_order:
                    progress(f"  {config_name.value}: ", end="")
                    for seq in range(1, max(WARMUP, 2) + 1):  # reach both decode targets
                        url, hdrs, _ = route_throughput(config_name, seq)
                        send(url, hdrs, warmup_prompt, min(max_tokens, 20))

                    def one(seq, inflight, config_name=config_name, ptokens=ptokens,
                            max_tokens=max_tokens, concurrency=concurrency):
                        prompt = build_prompt(
                            ptokens,
                            cache_bust=("exp12", RUN_NONCE, ptokens, max_tokens,
                                        config_name.value, concurrency, seq))
                        url, hdrs, tag = route_throughput(config_name, seq)
                        r = send(url, hdrs, prompt, max_tokens)
                        writer.write({
                            "experiment": "exp12",
                            "config": config_name,
                            "prompt_tokens_target": ptokens,
                            "max_tokens": max_tokens,
                            "concurrency": concurrency,
                            "run": seq,
                            "ttft_ms": r.ttft_ms,
                            "total_ms": r.total_ms,
                            "status_code": r.status,
                            "prompt_tokens_actual": r.prompt_tokens,
                            "completion_tokens": r.completion_tokens,
                            "target": tag,
                            "streaming": int(STREAMING),
                            "load_mode": LOAD_MODE,
                            "inflight_at_start": inflight,
                            "itl_mean_ms": round(mean_itl_ms(r), 2),
                            "server_completion_tokens": r.usage_completion_tokens,
                            "error": r.error,
                        })
                        dot()
                        return r

                    wall_start = time.monotonic()
                    results, total = run_load(LOAD_MODE, concurrency, TOTAL_REQUESTS, one)
                    wall_s = time.monotonic() - wall_start
                    errors = sum(1 for r in results if r.status != 200)
                    progress(f" {wall_s:.1f}s | {total} requests | {total / wall_s:.1f} req/s"
                             + (f" | errors: {errors}/{total}" if errors else ""))

                    tokens = sorted((r.usage_completion_tokens or r.completion_tokens)
                                    for r in results if r.status == 200)
                    if tokens and tokens[len(tokens) // 2] < 0.8 * max_tokens:
                        progress(f"    WARNING: median completion tokens {tokens[len(tokens) // 2]} "
                                 f"< 80% of target {max_tokens} (set IGNORE_EOS=1)")
            progress("")

    writer.close()
    progress(f"=== Experiment 12 Complete === ({outfile})")


if __name__ == "__main__":
    main()
