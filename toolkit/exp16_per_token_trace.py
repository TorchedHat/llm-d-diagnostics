#!/usr/bin/env python3
"""
Experiment 16: Per-Token ITL Trace

Records the arrival time of every streamed token of every request, so the
whole distribution of inter-token gaps can be analyzed, not just the mean.

What it shows:
  - The longest pause a user sees. In monolithic serving, a decode step
    that also processes another request's prompt chunk stalls every
    decoding request; that shows up as a few very long gaps (p99 ITL),
    which a mean ITL hides.
  - Whether a disaggregated decode pod keeps gaps flat under load.

Load: by default a steady closed loop (LOAD_MODE=steady): N workers each
send their next request as soon as the previous one finishes, so new
prompts keep arriving while other requests decode, as in real traffic.
The first and last N requests of each cell ramp the load up and down;
analysis drops them. LOAD_MODE=batch sends synchronized batches (all
prompts processed together at the start of each batch, then pure decode),
which measures a batch-start artifact rather than steady interference.

Output: one CSV row per streamed chunk with text, per request (token_idx 0
is the first; its gap_ms is the TTFT). server_completion_tokens is the
server's own count; if it exceeds the number of rows, some tokens decoded
to empty text and their time is merged into the next gap. A failed request
gets one row with token_idx -1.

Configs:
    BASELINE:   all requests to prefill vLLM (non-disaggregated)
    DISAGG-1D:  all requests through decode sidecar (1 decode target)
    DISAGG-2D:  round-robin across decode sidecar (2 decode targets)
    EPP:        through the EPP Gateway (when EPP_URL is set)

Usage: python3 toolkit/exp16_per_token_trace.py

Env vars:
    PROMPT_TOKENS        Prompt token target (default: 1000)
    MAX_TOKENS           Output tokens (default: 500)
    CONCURRENCY_LEVELS   Concurrency levels (default: 1,4,32,64)
    TOTAL_REQUESTS       Requests per cell, raised to at least 3x concurrency
                         in steady mode (default: 50)
    LOAD_MODE            steady (default) or batch
    STREAM_TIMEOUT       Socket timeout per read, seconds (default: 120)
    IGNORE_EOS           1 = generate exactly max_tokens (default), 0 = allow early stop
    CONFIGS              Comma-separated configs (default: BASELINE,DISAGG-2D,
                         plus EPP when EPP_URL is set)
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
    send_streaming,
    write_run_info,
)
from schemas import ConfigThroughput, Exp16Row, TypedCSVWriter

PROMPT_TOKENS = int(env("PROMPT_TOKENS", "1000"))
MAX_TOKENS = int(env("MAX_TOKENS", "500"))
CONCURRENCY_LEVELS = [int(x) for x in env("CONCURRENCY_LEVELS", "1,4,32,64").split(",")]
TOTAL_REQUESTS = int(env("TOTAL_REQUESTS", "50"))
LOAD_MODE = load_mode_from_env("steady")
STREAM_TIMEOUT = int(env("STREAM_TIMEOUT", "120"))
IGNORE_EOS = env("IGNORE_EOS", "1") == "1"

_CONFIG_MAP = {
    "BASELINE": ConfigThroughput.BASELINE,
    "DISAGG-1D": ConfigThroughput.DISAGG_1D,
    "DISAGG-2D": ConfigThroughput.DISAGG_2D,
    "EPP": ConfigThroughput.DISAGG_EPP,
}
_ALL_CONFIGS = [ConfigThroughput.BASELINE, ConfigThroughput.DISAGG_2D]
if epp_enabled():
    _ALL_CONFIGS.append(ConfigThroughput.DISAGG_EPP)
CONFIGS = ([_CONFIG_MAP[x.strip()] for x in env("CONFIGS", "").split(",") if x.strip()]
           or _ALL_CONFIGS)
if ConfigThroughput.DISAGG_EPP in CONFIGS and not epp_enabled():
    raise SystemExit("CONFIGS includes EPP but EPP_URL is not configured")


def token_rows(r):
    """Yield (token_idx, elapsed_ms, gap_ms, tokens_in_chunk) for one streamed request.

    token_times are monotonic timestamps relative to the request start, so
    the first gap is the TTFT. Each row is one timed chunk; tokens_in_chunk
    is how many tokens it carried ("" if the server did not report token
    IDs). A chunk with several tokens spans several decode steps in one gap.
    """
    prev = 0.0
    counts = r.token_counts or ("",) * len(r.token_times)
    for idx, (t, n) in enumerate(zip(r.token_times, counts, strict=True)):
        yield idx, round(t * 1000, 2), round((t - prev) * 1000, 2), n
        prev = t


def stream(url, headers, prompt, max_tokens):
    return send_streaming(url, prompt, max_tokens, extra_headers=headers,
                          timeout=STREAM_TIMEOUT, ignore_eos=IGNORE_EOS,
                          include_usage=True, token_ids=True)


def main():
    outfile = os.path.join(DATA_DIR, "exp16-results.csv")
    write_run_info("exp16", {
        "prompt_tokens": PROMPT_TOKENS,
        "max_tokens": MAX_TOKENS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "total_requests": TOTAL_REQUESTS,
        "load_mode": LOAD_MODE,
        "stream_timeout": STREAM_TIMEOUT,
        "ignore_eos": IGNORE_EOS,
    })
    writer = TypedCSVWriter(outfile, Exp16Row)

    progress("=== Experiment 16: Per-Token ITL Trace ===")
    print_config()
    progress(f"  Prompt tokens:  {PROMPT_TOKENS}")
    progress(f"  Output tokens:  {MAX_TOKENS}")
    progress(f"  Concurrency:    {CONCURRENCY_LEVELS}")
    progress(f"  Load mode:      {LOAD_MODE}")
    progress(f"  Requests/cell:  {TOTAL_REQUESTS} (raised to reach each concurrency)")
    progress(f"  Output: {outfile}")
    progress("")

    warmup_prompt = build_prompt(PROMPT_TOKENS)
    for concurrency in CONCURRENCY_LEVELS:
        progress(f"=== Concurrency: {concurrency} ===")
        config_order = list(CONFIGS)
        random.shuffle(config_order)

        for config_name in config_order:
            progress(f"  {config_name.value}: ", end="")
            for seq in range(1, max(WARMUP, 2) + 1):  # reach both decode targets
                url, hdrs, _ = route_throughput(config_name, seq)
                stream(url, hdrs, warmup_prompt, min(MAX_TOKENS, 20))

            def one(seq, inflight, config_name=config_name, concurrency=concurrency):
                prompt = build_prompt(
                    PROMPT_TOKENS,
                    cache_bust=("exp16", RUN_NONCE, PROMPT_TOKENS, MAX_TOKENS,
                                config_name.value, concurrency, seq))
                url, hdrs, tag = route_throughput(config_name, seq)
                r = stream(url, hdrs, prompt, MAX_TOKENS)
                base = {
                    "experiment": "exp16",
                    "config": config_name,
                    "load_mode": LOAD_MODE,
                    "prompt_tokens_target": PROMPT_TOKENS,
                    "max_tokens": MAX_TOKENS,
                    "concurrency": concurrency,
                    "run": seq,
                    "inflight_at_start": inflight,
                    "server_completion_tokens": r.usage_completion_tokens,
                    "status_code": r.status,
                    "target": tag,
                    "error": r.error,
                }
                if r.status != 200 or not r.token_times:
                    rows = [{**base, "token_idx": -1, "elapsed_ms": r.total_ms, "gap_ms": 0,
                             "tokens_in_chunk": ""}]
                else:
                    rows = [{**base, "token_idx": idx, "elapsed_ms": elapsed_ms, "gap_ms": gap_ms,
                             "tokens_in_chunk": n}
                            for idx, elapsed_ms, gap_ms, n in token_rows(r)]
                writer.write_many(rows)
                dot()
                return r

            wall_start = time.monotonic()
            results, total = run_load(LOAD_MODE, concurrency, TOTAL_REQUESTS, one)
            wall_s = time.monotonic() - wall_start
            errors = sum(1 for r in results if r.status != 200 or not r.token_times)
            progress(f" {wall_s:.1f}s | {total} requests | {total / wall_s:.1f} req/s"
                     + (f" | errors: {errors}/{total}" if errors else ""))
        progress("")

    writer.close()
    progress(f"=== Experiment 16 Complete === ({outfile})")


if __name__ == "__main__":
    main()
