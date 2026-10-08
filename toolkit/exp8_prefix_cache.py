#!/usr/bin/env python3
"""
Experiment 8: Prefix Cache Characterization

Measures prefix cache behavior on decode pods by using TTFT (time to first
token) as a proxy for cache hits. vLLM does not expose cache hit rate
directly, but a cache hit skips prefill computation, producing measurably
lower TTFT on the second request with the same prefix.

Methodology:
  - Uses streaming requests (send_streaming) for true TTFT measurement.
    Non-streaming TTFT ≈ total_ms because vLLM buffers responses; streaming
    captures the actual time to first generated token.
  - Trial order is randomized within each run to prevent order effects
    (GPU thermal drift, OS scheduling) from biasing one condition.
  - Negative control (different prompt, same length) validates that TTFT
    drops are from prefix cache, not GPU warmth or connection state.

Three direct-to-pod phases plus an optional EPP-routed comparison:

  Phase 1 — Cache Hit/Miss Baseline
    For each run, execute four conditions in random order:
      (a) Cold: first time this prefix on this pod
      (b) Warm: same prefix, same pod (should hit prefix cache)
      (c) Control: DIFFERENT prefix, same pod, same length (should be cold)
      (d) Cross-pod: same prefix, different pod (should be cold)
    A valid prefix cache produces: TTFT(warm) < TTFT(cold) ≈ TTFT(control).
    If TTFT(control) < TTFT(cold), the proxy is broken (GPU warmth, not cache).

  Phase 2 — Multi-Turn Simulation
    Simulate a 5-turn conversation where each turn extends the prefix.
    Later turns should benefit from prefix caching on the same pod.

  Phase 3 — Cache Decay Quick Check
    Send a prompt, wait CACHE_DECAY_S seconds, resend to the same pod.
    Checks whether the cache survives a short idle period.
    (exp10 does the full eviction sweep.)

  Phase 4 — EPP Routing (when EPP_URL is set)
    Repeat unique prompts through the Gateway. Backend pod identity is not
    observable here, so this measures cache behavior under the configured
    endpoint-selection policy rather than a guaranteed same-pod cache hit.

Phases 1-3 go to decode pods directly (port 8001, bypassing the sidecar) so
that sidecar overhead does not confound TTFT. Phase 4 intentionally uses EPP.

Usage:
    python3 toolkit/exp8_prefix_cache.py
    CACHE_LENGTHS=100,500 RUNS=5 python3 toolkit/exp8_prefix_cache.py

Env vars:
    CACHE_LENGTHS   Comma-separated prefix lengths (default: 100,500,1000)
    RUNS            Runs per config in phase 1 (default: 10)
    CONV_RUNS       Repetitions of multi-turn sim (default: 5)
    CACHE_DECAY_S   Seconds to wait in decay check (default: 30)
    MAX_TOKENS      Max completion tokens (default: 20)
    DATA_DIR        Output directory (default: data)
    WARMUP          Warmup requests (default: 3)
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    DATA_DIR,
    EPP_URL,
    MAX_TOKENS,
    PinnedConnection,
    build_prompt,
    decode_direct_url_by_ip,
    discover_pod_ips,
    dot,
    env,
    epp_enabled,
    print_config,
    progress,
    write_run_info,
)
from schemas import CachePhase, CacheState, Exp8Row, TypedCSVWriter

CACHE_LENGTHS = [int(x) for x in env("CACHE_LENGTHS", "100,500,1000").split(",")]
RUNS = int(env("RUNS", "10"))
CONV_RUNS = int(env("CONV_RUNS", "5"))
CACHE_DECAY_S = int(env("CACHE_DECAY_S", "30"))



def main():
    outfile = os.path.join(DATA_DIR, "exp8-results.csv")
    write_run_info("exp8", {
        "cache_lengths": CACHE_LENGTHS,
        "runs": RUNS,
        "conv_runs": CONV_RUNS,
        "cache_decay_s": CACHE_DECAY_S,
    })
    writer = TypedCSVWriter(outfile, Exp8Row)

    progress("=== Experiment 8: Prefix Cache Characterization ===")
    print_config()
    progress(f"  Cache lengths: {CACHE_LENGTHS}")
    progress(f"  Runs per length: {RUNS}")
    progress(f"  Conv runs: {CONV_RUNS}, Decay wait: {CACHE_DECAY_S}s")
    progress(f"  Output: {outfile}")
    progress("")

    # Discover decode pods
    pods = discover_pod_ips("app=vllm-decode")
    if not pods:
        progress("ERROR: no decode pods found (label=app=vllm-decode)")
        writer.close()
        return

    pod0_name, pod0_ip = pods[0]
    url0 = decode_direct_url_by_ip(pod0_ip)
    has_second_pod = len(pods) >= 2
    if has_second_pod:
        pod1_name, pod1_ip = pods[1]
        url1 = decode_direct_url_by_ip(pod1_ip)
    else:
        progress("WARNING: only 1 decode pod found, skipping cross-pod tests")

    progress(f"  Decode pods: {[p[0] for p in pods]}")
    progress("  Using streaming requests for true TTFT measurement")
    progress("")

    # Create pinned connections for connection reuse
    conn0 = PinnedConnection(url0, pod_name=pod0_name)
    conn1 = None
    conn_epp = PinnedConnection(EPP_URL, pod_name="epp-gateway") if epp_enabled() else None
    if has_second_pod:
        conn1 = PinnedConnection(url1, pod_name=pod1_name)

    def record(phase, config, run, pod, prompt_tokens, turn,
               cache_state, r, trial_order=0):
        writer.write({
            "experiment": "exp8",
            "phase": phase,
            "config": config,
            "run": run,
            "pod": pod,
            "prompt_tokens": prompt_tokens,
            "turn": turn,
            "cache_state": cache_state,
            "trial_order": trial_order,
            "ttft_ms": r.ttft_ms,
            "total_ms": r.total_ms,
            "status_code": r.status,
            "completion_tokens": r.completion_tokens,
            "error": r.error,
        })

    try:
        # ── Phase 1: Cache Hit/Miss Baseline ──────────────────────────────

        progress("--- Phase 1: Cache Hit/Miss Baseline ---")
        progress("  (trial order randomized within each run)")

        # Warm up the first pod (TLS handshake happens here)
        warmup_prompt = build_prompt(50)
        conn0.warmup(warmup_prompt, MAX_TOKENS)

        for ptokens in CACHE_LENGTHS:
            prompt = build_prompt(ptokens)
            # Negative control: prefix differs from first token to prevent
            # partial cache hits. build_prompt() repeats BASE_SENTENCE, so we
            # prepend unique text to guarantee a different token sequence.
            control_prompt = f"Control measurement for length {ptokens}: " + build_prompt(ptokens)
            config = f"len-{ptokens}"
            progress(f"  prefix_len={ptokens}: ", end="")

            for run in range(1, RUNS + 1):
                # Step 1: ALWAYS prime the cache first (cold then warm depend on this)
                # The cold measurement uses a DIFFERENT prompt for this run to
                # flush any prior cache entry for `prompt` on this pod.
                flush_prompt = f"Flush run {run} len {ptokens}: " + build_prompt(ptokens)
                conn0.send(flush_prompt, MAX_TOKENS)

                # Step 2: Send the test prompt cold (first time this run)
                r_cold = conn0.send_streaming(prompt, MAX_TOKENS)
                record(CachePhase.HIT_MISS, config, run, pod0_name, ptokens, 0,
                       CacheState.COLD, r_cold, trial_order=1)
                dot()

                # Step 3: Now the cache should contain `prompt`. Send the
                # remaining conditions in RANDOM order to prevent order effects.
                conditions = []
                conditions.append((CacheState.WARM_SAME_POD, conn0, prompt, pod0_name))
                conditions.append((CacheState.CONTROL_DIFF_PROMPT, conn0, control_prompt, pod0_name))
                if has_second_pod:
                    conditions.append((CacheState.WARM_DIFF_POD, conn1, prompt, pod1_name))

                random.shuffle(conditions)

                for order_idx, (state, conn, p, pod) in enumerate(conditions, start=2):
                    r = conn.send_streaming(p, MAX_TOKENS)
                    record(CachePhase.HIT_MISS, config, run, pod, ptokens, 0,
                           state, r, trial_order=order_idx)
                    dot()

                    # After warm_same_pod, re-prime the cache for the next
                    # condition (in case we shuffled control before warm).
                    # This ensures warm_same_pod always has a cache entry
                    # available regardless of shuffle order.
                    if state != CacheState.WARM_SAME_POD:
                        conn0.send(prompt, MAX_TOKENS)

            progress(" done")

        progress("")

        # ── Phase 2: Multi-Turn Simulation ────────────────────────────────

        progress("--- Phase 2: Multi-Turn Simulation ---")
        progress("  (streaming TTFT per turn)")
        turns = 5
        base_tokens = 100

        for conv in range(1, CONV_RUNS + 1):
            progress(f"  conversation {conv}/{CONV_RUNS}: ", end="")
            conversation = build_prompt(base_tokens)

            for turn in range(1, turns + 1):
                r = conn0.send_streaming(conversation, MAX_TOKENS)
                record(CachePhase.MULTI_TURN, f"conv-{conv}", conv, pod0_name,
                       base_tokens + (turn - 1) * 50, turn,
                       f"turn_{turn}", r)
                dot()
                # Extend conversation for next turn (~50 tokens)
                conversation += " " + build_prompt(50)

            progress(" done")

        progress("")

        # ── Phase 3: Cache Decay Quick Check ──────────────────────────────

        progress(f"--- Phase 3: Cache Decay Quick Check (wait={CACHE_DECAY_S}s) ---")
        decay_prompt = build_prompt(500)

        for run in range(1, 4):
            progress(f"  decay run {run}/3: ", end="")

            # Prime the cache with streaming request (measures true TTFT)
            r = conn0.send_streaming(decay_prompt, MAX_TOKENS)
            record(CachePhase.DECAY, "decay-500", run, pod0_name, 500, 0,
                   CacheState.DECAY_PRIME, r)
            dot()

            # Confirm cache is warm
            r = conn0.send_streaming(decay_prompt, MAX_TOKENS)
            record(CachePhase.DECAY, "decay-500", run, pod0_name, 500, 0,
                   CacheState.DECAY_WARM, r)
            dot()

            # Wait for potential cache eviction
            progress(f"waiting {CACHE_DECAY_S}s...", end="")
            time.sleep(CACHE_DECAY_S)

            # Resend — check if cache survived
            r = conn0.send_streaming(decay_prompt, MAX_TOKENS)
            record(CachePhase.DECAY, "decay-500", run, pod0_name, 500, 0,
                   CacheState.DECAY_AFTER_WAIT, r)
            dot()
            progress(" done")

        progress("")

        if conn_epp:
            progress("--- Phase 4: Prefix Cache Through EPP ---")
            progress("  Backend pod is intentionally not pinned; EPP selects each request.")
            for run in range(1, RUNS + 1):
                prompt = f"EPP cache probe {run}: " + build_prompt(500)
                r_first = conn_epp.send_streaming(prompt, MAX_TOKENS)
                record(CachePhase.EPP_ROUTING, "epp-gateway", run, "epp-gateway",
                       500, 0, CacheState.EPP_FIRST, r_first)
                r_repeat = conn_epp.send_streaming(prompt, MAX_TOKENS)
                record(CachePhase.EPP_ROUTING, "epp-gateway", run, "epp-gateway",
                       500, 0, CacheState.EPP_REPEAT, r_repeat)
                dot()
            progress("")

    finally:
        conn0.close()
        if conn1 is not None:
            conn1.close()
        if conn_epp is not None:
            conn_epp.close()
        writer.close()

    progress(f"=== Experiment 8 Complete === ({outfile})")


if __name__ == "__main__":
    main()
