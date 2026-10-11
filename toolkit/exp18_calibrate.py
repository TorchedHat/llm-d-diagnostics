#!/usr/bin/env python3
"""
Experiment 18: One-GPU Calibration

Measures, on one monolithic vLLM instance, the few numbers that decide
whether disaggregation can pay off for a workload:

  1. Prefill time vs prompt length (concurrency 1, one output token):
     fit TTFT(s) = t0 + a*s + b*s^2.
  2. Decode step time vs batch size and context: start B requests together
     and time the gaps between tokens once every request is decoding and
     before any finishes, at two or more prompt lengths; fit
     step(B, ctx) = w + B * (c0 + k * ctx): w per step (weights), c0 per
     request (sampling, scheduling), k per token of context (KV reads).
  3. KV cache capacity in tokens, from vLLM's cache_config_info metric
     (num_gpu_blocks x block_size).
  4. Overlap: how much of a decode step hides under a prefill chunk that
     shares the step. B requests decode; one prompt of OVERLAP_PROMPT_TOKENS
     (at most one chunk) is injected; the decoders' longest gap during the
     injection is the mixed step. theta = 1 - (mixed - chunk) / decode,
     where chunk is the prefill compute time of the injected prompt (from
     step 1) and decode is the step time just before injection.
     theta = 0: decode work adds fully to the chunk; theta = 1: decode is
     free inside the chunk. Values far outside [0, 1] mean the server did
     not stall decode for the chunk (the simulator does not) or the prompt
     spanned several chunks. The mixed step itself is the stall a prompt
     causes in every request decoding beside it, and is reported too; it
     depends on the server's chunk budget, so set CHUNK_BUDGET to record it.

Requests ask for token IDs (return_token_ids), so a chunk that carries
several tokens is counted as several: its gap is shared between them in
step 2, and an overlap run with such a chunk across the injection is
skipped (its longest gap may span more than one step).

The KV transfer cost comes from exp5b; pass it as TRANSFER_ALPHA_S and
TRANSFER_BW_BYTES_PER_S to include it in calibration.json.

Output:
    data/exp18-results.csv   raw measurements
    data/calibration.json    fitted coefficients, in the per-model format
                             the planning model reads (status "calibrated")

Usage: python3 toolkit/exp18_calibrate.py

Env vars:
    CALIBRATE_URL          Completions URL of one monolithic instance (default: BASELINE_URL)
    METRICS_URL            Its /metrics URL (default: derived from CALIBRATE_URL)
    PREFILL_LENGTHS        Prompt lengths for step 1 (default: 256,512,1024,2048,4096)
    PREFILL_RUNS           Requests per prompt length (default: 5)
    DECODE_BATCHES         Batch sizes for step 2 (default: 1,2,4,8,16)
    DECODE_PROMPT_TOKENS   Prompt lengths for step 2, comma-separated; two or more
                           separate per-request overhead from KV reads (default: 512,2048).
                           Step 4 uses the first.
    DECODE_OUTPUT_TOKENS   Output length for step 2 (default: 128)
    DECODE_RUNS            Repeats per batch size (default: 3)
    OVERLAP_DECODERS       Decoding requests in step 4 (default: 8)
    OVERLAP_PROMPT_TOKENS  Injected prompt length; keep it within one chunk (default: 2048)
    OVERLAP_OUTPUT_TOKENS  Decoders' output length in step 4 (default: 256)
    OVERLAP_RUNS           Repeats of step 4 (default: 3)
    CHUNK_BUDGET           The server's --max-num-batched-tokens, recorded (default: unset)
    KV_BYTES_PER_TOKEN     Recorded into calibration.json (default: unset)
    TRANSFER_ALPHA_S       KV transfer fixed cost from exp5b, seconds (default: unset)
    TRANSFER_BW_BYTES_PER_S  KV transfer bandwidth from exp5b (default: unset)
    STREAM_TIMEOUT         Socket timeout per read, seconds (default: 300)
    SKIP_PHASES            Comma-separated phases to skip: prefill,decode,kv,overlap
"""

import json
import os
import statistics
import sys
import threading
import time
import urllib.request
from itertools import pairwise
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))
from client import (
    BASELINE_URL,
    DATA_DIR,
    MODEL,
    RUN_NONCE,
    build_prompt,
    dot,
    env,
    print_config,
    progress,
    send_streaming,
    write_run_info,
)
from schemas import Exp18Row, TypedCSVWriter


def _ints(name, default):
    return [int(x) for x in env(name, default).split(",") if x.strip()]


def _opt_float(name):
    value = env(name, "").strip()
    return float(value) if value else None


CALIBRATE_URL = env("CALIBRATE_URL", BASELINE_URL)
METRICS_URL = env("METRICS_URL", "")
PREFILL_LENGTHS = _ints("PREFILL_LENGTHS", "256,512,1024,2048,4096")
PREFILL_RUNS = int(env("PREFILL_RUNS", "5"))
DECODE_BATCHES = _ints("DECODE_BATCHES", "1,2,4,8,16")
DECODE_PROMPT_TOKENS = _ints("DECODE_PROMPT_TOKENS", "512,2048")
DECODE_OUTPUT_TOKENS = int(env("DECODE_OUTPUT_TOKENS", "128"))
DECODE_RUNS = int(env("DECODE_RUNS", "3"))
OVERLAP_DECODERS = int(env("OVERLAP_DECODERS", "8"))
OVERLAP_PROMPT_TOKENS = int(env("OVERLAP_PROMPT_TOKENS", "2048"))
OVERLAP_OUTPUT_TOKENS = int(env("OVERLAP_OUTPUT_TOKENS", "256"))
OVERLAP_RUNS = int(env("OVERLAP_RUNS", "3"))
CHUNK_BUDGET = env("CHUNK_BUDGET", "").strip()
KV_BYTES_PER_TOKEN = _opt_float("KV_BYTES_PER_TOKEN")
TRANSFER_ALPHA_S = _opt_float("TRANSFER_ALPHA_S")
TRANSFER_BW = _opt_float("TRANSFER_BW_BYTES_PER_S")
STREAM_TIMEOUT = int(env("STREAM_TIMEOUT", "300"))
SKIP_PHASES = {x.strip() for x in env("SKIP_PHASES", "").split(",") if x.strip()}


# ── Pure helpers (unit-tested) ──────────────────────────────────────────────

def _solve(m, v):
    """Solve the small linear system m x = v by Gaussian elimination."""
    n = len(v)
    a = [[*row, v[i]] for i, row in enumerate(m)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular system")
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(n):
            if r != col:
                f = a[r][col] / a[col][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col], strict=True)]
    return [a[i][n] / a[i][i] for i in range(n)]


def fit_linear(rows, ys):
    """Least squares for y = rows . coef; returns (coef, r2)."""
    k = len(rows[0])
    if len(rows) < k:
        raise ValueError(f"need at least {k} points")
    m = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    v = [sum(r[i] * y for r, y in zip(rows, ys, strict=True)) for i in range(k)]
    coef = _solve(m, v)
    pred = [sum(c * x for c, x in zip(coef, r, strict=True)) for r in rows]
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred, strict=True))
    return coef, (1 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def fit_poly(xs, ys, degree):
    """Least-squares polynomial fit; returns (coefficients low→high, r2)."""
    if len(set(xs)) <= degree:
        raise ValueError(f"need more than {degree} distinct x values")
    k = degree + 1
    m = [[sum(x ** (i + j) for x in xs) for j in range(k)] for i in range(k)]
    v = [sum(y * x ** i for x, y in zip(xs, ys, strict=True)) for i in range(k)]
    coef = _solve(m, v)
    pred = [sum(c * x ** i for i, c in enumerate(coef)) for x in xs]
    mean = sum(ys) / len(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred, strict=True))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return coef, r2


def steady_decode_gaps(streams):
    """Per-token gaps (s) while every request decodes and none has finished.

    streams: list of (send_time, token_times[, token_counts]) with
    token_times relative to send_time. The window starts when the last
    request's first token arrives and ends when the first request's last
    token arrives. A chunk that carried n tokens contributes its gap / n,
    n times, so a server that batches tokens into chunks does not read as
    a slower decode step.
    """
    live = [(t0, tt, (rest[0] if rest else ()) or (1,) * len(tt))
            for t0, tt, *rest in streams if len(tt) >= 2]
    if len(live) != len(streams) or not live:
        return []
    start = max(t0 + tt[0] for t0, tt, _ in live)
    end = min(t0 + tt[-1] for t0, tt, _ in live)
    gaps = []
    for t0, tt, counts in live:
        absolute = [t0 + t for t in tt]
        for (prev, cur), n in zip(pairwise(absolute), counts[1:], strict=True):
            if prev >= start and cur <= end:
                gaps.extend([(cur - prev) / n] * n)
    return gaps


def overlap_theta(mixed_s, chunk_s, decode_s):
    """Share of a decode step hidden under a prefill chunk sharing the step."""
    if decode_s <= 0:
        raise ValueError("decode step time must be positive")
    return 1.0 - (mixed_s - chunk_s) / decode_s


def parse_kv_capacity(metrics_text):
    """KV capacity in tokens from vllm:cache_config_info, or None."""
    for line in metrics_text.splitlines():
        if not line.startswith("vllm:cache_config_info{"):
            continue
        labels = dict(
            part.split("=", 1) for part in line[line.index("{") + 1:line.index("}")].split(",")
            if "=" in part)
        try:
            blocks = int(labels["num_gpu_blocks"].strip('"'))
            block_size = int(labels["block_size"].strip('"'))
        except (KeyError, ValueError):
            return None
        return blocks * block_size
    return None


# ── Measurement ─────────────────────────────────────────────────────────────

def _stream(prompt, max_tokens):
    return send_streaming(CALIBRATE_URL, prompt, max_tokens, timeout=STREAM_TIMEOUT,
                          ignore_eos=True, include_usage=True, token_ids=True)


def _prompt(tokens, *key):
    return build_prompt(tokens, cache_bust=("exp18", RUN_NONCE, tokens, *key))


def _start_streams(n, prompt_tokens, max_tokens, key):
    """Start n streaming requests together; return threads and a result list."""
    out = [None] * n

    def run(i):
        t0 = time.monotonic()
        out[i] = (t0, _stream(_prompt(prompt_tokens, *key, i), max_tokens))

    threads = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(n)]
    for th in threads:
        th.start()
    return threads, out


def _row(writer, phase, metric, param, run, value, unit, r=None, detail=""):
    writer.write({
        "experiment": "exp18", "phase": phase, "metric": metric, "param": param,
        "run": run, "value": round(value, 6), "unit": unit,
        "prompt_tokens_actual": r.prompt_tokens if r else "",
        "status_code": r.status if r else "", "detail": detail,
        "error": r.error if r else "",
    })


def phase_prefill(writer, warnings):
    progress("--- Step 1: prefill time vs prompt length ---")
    points = []
    for s in PREFILL_LENGTHS:
        ttfts = []
        for run in range(1, PREFILL_RUNS + 1):
            r = _stream(_prompt(s, "prefill", run), 1)
            _row(writer, "prefill", "ttft", s, run, r.ttft_ms / 1000, "s", r)
            if r.status == 200 and r.token_times:
                ttfts.append((r.prompt_tokens or s, r.ttft_ms / 1000))
            dot()
        if ttfts:
            points.append((statistics.median(p for p, _ in ttfts),
                           statistics.median(t for _, t in ttfts)))
    progress("")
    if len(points) < 3:
        warnings.append("prefill: fewer than 3 prompt lengths succeeded; no fit")
        return None
    xs, ys = [p for p, _ in points], [t for _, t in points]
    (t0, a, b), r2 = fit_poly(xs, ys, 2)
    if b < 0:
        warnings.append("prefill: negative quadratic term; using a linear fit")
        (t0, a), r2 = fit_poly(xs, ys, 1)
        b = 0.0
    progress(f"  t0={t0 * 1000:.1f} ms  a={a * 1e6:.2f} us/token  b={b * 1e9:.3f} ns/token^2  "
             f"R^2={r2:.4f}")
    return {"prefill_t0_s": t0, "prefill_a_s_per_token": a, "prefill_b_s_per_token2": b,
            "prefill_fit_r2": r2, "prefill_points": points}


def phase_decode(writer, warnings):
    progress("--- Step 2: decode step time vs batch size and context ---")
    points = []  # (batch, mean context tokens, step seconds)
    for prompt_tokens in DECODE_PROMPT_TOKENS:
        ctx = prompt_tokens + DECODE_OUTPUT_TOKENS / 2
        for batch in DECODE_BATCHES:
            steps = []
            for run in range(1, DECODE_RUNS + 1):
                threads, out = _start_streams(batch, prompt_tokens, DECODE_OUTPUT_TOKENS,
                                              ("decode", prompt_tokens, batch, run))
                for th in threads:
                    th.join()
                failed = [r for _, r in out if r.status != 200]
                if failed:
                    warnings.append(f"decode: B={batch} prompt={prompt_tokens} run {run}: "
                                    f"{len(failed)} failed requests")
                    continue
                gaps = steady_decode_gaps([(t0, r.token_times, r.token_counts) for t0, r in out])
                if len(gaps) < batch:
                    warnings.append(f"decode: B={batch} prompt={prompt_tokens} run {run}: no "
                                    f"steady window (raise DECODE_OUTPUT_TOKENS, or B exceeds "
                                    f"--max-num-seqs so requests queued)")
                    continue
                step = statistics.median(gaps)
                steps.append(step)
                _row(writer, "decode", "step", f"B={batch};ctx={ctx:.0f}", run, step, "s",
                     detail=f"gaps={len(gaps)}")
                dot()
            if steps:
                points.append((batch, ctx, statistics.median(steps)))
    progress("")
    contexts = sorted({c for _, c, _ in points})
    if len({b for b, _, _ in points}) < 2:
        warnings.append("decode: fewer than 2 batch sizes succeeded; no fit")
        return None
    if len(contexts) >= 2:
        # step = w + B * (c0 + k * ctx): per-step weights, per-request overhead, KV reads.
        (w, c0, k), r2 = fit_linear([[1.0, b, b * c] for b, c, _ in points],
                                    [s for _, _, s in points])
        # A negative term is noise, and extrapolating it to long contexts or large
        # batches would be wrong: drop it and refit.
        if k < 0:
            warnings.append(f"decode: negative KV-read term ({k * 1e9:.1f} ns/token) is noise; "
                            "refitted without it (more DECODE_RUNS or wider contexts help)")
            (w, c0), r2 = fit_linear([[1.0, b] for b, _, _ in points], [s for _, _, s in points])
            k = 0.0
        if c0 < 0:
            warnings.append(f"decode: negative per-request term ({c0 * 1000:.3f} ms) is noise; "
                            "refitted without it")
            (w, k), r2 = fit_linear([[1.0, b * c] for b, c, _ in points],
                                    [s for _, _, s in points])
            c0 = 0.0
    else:
        warnings.append("decode: one context only; the per-request slope is attributed to KV "
                        "reads (set two DECODE_PROMPT_TOKENS values to separate them)")
        (w, slope), r2 = fit_poly([b for b, _, _ in points], [s for _, _, s in points], 1)
        c0, k = 0.0, slope / contexts[0]
    progress(f"  step(B, ctx) = {w * 1000:.2f} ms + B x ({c0 * 1000:.3f} ms + "
             f"{k * 1e9:.2f} ns x ctx)   R^2={r2:.4f}")
    return {"decode_weight_s": w, "decode_per_request_s": c0, "decode_kv_s_per_token": k,
            "decode_fit_r2": r2, "decode_points": points}


def phase_kv(warnings):
    progress("--- Step 3: KV cache capacity ---")
    url = METRICS_URL
    if not url:
        p = urlparse(CALIBRATE_URL)
        url = f"{p.scheme}://{p.netloc}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            tokens = parse_kv_capacity(resp.read().decode("utf-8", errors="replace"))
    except OSError as exc:
        warnings.append(f"kv: could not read {url}: {exc}")
        return None
    if tokens is None:
        warnings.append(f"kv: vllm:cache_config_info not found at {url}")
        return None
    progress(f"  {tokens} tokens")
    return {"kv_capacity_tokens": tokens}


def phase_overlap(writer, warnings, prefill, decode):
    progress("--- Step 4: overlap of decode under a prefill chunk ---")
    if not prefill or not decode:
        warnings.append("overlap: needs steps 1 and 2")
        return None
    if not CHUNK_BUDGET:
        warnings.append("overlap: CHUNK_BUDGET not set; the stall a prompt causes depends on "
                        "the server's --max-num-batched-tokens (with vLLM's default of 8192, "
                        "an 8k prompt prefills in one step), so record it")
    elif OVERLAP_PROMPT_TOKENS > int(CHUNK_BUDGET):
        warnings.append("overlap: OVERLAP_PROMPT_TOKENS exceeds CHUNK_BUDGET; the prompt "
                        "spans several chunks and theta is not meaningful")
    b = OVERLAP_DECODERS
    chunk = (prefill["prefill_a_s_per_token"] * OVERLAP_PROMPT_TOKENS
             + prefill["prefill_b_s_per_token2"] * OVERLAP_PROMPT_TOKENS ** 2)
    dec_prompt = DECODE_PROMPT_TOKENS[0]
    ctx = dec_prompt + OVERLAP_OUTPUT_TOKENS / 2
    step = decode["decode_weight_s"] + b * (decode["decode_per_request_s"]
                                            + decode["decode_kv_s_per_token"] * ctx)
    ttft_dec = (prefill["prefill_t0_s"] + prefill["prefill_a_s_per_token"] * dec_prompt
                + prefill["prefill_b_s_per_token2"] * dec_prompt ** 2)
    # Inject once every decoder is surely decoding: all prompts, then 20 steps.
    delay = b * ttft_dec + 20 * step
    if delay + chunk + 20 * step > OVERLAP_OUTPUT_TOKENS * step:
        warnings.append("overlap: decoders may finish before the injection; "
                        "raise OVERLAP_OUTPUT_TOKENS")
    thetas, mixed_steps, decode_steps = [], [], []
    for run in range(1, OVERLAP_RUNS + 1):
        threads, out = _start_streams(b, dec_prompt, OVERLAP_OUTPUT_TOKENS,
                                      ("overlap", run))
        time.sleep(delay)
        inject_start = time.monotonic()
        injected = _stream(_prompt(OVERLAP_PROMPT_TOKENS, "inject", run), 1)
        inject_end = inject_start + injected.ttft_ms / 1000
        for th in threads:
            th.join()
        if injected.status != 200 or any(r.status != 200 for _, r in out):
            warnings.append(f"overlap: run {run} had failed requests")
            continue
        before, longest, merged = [], [], 0
        for t0, r in out:
            absolute = [t0 + t for t in r.token_times]
            counts = r.token_counts or (1,) * len(absolute)
            during = []
            for (prev, cur), n in zip(pairwise(absolute), counts[1:], strict=True):
                if cur <= inject_start:
                    before.append((cur, (cur - prev) / n))
                elif prev < inject_end and cur > inject_start:
                    during.append(cur - prev)
                    merged += n > 1
            if during:
                longest.append(max(during))
        if merged:
            # Several tokens in one chunk: the longest gap may span more than
            # the mixed step, so this run cannot measure it.
            warnings.append(f"overlap: run {run}: {merged} chunks across the injection carried "
                            "several tokens; run skipped")
            continue
        if not before or len(longest) < b:
            warnings.append(f"overlap: run {run}: decoders not running across the injection")
            continue
        mixed = statistics.median(longest)
        # Step time just before the injection: the last few steps of every decoder.
        decode_step = statistics.median(g for _, g in sorted(before)[-5 * b:])
        theta = overlap_theta(mixed, chunk, decode_step)
        thetas.append(theta)
        mixed_steps.append(mixed)
        decode_steps.append(decode_step)
        _row(writer, "overlap", "mixed_step", OVERLAP_PROMPT_TOKENS, run, mixed, "s", injected)
        _row(writer, "overlap", "decode_step", b, run, decode_step, "s")
        _row(writer, "overlap", "theta", OVERLAP_PROMPT_TOKENS, run, theta, "ratio",
             detail=f"chunk_compute_s={chunk:.6f}")
        dot()
    progress("")
    if not thetas:
        return None
    theta = statistics.median(thetas)
    if not -0.5 <= theta <= 1.5:
        warnings.append(f"overlap: theta={theta:.2f} is far outside [0, 1]: the server did not "
                        "stall decode for the chunk, or the prompt spanned several chunks")
    mixed = statistics.median(mixed_steps)
    progress(f"  theta={theta:.2f} (runs: {', '.join(f'{t:.2f}' for t in thetas)})")
    progress(f"  stall: decoders' longest gap during a {OVERLAP_PROMPT_TOKENS}-token prompt "
             f"{mixed * 1000:.1f} ms (decode step {statistics.median(decode_steps) * 1000:.1f} ms)")
    return {"overlap_theta": theta, "overlap_runs": thetas,
            "overlap_chunk_compute_s": chunk, "overlap_prompt_tokens": OVERLAP_PROMPT_TOKENS,
            "overlap_mixed_step_s": mixed,
            "overlap_decode_step_s": statistics.median(decode_steps)}


def main():
    outfile = os.path.join(DATA_DIR, "exp18-results.csv")
    calfile = os.path.join(DATA_DIR, "calibration.json")
    settings = {
        "calibrate_url": CALIBRATE_URL, "prefill_lengths": PREFILL_LENGTHS,
        "prefill_runs": PREFILL_RUNS, "decode_batches": DECODE_BATCHES,
        "decode_prompt_tokens": DECODE_PROMPT_TOKENS,
        "decode_output_tokens": DECODE_OUTPUT_TOKENS, "decode_runs": DECODE_RUNS,
        "overlap_decoders": OVERLAP_DECODERS, "overlap_prompt_tokens": OVERLAP_PROMPT_TOKENS,
        "overlap_output_tokens": OVERLAP_OUTPUT_TOKENS, "overlap_runs": OVERLAP_RUNS,
        "chunk_budget": CHUNK_BUDGET or None, "skip_phases": sorted(SKIP_PHASES),
    }
    write_run_info("exp18", settings)
    writer = TypedCSVWriter(outfile, Exp18Row)

    progress("=== Experiment 18: One-GPU Calibration ===")
    print_config()
    progress(f"  Target: {CALIBRATE_URL}")
    progress(f"  Output: {outfile}, {calfile}")
    progress("")

    warnings = []
    prefill = None if "prefill" in SKIP_PHASES else phase_prefill(writer, warnings)
    decode = None if "decode" in SKIP_PHASES else phase_decode(writer, warnings)
    kv = None if "kv" in SKIP_PHASES else phase_kv(warnings)
    overlap = (None if "overlap" in SKIP_PHASES
               else phase_overlap(writer, warnings, prefill, decode))
    writer.close()

    calibration = {
        "status": "calibrated",
        "model": MODEL,
        "run_dir": os.path.abspath(DATA_DIR),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "chunk_budget_tokens": int(CHUNK_BUDGET) if CHUNK_BUDGET else None,
        "moe": None,
        "transfer": ({"alpha_s": TRANSFER_ALPHA_S, "bw_bytes_per_s": TRANSFER_BW}
                     if TRANSFER_ALPHA_S is not None and TRANSFER_BW is not None else None),
        "settings": settings,
        "warnings": warnings,
    }
    for part in (prefill, decode, kv, overlap):
        if part:
            calibration.update(part)
    with open(calfile, "w") as fh:
        json.dump(calibration, fh, indent=2)

    for w in warnings:
        progress(f"  WARNING: {w}")
    progress(f"=== Experiment 18 Complete === ({calfile})")


if __name__ == "__main__":
    main()
