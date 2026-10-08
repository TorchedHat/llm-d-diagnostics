# Measurement Protocol for llm-d Disaggregated Inference

## Scope

This protocol governs how we collect, validate, analyze, and report
performance measurements of the Phi-3 3.8B deployment on 5×T4 GPUs.

We are characterizing **this system** — not extrapolating to other
hardware or models. Claims must be traceable to measurements.

---

## 1. Principles

**P1. Measure, don't model.** Every claim must have a direct measurement
behind it. If we fit a line, we report the residuals. If we extrapolate,
we say "extrapolation" and quantify how far beyond the data we've gone.

**P2. State your hypothesis before running.** Every experiment answers a
specific question. Write the question and your prediction *before*
collecting data. If the result surprises you, that's the interesting part.

**P3. Control what you can, measure what you can't.** Document GPU state,
pod restart count, time of day, other cluster activity. If you can't
control it, at least record it so you can check later.

**P4. Report uncertainty honestly.** A number without an error bar is not
a measurement. Use 95% confidence intervals. Report n. Report what you
filtered and why.

**P5. Anomalies are data.** Don't drop outliers silently. Investigate
them. A "bad" data point that you understand teaches more than a clean
dataset you don't.

---

## 2. Pre-Flight Checklist

Before any experiment run:

- [ ] All pods Running and Ready (`oc get pods -n $NS`)
- [ ] Pod restart counts recorded (they affect warmup state)
- [ ] Verify each request path with a single test request:
  - A: prefill headless Service (pod port 8000, HTTP; SIM uses 8100)
  - B: decode-direct (port 8001, HTTP)
  - C: sidecar-only (port 8000; scheme follows `SIDECAR_SCHEME`)
  - D: disaggregated (sidecar port 8000; scheme follows `SIDECAR_SCHEME`, routed through prefill pod port 8000)
  - E: EPP Gateway route (when configured; record its URL and scheduler metrics)
- [ ] Record `oc get pods -o wide` (node placement, IPs, ages)
- [ ] Note time and any known cluster activity

## 3. Warmup Protocol

**Problem:** First requests to a cold model are slower (CUDA JIT, KV cache
allocation, connection setup, cuDNN autotuning). If warmup requests leak
into measurement, short-sequence data is contaminated.

**Protocol:**
1. Send `warmup_runs` requests (default: 5) to EACH config at EACH
   sequence length before measuring. Not just 5 total — 5 per condition.
2. Wait 2 seconds between warmup and measurement to let GPU clocks settle.
3. Record warmup latencies separately (they're useful for cold-start analysis).

**Validation:** Compare first 3 measurement runs to last 3. If they differ
by >10%, warmup was insufficient.

## 4. Sequence Length Sweep (Exp5)

### 4.1 What We're Measuring

For each sequence length L, we measure four TTFT values per run:
- A(L): prefill GPU, direct — baseline compute time
- B(L): decode GPU, direct — same model, different GPU
- C(L): decode GPU via sidecar — adds sidecar routing overhead
- D(L): disaggregated — adds KV transfer from prefill

The paired difference T_transfer(L) = D(L) - C(L) isolates the KV
transfer cost from all other latency components.

### 4.2 What We Expect (Physics)

T_transfer has two components:
```
T_transfer(L) = T_fixed + T_bandwidth(L)
```

- **T_fixed**: protocol overhead — NIXL handshake, sidecar routing, HTTP
  round-trips. Should be constant with L.
- **T_bandwidth(L)**: time to move KV cache data. Proportional to
  KV_bytes(L) = 2 × n_layers × n_kv_heads × d_head × dtype × L.

For Phi-3 (MHA, 32 KV heads, 32 layers, 96 d_head, fp16):
```
KV_bytes(L) = 2 × 32 × 32 × 96 × 2 × L = 393,216 × L bytes
```

At T4 memory bandwidth (300 GB/s theoretical):
```
T_bandwidth(L) = 393,216 × L / (300 × 10^9) × 1000 ms
              = 0.00131 × L ms
```

So at 1000 tokens: T_bandwidth ≈ 1.3ms from raw bandwidth alone.

**But we measure ~100ms.** The gap is the interesting part — it tells us
about protocol overhead, serialization, network transit, and NIXL
implementation efficiency.

### 4.3 Analysis Standards

**Regression:** Fit T_transfer(L) = a + b×L on ALL individual paired
differences (not on medians). This gives proper standard errors from the
actual data variance, not from 6-point residuals.

**Diagnostics to report:**
- Pearson r and R²
- Residual plot (are residuals random, or is there curvature?)
- Cook's distance (do any points dominate the fit?)
- Breusch-Pagan or visual check for heteroscedasticity (does variance
  change with L?)

**If the fit is poor (R² < 0.9):** Don't force a linear model. Report
per-length statistics instead. A table of medians with CIs is more
honest than a bad regression.

**If there's a regime change:** Fit piecewise. Report the breakpoint
and both slopes. The breakpoint is interesting — it tells you when
a different physical mechanism kicks in.

### 4.4 Comparisons Across Topologies

When comparing 1P+2D vs 2P+3D:
- **Same conditions?** Check that configs A and B (which don't use
  disaggregation) give similar values. If C differs between topologies,
  something besides the transfer path changed.
- **Report confounds:** Different nodes, different pod ages, different
  times of day, different restart counts.

## 5. Latency Decomposition (Exp1b)

### 5.1 The Decomposition

```
T_overhead = D - A = (D - C) + (C - B) + (B - A)
           = T_transfer + T_sidecar + T_gpu_diff
```

- **T_transfer = D - C:** KV transfer cost (should be ≥0)
- **T_sidecar = C - B:** sidecar routing overhead (should be ≥0)
- **T_gpu_diff = B - A:** difference between decode and prefill GPU
  (should be ≈0 for same model; nonzero means GPU or load asymmetry)

### 5.2 Sum Check

T_transfer + T_sidecar + T_gpu_diff MUST equal T_overhead exactly
(it's algebraic, not approximate). If the analysis reports a "residual",
something is wrong with the pairing — missed runs, filtered runs, or
a bug in the analysis code.

**The current analysis reports 31% residual.** This must be investigated
and explained before we trust any decomposition numbers.

### 5.3 Paired vs Unpaired

Always use paired differences (D_i - C_i for the same run i), not
difference of medians. Paired analysis cancels run-to-run variation
(load, scheduling, temperature). Difference of medians does not.

## 6. Data Quality Gates

Before analyzing, every dataset must pass:

| Check | Threshold | Action if failed |
|-------|-----------|------------------|
| Error rate | <5% per config | Investigate cause, exclude errors, report rate |
| Truncated responses | <2% (completion_tokens < target) | Exclude, report count |
| Warmup contamination | First 3 runs ≈ last 3 runs (within 10%) | Exclude warmup runs |
| A ≈ B validation | Differ <5% at each seq_len | Investigate GPU asymmetry |
| C topology-independent | C similar across topologies (within 10%) | Investigate confound |
| Monotonicity | TTFT increases with seq_len for A, B (compute-bound) | Investigate non-monotonic points |

## 7. Reporting Standards

### What to report for each metric:
- Median (robust to outliers)
- Mean (for comparison and sum checks)
- 95% CI (from t-distribution, not assumed normal)
- n (sample size after filtering)
- Number filtered and why
- CV (coefficient of variation — flags noisy measurements)

### What NOT to do:
- Don't report "effective bandwidth" or "effective TFLOPS" derived from
  slopes unless the slope is physically meaningful (R² > 0.95, residuals
  are random, and the derived value is physically plausible).
- Don't extrapolate beyond the measured range without labeling it
  "EXTRAPOLATION" and stating the extrapolation factor.
- Don't report a number from a failed validation as if it passed.

### Figures of merit:
- **T_transfer at each measured length** with CI — the primary result
- **T_fixed** (transfer intercept) — only if linear fit is good
- **Marginal cost per token** (slope) — only if linear fit is good
- **Regime change point** — if nonlinearity detected

## 8. Known Issues to Investigate

These are open questions from our current data that must be resolved
before we can trust the measurements:

1. **Exp1b 31% residual.** The decomposition doesn't add up. Is this
   a pairing bug in analyze.py, or a real unmeasured component?

2. **C differs across topologies at short sequences.** C (sidecar-only)
   should be topology-independent. At 10tok: 682ms (1P+2D) vs 849ms
   (2P+3D). At 100tok they converge. Why?

3. **High variance at short sequences.** Std of D-C is 24-30ms at
   10-50tok, drops to 1-5ms at 100-500tok. Is this warmup contamination,
   connection reuse, or something else?

4. **1000-token regime change in 2P+3D.** D-C jumps from 72ms (500tok)
   to 142ms (1000tok), a 97% increase for 2x tokens. The 1P+2D cluster
   shows 89ms→108ms (21%). What's different?

5. **Protocol overhead model-dependence.** Intercept varies from 29ms
   (TinyLlama) to 45-66ms (Phi-3). If protocol overhead is model-
   independent, why? If it's not, what model-dependent cost is hiding
   in the intercept?

---

## 9. Revision History

- 2026-04-25: Initial protocol. Motivated by discovering that our
  scaling model passed validation checks that should have stopped it.
