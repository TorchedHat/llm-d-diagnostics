#!/usr/bin/env python3
"""
llm-d Diagnostics Toolkit — Analysis

Reads CSV data from experiments and produces rigorous statistical summaries.
Flags data quality issues, computes confidence intervals, and separates
findings from predictions.

Usage:
    python3 toolkit/analyze.py [data_dir]

Default data_dir: data/
"""

import csv
import math
import os
import sys
from collections import defaultdict


def load_csv(filepath):
    """Load a CSV file and return list of dicts."""
    if not os.path.exists(filepath):
        return []
    with open(filepath) as f:
        return list(csv.DictReader(f))


def safe_float(val, default=0.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_status(row):
    """Get HTTP status code from a row, handling both 'status_code' and 'status' fields."""
    return safe_int(row.get("status_code", row.get("status", 0)))


# ── Statistics ───────────────────────────────────────────────────────────────

# Two-tailed t critical values at 95% for common df (scipy-free)
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 19: 2.093, 20: 2.086, 24: 2.064, 29: 2.045,
    30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
}


def _t95(df):
    """Return t critical value for 95% CI. Uses table for small df, 1.96 for large."""
    if df in _T95:
        return _T95[df]
    # Interpolate from nearest keys
    keys = sorted(_T95.keys())
    if df > keys[-1]:
        return 1.96
    lo = max(k for k in keys if k <= df)
    hi = min(k for k in keys if k >= df)
    if lo == hi:
        return _T95[lo]
    frac = (df - lo) / (hi - lo)
    return _T95[lo] * (1 - frac) + _T95[hi] * frac


def stats(values):
    """Compute summary statistics for a list of floats.

    Returns dict with: n, mean, median, std, min, max, p10, p25, p75, p90,
    iqr, ci95_lo, ci95_hi (95% confidence interval for the mean),
    cv (coefficient of variation), outliers (count beyond 3×IQR from median).
    """
    if not values:
        return {"n": 0}

    s = sorted(values)
    n = len(s)
    mean = sum(s) / n

    if n >= 2:
        variance = sum((x - mean) ** 2 for x in s) / (n - 1)
        std = math.sqrt(variance)
        # 95% CI for the mean (t-approximation for small n)
        t_val = _t95(n - 1)
        ci_half = t_val * std / math.sqrt(n)
    else:
        std = float('inf')  # unknown with n=1
        ci_half = float('inf')  # CI is undefined, not zero-width

    def percentile(p):
        idx = p / 100 * (n - 1)
        lo = math.floor(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] * (1 - frac) + s[hi] * frac

    p25 = percentile(25)
    p75 = percentile(75)
    iqr = p75 - p25

    # Outlier detection: values beyond 3×IQR from the quartiles (Tukey's far fence).
    # These are extreme outliers that likely represent transient system events
    # (GC pauses, network retransmits, pod scheduling) rather than steady-state.
    fence_lo = p25 - 3.0 * iqr
    fence_hi = p75 + 3.0 * iqr
    outlier_count = sum(1 for x in s if x < fence_lo or x > fence_hi)

    # Coefficient of variation: std/mean. Dimensionless measure of noise.
    # CV < 0.10 = precise measurement, 0.10-0.30 = moderate noise, > 0.30 = noisy.
    cv = std / mean if mean > 0 and n >= 2 else float('inf')

    return {
        "n": n,
        "mean": round(mean, 1),
        "median": round(percentile(50), 1),
        "std": round(std, 1),
        "min": round(s[0], 1),
        "max": round(s[-1], 1),
        "p10": round(percentile(10), 1),
        "p25": round(p25, 1),
        "p50": round(percentile(50), 1),
        "p75": round(p75, 1),
        "p90": round(percentile(90), 1),
        "p99": round(percentile(99), 1),
        "iqr": round(iqr, 1),
        "ci95_lo": round(mean - ci_half, 1),
        "ci95_hi": round(mean + ci_half, 1),
        "cv": round(cv, 3),
        "outliers": outlier_count,
    }


def fmt_stats(s):
    """Format stats dict for display.

    Reports median with IQR (robust), plus mean with 95% CI (parametric).
    These are separate: the CI describes uncertainty about the mean, not the median.
    """
    if s["n"] == 0:
        return "no data"
    return (f'{s["median"]:.1f}ms (n={s["n"]}, '
            f'IQR=[{s["p25"]:.1f},{s["p75"]:.1f}], '
            f'mean={s["mean"]:.1f}, '
            f'95%CI=[{s["ci95_lo"]:.1f},{s["ci95_hi"]:.1f}])')


def sample_adequacy(n, context=""):
    """Return warning string if n is too small for the statistics being reported.

    Rules of thumb (conservative):
      - p99 needs n >= 100 (below that, p99 IS the single largest value)
      - p90 needs n >= 30
      - Median/CI need n >= 5
      - Any comparison needs n >= 10 per group for rank-sum test power
    """
    warnings = []
    prefix = f"{context}: " if context else ""
    if n < 5:
        warnings.append(f"{prefix}n={n} — too few samples for any reliable statistic")
    elif n < 10:
        warnings.append(f"{prefix}n={n} — median is approximate, CI is wide")
    if 10 <= n < 30:
        warnings.append(f"{prefix}n={n} — p90 is interpolated from ~{max(1,int(n*0.1))} "
                        f"extreme values")
    if n < 100:
        p99_count = max(1, int(n * 0.01))
        if n >= 10:
            warnings.append(f"{prefix}n={n} — p99 estimate rests on "
                            f"{p99_count} observation{'s' if p99_count > 1 else ''}")
    return warnings


def mann_whitney_u(x, y):
    """Mann-Whitney U test (two-sided, normal approximation).

    Non-parametric test for whether two independent samples come from the
    same distribution. No normality assumption. Scipy-free.

    Returns (U, z, p_approx, effect_size_r) where:
      - U: the U statistic
      - z: z-score (normal approximation, valid for n >= 10)
      - p_approx: two-sided p-value from normal approximation
      - effect_size_r: r = z / sqrt(n1+n2), rank-biserial effect size
        (0 = no effect, 0.1 = small, 0.3 = medium, 0.5 = large)

    Returns None if either sample has < 5 observations.
    """
    n1, n2 = len(x), len(y)
    if n1 < 5 or n2 < 5:
        return None

    # Rank all values together
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])

    # Assign ranks with tie correction
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1  # 1-based
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of ranks for group 0 (x)
    r1 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 0)

    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    # Normal approximation (with tie correction)
    mean_u = n1 * n2 / 2
    # Tie correction factor
    n = n1 + n2
    # Count tie groups
    tie_sum = 0
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        t = j - i
        if t > 1:
            tie_sum += t ** 3 - t
        i = j

    std_u = math.sqrt(n1 * n2 / 12 * (n + 1 - tie_sum / (n * (n - 1))))
    if std_u == 0:
        return None

    z = (u1 - mean_u) / std_u

    # Two-sided p-value from normal approximation (Abramowitz & Stegun 26.2.17)
    az = abs(z)
    # Rational approximation of erfc
    t = 1.0 / (1.0 + 0.2316419 * az)
    d = 0.3989422804014327  # 1/sqrt(2π)
    p_one = d * math.exp(-az * az / 2.0) * t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 +
        t * (-1.821255978 + t * 1.330274429))))
    p_approx = 2.0 * p_one  # two-sided

    effect_r = abs(z) / math.sqrt(n)

    return u, z, min(p_approx, 1.0), round(effect_r, 3)


def compare_error_rates(rows, group_field="config"):
    """Compare error rates across groups. Returns (per_group_rates, warning_str).

    If error rates differ by >5 percentage points between groups, the latency
    comparison is biased — the group with more errors has filtered out its
    worst-performing requests (survivorship bias).
    """
    groups = defaultdict(lambda: {"ok": 0, "total": 0})
    for r in rows:
        g = r.get(group_field, "unknown")
        groups[g]["total"] += 1
        if get_status(r) == 200:
            groups[g]["ok"] += 1

    rates = {}
    for g, counts in sorted(groups.items()):
        rate = 1.0 - counts["ok"] / counts["total"] if counts["total"] > 0 else 0
        rates[g] = rate

    warning = None
    rate_vals = list(rates.values())
    if len(rate_vals) >= 2:
        spread = max(rate_vals) - min(rate_vals)
        if spread > 0.05:
            worst = max(rates, key=rates.get)
            best = min(rates, key=rates.get)
            warning = (f"WARNING: Error rates differ — {worst}: {rates[worst]:.0%} vs "
                       f"{best}: {rates[best]:.0%}. Latency comparison is biased "
                       f"(survivorship: {worst} has filtered out its slowest requests).")
    return rates, warning


# ── Data quality ─────────────────────────────────────────────────────────────

def check_data_quality(rows, max_tokens_field="max_tokens",
                       ct_field="completion_tokens"):
    """Check for truncated responses and errors."""
    issues = []
    total = len(rows)
    if total == 0:
        return issues

    errors = [r for r in rows if get_status(r) not in (0, 200)]
    if errors:
        issues.append(f"  {len(errors)}/{total} non-200 responses ({100*len(errors)/total:.1f}%)")

    if ct_field in rows[0]:
        truncated = [r for r in rows if safe_int(r.get(ct_field, 0)) < 10
                     and get_status(r) == 200]
        if truncated:
            issues.append(f"  {len(truncated)}/{total} truncated responses "
                          f"(<10 completion tokens, {100*len(truncated)/total:.1f}%)")

    error_field = [r for r in rows if r.get("error", "")]
    if error_field:
        issues.append(f"  {len(error_field)}/{total} requests with error field set")

    return issues


def validate_completeness(rows, group_fields, expected_per_group=None):
    """Check that each (config, prompt_length, etc.) group has consistent row counts.

    Args:
        rows: List of dicts from CSV.
        group_fields: List of field names to group by (e.g., ["config", "prompt_tokens_target"]).
        expected_per_group: Expected rows per group (optional). If None, uses the mode.

    Returns:
        List of warning strings (empty if data is complete).
    """
    groups = defaultdict(int)
    for r in rows:
        key = tuple(r.get(f, "") for f in group_fields)
        groups[key] += 1

    if not groups:
        return ["No data rows found"]

    counts = list(groups.values())
    if expected_per_group is None:
        # Mode of group counts. Ties broken by largest count (most complete).
        expected_per_group = max(set(counts), key=lambda c: (counts.count(c), c))

    warnings = []
    for key, count in sorted(groups.items()):
        if count != expected_per_group:
            label = "/".join(f"{f}={v}" for f, v in zip(group_fields, key))
            warnings.append(f"  {label}: {count} rows (expected {expected_per_group})")

    error_rate = sum(1 for r in rows if get_status(r) != 200) / len(rows) if rows else 0
    if error_rate > 0.1:
        warnings.append(f"  High error rate: {error_rate:.0%} of requests returned non-200")

    return warnings


# ── Experiment analyses ──────────────────────────────────────────────────────

def analyze_exp1(data_dir):
    """Experiment 1: Single-Request Latency."""
    rows = load_csv(os.path.join(data_dir, "exp1-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "prompt_tokens_target"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")

    # Per-config error rates (survivorship bias check)
    err_rates, err_warning = compare_error_rates(rows)
    for cfg, rate in err_rates.items():
        print(f"    {cfg}: {rate:.1%} error rate")
    if err_warning:
        print(f"    {err_warning}")
    print()

    # Group by (config, prompt_tokens_target)
    print(f"  {'Prompt':>6} | {'Config':>10} | {'Median TTFT':>12} | "
          f"{'Mean':>8} | {'CV':>5} | {'IQR':>15} | {'95% CI':>17} | {'n':>3} | outliers")
    print(f"  {'':->6}-+-{'':->10}-+-{'':->12}-+-"
          f"{'':->8}-+-{'':->5}-+-{'':->15}-+-{'':->17}-+-{'':->3}-+-{'':->8}")

    prompt_targets = sorted(set(r["prompt_tokens_target"] for r in rows), key=lambda x: safe_int(x))
    configs = sorted(set(r["config"] for r in rows))

    for pt in prompt_targets:
        for cfg in configs:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg and r["prompt_tokens_target"] == pt
                 and get_status(r) == 200]
            s = stats(v)
            if s["n"] == 0:
                continue
            outlier_str = f'{s["outliers"]}' if s["outliers"] > 0 else ""
            print(f'  {pt:>6} | {cfg:>10} | {s["median"]:>10.1f}ms | '
                  f'{s["mean"]:>6.1f} | {s["cv"]:.2f} | '
                  f'[{s["p25"]:>5.1f},{s["p75"]:>5.1f}] | '
                  f'[{s["ci95_lo"]:>6.1f},{s["ci95_hi"]:>6.1f}] | {s["n"]:>3} | '
                  f'{outlier_str:>8}')

    # Overhead analysis: compare each non-baseline config to baseline
    bl_name = "BASELINE"
    disagg_configs = [c for c in configs if c != bl_name]
    if not disagg_configs:
        return

    print()
    print(f"  Overhead (disagg median - {bl_name} median):")
    overheads = {c: [] for c in disagg_configs}
    for pt in prompt_targets:
        bl = stats([safe_float(r["ttft_ms"]) for r in rows
                    if r["config"] == bl_name and r["prompt_tokens_target"] == pt
                    and get_status(r) == 200])
        if bl["n"] == 0:
            continue

        parts = []
        skip = False
        for dc in disagg_configs:
            ds = stats([safe_float(r["ttft_ms"]) for r in rows
                        if r["config"] == dc and r["prompt_tokens_target"] == pt
                        and get_status(r) == 200])
            if ds["n"] == 0:
                skip = True
                break
            oh = ds["median"] - bl["median"]
            overheads[dc].append(oh)
            parts.append(f"{dc}=+{oh:.1f}ms")

        if skip:
            print(f"    {pt:>5} tok: insufficient data")
        else:
            print(f"    {pt:>5} tok: {'  '.join(parts)}")

    any_data = False
    for dc in disagg_configs:
        s = stats(overheads[dc])
        if s["n"] == 0:
            continue
        any_data = True
        print(f"    {dc} overhead: {s['mean']:.1f} +/- {s['std']:.1f}ms "
              f"(range {s['min']:.1f}-{s['max']:.1f})")

    if not any_data:
        print("    No overhead data to summarize")
        return

    # Statistical significance of overhead (Mann-Whitney U on raw TTFT values)
    print()
    print("  Statistical significance of overhead (Mann-Whitney U, α=0.05):")
    for pt in prompt_targets:
        bl_v = [safe_float(r["ttft_ms"]) for r in rows
                if r["config"] == bl_name and r["prompt_tokens_target"] == pt
                and get_status(r) == 200]
        for dc in disagg_configs:
            dc_v = [safe_float(r["ttft_ms"]) for r in rows
                    if r["config"] == dc and r["prompt_tokens_target"] == pt
                    and get_status(r) == 200]
            result = mann_whitney_u(bl_v, dc_v)
            if result:
                _u, _z, p, r_eff = result
                sig = "significant" if p < 0.05 else "NOT significant"
                size = ("large" if r_eff >= 0.5 else "medium" if r_eff >= 0.3
                        else "small" if r_eff >= 0.1 else "negligible")
                print(f"    {pt:>5} tok {dc}: p={p:.4f} ({sig}), "
                      f"effect size r={r_eff:.3f} ({size})")
            else:
                print(f"    {pt:>5} tok {dc}: insufficient data for test")

    # Sample adequacy warnings
    all_warnings = []
    for pt in prompt_targets:
        for cfg in configs:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg and r["prompt_tokens_target"] == pt
                 and get_status(r) == 200]
            all_warnings.extend(sample_adequacy(len(v), f"{cfg}/{pt}tok"))
    if all_warnings:
        print()
        print("  Sample adequacy:")
        for w in all_warnings:
            print(f"    {w}")

    # Flag variance asymmetry between disagg configs
    oh_stats = {dc: stats(overheads[dc]) for dc in disagg_configs if overheads[dc]}
    std_vals = [(dc, oh_stats[dc]["std"]) for dc in oh_stats if oh_stats[dc]["std"] > 0]
    if len(std_vals) >= 2:
        min_dc, min_std = min(std_vals, key=lambda x: x[1])
        max_dc, max_std = max(std_vals, key=lambda x: x[1])
        if max_std > min_std * 1.5:
            print(f"    WARNING: {max_dc} overhead variance ({max_std:.1f}ms) is "
                  f"{max_std/min_std:.1f}x {min_dc} ({min_std:.1f}ms). "
                  f"Investigate node asymmetry.")


def analyze_exp1b(data_dir):
    """Experiment 1b: Latency Decomposition."""
    rows = load_csv(os.path.join(data_dir, "exp1b-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")
    print()

    # Filter: only full responses (completion_tokens >= 10)
    configs = {}
    required_configs = ["A-prefill-direct", "B-decode-direct",
                        "C-sidecar-only", "D-disaggregated"]
    config_names = list(required_configs)
    if any(r["config"] == "E-EPP-gateway" for r in rows):
        config_names.append("E-EPP-gateway")
    for cfg in config_names:
        all_v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg and get_status(r) == 200]
        full_v = [safe_float(r["ttft_ms"]) for r in rows
                  if r["config"] == cfg and get_status(r) == 200
                  and safe_int(r.get("completion_tokens", 0)) >= 10]
        trimmed = len(all_v) - len(full_v)

        s_all = stats(all_v)
        s_full = stats(full_v)
        configs[cfg] = s_full

        trim_note = f" ({trimmed} truncated excluded)" if trimmed else ""
        print(f"  {cfg}:{trim_note}")
        print(f"    All data: {fmt_stats(s_all)}")
        if trimmed:
            print(f"    Trimmed:  {fmt_stats(s_full)}")
        print()

    # Decomposition
    if any(configs[c]["n"] == 0 for c in required_configs):
        print("  Decomposition: insufficient data (one or more configs have no valid rows)")
        return

    # ── PRIMARY: Paired-difference decomposition ───────────────────────────
    # This is the statistically correct method: compute the difference WITHIN
    # each run (A,B,C,D measured back-to-back), then take the mean of those
    # differences. This cancels time-varying noise (GPU thermal drift, OS
    # scheduling) that is shared across configs within the same run.
    #
    # The alternative (median(D) - median(C)) is statistically invalid for
    # independent samples because median(X-Y) ≠ median(X) - median(Y).
    # With interleaved runs, the samples ARE paired, so use them that way.

    run_data = defaultdict(dict)
    for r in rows:
        if get_status(r) == 200 and safe_int(r.get("completion_tokens", 0)) >= 10:
            run_data[r["run"]][r["config"]] = safe_float(r["ttft_ms"])

    paired_runs = [run for run, cfgs in run_data.items()
                   if all(c in cfgs for c in ["A-prefill-direct", "B-decode-direct",
                                              "C-sidecar-only", "D-disaggregated"])]

    if len(paired_runs) >= 5:
        diffs_sidecar = []
        diffs_prefill_rt = []
        diffs_overhead = []
        for run in paired_runs:
            d_ = run_data[run]
            diffs_sidecar.append(d_["C-sidecar-only"] - d_["B-decode-direct"])
            diffs_prefill_rt.append(d_["D-disaggregated"] - d_["C-sidecar-only"])
            diffs_overhead.append(d_["D-disaggregated"] - d_["A-prefill-direct"])

        s_sc = stats(diffs_sidecar)
        s_pr = stats(diffs_prefill_rt)
        s_oh = stats(diffs_overhead)

        print(f"  Paired-difference decomposition (n={len(paired_runs)} matched runs):")
        print(f"    T_sidecar    = mean(C_i - B_i) = {s_sc['mean']:.1f}ms "
              f"(95%CI [{s_sc['ci95_lo']:.1f}, {s_sc['ci95_hi']:.1f}], "
              f"CV={s_sc['cv']:.2f})")
        print(f"    T_prefill_rt = mean(D_i - C_i) = {s_pr['mean']:.1f}ms "
              f"(95%CI [{s_pr['ci95_lo']:.1f}, {s_pr['ci95_hi']:.1f}], "
              f"CV={s_pr['cv']:.2f})")
        print(f"    T_overhead   = mean(D_i - A_i) = {s_oh['mean']:.1f}ms "
              f"(95%CI [{s_oh['ci95_lo']:.1f}, {s_oh['ci95_hi']:.1f}], "
              f"CV={s_oh['cv']:.2f})")
        print()

        # Sum check: T_sidecar + T_prefill_rt should ≈ T_overhead
        sum_parts = s_sc["mean"] + s_pr["mean"]
        residual = s_oh["mean"] - sum_parts
        residual_pct = abs(residual) / s_oh["mean"] * 100 if s_oh["mean"] > 0 else 0
        print(f"    Sum check: {s_sc['mean']:.1f} + {s_pr['mean']:.1f} = {sum_parts:.1f}ms "
              f"vs T_overhead = {s_oh['mean']:.1f}ms "
              f"(residual: {residual:.1f}ms, {residual_pct:.1f}%)")
        if residual_pct > 10:
            print("    WARNING: decomposition does not add up — "
                  "investigate unmeasured component (e.g., decode scheduling)")
        print()

        # Significance: is T_sidecar > 0? Is T_prefill_rt > 0?
        for label, s in [("T_sidecar", s_sc), ("T_prefill_rt", s_pr),
                         ("T_overhead", s_oh)]:
            if s["ci95_lo"] > 0:
                print(f"    {label}: significantly > 0 "
                      f"(CI lower bound {s['ci95_lo']:.1f}ms > 0)")
            elif s["ci95_hi"] < 0:
                print(f"    {label}: significantly < 0 — unexpected!")
            else:
                print(f"    {label}: NOT significantly different from 0 "
                      f"(CI spans zero: [{s['ci95_lo']:.1f}, {s['ci95_hi']:.1f}])")
    else:
        print(f"  Paired-difference decomposition: insufficient matched runs "
              f"({len(paired_runs)} found, need >= 5)")

    if "E-EPP-gateway" in configs and configs["E-EPP-gateway"]["n"] > 0:
        epp = configs["E-EPP-gateway"]
        manual = configs["D-disaggregated"]
        delta = epp["median"] - manual["median"]
        print("  EPP route comparison (end-to-end TTFT; EPP selects the backend):")
        print(f"    Manual D: {manual['median']:.1f}ms (n={manual['n']})")
        print(f"    EPP:      {epp['median']:.1f}ms (n={epp['n']})")
        print(f"    EPP - D:  {delta:+.1f}ms ({delta / manual['median'] * 100:+.1f}%)"
              if manual["median"] > 0 else f"    EPP - D:  {delta:+.1f}ms")
        paired_epp_runs = [run for run, cfgs in run_data.items()
                           if "D-disaggregated" in cfgs and "E-EPP-gateway" in cfgs]
        if len(paired_epp_runs) >= 5:
            epp_diffs = [run_data[run]["E-EPP-gateway"] -
                         run_data[run]["D-disaggregated"]
                         for run in paired_epp_runs]
            s_epp = stats(epp_diffs)
            print(f"    Paired EPP - D: {s_epp['mean']:+.1f}ms "
                  f"(95%CI [{s_epp['ci95_lo']:.1f}, {s_epp['ci95_hi']:.1f}], "
                  f"n={s_epp['n']})")
        print()

    print()

    # ── SECONDARY: Difference of medians (for reference) ───────────────────
    # Less precise but works when paired data is unavailable.
    a = configs["A-prefill-direct"]["median"]
    b = configs["B-decode-direct"]["median"]
    c = configs["C-sidecar-only"]["median"]
    d = configs["D-disaggregated"]["median"]

    print("  Reference: difference of medians (less precise than paired method):")
    print(f"    T_sidecar    = C - B = {c-b:.1f}ms")
    print(f"    T_prefill_rt = D - C = {d-c:.1f}ms")
    print(f"    T_overhead   = D - A = {d-a:.1f}ms")

    # Check A ≈ B (same model, different GPU)
    a_mean = configs["A-prefill-direct"]["mean"]
    b_mean = configs["B-decode-direct"]["mean"]
    a_std = configs["A-prefill-direct"]["std"]
    b_std = configs["B-decode-direct"]["std"]
    print()
    print("    Validation: A ≈ B (same model, different GPU)")
    print(f"      A: {a_mean:.1f} +/- {a_std:.1f}ms")
    print(f"      B: {b_mean:.1f} +/- {b_std:.1f}ms")
    diff_pct = abs(a - b) / min(a, b) * 100 if min(a, b) > 0 else float('inf')
    print(f"      Difference: {abs(a-b):.1f}ms ({diff_pct:.1f}%) — "
          f"{'OK' if diff_pct < 5 else 'WARNING: significant'}")


def analyze_exp2(data_dir):
    """Experiment 2: Throughput Under Load."""
    rows = load_csv(os.path.join(data_dir, "exp2-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows, ct_field="_none_"):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "concurrency"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")

    suspicious = sum(1 for r in rows
                     if safe_float(r["total_ms"]) < 100
                     and get_status(r) == 200)
    if suspicious:
        print(f"    {suspicious}/{len(rows)} requests with <100ms total latency "
              f"(likely truncated responses)")
    print()

    print(f"  {'C':>3} | {'Config':>10} | {'Med Total':>10} | {'P90':>8} | "
          f"{'Mean':>8} | {'Std':>6} | n | {'Latency Ratio':>14}")
    print(f"  {'':->3}-+-{'':->10}-+-{'':->10}-+-{'':->8}-+-"
          f"{'':->8}-+-{'':->6}-+---+-{'':->14}")

    concurrency_levels = sorted(set(r["concurrency"] for r in rows), key=lambda x: safe_int(x))

    for conc in concurrency_levels:
        bl_s = None
        for cfg in ["BASELINE", "DISAGG-1D", "DISAGG-2D"]:
            v = [safe_float(r["total_ms"]) for r in rows
                 if r["config"] == cfg and r["concurrency"] == conc
                 and get_status(r) == 200]
            s = stats(v)
            if s["n"] == 0:
                continue

            if cfg == "BASELINE":
                bl_s = s
                ratio = "1.00x (ref)"
            else:
                ratio = (f'{s["median"]/bl_s["median"]:.2f}x'
                         if bl_s and bl_s["median"] > 0 else "N/A")

            print(f'  {conc:>3} | {cfg:>10} | {s["median"]:>8.0f}ms | '
                  f'{s["p90"]:>6.0f}ms | {s["mean"]:>6.0f}ms | '
                  f'{s["std"]:>5.0f} | {s["n"]:>2} | {ratio:>14}')

    print()
    print("  Note: Latency ratio >1 means disagg is slower than baseline.")
    print("  Throughput numbers require wall-clock timing to be precise.")


def analyze_exp3(data_dir):
    """Experiment 3: Prefill Isolation.

    Design note: with a single prefill pod, all requests (heavy + light) share
    the same prefill queue. TTFT measures prefill queuing, not decode isolation.
    ITL (inter-token latency) is the valid metric for decode isolation — it
    measures whether heavy prefills on the shared GPU degrade decode quality
    for light requests on separate decode GPUs.
    """
    rows = load_csv(os.path.join(data_dir, "exp3-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "weight"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")
    print()

    # ── ITL analysis (primary metric for decode isolation) ────────────────
    has_itl = "itl_ms" in rows[0]
    if has_itl:
        all_itl = [safe_float(r["itl_ms"]) for r in rows
                   if get_status(r) == 200 and safe_float(r["itl_ms"]) > 0]
        if all_itl and max(all_itl) >= 1.0:
            print("  Decode Isolation (ITL — inter-token latency during mixed load):")
            print("  This is the primary isolation metric: does a heavy prefill on the")
            print("  shared GPU degrade token generation on the decode GPUs?")
            print()

            configs = sorted(set(r["config"] for r in rows))
            weights = sorted(set(r["weight"] for r in rows))
            for cfg in configs:
                for weight in weights:
                    v = [safe_float(r["itl_ms"]) for r in rows
                         if r["config"] == cfg and r["weight"] == weight
                         and get_status(r) == 200 and safe_float(r["itl_ms"]) > 0]
                    s = stats(v)
                    if s["n"] > 0:
                        print(f"    {cfg:>10} {weight:>5}: {fmt_stats(s)}")

            # ITL isolation ratio: BL light ITL / DG light ITL
            bl_itl = [safe_float(r["itl_ms"]) for r in rows
                      if r["config"] == "BASELINE" and r["weight"] == "light"
                      and get_status(r) == 200 and safe_float(r["itl_ms"]) > 0]
            dg_itl = [safe_float(r["itl_ms"]) for r in rows
                      if r["config"] == "DISAGG-2D" and r["weight"] == "light"
                      and get_status(r) == 200 and safe_float(r["itl_ms"]) > 0]
            bl_itl_s = stats(bl_itl)
            dg_itl_s = stats(dg_itl)
            if bl_itl_s["n"] > 0 and dg_itl_s["n"] > 0 and dg_itl_s["median"] > 0:
                itl_ratio = bl_itl_s["median"] / dg_itl_s["median"]
                print()
                print(f"  ITL isolation ratio (BL / DG for light requests): {itl_ratio:.3f}")
                print("    >1 = disagg decode is faster (better isolation)")
                print("    ≈1 = no difference (decode not affected either way)")
                print("    <1 = disagg decode is slower")
                if abs(itl_ratio - 1.0) < 0.10:
                    print(f"    Observed: no meaningful difference "
                          f"({abs(itl_ratio - 1.0)*100:.0f}% — within noise)")
                elif itl_ratio > 1.0:
                    print("    Observed: disagg improves decode quality")
                else:
                    print("    Observed: disagg degrades decode quality")
            epp_itl = [safe_float(r["itl_ms"]) for r in rows
                       if r["config"] == "DISAGG-EPP" and r["weight"] == "light"
                       and get_status(r) == 200 and safe_float(r["itl_ms"]) > 0]
            epp_itl_s = stats(epp_itl)
            if bl_itl_s["n"] > 0 and epp_itl_s["n"] > 0:
                print(f"  EPP light-request ITL: baseline p50={bl_itl_s['median']:.2f}ms, "
                      f"EPP p50={epp_itl_s['median']:.2f}ms "
                      f"(n={epp_itl_s['n']}; EPP selects backend dynamically)")
            print()
        elif all_itl:
            print("  ITL: all values <1ms (model generates faster than measurement resolution)")
            print()

    # ── TTFT analysis (secondary — measures prefill queuing, not isolation) ──
    print("  TTFT under mixed load (NOTE: with single-prefill topology, TTFT reflects")
    print("  prefill queue contention, not decode isolation):")
    print()

    ttft_configs = ["BASELINE", "DISAGG-2D"]
    if any(r["config"] == "DISAGG-EPP" for r in rows):
        ttft_configs.append("DISAGG-EPP")
    for cfg in ttft_configs:
        for weight in ["heavy", "light"]:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg and r["weight"] == weight
                 and get_status(r) == 200]
            s = stats(v)
            print(f"    {cfg:>10} {weight:>5}: {fmt_stats(s)}")

    # TTFT ratio with caveat
    bl_light = [safe_float(r["ttft_ms"]) for r in rows
                if r["config"] == "BASELINE" and r["weight"] == "light"
                and get_status(r) == 200]
    dg_light = [safe_float(r["ttft_ms"]) for r in rows
                if r["config"] == "DISAGG-2D" and r["weight"] == "light"
                and get_status(r) == 200]

    bl_s = stats(bl_light)
    dg_s = stats(dg_light)

    if bl_s["n"] > 0 and dg_s["n"] > 0 and dg_s["median"] > 0:
        ratio_median = bl_s["median"] / dg_s["median"]
        print()
        print(f"  TTFT ratio (BL / DG for light requests): {ratio_median:.3f}")
        if ratio_median < 1:
            print(f"    Disagg light TTFT is {1/ratio_median:.1f}x slower — expected with single")
            print("    prefill pod (all requests queue through same sidecar+NIXL pipeline)")
        else:
            print("    Disagg light TTFT is faster — unexpected, investigate")
    epp_light = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == "DISAGG-EPP" and r["weight"] == "light"
                 and get_status(r) == 200]
    epp_light_s = stats(epp_light)
    if bl_s["n"] > 0 and epp_light_s["n"] > 0 and epp_light_s["median"] > 0:
        print(f"  EPP light TTFT vs baseline: "
              f"{bl_s['median'] / epp_light_s['median']:.3f} "
              f"(baseline/EPP; EPP selects backend dynamically)")
        print()


def analyze_exp4(data_dir):
    """Experiment 4: Fault Tolerance."""
    rows = load_csv(os.path.join(data_dir, "exp4-results.csv"))
    if not rows:
        print("  No data found")
        return

    # Group rows by sub-experiment
    subs = defaultdict(list)
    for r in rows:
        subs[r["sub"]].append(r)

    # Display results per sub-experiment with detection gap analysis
    for sub in sorted(subs.keys()):
        sub_rows = subs[sub]
        print(f"  --- {sub} ---")

        kill_epoch = None
        for r in sub_rows:
            status = r.get("status_code", "")
            ttft = safe_float(r.get("ttft_ms", 0))
            total = safe_float(r.get("total_ms", 0))
            epoch = safe_float(r.get("epoch_ms", 0))
            note = r.get("note", "")
            error = r.get("error", "")
            phase = r.get("phase", "")

            # Track kill/partition events for detection gap
            if phase in ("kill", "partition-on", "rollout"):
                kill_epoch = epoch
                print(f"    [{phase}] {note}")
                continue
            if phase in ("partition-off", "netem-on", "netem-off", "skip",
                         "log-event"):
                print(f"    [{phase}] {note}")
                continue

            ok = "200" in str(status)
            symbol = "OK" if ok else "FAIL"
            err_msg = f" [{error}]" if error else ""

            # Detection gap: time from kill/partition to this request's epoch
            gap_str = ""
            if kill_epoch and epoch and phase.startswith("during"):
                gap_ms = epoch - kill_epoch
                gap_str = f" (detection gap: {gap_ms/1000:.1f}s)"

            # ITL and token info when available
            itl_str = ""
            itl_mean = safe_float(r.get("itl_mean_ms", 0))
            token_count = safe_float(r.get("token_count", 0))
            if itl_mean > 0:
                itl_p99 = safe_float(r.get("itl_p99_ms", 0))
                itl_str = f" itl={itl_mean:.0f}ms(p99={itl_p99:.0f}ms) tokens={int(token_count)}"

            # Probe-based detection/recovery timing
            probe_str = ""
            detect = safe_float(r.get("detect_epoch_ms", 0))
            recover = safe_float(r.get("recover_epoch_ms", 0))
            if detect and recover:
                gap = recover - detect
                probe_str = f" (detect->recover: {gap/1000:.1f}s)"
            elif r.get("probes_to_detect"):
                probe_str = f" (probes={r['probes_to_detect']})"
            elif r.get("probes_to_recover"):
                probe_str = f" (probes={r['probes_to_recover']})"

            print(f"    [{phase}] {symbol} "
                  f"ttft={ttft:.0f}ms total={total:.0f}ms — "
                  f"{note}{err_msg}{gap_str}{itl_str}{probe_str}")

        # Sub-experiment summary
        request_rows = [r for r in sub_rows
                        if r.get("phase", "") not in
                        ("kill", "partition-on", "partition-off",
                         "netem-on", "netem-off", "skip")]
        if request_rows:
            ok_count = sum(1 for r in request_rows if r.get("status_code") == "200")
            fail_count = len(request_rows) - ok_count
            print(f"    Summary: {ok_count}/{len(request_rows)} OK, "
                  f"{fail_count} failed")

        # Recovery analysis: compare pre vs post TTFT
        pre_rows = [r for r in request_rows if r.get("phase", "").startswith("pre")]
        post_rows = [r for r in request_rows if r.get("phase", "").startswith("post")]
        if pre_rows and post_rows:
            pre_ttft = safe_float(pre_rows[0].get("ttft_ms", 0))
            post_ttft = safe_float(post_rows[0].get("ttft_ms", 0))
            if pre_ttft > 0:
                ratio = post_ttft / pre_ttft
                label = "cold start" if ratio > 3.0 else "warm" if ratio < 1.5 else "elevated"
                print(f"    Recovery latency: {post_ttft:.0f}ms vs pre-kill {pre_ttft:.0f}ms "
                      f"({ratio:.1f}x — {label})")
        print()

    # Log analysis — scan exp4-logs/ for NIXL/transport events
    log_dir = os.path.join(data_dir, "exp4-logs")
    if os.path.isdir(log_dir):
        _analyze_exp4_logs(log_dir)


# Patterns to search for in exp4 pod logs, grouped by category.
_LOG_PATTERNS = {
    "NIXL connection": [
        "nixl", "kv_transfer", "kv transfer", "pull_async", "push_async",
        "handshake", "register_agent", "deregister",
    ],
    "ZMQ discovery": [
        "zmq", "discovery", "new peer", "peer lost", "endpoint",
        "service_discovery",
    ],
    "Transport errors": [
        "ECONNRESET", "ECONNREFUSED", "ETIMEDOUT", "broken pipe",
        "connection reset", "connection refused", "timed out",
    ],
    "Reconnection": [
        "reconnect", "retry", "backoff", "re-establish",
        "connection restored", "recovered",
    ],
    "Sidecar routing": [
        "prefill.*fail", "prefill.*error", "fallback", "no backend",
        "upstream.*error", "502", "503", "504",
    ],
}


def _analyze_exp4_logs(log_dir):
    """Scan exp4 pod logs for NIXL, ZMQ, and transport events."""
    import glob
    import re

    log_files = sorted(glob.glob(os.path.join(log_dir, "*.log")))
    if not log_files:
        return

    print("  --- Log Analysis ---")

    for log_path in log_files:
        filename = os.path.basename(log_path)
        try:
            with open(log_path) as f:
                lines = f.readlines()
        except OSError:
            continue

        if not lines:
            continue

        hits = defaultdict(list)
        for i, line in enumerate(lines):
            lower = line.lower()
            for category, patterns in _LOG_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, lower):
                        hits[category].append((i + 1, line.rstrip()))
                        break  # one match per category per line

        if hits:
            print(f"    {filename}:")
            for category, matches in sorted(hits.items()):
                print(f"      {category}: {len(matches)} events")
                # Show first and last match for context
                if len(matches) <= 3:
                    for lineno, text in matches:
                        print(f"        L{lineno}: {text[:120]}")
                else:
                    lineno, text = matches[0]
                    print(f"        L{lineno}: {text[:120]}")
                    print(f"        ... ({len(matches) - 2} more)")
                    lineno, text = matches[-1]
                    print(f"        L{lineno}: {text[:120]}")
    print()


def pearson_r(xs, ys):
    """Pearson correlation coefficient (stdlib-only)."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den > 0 else 0.0


def analyze_exp5(data_dir):
    """Experiment 5: Sequence Length Sweep — Transfer Tax Crossover."""
    rows = load_csv(os.path.join(data_dir, "exp5-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "prompt_tokens_target"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")

    # Per-config error rates
    err_rates, err_warning = compare_error_rates(rows)
    for cfg, rate in err_rates.items():
        print(f"    {cfg}: {rate:.1%} error rate")
    if err_warning:
        print(f"    {err_warning}")
    print()

    # Check actual vs target prompt tokens.
    # build_prompt() systematically produces ~1.2x the target token count
    # (tokenizer behavior). Only warn if the ratio varies within a given
    # target length, which would indicate a data quality issue.
    by_target = defaultdict(set)
    for r in rows:
        target = safe_int(r.get("prompt_tokens_target", 0))
        actual = safe_int(r.get("prompt_tokens_actual", 0))
        if actual > 0 and target > 0:
            by_target[target].add(actual)
    variable_targets = {t: actuals for t, actuals in by_target.items()
                        if len(actuals) > 1}
    if variable_targets:
        print("  WARNING: prompt token counts vary within same target length:")
        for target, actuals in sorted(variable_targets.items()):
            print(f"    target={target}, actual values={sorted(actuals)}")
        print()
    elif by_target:
        # Report the systematic ratio as a note, not a warning
        sample = next(iter(by_target.items()))
        ratio = next(iter(sample[1])) / sample[0] if sample[0] > 0 else 0
        if ratio > 1.1:
            print(f"  Note: build_prompt produces ~{ratio:.1f}x target token count "
                  f"(consistent across all lengths)")
            print()

    # Prompt lengths and configs
    prompt_lengths = sorted(set(safe_int(r["prompt_tokens_target"]) for r in rows))
    config_names = ["A-prefill-direct", "B-decode-direct",
                    "C-sidecar-only", "D-disaggregated"]
    if any(r["config"] == "E-EPP-gateway" for r in rows):
        config_names.append("E-EPP-gateway")

    # Per-length decomposition table
    print(f"  {'Length':>8} | {'T_sidecar':>10} | {'T_transfer':>10} | "
          f"{'T_overhead':>10} | {'A(prefill)':>10} | {'B(decode)':>10} | "
          f"{'C(sidecar)':>10} | {'D(disagg)':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-"
          f"{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    # Collect per-length stats for crossover analysis
    length_data = {}  # prompt_length -> {config -> stats_dict}
    transfer_points = []  # (prompt_length, T_transfer_mean, T_prefill_median)

    for pt in prompt_lengths:
        cfg_stats = {}
        for cfg in config_names:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg
                 and safe_int(r["prompt_tokens_target"]) == pt
                 and get_status(r) == 200
                 and safe_int(r.get("completion_tokens", 0)) >= 10]
            cfg_stats[cfg] = stats(v)
        length_data[pt] = cfg_stats

        a = cfg_stats["A-prefill-direct"]
        b = cfg_stats["B-decode-direct"]
        c = cfg_stats["C-sidecar-only"]
        d = cfg_stats["D-disaggregated"]

        if all(s["n"] > 0 for s in [a, b, c, d]):
            t_sc = c["median"] - b["median"]
            t_tr = d["median"] - c["median"]
            t_oh = d["median"] - a["median"]
            print(f"  {pt:>8} | {t_sc:>9.1f}ms | {t_tr:>9.1f}ms | "
                  f"{t_oh:>9.1f}ms | {a['median']:>9.1f}ms | {b['median']:>9.1f}ms | "
                  f"{c['median']:>9.1f}ms | {d['median']:>9.1f}ms")
            transfer_points.append((pt, t_tr, a["median"]))
        else:
            missing = [cfg for cfg in config_names if cfg_stats[cfg]["n"] == 0]
            # Check if ALL configs have 0 successful responses at this length
            all_rows_at_pt = [r for r in rows
                              if safe_int(r["prompt_tokens_target"]) == pt]
            errors_at_pt = [r for r in all_rows_at_pt if get_status(r) != 200]
            if all_rows_at_pt and len(errors_at_pt) == len(all_rows_at_pt):
                # Check for HTTP 400 (context window exceeded)
                status_codes = set(get_status(r) for r in errors_at_pt)
                if 400 in status_codes:
                    print(f"  {pt:>8} | (all requests failed — HTTP 400: "
                          f"prompt likely exceeds model context window)")
                else:
                    print(f"  {pt:>8} | (all requests failed — "
                          f"status codes: {', '.join(str(s) for s in sorted(status_codes))})")
            else:
                print(f"  {pt:>8} | (insufficient data: {', '.join(missing)})")

        epp_stats = cfg_stats.get("E-EPP-gateway", stats([]))
        if epp_stats["n"] > 0:
            epp_delta = epp_stats["median"] - d["median"] if d["n"] else None
            baseline_delta = epp_stats["median"] - a["median"] if a["n"] else None
            comparison = []
            if epp_delta is not None:
                comparison.append(f"vs D {epp_delta:+.1f}ms")
            if baseline_delta is not None:
                comparison.append(f"vs A {baseline_delta:+.1f}ms")
            print(f"    EPP gateway: {epp_stats['median']:.1f}ms (n={epp_stats['n']}; "
                  f"{', '.join(comparison)})")

    print()

    # Monotonicity check: TTFT should increase with prompt length for each config.
    # Non-monotonic behavior suggests measurement noise, caching, or system instability.
    print("  Monotonicity check (TTFT should increase with sequence length):")
    for cfg in config_names:
        medians = [(pt, length_data[pt][cfg]["median"])
                   for pt in prompt_lengths
                   if length_data[pt][cfg]["n"] > 0]
        if len(medians) < 2:
            continue
        violations = []
        for i in range(len(medians) - 1):
            if medians[i + 1][1] < medians[i][1]:
                violations.append((medians[i][0], medians[i][1],
                                   medians[i + 1][0], medians[i + 1][1]))
        if violations:
            print(f"    {cfg}: NON-MONOTONIC at {len(violations)} point(s)")
            for pt_lo, v_lo, pt_hi, v_hi in violations:
                print(f"      {pt_lo}tok={v_lo:.1f}ms > {pt_hi}tok={v_hi:.1f}ms "
                      f"(decrease of {v_lo - v_hi:.1f}ms)")
        else:
            print(f"    {cfg}: OK (monotonically increasing)")
    print()

    # Paired-difference analysis at each length
    print("  Paired-difference T_transfer = mean(D_i - C_i) at each length:")
    paired_transfers = []  # (prompt_length, mean, ci_lo, ci_hi)

    for pt in prompt_lengths:
        run_data = defaultdict(dict)
        for r in rows:
            if (safe_int(r["prompt_tokens_target"]) == pt
                    and get_status(r) == 200
                    and safe_int(r.get("completion_tokens", 0)) >= 10):
                run_data[r["run"]][r["config"]] = safe_float(r["ttft_ms"])

        paired_runs = [run for run, cfgs in run_data.items()
                       if "C-sidecar-only" in cfgs and "D-disaggregated" in cfgs]

        if len(paired_runs) >= 5:
            diffs = [run_data[run]["D-disaggregated"] - run_data[run]["C-sidecar-only"]
                     for run in paired_runs]
            s = stats(diffs)
            paired_transfers.append((pt, s["mean"], s["ci95_lo"], s["ci95_hi"]))
            print(f"    {pt:>6} tokens: {s['mean']:.1f}ms "
                  f"(95%CI [{s['ci95_lo']:.1f}, {s['ci95_hi']:.1f}], "
                  f"n={s['n']})")
        else:
            print(f"    {pt:>6} tokens: insufficient paired runs ({len(paired_runs)})")
    print()

    # Crossover detection
    if len(transfer_points) >= 2:
        print("  Crossover analysis (T_transfer vs T_prefill):")
        crossover_found = False
        for i in range(len(transfer_points) - 1):
            pt_lo, t_tr_lo, t_pf_lo = transfer_points[i]
            pt_hi, t_tr_hi, t_pf_hi = transfer_points[i + 1]
            if t_tr_lo <= t_pf_lo and t_tr_hi > t_pf_hi:
                print(f"    CROSSOVER: T_transfer overtakes T_prefill between "
                      f"{pt_lo} and {pt_hi} tokens")
                print(f"      At {pt_lo}: T_transfer={t_tr_lo:.1f}ms, "
                      f"T_prefill={t_pf_lo:.1f}ms")
                print(f"      At {pt_hi}: T_transfer={t_tr_hi:.1f}ms, "
                      f"T_prefill={t_pf_hi:.1f}ms")
                crossover_found = True
        if not crossover_found:
            if all(t_tr > t_pf for _, t_tr, t_pf in transfer_points):
                print("    T_transfer > T_prefill at ALL measured lengths "
                      "(transfer-dominated)")
            elif all(t_tr <= t_pf for _, t_tr, t_pf in transfer_points):
                print("    T_transfer < T_prefill at ALL measured lengths "
                      "(compute-dominated)")
                print(f"    Crossover may occur beyond {transfer_points[-1][0]} tokens")
            else:
                print("    Non-monotonic behavior — no clean crossover detected")
        print()

    # Linearity check: is T_transfer proportional to sequence length?
    if len(transfer_points) >= 3:
        xs = [float(pt) for pt, _, _ in transfer_points]
        ys = [t_tr for _, t_tr, _ in transfer_points]
        r = pearson_r(xs, ys)
        label = "linear (bandwidth-limited)" if r > 0.95 else "non-linear"
        print(f"  Linearity: Pearson r(seq_len, T_transfer) = {r:.3f} — {label}")
        if r <= 0.95:
            print("    Transfer cost is not purely bandwidth-limited; "
                  "investigate protocol overhead or memory allocation")
        print()

    # ── Scaling model: extrapolate to other deployments ──────────────────
    # Fit linear models: T = intercept + slope * seq_len
    # Transfer: intercept = protocol overhead, slope = bandwidth cost/token
    # Prefill: intercept = launch overhead, slope = compute cost/token
    # The ratio slope_prefill/slope_transfer tells us how model size affects crossover.
    if len(transfer_points) >= 3:
        xs = [float(pt) for pt, _, _ in transfer_points]
        ys_tr = [t_tr for _, t_tr, _ in transfer_points]
        ys_pf = [t_pf for _, _, t_pf in transfer_points]

        # Simple linear regression (stdlib only)
        def linreg(xs, ys):
            n = len(xs)
            mx, my = sum(xs)/n, sum(ys)/n
            ss_xx = sum((x - mx)**2 for x in xs)
            ss_xy = sum((x - mx)*(y - my) for x, y in zip(xs, ys))
            if ss_xx == 0:
                return my, 0
            slope = ss_xy / ss_xx
            intercept = my - slope * mx
            return intercept, slope

        tr_intercept, tr_slope = linreg(xs, ys_tr)
        pf_intercept, pf_slope = linreg(xs, ys_pf)

        print("  Scaling Model (linear fit):")
        print(f"    T_transfer  = {tr_intercept:.1f}ms + {tr_slope:.3f}ms/token")
        print(f"    T_prefill   = {pf_intercept:.1f}ms + {pf_slope:.3f}ms/token")
        print()

        # Protocol overhead is the transfer intercept (cost at seq_len→0)
        print(f"    Protocol overhead (transfer at 0 tokens): {tr_intercept:.1f}ms")
        print(f"    Bandwidth cost: {tr_slope:.3f}ms/token "
              f"= {1/tr_slope:.0f} tokens/ms" if tr_slope > 0 else "")
        print()

        # Predict crossover: T_transfer = T_prefill
        # tr_intercept + tr_slope*L = pf_intercept + pf_slope*L
        # L = (pf_intercept - tr_intercept) / (tr_slope - pf_slope)
        if tr_slope != pf_slope:
            crossover_tokens = (pf_intercept - tr_intercept) / (tr_slope - pf_slope)
            if crossover_tokens > 0:
                print(f"    Predicted crossover: {crossover_tokens:.0f} tokens "
                      f"(T_transfer = T_prefill = "
                      f"{tr_intercept + tr_slope * crossover_tokens:.0f}ms)")
            elif crossover_tokens < 0:
                print("    Transfer always exceeds prefill (crossover at negative tokens)")
            print()

        # Key insight: prefill compute scales with model size, transfer does not.
        # At model_size_ratio × current model, prefill becomes:
        #   T_prefill_new = pf_intercept + (pf_slope * model_size_ratio) * seq_len
        # Transfer stays the same (same KV cache size per token regardless of model depth).
        # Exception: KV cache SIZE per token scales with num_layers * d_model, so
        # transfer also scales — but sub-linearly compared to compute.
        print("  Predictions for larger models:")
        print("    (Prefill compute scales ~linearly with model params.)")
        print("    (Transfer cost scales with KV cache size = num_layers × d_head × 2.)")
        print()

        # Read model info to determine current model size
        import json
        run_info_path = os.path.join(data_dir, "run-info.json")
        current_model = "unknown"
        if os.path.exists(run_info_path):
            try:
                with open(run_info_path) as f:
                    info = json.load(f)
                current_model = info.get("toolkit", {}).get("model", "unknown")
            except (ValueError, KeyError):
                pass

        # Model size scaling table
        # Physics:
        #   Prefill FLOPs ≈ 2 × params × seq_len (ignoring attention O(seq²))
        #   KV cache = 2 × num_layers × num_kv_heads × d_head × seq_len × dtype
        #   Both scale ~linearly with model params (num_layers ∝ params).
        #   GQA/MQA reduces KV heads → KV grows slower than compute.
        #
        # What this model CANNOT tell you:
        #   - Actual MFU on different GPUs (varies 30-60%)
        #   - Attention quadratic at long sequences (>4K tokens)
        #   - GQA factor (Llama 3.1 uses GQA: 8 KV heads vs 32 query heads)
        #   - Memory bandwidth effects (KV transfer is bandwidth-bound)
        print(f"    Current model: {current_model}")
        print("    Both prefill and transfer scale ~linearly with model params.")
        print("    The ratio prefill/transfer determines if disagg helps.")
        print()
        print(f"    {'Model scale':>12} | {'T_prefill @500tok':>18} | "
              f"{'T_transfer @500tok':>18} | {'Ratio pf/tr':>12}")
        print(f"    {'-'*12}-+-{'-'*18}-+-{'-'*18}-+-{'-'*12}")
        for scale_label, param_scale in [
            ("1x (current)", 1.0),
            ("3x (~10B)", 3.0),
            ("8x (~30B)", 8.0),
            ("20x (~70B)", 20.0),
        ]:
            # Both prefill and transfer scale with model size
            t_pf_500 = pf_intercept * param_scale + pf_slope * param_scale * 500
            t_tr_500 = tr_intercept + tr_slope * param_scale * 500
            ratio = t_pf_500 / t_tr_500 if t_tr_500 > 0 else float('inf')
            print(f"    {scale_label:>12} | {t_pf_500:>15.0f}ms | "
                  f"{t_tr_500:>15.0f}ms | {ratio:>10.1f}x")
        print()
        print("    Ratio > 1: prefill dominates → disagg isolates decode (good).")
        print("    Ratio ≈ 1: marginal — overhead may not be worth it.")
        print("    Ratio < 1: transfer dominates → disagg adds pure overhead.")
        print(f"    Protocol overhead ({tr_intercept:.0f}ms) is the floor regardless of model.")
        print("    GQA models (Llama 3.1: 8 KV heads vs 32 Q heads) have ~4x")
        print("    less KV data → shift ratio further toward prefill-dominated.")
        print()

        # Cross-hardware prediction: what changes on faster GPUs/networks?
        # What scales: prefill compute (1/GPU_FLOPS), bandwidth cost (1/network_BW)
        # What doesn't scale: protocol overhead (sidecar routing, NIXL handshake,
        #   HTTP round-trips). Measured as tr_intercept on THIS deployment.
        # CAVEAT: protocol overhead is NOT truly constant — it includes components
        #   that scale with network latency (TCP RTT, TLS handshake). On faster
        #   networks these shrink, but sidecar processing time doesn't. We report
        #   the measured value as a conservative upper bound.
        print("  Cross-hardware predictions (order-of-magnitude):")
        print("    Measured on this deployment. Protocol overhead is an upper bound")
        print("    (includes TCP/TLS components that shrink on faster networks).")
        print()

        protocol_overhead = tr_intercept  # ms, measured on this hardware
        compute_per_token = pf_slope  # ms/token on THIS GPU
        transfer_per_token = tr_slope  # ms/token on THIS network

        hw_table = [
            # (label, compute_speedup, transfer_speedup, protocol_fraction)
            # protocol_fraction: how much of protocol overhead survives on this HW
            ("T4 + TCP (current)", 1.0, 1.0, 1.0),
            ("H100 + TCP", 15.0, 1.0, 1.0),
            ("H100 + RoCE 100G", 15.0, 10.0, 0.7),
            ("H200 + IB 400G", 15.0, 40.0, 0.5),
        ]
        print(f"    {'Hardware':>22} | {'T_prefill':>10} | {'T_transfer':>10} | "
              f"{'Protocol':>10} | {'Disagg cost':>12}")
        print(f"    {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
        for hw_label, compute_speedup, transfer_speedup, proto_frac in hw_table:
            t_pf = pf_intercept / compute_speedup + compute_per_token / compute_speedup * 500
            t_proto = protocol_overhead * proto_frac
            t_bw = transfer_per_token / transfer_speedup * 500
            t_tr = t_proto + t_bw
            cost = t_tr - t_pf
            print(f"    {hw_label:>22} | {t_pf:>8.0f}ms | {t_tr:>8.0f}ms | "
                  f"{t_proto:>8.0f}ms | {cost:>+10.0f}ms")
        print()
        print("    'Disagg cost' = T_transfer - T_prefill at 500 tokens.")
        print("    Negative = disagg slower than monolithic for single requests.")
        print("    But disagg value is ISOLATION, not single-request speed.")
        print("    At high QPS, prefill contention degrades decode ITL —")
        print("    that's where disagg pays off regardless of per-request cost.")
        print()

    # Validation: A ≈ B at each length
    print("  Validation: A ≈ B (same model, different GPU)")
    for pt in prompt_lengths:
        a = length_data[pt]["A-prefill-direct"]
        b = length_data[pt]["B-decode-direct"]
        if a["n"] > 0 and b["n"] > 0:
            diff_pct = abs(a["median"] - b["median"]) / min(a["median"], b["median"]) * 100 \
                if min(a["median"], b["median"]) > 0 else float('inf')
            flag = "" if diff_pct < 10 else " ← WARNING"
            print(f"    {pt:>6}: A={a['median']:.1f}ms, B={b['median']:.1f}ms, "
                  f"diff={diff_pct:.1f}%{flag}")
    print()


def analyze_exp6(data_dir):
    """Experiment 6: Saturation Profiling — Scaling Dividend."""
    rows = load_csv(os.path.join(data_dir, "exp6-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows, ct_field="_none_"):
        print(f"    {issue}")

    # Exp6 has different expected row counts per QPS level (QPS × duration_s),
    # so we validate per-group rather than using the mode-based generic check.
    duration_s = 30  # default
    run_info_path = os.path.join(data_dir, "run-info.json")
    if os.path.exists(run_info_path):
        try:
            import json
            with open(run_info_path) as f:
                info = json.load(f)
            duration_s = info.get("experiments", {}).get("exp6", {}).get("duration_s", 30)
        except (ValueError, KeyError):
            pass

    completeness_warnings = []
    groups = defaultdict(int)
    for r in rows:
        key = (r.get("config", ""), r.get("qps_target", ""))
        groups[key] += 1
    for (cfg, qps_str), count in sorted(groups.items()):
        expected = int(safe_float(qps_str) * duration_s)
        if expected > 0 and count != expected:
            completeness_warnings.append(
                f"  config={cfg}/qps={qps_str}: {count} rows (expected {expected})")
    if completeness_warnings:
        print("  Completeness warnings:")
        for w in completeness_warnings:
            print(f"    {w}")
    print()

    configs = sorted(set(r["config"] for r in rows))
    qps_levels = sorted(set(safe_float(r["qps_target"]) for r in rows))

    # Latency vs QPS table
    print(f"  {'Config':>10} | {'QPS':>5} | {'n':>4} | {'p50':>8} | "
          f"{'p90':>8} | {'p99':>8} | {'err%':>5} | {'delay_p50':>10} | "
          f"{'delay_p99':>10}")
    print(f"  {'-'*10}-+-{'-'*5}-+-{'-'*4}-+-{'-'*8}-+-{'-'*8}-+-"
          f"{'-'*8}-+-{'-'*5}-+-{'-'*10}-+-{'-'*10}")

    # Collect per-(config, qps) stats for saturation analysis
    cfg_qps_stats = {}  # (config, qps) -> stats_dict
    cfg_baselines = {}  # config -> p50 at lowest QPS

    for cfg in configs:
        for qps in qps_levels:
            v = [safe_float(r["ttft_ms"]) for r in rows
                 if r["config"] == cfg
                 and safe_float(r["qps_target"]) == qps
                 and get_status(r) == 200]
            delays = [safe_float(r["depart_delay_ms"]) for r in rows
                      if r["config"] == cfg
                      and safe_float(r["qps_target"]) == qps]
            errors = [r for r in rows
                      if r["config"] == cfg
                      and safe_float(r["qps_target"]) == qps
                      and get_status(r) != 200]

            s = stats(v)
            d = stats(delays) if delays else stats([])
            cfg_qps_stats[(cfg, qps)] = s

            # Track baseline (lowest QPS level)
            if qps == qps_levels[0] and s["n"] > 0:
                cfg_baselines[cfg] = s["p50"]

            total_n = len(v) + len(errors)
            err_pct = len(errors) / total_n * 100 if total_n > 0 else 0

            if s["n"] > 0:
                n_warn = "*" if s["n"] < 100 else " "
                delay_warn = " !" if d["n"] > 0 and d["p50"] > 50 else "  "
                print(f"  {cfg:>10} | {qps:>5.0f} | {s['n']:>3}{n_warn}| "
                      f"{s['p50']:>7.1f}ms | {s['p90']:>7.1f}ms | "
                      f"{s['p99']:>7.1f}ms | {err_pct:>4.0f}% | "
                      f"{d['p50']:>9.1f}ms | {d['p99']:>9.1f}ms{delay_warn}")
            else:
                print(f"  {cfg:>10} | {qps:>5.0f} | {0:>4} | "
                      f"{'no data':>8} | {'':>8} | {'':>8} | "
                      f"{err_pct:>4.0f}% |")

    print()
    print("  * = n<100 (p99 estimate is approximate)  ! = depart delay >50ms (harness saturated)")
    print()

    # Saturation point detection
    # Read SLO_MULT from run-info.json if available, otherwise default 2.0
    slo_mult = 2.0
    run_info_path = os.path.join(data_dir, "run-info.json")
    if os.path.exists(run_info_path):
        try:
            import json
            with open(run_info_path) as f:
                info = json.load(f)
            slo_mult = info.get("experiments", {}).get("exp6", {}).get("slo_mult", 2.0)
        except (ValueError, KeyError):
            pass

    print(f"  Saturation Analysis (SLO: p99 > {slo_mult}x baseline p50):")
    saturation_points = {}

    for cfg in configs:
        baseline_p50 = cfg_baselines.get(cfg)
        if baseline_p50 is None or baseline_p50 <= 0:
            print(f"    {cfg}: no baseline data")
            continue

        slo_threshold = baseline_p50 * slo_mult
        sat_qps = None

        for qps in qps_levels:
            s = cfg_qps_stats.get((cfg, qps))
            if s and s["n"] > 0 and s["p99"] > slo_threshold:
                sat_qps = qps
                break

        saturation_points[cfg] = sat_qps
        if sat_qps is not None:
            s = cfg_qps_stats[(cfg, sat_qps)]
            print(f"    {cfg}: saturates at QPS={sat_qps} "
                  f"(p99={s['p99']:.1f}ms > threshold={slo_threshold:.1f}ms)")
        else:
            print(f"    {cfg}: no saturation observed up to QPS={qps_levels[-1]}")

    print()

    # Scaling dividend
    baseline_sat = saturation_points.get("BASELINE")
    if baseline_sat:
        for cfg in configs:
            if cfg == "BASELINE":
                continue
            disagg_sat = saturation_points.get(cfg)
            if disagg_sat:
                dividend = disagg_sat / baseline_sat
                print(f"  Scaling dividend ({cfg} vs BASELINE): "
                      f"{dividend:.1f}x (saturates at QPS={disagg_sat} vs {baseline_sat})")
            else:
                print(f"  Scaling dividend ({cfg} vs BASELINE): "
                      f">={qps_levels[-1] / baseline_sat:.1f}x "
                      f"(disagg did not saturate)")
        print()
    elif baseline_sat is None and any(saturation_points.get(c) for c in configs):
        print("  Scaling dividend: BASELINE did not saturate, cannot compute ratio")
        print()

    # Depart delay warnings
    delay_issues = []
    for cfg in configs:
        for qps in qps_levels:
            delays = [safe_float(r["depart_delay_ms"]) for r in rows
                      if r["config"] == cfg
                      and safe_float(r["qps_target"]) == qps]
            if delays:
                d = stats(delays)
                if d["p50"] > 50:
                    delay_issues.append((cfg, qps, d["p50"]))
    if delay_issues:
        print("  WARNING: Test harness saturation detected (depart delay > 50ms):")
        for cfg, qps, delay in delay_issues:
            print(f"    {cfg} at QPS={qps}: median depart delay = {delay:.1f}ms")
        print("    Results at these QPS levels reflect closed-loop behavior.")
        print()


def analyze_exp7(data_dir):
    """Experiment 7: Mixed Workload — The Essential Characterization.

    Distills disaggregated inference to three numbers:
        1. TTFT overhead: what does disagg cost per request?
        2. ITL stability: does disagg protect decode quality under load?
        3. Throughput: how many requests can each topology sustain?
    """
    rows = load_csv(os.path.join(data_dir, "exp7-results.csv"))
    if not rows:
        print("  No data found")
        return

    print("  Data quality:")
    for issue in check_data_quality(rows, ct_field="_none_"):
        print(f"    {issue}")
    completeness = validate_completeness(rows, ["config", "workload_class"])
    if completeness:
        print("  Completeness warnings:")
        for w in completeness:
            print(f"    {w}")

    # Per-config error rates (survivorship bias check)
    err_rates, err_warning = compare_error_rates(rows)
    for cfg, rate in err_rates.items():
        print(f"    {cfg}: {rate:.1%} error rate")
    if err_warning:
        print(f"    {err_warning}")
    print()

    configs = sorted(set(r["config"] for r in rows))
    classes = sorted(set(r.get("workload_class", "unknown") for r in rows))

    # Workload mix summary
    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        total = len(cfg_rows)
        for wl in classes:
            wl_rows = [r for r in cfg_rows if r.get("workload_class") == wl]
            ok = sum(1 for r in wl_rows if get_status(r) == 200)
            print(f"  {cfg} / {wl}: {ok}/{len(wl_rows)} OK ({len(wl_rows)/total*100:.0f}% of mix)")
    print()

    # Per-(config, workload_class) TTFT and ITL stats
    print(f"  {'Config':>10} | {'Class':>6} | {'n':>4} | {'TTFT p50':>9} | "
          f"{'TTFT p99':>9} | {'ITL mean':>9} | {'ITL p99':>9}")
    print(f"  {'-'*10}-+-{'-'*6}-+-{'-'*4}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")

    cfg_wl_stats = {}  # (config, workload_class) -> (ttft_stats, itl_stats)

    for cfg in configs:
        for wl in classes:
            ok_rows = [r for r in rows
                       if r["config"] == cfg
                       and r.get("workload_class") == wl
                       and get_status(r) == 200]

            ttfts = [safe_float(r["ttft_ms"]) for r in ok_rows]
            itl_means = [safe_float(r.get("itl_mean_ms", 0)) for r in ok_rows
                         if safe_float(r.get("itl_mean_ms", 0)) > 0]
            itl_p99s = [safe_float(r.get("itl_p99_ms", 0)) for r in ok_rows
                        if safe_float(r.get("itl_p99_ms", 0)) > 0]

            s_ttft = stats(ttfts)
            s_itl_mean = stats(itl_means)
            s_itl_p99 = stats(itl_p99s)
            cfg_wl_stats[(cfg, wl)] = (s_ttft, s_itl_mean, s_itl_p99)

            if s_ttft["n"] > 0:
                itl_m = f"{s_itl_mean['median']:.1f}ms" if s_itl_mean["n"] > 0 else "n/a"
                itl_p = f"{s_itl_p99['median']:.1f}ms" if s_itl_p99["n"] > 0 else "n/a"
                print(f"  {cfg:>10} | {wl:>6} | {s_ttft['n']:>4} | "
                      f"{s_ttft['p50']:>8.1f}ms | {s_ttft['p99']:>8.1f}ms | "
                      f"{itl_m:>9} | {itl_p:>9}")
    print()

    # ── The Three Essential Numbers ──────────────────────────────────────

    print("  ═══════════════════════════════════════════════════")
    print("  THE VERDICT")
    print("  ═══════════════════════════════════════════════════")
    print()

    baseline = "BASELINE"
    disagg = [c for c in configs if c != baseline]
    if not disagg:
        print("  Cannot compute: need both BASELINE and DISAGG configs")
        return
    disagg_cfg = disagg[0]  # typically DISAGG-2D

    # 1. TTFT Overhead
    bl_ttft = cfg_wl_stats.get((baseline, "short"), (stats([]),))[0]
    dg_ttft = cfg_wl_stats.get((disagg_cfg, "short"), (stats([]),))[0]
    overhead_pct = 0
    if bl_ttft["n"] > 0 and dg_ttft["n"] > 0:
        overhead = dg_ttft["median"] - bl_ttft["median"]
        overhead_pct = overhead / bl_ttft["median"] * 100 if bl_ttft["median"] > 0 else 0
        print("  1. TTFT OVERHEAD (short requests):")
        print(f"     Baseline:      {bl_ttft['median']:.1f}ms (p50, "
              f"CV={bl_ttft['cv']:.2f}, n={bl_ttft['n']})")
        print(f"     Disaggregated: {dg_ttft['median']:.1f}ms (p50, "
              f"CV={dg_ttft['cv']:.2f}, n={dg_ttft['n']})")
        print(f"     Overhead:      {overhead:+.1f}ms ({overhead_pct:+.0f}%)")

        # Statistical test: is the overhead real?
        bl_raw = [safe_float(r["ttft_ms"]) for r in rows
                  if r["config"] == baseline and r.get("workload_class") == "short"
                  and get_status(r) == 200]
        dg_raw = [safe_float(r["ttft_ms"]) for r in rows
                  if r["config"] == disagg_cfg and r.get("workload_class") == "short"
                  and get_status(r) == 200]
        mw = mann_whitney_u(bl_raw, dg_raw)
        if mw:
            _u, _z, p, r_eff = mw
            if p < 0.05:
                print(f"     Statistical test: p={p:.4f} — overhead IS significant "
                      f"(effect r={r_eff:.3f})")
            else:
                print(f"     Statistical test: p={p:.4f} — overhead is NOT significant "
                      f"(cannot distinguish from noise)")

        if overhead < 0:
            print("     → Disagg is FASTER (likely noise or routing benefit)")
        elif overhead_pct < 20:
            print("     → Acceptable overhead")
        else:
            print("     → Significant overhead — investigate sidecar/transfer cost")
    else:
        print("  1. TTFT OVERHEAD: insufficient data")
    print()

    # 2. ITL Stability (the core value proposition)
    bl_itl_short = cfg_wl_stats.get((baseline, "short"), (stats([]), stats([]), stats([])))[1]
    dg_itl_short = cfg_wl_stats.get((disagg_cfg, "short"), (stats([]), stats([]), stats([])))[1]
    bl_itl_p99_short = cfg_wl_stats.get((baseline, "short"), (stats([]), stats([]), stats([])))[2]
    dg_itl_p99_short = cfg_wl_stats.get((disagg_cfg, "short"), (stats([]), stats([]), stats([])))[2]

    if bl_itl_short["n"] > 0 and dg_itl_short["n"] > 0:
        print("  2. ITL STABILITY (short requests — decode quality under mixed load):")
        print(f"     Baseline      ITL: mean={bl_itl_short['median']:.1f}ms, "
              f"p99={bl_itl_p99_short['median']:.1f}ms (n={bl_itl_short['n']})")
        print(f"     Disaggregated ITL: mean={dg_itl_short['median']:.1f}ms, "
              f"p99={dg_itl_p99_short['median']:.1f}ms (n={dg_itl_short['n']})")

        # Mann-Whitney U on raw ITL values
        bl_itl_raw = [safe_float(r.get("itl_mean_ms", 0)) for r in rows
                      if r["config"] == baseline and r.get("workload_class") == "short"
                      and get_status(r) == 200 and safe_float(r.get("itl_mean_ms", 0)) > 0]
        dg_itl_raw = [safe_float(r.get("itl_mean_ms", 0)) for r in rows
                      if r["config"] == disagg_cfg and r.get("workload_class") == "short"
                      and get_status(r) == 200 and safe_float(r.get("itl_mean_ms", 0)) > 0]
        mw_itl = mann_whitney_u(bl_itl_raw, dg_itl_raw)

        if bl_itl_p99_short["median"] > 0 and dg_itl_p99_short["median"] > 0:
            ratio = dg_itl_p99_short["median"] / bl_itl_p99_short["median"]
            print(f"     ITL p99 ratio: {ratio:.2f}x")

            if mw_itl:
                _u, _z, p, r_eff = mw_itl
                if p < 0.05:
                    direction = "lower" if _z < 0 else "higher"
                    print(f"     Statistical test: p={p:.4f} — ITL difference IS significant "
                          f"(disagg {direction}, effect r={r_eff:.3f})")
                else:
                    print(f"     Statistical test: p={p:.4f} — ITL difference is NOT significant")

            if ratio < 0.7:
                print("     → Disagg SIGNIFICANTLY improves decode quality")
            elif ratio < 1.0:
                print("     → Disagg improves decode quality")
            elif ratio < 1.3:
                print("     → Similar decode quality (disagg not helping here)")
            else:
                print("     → Disagg DEGRADES decode quality — investigate")
    else:
        print("  2. ITL STABILITY: insufficient data (need streaming responses with >=2 tokens)")
    print()

    # 3. Goodput (successful requests / offered duration)
    # Use duration_s from run-info.json for a stable denominator
    exp7_duration = None
    run_info_path = os.path.join(data_dir, "run-info.json")
    if os.path.exists(run_info_path):
        try:
            import json
            with open(run_info_path) as f:
                info = json.load(f)
            exp7_duration = info.get("experiments", {}).get("exp7", {}).get("duration_s")
        except (ValueError, KeyError):
            pass

    bl_qps = 0.0
    dg_qps = 0.0
    goodput_by_config = {}
    for cfg in configs:
        cfg_rows = [r for r in rows if r["config"] == cfg]
        ok = sum(1 for r in cfg_rows if get_status(r) == 200)

        # Use configured duration; fall back to schedule span
        if exp7_duration:
            duration = float(exp7_duration)
        else:
            departures = sorted(safe_float(r.get("scheduled_at_s", 0)) for r in cfg_rows)
            duration = departures[-1] - departures[0] if len(departures) >= 2 else 0

        goodput = ok / duration if duration > 0 else 0
        goodput_by_config[cfg] = goodput

        if cfg == baseline:
            bl_qps = goodput
        else:
            dg_qps = goodput

    if bl_qps > 0 and dg_qps > 0:
        print("  3. GOODPUT (successful req/s at offered load):")
        print(f"     Baseline:      {bl_qps:.1f} req/s")
        print(f"     Disaggregated: {dg_qps:.1f} req/s")
        ratio = dg_qps / bl_qps
        print(f"     Ratio: {ratio:.2f}x")
        if ratio > 1.2:
            print(f"     → Disagg handles {ratio:.1f}x more load")
        elif ratio > 0.9:
            print("     → Similar goodput (neither wins at this QPS)")
        else:
            print("     → Disagg goodput is LOWER — overhead exceeds benefit at this QPS")
    else:
        print("  3. GOODPUT: insufficient data")
    print()

    epp_cfg = "DISAGG-EPP"
    if epp_cfg in configs:
        epp_ttft = cfg_wl_stats.get((epp_cfg, "short"), (stats([]),))[0]
        epp_itl = cfg_wl_stats.get((epp_cfg, "short"),
                                   (stats([]), stats([]), stats([])))[2]
        epp_qps = goodput_by_config.get(epp_cfg, 0.0)
        print("  EPP Gateway comparison (supplemental; backend selection is dynamic):")
        if bl_ttft["n"] > 0 and epp_ttft["n"] > 0:
            print(f"    Short TTFT p50: baseline={bl_ttft['median']:.1f}ms, "
                  f"EPP={epp_ttft['median']:.1f}ms "
                  f"(EPP - baseline {epp_ttft['median'] - bl_ttft['median']:+.1f}ms)")
        else:
            print("    Short TTFT p50: insufficient data")
        if epp_itl["n"] > 0:
            print(f"    Short-request ITL p99: {epp_itl['median']:.2f}ms "
                  f"(n={epp_itl['n']})")
        else:
            print("    Short-request ITL p99: unavailable")
        print(f"    Goodput: baseline={bl_qps:.2f} req/s, EPP={epp_qps:.2f} req/s")
        print()

    # Overall verdict
    print("  ─────────────────────────────────────────────────")

    # GPU cost check: disagg uses more GPUs than monolithic.
    # Read topology from run-info.json if available.
    run_info_path = os.path.join(data_dir, "run-info.json")
    bl_gpus = 1  # monolithic baseline = 1 GPU
    dg_gpus = 3  # default disagg topology = 1P + 2D
    if os.path.exists(run_info_path):
        try:
            import json
            with open(run_info_path) as f:
                info = json.load(f)
            # Try to count decode URLs to infer GPU count
            d1 = info.get("toolkit", {}).get("disagg_d1_url", "")
            d2 = info.get("toolkit", {}).get("disagg_d2_url", "")
            if d1 and d2 and d1 != d2:
                dg_gpus = 3  # 1 prefill + 2 decode
            elif d1:
                dg_gpus = 2  # 1 prefill + 1 decode
        except (ValueError, KeyError):
            pass
    gpu_ratio = dg_gpus / bl_gpus

    # Only give a verdict if we have enough data
    if bl_ttft["n"] > 0 and dg_ttft["n"] > 0:
        print()
        print("  SCORECARD:")

        # Check 1: Overhead acceptable?
        overhead_ok = overhead_pct < 20
        print(f"    Overhead < 20%:     {'PASS' if overhead_ok else 'FAIL'} "
              f"({overhead_pct:+.0f}%)")

        # Check 2: ITL improves?
        itl_ok = False
        if (bl_itl_p99_short["n"] > 0 and dg_itl_p99_short["n"] > 0
                and bl_itl_p99_short["median"] > 0):
            itl_ratio = dg_itl_p99_short["median"] / bl_itl_p99_short["median"]
            itl_ok = itl_ratio < 0.85
            print(f"    ITL p99 improved:   {'PASS' if itl_ok else 'FAIL'} "
                  f"(ratio={itl_ratio:.2f}x, need <0.85)")
        else:
            print("    ITL p99 improved:   N/A (insufficient data)")

        # Check 3: Goodput improves?
        goodput_ok = dg_qps > bl_qps * 1.1
        if bl_qps > 0:
            gp_ratio = dg_qps / bl_qps
            print(f"    Goodput improved:   {'PASS' if goodput_ok else 'FAIL'} "
                  f"(ratio={gp_ratio:.2f}x, need >1.10)")
        else:
            print("    Goodput improved:   N/A (insufficient data)")

        # Check 4: Cost efficiency (NEW — Fermi check)
        # Disagg uses more GPUs. Per-GPU goodput must justify the cost.
        per_gpu_bl = bl_qps / bl_gpus if bl_gpus > 0 else 0
        per_gpu_dg = dg_qps / dg_gpus if dg_gpus > 0 else 0
        cost_ok = per_gpu_dg >= per_gpu_bl * 0.8  # within 80% of baseline efficiency
        if per_gpu_bl > 0:
            efficiency = per_gpu_dg / per_gpu_bl
            print(f"    GPU efficiency:     {'PASS' if cost_ok else 'FAIL'} "
                  f"({per_gpu_dg:.2f} vs {per_gpu_bl:.2f} req/s/GPU, "
                  f"ratio={efficiency:.2f}x)")
        print()

        # Sample adequacy
        min_n = min(bl_ttft["n"], dg_ttft["n"])
        if min_n < 30:
            print(f"  CAUTION: n={min_n} — verdict has limited statistical power.")
            print("    Increase DURATION_S or QPS for more confident results.")
            print()

        # Error rate asymmetry check
        bl_err = err_rates.get(baseline, 0)
        dg_err = err_rates.get(disagg_cfg, 0)
        if abs(bl_err - dg_err) > 0.05:
            print(f"  CAUTION: Unequal error rates ({baseline}={bl_err:.0%}, "
                  f"{disagg_cfg}={dg_err:.0%}). Latency comparison may be biased.")
            print()

        wins = sum([overhead_ok, itl_ok, goodput_ok])

        print(f"  COST CONTEXT: {disagg_cfg} uses {dg_gpus} GPUs vs "
              f"{baseline} {bl_gpus} GPU ({gpu_ratio:.0f}x hardware cost).")

        if wins >= 2 and cost_ok:
            print("  VERDICT: Disaggregation is BENEFICIAL for this workload/QPS.")
        elif wins >= 2 and not cost_ok:
            print("  VERDICT: Disaggregation IMPROVES latency but at POOR GPU efficiency.")
            print("           Consider whether the latency benefit justifies "
                  f"{gpu_ratio:.0f}x hardware cost.")
        elif wins == 1:
            print("  VERDICT: Disaggregation shows MARGINAL benefit. "
                  "Consider higher QPS or longer prompts.")
        else:
            print("  VERDICT: Disaggregation is NOT justified at this operating point.")
            print("           Overhead exceeds benefit.")
        print("  ─────────────────────────────────────────────────")
    print()

    # GPU utilization (if available)
    gpu_rows = load_csv(os.path.join(data_dir, "exp7-gpu.csv"))
    if gpu_rows:
        print("  GPU Utilization Summary:")
        for cfg in configs:
            cfg_gpu = [r for r in gpu_rows if r.get("config") == cfg]
            gpus = sorted(set(r.get("gpu_index", "") for r in cfg_gpu))
            for gpu in gpus:
                utils = [safe_float(r["gpu_util_pct"]) for r in cfg_gpu
                         if r.get("gpu_index") == gpu]
                mems = [safe_float(r["mem_util_pct"]) for r in cfg_gpu
                        if r.get("gpu_index") == gpu]
                if utils:
                    s_u = stats(utils)
                    s_m = stats(mems)
                    print(f"    {cfg} GPU {gpu}: "
                          f"compute={s_u['mean']:.0f}% (p90={s_u['p90']:.0f}%), "
                          f"memory={s_m['mean']:.0f}% (p90={s_m['p90']:.0f}%)")
        print()


def analyze_exp8(data_dir):
    """Experiment 8: Prefix Cache Characterization, including optional EPP."""
    rows = load_csv(os.path.join(data_dir, "exp8-results.csv"))
    if not rows:
        print("  No data found")
        return

    errors = sum(1 for r in rows if get_status(r) != 200)
    print(f"  Data quality: {len(rows)} rows, {errors} errors")
    groups = defaultdict(list)
    for row in rows:
        if get_status(row) == 200:
            groups[(row.get("phase", ""), row.get("config", ""),
                    row.get("cache_state", ""))].append(safe_float(row["ttft_ms"]))

    print(f"  {'Phase':>14} | {'Config':>16} | {'Cache state':>20} | "
          f"{'n':>4} | {'TTFT p50':>10} | {'p90':>9}")
    print(f"  {'-'*14}-+-{'-'*16}-+-{'-'*20}-+-{'-'*4}-+-{'-'*10}-+-{'-'*9}")
    for (phase, config, state), values in sorted(groups.items()):
        s = stats(values)
        print(f"  {phase:>14} | {config:>16} | {state:>20} | "
              f"{s['n']:>4} | {s['median']:>8.1f}ms | {s['p90']:>7.1f}ms")

    epp_by_run = defaultdict(dict)
    for row in rows:
        if (row.get("phase") == "epp_routing" and get_status(row) == 200
                and row.get("cache_state") in {"epp_first", "epp_repeat"}):
            epp_by_run[row.get("run", "")][row["cache_state"]] = safe_float(row["ttft_ms"])
    paired = [(values["epp_first"], values["epp_repeat"])
              for values in epp_by_run.values()
              if "epp_first" in values and "epp_repeat" in values]
    if paired:
        first = stats([a for a, _ in paired])
        repeat = stats([b for _, b in paired])
        deltas = stats([b - a for a, b in paired])
        print("\n  EPP cache probes (backend pod is not pinned):")
        print(f"    First p50={first['median']:.1f}ms; repeat p50={repeat['median']:.1f}ms; "
              f"paired repeat-first={deltas['mean']:+.1f}ms "
              f"(95%CI [{deltas['ci95_lo']:.1f}, {deltas['ci95_hi']:.1f}], "
              f"n={deltas['n']})")
        print("    Interpret as endpoint-policy behavior; repeats may reach different backends.")


# ── Experiment 11: Throughput vs Prompt Length ────────────────────────────────

def analyze_exp11(data_dir):
    """Experiment 11: Throughput vs Prompt Length — Crossover Detection."""
    rows = load_csv(os.path.join(data_dir, "exp11-results.csv"))
    if not rows:
        print("  No data found")
        return

    ok = [r for r in rows if get_status(r) == 200]
    errs = len(rows) - len(ok)
    if errs:
        print(f"  Data quality: {errs}/{len(rows)} errors ({errs/len(rows):.0%})")
    else:
        print(f"  Data quality: {len(rows)} rows, 0 errors")
    print()

    configs = sorted(set(r["config"] for r in ok))
    concurrencies = sorted(set(safe_int(r["concurrency"]) for r in ok))
    prompt_lens = sorted(set(safe_int(r["prompt_tokens_target"]) for r in ok))

    by_key = defaultdict(list)
    for r in ok:
        key = (r["config"], safe_int(r["concurrency"]), safe_int(r["prompt_tokens_target"]))
        by_key[key].append(safe_float(r["ttft_ms"]))

    for conc in concurrencies:
        print(f"  TTFT by config (concurrency={conc}):")
        print(f"  {'Config':>12} | {'Prompt':>6} | {'n':>4} | {'p50':>8} | "
              f"{'p90':>8} | {'CV':>5} | {'95%CI':>18}")
        print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*4}-+-{'-'*8}-+-"
              f"{'-'*8}-+-{'-'*5}-+-{'-'*18}")

        for cfg in configs:
            for pt in prompt_lens:
                vals = by_key.get((cfg, conc, pt), [])
                if not vals:
                    continue
                s = stats(vals)
                print(f"  {cfg:>12} | {pt:>6} | {s['n']:>4} | {s['p50']:>7.1f}ms | "
                      f"{s['p90']:>7.1f}ms | {s['cv']:>.3f} | "
                      f"[{s['ci95_lo']:.1f}, {s['ci95_hi']:.1f}]")
        print()

    baseline_cfg = "BASELINE"
    disagg_cfgs = [c for c in configs if c != baseline_cfg]

    print("  Disagg overhead (% slower than BASELINE):")
    print(f"  {'Concurrency':>12} | {'Prompt':>6}", end="")
    for dc in disagg_cfgs:
        print(f" | {dc:>12}", end="")
    print()
    print(f"  {'-'*12}-+-{'-'*6}", end="")
    for _ in disagg_cfgs:
        print(f"-+-{'-'*12}", end="")
    print()

    crossovers = []
    for conc in concurrencies:
        for pt in prompt_lens:
            bl_vals = by_key.get((baseline_cfg, conc, pt), [])
            if not bl_vals:
                continue
            bl_med = stats(bl_vals)["median"]
            row_str = f"  {conc:>12} | {pt:>6}"
            for dc in disagg_cfgs:
                dc_vals = by_key.get((dc, conc, pt), [])
                if not dc_vals or bl_med == 0:
                    row_str += f" | {'':>12}"
                    continue
                dc_med = stats(dc_vals)["median"]
                overhead = (dc_med - bl_med) / bl_med * 100
                marker = " *" if overhead < 0 else ""
                row_str += f" | {overhead:>+10.1f}%{marker}"
                if overhead < 0:
                    crossovers.append((conc, pt, dc, overhead))
            print(row_str)
    print()

    if crossovers:
        print("  CROSSOVER DETECTED (* = disagg faster than mono):")
        for conc, pt, dc, overhead in crossovers:
            print(f"    c={conc}, {pt} tokens, {dc}: {overhead:+.1f}%")
        print()
    else:
        print("  No crossover detected: mono wins at all conditions.")
        print()

    if len(concurrencies) >= 2:
        c_lo, c_hi = concurrencies[0], concurrencies[-1]
        print(f"  Concurrency scaling (TTFT ratio c={c_hi} / c={c_lo}):")
        for cfg in configs:
            for pt in prompt_lens:
                lo_vals = by_key.get((cfg, c_lo, pt), [])
                hi_vals = by_key.get((cfg, c_hi, pt), [])
                if lo_vals and hi_vals:
                    lo_med = stats(lo_vals)["median"]
                    hi_med = stats(hi_vals)["median"]
                    ratio = hi_med / lo_med if lo_med > 0 else 0
                    print(f"    {cfg:>12} @ {pt:>5}tok: {lo_med:.0f}ms → "
                          f"{hi_med:.0f}ms ({ratio:.1f}x)")
        print()


# ── Helpers for streaming experiments (exp12, exp16) ────────────────────────

def _tokens(r):
    """Output tokens of one request: the server's count if reported, else streamed chunks."""
    return safe_int(r.get("server_completion_tokens")) or safe_int(r.get("completion_tokens"))


def _quality(rows, ok):
    errs = len(rows) - len(ok)
    if errs:
        print(f"  Data quality: {errs}/{len(rows)} errors ({errs/len(rows):.0%})")
    else:
        print(f"  Data quality: {len(rows)} rows, 0 errors")
    short = sum(1 for r in ok if _tokens(r) < safe_int(r.get("max_tokens")))
    if short:
        print(f"  WARNING: {short}/{len(ok)} requests stopped before max_tokens "
              f"(run with IGNORE_EOS=1 to control output length)")
    merged = sum(1 for r in ok
                 if safe_int(r.get("server_completion_tokens")) > safe_int(r.get("completion_tokens")))
    if merged:
        print(f"  NOTE: {merged}/{len(ok)} requests had tokens with empty text; their time is "
              f"merged into the next streamed gap")
    print()


def _steady_window(rows, request_key):
    """Keep requests from the steady middle of each steady-mode cell.

    In LOAD_MODE=steady, the first and last `concurrency` requests of a cell
    (by start order, column `run`) ramp the load up and down. Batch-mode rows
    and rows without a load_mode column are kept as they are.
    Returns (kept_rows, dropped_request_count).
    """
    max_run = defaultdict(int)
    for r in rows:
        if r.get("load_mode") == "steady":
            k = request_key(r)
            max_run[k] = max(max_run[k], safe_int(r["run"]))
    kept, dropped = [], set()
    for r in rows:
        if r.get("load_mode") != "steady":
            kept.append(r)
            continue
        c = safe_int(r["concurrency"])
        run = safe_int(r["run"])
        if c < run <= max_run[request_key(r)] - c:
            kept.append(r)
        else:
            dropped.add((request_key(r), run))
    return kept, len(dropped)


# ── Experiment 12: Throughput vs Output Length ────────────────────────────────

def analyze_exp12(data_dir):
    """Experiment 12: TTFT, ITL and total latency vs output length."""
    rows = load_csv(os.path.join(data_dir, "exp12-results.csv"))
    if not rows:
        print("  No data found")
        return

    rows, dropped = _steady_window(
        rows, lambda r: (r["config"], r["concurrency"], r["prompt_tokens_target"], r["max_tokens"]))
    ok = [r for r in rows if get_status(r) == 200]
    _quality(rows, ok)
    if dropped:
        print(f"  Steady load: dropped {dropped} ramp-up/ramp-down requests "
              f"(first and last `concurrency` per cell)")
        print()
    streaming = any(r.get("streaming") == "1" for r in ok)
    if not streaming:
        print("  Non-streaming data: ttft_ms is close to total_ms, so 'Decode p50' is not")
        print("  meaningful. Re-run with STREAMING=1 for true TTFT and ITL.")
        print()

    cell = defaultdict(lambda: {"ttft": [], "itl": [], "total": []})
    for r in ok:
        key = (safe_int(r["concurrency"]), safe_int(r["prompt_tokens_target"]),
               safe_int(r["max_tokens"]), r["config"])
        cell[key]["ttft"].append(safe_float(r["ttft_ms"]))
        cell[key]["total"].append(safe_float(r["total_ms"]))
        itl = safe_float(r.get("itl_mean_ms"))
        if itl > 0:
            cell[key]["itl"].append(itl)

    print(f"  {'c':>3} | {'prompt':>6} | {'out':>5} | {'Config':>10} | {'n':>4} | "
          f"{'TTFT p50':>9} | {'ITL p50':>8} | {'Total p50':>10} | {'Decode p50':>10} | "
          f"{'CV(total)':>9}")
    for key in sorted(cell):
        c, s, out, cfg = key
        d = cell[key]
        s_ttft, s_total = stats(d["ttft"]), stats(d["total"])
        itl = f"{stats(d['itl'])['median']:>6.1f}ms" if d["itl"] else f"{'-':>8}"
        print(f"  {c:>3} | {s:>6} | {out:>5} | {cfg:>10} | {s_total['n']:>4} | "
              f"{s_ttft['median']:>7.1f}ms | {itl} | {s_total['median']:>8.1f}ms | "
              f"{s_total['median'] - s_ttft['median']:>8.1f}ms | {s_total['cv']:>9.3f}")
    print()

    configs = sorted(set(k[3] for k in cell))
    if "BASELINE" in configs and len(configs) > 1:
        print("  Per-request latency, config vs BASELINE. BASELINE is one vLLM instance;")
        print("  DISAGG configs use more GPUs, so this is not throughput per GPU.")
        for c, s, out in sorted(set(k[:3] for k in cell)):
            bl = cell.get((c, s, out, "BASELINE"))
            if not bl or not bl["total"]:
                continue
            bl_total = stats(bl["total"])["median"]
            for cfg in configs:
                d = cell.get((c, s, out, cfg))
                if cfg == "BASELINE" or not d or not d["total"]:
                    continue
                cfg_total = stats(d["total"])["median"]
                overhead = cfg_total - bl_total
                pct = overhead / cfg_total * 100 if cfg_total > 0 else 0
                itl_part = ""
                if bl["itl"] and d["itl"]:
                    itl_part = (f", ITL ratio BASELINE/config "
                                f"{stats(bl['itl'])['median'] / stats(d['itl'])['median']:.2f}")
                print(f"    c={c} prompt={s} out={out} {cfg}: overhead {overhead:.0f}ms "
                      f"({pct:.0f}% of total){itl_part}")
        print()


# ── Experiment 13: Saturation Ceiling ─────────────────────────────────────────

def analyze_exp13(data_dir):
    """Experiment 13: Saturation Ceiling — Max Sustainable Throughput."""
    rows = load_csv(os.path.join(data_dir, "exp13-results.csv"))
    if not rows:
        print("  No data found")
        return

    configs = sorted(set(r["config"] for r in rows))
    concurrencies = sorted(set(safe_int(r["concurrency"]) for r in rows))

    print(f"  {'Config':>12} | {'Conc':>5} | {'Total':>5} | {'OK':>4} | "
          f"{'Err%':>5} | {'TTFT p50':>9} | {'Goodput':>10}")
    print(f"  {'-'*12}-+-{'-'*5}-+-{'-'*5}-+-{'-'*4}-+-"
          f"{'-'*5}-+-{'-'*9}-+-{'-'*10}")

    prev_goodput = {}
    for cfg in configs:
        for conc in concurrencies:
            cfg_rows = [r for r in rows
                        if r["config"] == cfg
                        and safe_int(r["concurrency"]) == conc]
            if not cfg_rows:
                continue

            ok_rows = [r for r in cfg_rows if get_status(r) == 200]
            total_n = len(cfg_rows)
            ok_n = len(ok_rows)
            err_pct = (total_n - ok_n) / total_n * 100 if total_n > 0 else 0

            ttft_vals = [safe_float(r["ttft_ms"]) for r in ok_rows]
            s = stats(ttft_vals)
            total_vals = [safe_float(r["total_ms"]) for r in ok_rows]
            s_total = stats(total_vals)
            goodput = ok_n / (s_total["max"] / 1000) if s_total.get("max", 0) > 0 else 0

            marker = ""
            if err_pct > 10:
                marker = " SATURATED"
            elif cfg in prev_goodput and goodput < prev_goodput[cfg] * 0.9:
                marker = " DECLINING"

            prev_goodput[cfg] = goodput

            if s["n"] > 0:
                print(f"  {cfg:>12} | {conc:>5} | {total_n:>5} | {ok_n:>4} | "
                      f"{err_pct:>4.0f}% | {s['p50']:>7.1f}ms | "
                      f"{goodput:>8.1f} r/s{marker}")
            else:
                print(f"  {cfg:>12} | {conc:>5} | {total_n:>5} | {ok_n:>4} | "
                      f"{err_pct:>4.0f}% | {'N/A':>9} | {'N/A':>10}{marker}")
    print()

    print("  Max sustainable throughput per config:")
    for cfg in configs:
        best_gp = 0
        best_conc = 0
        for conc in concurrencies:
            cfg_rows = [r for r in rows
                        if r["config"] == cfg
                        and safe_int(r["concurrency"]) == conc]
            ok_rows = [r for r in cfg_rows if get_status(r) == 200]
            err_pct = (len(cfg_rows) - len(ok_rows)) / len(cfg_rows) * 100 if cfg_rows else 100
            if err_pct > 10:
                continue
            total_vals = [safe_float(r["total_ms"]) for r in ok_rows]
            s_total = stats(total_vals)
            if s_total.get("max", 0) > 0:
                gp = len(ok_rows) / (s_total["max"] / 1000)
                if gp > best_gp:
                    best_gp = gp
                    best_conc = conc
        if best_gp > 0:
            print(f"    {cfg}: {best_gp:.1f} req/s at c={best_conc}")
        else:
            print(f"    {cfg}: no error-free data")
    print()


# ── Experiment 14: Overhead Decomposition Under Load ─────────────────────────

def analyze_exp14(data_dir):
    """Experiment 14: Overhead Decomposition Under Load — Sidecar vs NIXL."""
    rows = load_csv(os.path.join(data_dir, "exp14-results.csv"))
    if not rows:
        print("  No data found")
        return

    ok = [r for r in rows if get_status(r) == 200]
    errs = len(rows) - len(ok)
    if errs:
        print(f"  Data quality: {errs}/{len(rows)} errors ({errs/len(rows):.0%})")
    else:
        print(f"  Data quality: {len(rows)} rows, 0 errors")

    concurrency = safe_int(ok[0].get("concurrency", "1")) if ok else 1
    print(f"  Concurrency: {concurrency}")
    print()

    configs = sorted(set(r["config"] for r in ok))
    prompt_lens = sorted(set(safe_int(r["prompt_tokens_target"]) for r in ok))

    by_key = defaultdict(list)
    for r in ok:
        key = (r["config"], safe_int(r["prompt_tokens_target"]))
        by_key[key].append(safe_float(r["ttft_ms"]))

    # Per-config TTFT table
    print(f"  {'Config':>22} | {'Prompt':>6} | {'n':>4} | {'p50':>8} | "
          f"{'p90':>8} | {'CV':>5}")
    print(f"  {'-'*22}-+-{'-'*6}-+-{'-'*4}-+-{'-'*8}-+-"
          f"{'-'*8}-+-{'-'*5}")

    for cfg in configs:
        for pt in prompt_lens:
            vals = by_key.get((cfg, pt), [])
            if not vals:
                continue
            s = stats(vals)
            print(f"  {cfg:>22} | {pt:>6} | {s['n']:>4} | {s['p50']:>7.1f}ms | "
                  f"{s['p90']:>7.1f}ms | {s['cv']:>.3f}")
    print()

    # Decomposition table (requires configs A, B, C, D)
    has_a = any(c.startswith("A-") for c in configs)
    has_b = any(c.startswith("B-") for c in configs)
    has_c = any(c.startswith("C-") for c in configs)
    has_d = any(c.startswith("D-") for c in configs)

    cfg_a = next((c for c in configs if c.startswith("A-")), None)
    cfg_b = next((c for c in configs if c.startswith("B-")), None)
    cfg_c = next((c for c in configs if c.startswith("C-")), None)
    cfg_d = next((c for c in configs if c.startswith("D-")), None)

    if has_b and has_c and has_d:
        print("  Overhead Decomposition (median TTFT):")
        print(f"  {'Prompt':>6} | {'Sidecar(C-B)':>13} | {'NIXL(D-C)':>12} | "
              f"{'Total(D-B)':>12}")
        print(f"  {'-'*6}-+-{'-'*13}-+-{'-'*12}-+-{'-'*12}")

        sidecar_vals = []
        nixl_vals = []
        for pt in prompt_lens:
            b_med = stats(by_key.get((cfg_b, pt), []))
            c_med = stats(by_key.get((cfg_c, pt), []))
            d_med = stats(by_key.get((cfg_d, pt), []))

            if b_med["n"] == 0 or c_med["n"] == 0 or d_med["n"] == 0:
                continue

            sidecar = c_med["median"] - b_med["median"]
            nixl = d_med["median"] - c_med["median"]
            total = d_med["median"] - b_med["median"]

            sidecar_vals.append(sidecar)
            nixl_vals.append((pt, nixl))

            print(f"  {pt:>6} | {sidecar:>11.0f}ms | {nixl:>10.0f}ms | "
                  f"{total:>10.0f}ms")
        print()

        # Compare to advisor constants
        if sidecar_vals:
            avg_sidecar = sum(sidecar_vals) / len(sidecar_vals)
            print("  Advisor comparison:")
            print(f"    Sidecar overhead:  measured={avg_sidecar:.0f}ms  "
                  f"advisor=12ms (SCALING_SIDECAR_MS)")

        if nixl_vals and len(nixl_vals) >= 3:
            xs = [float(pt) for pt, _ in nixl_vals]
            ys = [v for _, v in nixl_vals]
            r = pearson_r(xs, ys)
            print(f"    NIXL linearity:    Pearson r={r:.3f} "
                  f"({'linear' if abs(r) > 0.95 else 'NON-LINEAR'})")
            if abs(r) <= 0.95:
                peak_pt, peak_val = max(nixl_vals, key=lambda x: x[1])
                print(f"    NIXL peaks at {peak_pt} tokens ({peak_val:.0f}ms), "
                      "then drops — likely prefill pipeline amortization under load")
        print()

    if has_a and has_b:
        print("  Pod comparison (A=prefill GPU, B=decode GPU):")
        for pt in prompt_lens:
            a = stats(by_key.get((cfg_a, pt), []))
            b = stats(by_key.get((cfg_b, pt), []))
            if a["n"] > 0 and b["n"] > 0:
                diff_pct = (a["median"] - b["median"]) / b["median"] * 100 if b["median"] > 0 else 0
                print(f"    {pt:>5}tok: prefill={a['median']:.0f}ms, "
                      f"decode={b['median']:.0f}ms ({diff_pct:+.0f}%)")
        print()

    if "E-EPP-gateway" in configs and cfg_d:
        print("  EPP route comparison (end-to-end median TTFT; dynamic backend selection):")
        for pt in prompt_lens:
            epp = stats(by_key.get(("E-EPP-gateway", pt), []))
            manual = stats(by_key.get((cfg_d, pt), []))
            if epp["n"] and manual["n"]:
                delta = epp["median"] - manual["median"]
                print(f"    {pt:>5}tok: EPP={epp['median']:.1f}ms, "
                      f"manual D={manual['median']:.1f}ms, EPP-D={delta:+.1f}ms")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

# ── Experiment 16: Per-Token ITL Trace ──────────────────────────────────────

def analyze_exp16(data_dir):
    """Experiment 16: distribution of inter-token gaps, including the longest pause."""
    rows = load_csv(os.path.join(data_dir, "exp16-results.csv"))
    if not rows:
        print("  No data found")
        return
    rows, dropped = _steady_window(rows, lambda r: (r["config"], r["concurrency"]))
    failed = [r for r in rows if safe_int(r["token_idx"], -1) < 0]
    ok = [r for r in rows if safe_int(r["token_idx"], -1) >= 0]
    n_requests = len(set((r["config"], r["concurrency"], r["run"]) for r in ok))
    chunks = defaultdict(int)
    server = {}
    for r in ok:
        k = (r["config"], r["concurrency"], r["run"], safe_int(r["max_tokens"]))
        chunks[k] += 1
        server[k] = safe_int(r.get("server_completion_tokens"))
    short = sum(1 for k, n in chunks.items() if (server[k] or n) < k[3])
    merged = sum(1 for k, n in chunks.items() if server[k] > n)
    if failed:
        print(f"  Data quality: {len(failed)} failed requests, {n_requests} complete")
    else:
        print(f"  Data quality: {n_requests} complete requests, 0 failed")
    if short:
        print(f"  WARNING: {short}/{n_requests} requests stopped before max_tokens "
              f"(run with IGNORE_EOS=1 to control output length)")
    if merged:
        print(f"  NOTE: {merged}/{n_requests} requests had tokens with empty text; their "
              f"time is merged into the next gap")
    if dropped:
        print(f"  Steady load: dropped {dropped} ramp-up/ramp-down requests")
    slo_ms = safe_float(os.environ.get("ITL_SLO_MS", "50"), 50.0)
    print(f"  ITL threshold for the 'gaps over' column: {slo_ms:.0f} ms (ITL_SLO_MS)")
    print()

    ttft = defaultdict(list)
    gaps = defaultdict(list)
    by_pos = defaultdict(lambda: defaultdict(list))
    req_max = defaultdict(lambda: defaultdict(float))
    for r in ok:
        key = (r["config"], safe_int(r["concurrency"]))
        idx = safe_int(r["token_idx"])
        gap = safe_float(r["gap_ms"])
        if idx == 0:
            ttft[key].append(gap)
            continue
        gaps[key].append(gap)
        bucket = "1-5" if idx <= 5 else ("6-20" if idx <= 20 else "21+")
        by_pos[key][bucket].append(gap)
        req_max[key][r["run"]] = max(req_max[key][r["run"]], gap)

    print(f"  {'Config':>10} | {'c':>3} | {'TTFT p50':>9} | {'ITL p50':>8} | {'p90':>7} | "
          f"{'p99':>7} | {'max':>8} | {'longest pause p50':>17} | {'gaps over':>9}")
    for key in sorted(gaps):
        cfg, c = key
        g = stats(gaps[key])
        longest = stats(list(req_max[key].values()))
        over = sum(1 for x in gaps[key] if x > slo_ms) / len(gaps[key])
        t = stats(ttft[key])["median"] if ttft[key] else float("nan")
        print(f"  {cfg:>10} | {c:>3} | {t:>7.1f}ms | {g['p50']:>6.1f}ms | {g['p90']:>5.1f}ms | "
              f"{g['p99']:>5.1f}ms | {g['max']:>6.1f}ms | {longest['median']:>15.1f}ms | "
              f"{over:>8.1%}")
    print()

    print("  Mean gap by token position (under batch load, early gaps include the batch's")
    print("  own prompt processing; under steady load they should be flat):")
    for key in sorted(by_pos):
        cfg, c = key
        parts = []
        for bucket in ("1-5", "6-20", "21+"):
            vals = by_pos[key].get(bucket, [])
            if vals:
                parts.append(f"{bucket}: {sum(vals) / len(vals):.1f}ms")
        print(f"    {cfg} c={c}: " + ", ".join(parts))
    print()

    for key in sorted(gaps):
        if key[1] == 1:
            cv = stats(gaps[key])["cv"]
            note = "flat" if cv < 0.3 else "structured: check client/network before reading higher c"
            print(f"  Instrument check, {key[0]} at c=1: gap CV {cv:.2f} ({note})")
    print()


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    if not os.path.isdir(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Usage: python3 analyze.py [data_dir]")
        sys.exit(1)

    experiments = [
        ("Experiment 1: Single-Request Latency", "exp1-results.csv", analyze_exp1),
        ("Experiment 1b: Latency Decomposition", "exp1b-results.csv", analyze_exp1b),
        ("Experiment 2: Throughput Under Load", "exp2-results.csv", analyze_exp2),
        ("Experiment 3: Prefill Isolation", "exp3-results.csv", analyze_exp3),
        ("Experiment 4: Fault Tolerance", "exp4-results.csv", analyze_exp4),
        ("Experiment 5: Sequence Length Sweep", "exp5-results.csv", analyze_exp5),
        ("Experiment 6: Saturation Profiling", "exp6-results.csv", analyze_exp6),
        ("Experiment 7: Mixed Workload", "exp7-results.csv", analyze_exp7),
        ("Experiment 8: Prefix Cache Characterization", "exp8-results.csv", analyze_exp8),
        ("Experiment 11: Throughput vs Prompt Length", "exp11-results.csv", analyze_exp11),
        ("Experiment 12: Throughput vs Output Length", "exp12-results.csv", analyze_exp12),
        ("Experiment 13: Saturation Ceiling", "exp13-results.csv", analyze_exp13),
        ("Experiment 14: Overhead Under Load", "exp14-results.csv", analyze_exp14),
        ("Experiment 16: Per-Token ITL Trace", "exp16-results.csv", analyze_exp16),
    ]

    print("llm-d Diagnostics — Analysis")
    print(f"Data directory: {data_dir}")
    print()

    for title, filename, analyzer in experiments:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"{'=' * 60}")
            print(f"  {title}")
            print(f"{'=' * 60}")
            print()
            analyzer(data_dir)
            print()
        else:
            print(f"  {title}: {filename} not found, skipping")
            print()


if __name__ == "__main__":
    main()
