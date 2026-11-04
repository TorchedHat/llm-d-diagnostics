"""Versioned, offline comparison of measured serving candidates.

This module consumes request-level evidence. It never sends requests or changes a deployment.
Callers must verify referenced artifacts before passing ``artifacts_verified=True``.
"""

import hashlib
import json
import math
import statistics
from pathlib import Path

VERSION = "llmd-diagnostics-evidence/v1"
OBJECTIVES = {"ttft_p50", "ttft_p90", "joint_goodput_per_gpu"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def verify_sources(sources, base_dir):
    """Check referenced artifact contents, not the truth of their metadata."""
    if not isinstance(sources, list) or not sources:
        return False
    for source in sources:
        if not isinstance(source, dict):
            return False
        path, expected = source.get("path"), source.get("sha256")
        if not isinstance(path, str) or not isinstance(expected, str) or len(expected) != 64:
            return False
        try:
            if sha256_file(Path(base_dir) / path) != expected.lower():
                return False
        except OSError:
            return False
    return True


def verify_bundle_sources(bundle, base_dir):
    """Bind every normalized cell to a hashed source file before analysis.

    This detects altered or mismatched normalized cells. It does not independently
    establish that an external importer faithfully interpreted its original raw data.
    """
    sources = bundle.get("sources") if isinstance(bundle, dict) else None
    if not verify_sources(sources, base_dir):
        return False
    by_hash = {s["sha256"].lower(): s for s in sources}
    try:
        for candidate in bundle["candidates"]:
            for block in candidate["blocks"]:
                source = by_hash[block["source_sha256"].lower()]
                stored = json.loads((Path(base_dir) / source["path"]).read_text())
                if (stored.get("candidate_id") != candidate["id"] or
                        stored.get("comparison_fingerprint") != candidate["comparison_fingerprint"] or
                        stored.get("block_id") != block["id"] or
                        stored.get("wall_seconds") != block["wall_seconds"] or
                        stored.get("requests") != block["requests"]):
                    return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return True


def percentile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def _positive_number(value, allow_zero=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value >= 0 if allow_zero else value > 0))


def _out(status, reasons, **values):
    return {"schema_version": VERSION, "status": status, "choice": None,
            "reasons": reasons, **values}


def compare(bundle, *, artifacts_verified=False):
    """Compare equal-budget candidates with paired blocks and observed SLO gates.

    The decision applies only to the recorded workload and configured objective. Reported
    attainment is a sample statistic, not a population-level reliability guarantee.
    """
    if not isinstance(bundle, dict) or bundle.get("schema_version") != VERSION:
        return _out("insufficient_evidence", ["unsupported evidence schema"])
    policy = bundle.get("policy", {})
    if not isinstance(policy, dict):
        return _out("insufficient_evidence", ["policy is missing"])
    objective = policy.get("objective")
    min_blocks, min_requests = policy.get("min_blocks"), policy.get("min_requests_per_block")
    margin, attainment = policy.get("practical_margin"), policy.get("min_observed_attainment")
    ttft_limit, tpot_limit = policy.get("ttft_slo_ms"), policy.get("tpot_slo_ms")
    quality_required = policy.get("require_quality", True)
    if (objective not in OBJECTIVES or not isinstance(min_blocks, int) or min_blocks < 2
            or not isinstance(min_requests, int) or min_requests < 2
            or not _positive_number(margin, True) or margin >= 1
            or not _positive_number(attainment) or attainment > 1
            or not _positive_number(ttft_limit) or not _positive_number(tpot_limit)
            or type(quality_required) is not bool
            or (policy.get("max_reserved_gpus") is not None
                and (type(policy["max_reserved_gpus"]) is not int or policy["max_reserved_gpus"] < 1))):
        return _out("insufficient_evidence", ["missing or invalid objective, SLO or sampling criteria"])
    if not artifacts_verified:
        return _out("insufficient_evidence", ["source artifacts were not verified"])
    workload = bundle.get("workload", {})
    keys = ("model_revision", "tokenizer_revision", "input_tokens", "output_tokens",
            "load", "sampling", "comparison_fingerprint")
    if not isinstance(workload, dict) or any(not workload.get(k) for k in keys):
        return _out("insufficient_evidence", ["workload identity is incomplete"])
    if any(not isinstance(workload[k], str) or workload[k].lower() in {"unknown", "unverified", "none"}
           for k in ("model_revision", "tokenizer_revision", "comparison_fingerprint")):
        return _out("insufficient_evidence", ["workload revision or fingerprint is unknown"])
    candidates = bundle.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        return _out("insufficient_evidence", ["at least two candidates are required"])
    if any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"]
           for c in candidates) or len({c["id"] for c in candidates}) != len(candidates):
        return _out("insufficient_evidence", ["candidate identifiers are missing or duplicated"])

    reasons, summaries, requests_by_block, budgets, deployments = [], {}, {}, set(), set()
    if bundle.get("comparison_verified") is not True:
        reasons.append("workload and serving configuration equivalence is unverified")
    for candidate in candidates:
        cid = candidate["id"]
        allocation = candidate.get("allocation", {})
        if candidate.get("comparison_fingerprint") != workload["comparison_fingerprint"]:
            reasons.append(f"{cid}: workload or common serving configuration differs")
        if not isinstance(candidate.get("deployment_fingerprint"), str) or not candidate["deployment_fingerprint"]:
            reasons.append(f"{cid}: deployment fingerprint is missing")
        elif candidate["deployment_fingerprint"] in deployments:
            reasons.append(f"{cid}: deployment fingerprint is duplicated")
        else:
            deployments.add(candidate["deployment_fingerprint"])
        if candidate.get("topology") not in {"aggregated", "disaggregated"}:
            reasons.append(f"{cid}: topology is missing or unsupported")
        if (not isinstance(allocation, dict) or type(allocation.get("active_gpus")) is not int
                or type(allocation.get("reserved_gpus")) is not int
                or allocation.get("active_gpus", 0) < 1
                or allocation.get("reserved_gpus", 0) < allocation.get("active_gpus", 0)):
            reasons.append(f"{cid}: GPU allocation or reservation is invalid")
            continue
        if allocation.get("verified") is not True:
            reasons.append(f"{cid}: GPU allocation is unverified")
        if allocation.get("exclusive_verified") is not True:
            reasons.append(f"{cid}: GPU isolation is unverified")
        active, reserved = allocation["active_gpus"], allocation["reserved_gpus"]
        budgets.add((active, reserved))
        if policy.get("max_reserved_gpus") is not None and reserved > policy["max_reserved_gpus"]:
            reasons.append(f"{cid}: reserved GPU budget exceeded")
        if quality_required and candidate.get("quality_status") != "verified_pass":
            reasons.append(f"{cid}: task quality is unverified")
        blocks = candidate.get("blocks")
        if not isinstance(blocks, list) or len(blocks) < min_blocks:
            reasons.append(f"{cid}: too few blocks")
            continue
        scored, seen_blocks = [], set()
        for block in blocks:
            if not isinstance(block, dict) or not isinstance(block.get("id"), str):
                reasons.append(f"{cid}: invalid block identity")
                continue
            bid = block["id"]
            rows, wall = block.get("requests"), block.get("wall_seconds")
            if bid in seen_blocks or not _positive_number(wall) or not isinstance(rows, list) or len(rows) < min_requests:
                reasons.append(f"{cid}/{bid}: duplicate block or incomplete cell")
                continue
            seen_blocks.add(bid)
            requests, successful, tpots, end_to_end, joint, failed, unknown = {}, [], [], [], 0, 0, 0
            for row in rows:
                if not isinstance(row, dict):
                    reasons.append(f"{cid}/{bid}: invalid request")
                    continue
                rid, prompt = row.get("id"), row.get("prompt_sha256")
                if (not isinstance(rid, str) or not rid or rid in requests
                        or not isinstance(prompt, str) or len(prompt) != 64
                        or any(ch not in "0123456789abcdef" for ch in prompt.lower())):
                    reasons.append(f"{cid}/{bid}: missing or duplicate request identity")
                    continue
                requests[rid] = prompt
                outcome = row.get("outcome")
                if outcome not in {"completed", "failed", "unknown"}:
                    reasons.append(f"{cid}/{bid}: request outcome is missing")
                elif outcome == "completed":
                    ttft, tpot = row.get("ttft_ms"), row.get("tpot_ms")
                    if not _positive_number(ttft) or not _positive_number(tpot, True):
                        reasons.append(f"{cid}/{bid}: completed request lacks valid TTFT or TPOT")
                        continue
                    successful.append(ttft)
                    tpots.append(tpot)
                    elapsed = row.get("e2e_ms")
                    if elapsed is not None:
                        if not _positive_number(elapsed) or elapsed < ttft:
                            reasons.append(f"{cid}/{bid}: invalid end-to-end request timing")
                        else:
                            end_to_end.append(elapsed)
                    joint += int(ttft <= ttft_limit and tpot <= tpot_limit)
                elif outcome == "failed":
                    failed += 1
                else:
                    unknown += 1
            if len(requests) != len(rows):
                continue
            requests_by_block.setdefault(bid, []).append((cid, requests))
            scored.append({"id": bid, "offered": len(rows), "completed": len(successful),
                           "failed": failed, "unknown": unknown,
                           "failed_or_unknown": len(rows) - len(successful),
                           "joint_pass": joint, "joint_attainment": joint / len(rows),
                           "ttft_p50": percentile(successful, .5) if successful else None,
                           "ttft_p90": percentile(successful, .9) if successful else None,
                           "tpot_p50": percentile(tpots, .5) if tpots else None,
                           "tpot_p90": percentile(tpots, .9) if tpots else None,
                           "e2e_p50": percentile(end_to_end, .5) if end_to_end and len(end_to_end) == len(successful) else None,
                           "e2e_p90": percentile(end_to_end, .9) if end_to_end and len(end_to_end) == len(successful) else None,
                           "finite_completed_per_active_gpu": len(successful) / wall / active,
                           "joint_goodput_per_gpu": joint / wall / active,
                           "joint_goodput_per_reserved_gpu": joint / wall / reserved,
                           "active_gpu_seconds_during_cell": wall * active,
                           "reserved_gpu_seconds_during_cell": wall * reserved,
                           "wall_seconds": wall})
        summaries[cid] = {"topology": candidate.get("topology"), "active_gpus": active,
                          "reserved_gpus": reserved, "blocks": scored}
    if len(budgets) > 1:
        reasons.append("candidate active or reserved GPU budgets differ")
    if len(summaries) != len(candidates):
        reasons.append("candidate evidence is incomplete")
    for bid, observed in requests_by_block.items():
        if len(observed) != len(candidates) or any(m != observed[0][1] for _, m in observed[1:]):
            reasons.append(f"{bid}: paired request IDs or prompt hashes differ")
    block_sets = [{b["id"] for b in summary["blocks"]} for summary in summaries.values()]
    if block_sets and any(blocks != block_sets[0] for blocks in block_sets[1:]):
        reasons.append("candidate block sets differ")
    if reasons:
        return _out("insufficient_evidence", reasons, candidates=summaries, objective=objective)

    eligible = [cid for cid, summary in summaries.items()
                if all(b["joint_attainment"] >= attainment for b in summary["blocks"])]
    common = {"candidates": summaries, "objective": objective,
              "scope": "Observed paired blocks only; not a population SLO guarantee or model-quality claim."}
    if not eligible:
        return _out("no_candidate_meets_slo", ["no candidate met the observed attainment gate in every block"], **common)
    higher = objective == "joint_goodput_per_gpu"
    scores = {cid: statistics.median(b[objective] for b in summaries[cid]["blocks"]) for cid in eligible}
    best = sorted(eligible, key=lambda cid: scores[cid], reverse=higher)[0]
    for other in eligible:
        if other == best:
            continue
        other_blocks = {b["id"]: b for b in summaries[other]["blocks"]}
        for block in summaries[best]["blocks"]:
            a, b = block[objective], other_blocks[block["id"]][objective]
            if (a is None or b is None or
                    (a <= b or a < b * (1 + margin) if higher
                     else a >= b or a > b * (1 - margin))):
                return _out("inconclusive", ["paired blocks do not consistently exceed the practical margin"], **common)
    return {**_out("measured", ["observed service gates and paired comparison passed"], **common),
            "choice": best, "eligible": eligible}


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", help="Versioned request-level evidence JSON")
    parser.add_argument("--output", required=True, help="New decision JSON file")
    args = parser.parse_args()
    try:
        source = Path(args.evidence)
        bundle = json.loads(source.read_text())
        result = compare(bundle, artifacts_verified=verify_bundle_sources(bundle, source.parent))
        result["evidence_sha256"] = sha256_file(source)
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Cannot analyze evidence: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
