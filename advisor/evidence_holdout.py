#!/usr/bin/env python3
"""Freeze and score a measured evidence decision on fresh, matched serving inputs."""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .evidence import VERSION, compare, sha256_file, verify_bundle_sources
except ImportError:
    from evidence import (  # type: ignore[no-redef]
        VERSION,
        compare,
        sha256_file,
        verify_bundle_sources,
    )


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("recorded time is missing")
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise ValueError("recorded time lacks a timezone")
    return result


def provenance(bundle, base):
    info = bundle.get("serving_provenance")
    if not isinstance(info, dict):
        raise ValueError("serving provenance is missing")
    for key in ("started_at", "cluster", "namespace", "worker_sha256", "controller_sha256", "result_sha256"):
        if not isinstance(info.get(key), str) or not info[key]:
            raise ValueError(f"serving provenance is missing {key}")
    timestamp(info["started_at"])
    for key in ("worker_sha256", "controller_sha256", "result_sha256"):
        if len(info[key]) != 64 or any(ch not in "0123456789abcdef" for ch in info[key]):
            raise ValueError(f"invalid serving provenance {key}")
    matches = [s for s in bundle.get("sources", []) if s.get("sha256") == info["result_sha256"]]
    if not matches:
        raise ValueError("serving result is not linked to evidence sources")
    result = json.loads((Path(base) / matches[0]["path"]).read_text())
    metadata = result.get("metadata", {})
    for key, source_key in (("started_at", "created_at"), ("cluster", "cluster"),
                            ("namespace", "namespace"), ("worker_sha256", "worker_sha256"),
                            ("controller_sha256", "controller_sha256")):
        if info[key] != metadata.get(source_key):
            raise ValueError(f"serving provenance differs from result: {key}")
    return info


def freeze(bundle, base, evidence_sha256, now):
    if not verify_bundle_sources(bundle, base):
        raise ValueError("calibration evidence sources are not verified")
    info = provenance(bundle, base)
    if timestamp(info["started_at"]) >= now:
        raise ValueError("calibration must start before advice is frozen")
    decision = compare(bundle, artifacts_verified=True)
    return {"schema_version": VERSION, "frozen_at": now.isoformat(),
            "calibration_evidence_sha256": evidence_sha256,
            "comparator_sha256": sha256_file(Path(__file__).with_name("evidence.py")),
            "decision": decision,
            "scope": "Recorded chronology and hashes are not independent proof of deployment equivalence or clock integrity."}


def _candidate_identity(bundle):
    fields = ("id", "topology", "comparison_fingerprint", "deployment_fingerprint", "allocation")
    return sorted(({key: c.get(key) for key in fields} for c in bundle["candidates"]), key=lambda x: x["id"])


def _prompts(bundle):
    first = [request["prompt_sha256"] for block in bundle["candidates"][0]["blocks"]
             for request in block["requests"]]
    if len(first) != len(set(first)):
        raise ValueError("a serving run reused prompt hashes across measured requests")
    return {request["prompt_sha256"] for candidate in bundle["candidates"]
            for block in candidate["blocks"] for request in block["requests"]}


def _objective_regret(decision):
    """Compare constant candidates with the per-block best observed objective."""
    objective = decision.get("objective")
    candidates = decision.get("candidates", {})
    if not candidates or objective not in {"ttft_p50", "ttft_p90", "joint_goodput_per_gpu"}:
        return {}
    blocks = {cid: {block["id"]: block.get(objective) for block in candidate.get("blocks", [])}
              for cid, candidate in candidates.items()}
    sets = [set(items) for items in blocks.values()]
    if not sets or not sets[0] or any(items != sets[0] for items in sets[1:]):
        return {}
    losses = {cid: [] for cid in blocks}
    for block in sorted(sets[0]):
        values = {cid: items[block] for cid, items in blocks.items()}
        if any(not isinstance(value, (int, float)) or value < 0 for value in values.values()):
            return {}
        best = (max(values.values()) if objective == "joint_goodput_per_gpu" else min(values.values()))
        if best <= 0:
            return {}
        for cid, value in values.items():
            losses[cid].append(1 - value / best if objective == "joint_goodput_per_gpu" else value / best - 1)
    return {cid: statistics.mean(values) for cid, values in losses.items()}


def score(frozen, calibration, calibration_base, calibration_sha256, holdout, holdout_base):
    if frozen.get("schema_version") != VERSION or frozen.get("calibration_evidence_sha256") != calibration_sha256:
        raise ValueError("frozen advice does not match calibration evidence")
    if frozen.get("comparator_sha256") != sha256_file(Path(__file__).with_name("evidence.py")):
        raise ValueError("evidence comparator changed since advice was frozen")
    if not verify_bundle_sources(calibration, calibration_base) or not verify_bundle_sources(holdout, holdout_base):
        raise ValueError("calibration or holdout evidence sources are not verified")
    train, test = provenance(calibration, calibration_base), provenance(holdout, holdout_base)
    if not timestamp(train["started_at"]) < timestamp(frozen.get("frozen_at")) < timestamp(test["started_at"]):
        raise ValueError("recorded advice must be frozen between calibration and holdout")
    for key in ("cluster", "namespace", "worker_sha256", "controller_sha256"):
        if train[key] != test[key]:
            raise ValueError(f"serving provenance differs: {key}")
    if (calibration.get("workload") != holdout.get("workload")
            or calibration.get("policy") != holdout.get("policy")
            or _candidate_identity(calibration) != _candidate_identity(holdout)):
        raise ValueError("workload, policy or deployment identities differ")
    before, after = _prompts(calibration), _prompts(holdout)
    if not before or not after or before & after:
        raise ValueError("calibration and holdout prompts overlap or are missing")
    expected = compare(calibration, artifacts_verified=True)
    if frozen.get("decision") != expected:
        raise ValueError("frozen advice differs from the calibration decision")
    observed = compare(holdout, artifacts_verified=True)
    choice = expected.get("choice")
    if expected.get("status") != "measured" or choice is None:
        verdict = "abstained"
    elif observed.get("status") in {"insufficient_evidence", "inconclusive"}:
        verdict = "inconclusive"
    elif observed.get("status") == "no_candidate_meets_slo":
        verdict = "not_confirmed"
    else:
        verdict = "confirmed" if observed.get("choice") == choice else "not_confirmed"
    def cost(decision, key):
        return sum(block[key] for candidate in decision.get("candidates", {}).values()
                   for block in candidate.get("blocks", []))
    return {"schema_version": VERSION, "verdict": verdict, "frozen_choice": choice,
            "holdout": observed, "calibration_active_gpu_seconds_in_cells": cost(expected, "active_gpu_seconds_during_cell"),
            "calibration_reserved_gpu_seconds_in_cells": cost(expected, "reserved_gpu_seconds_during_cell"),
            "holdout_constant_candidate_regret": _objective_regret(observed),
            "regret_scope": "Equal-block mean objective regret against each block's observed best, without SLO eligibility filtering.",
            "scope": "Fresh synthetic inputs on recorded matching deployments; per-cell costs exclude warmup and rollout. Recorded clocks and operator assertions are not independently attested."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    first = sub.add_parser("freeze")
    first.add_argument("--evidence", required=True)
    first.add_argument("--output", required=True)
    second = sub.add_parser("score")
    second.add_argument("--frozen", required=True)
    second.add_argument("--calibration", required=True)
    second.add_argument("--holdout", required=True)
    second.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            path = Path(args.evidence)
            bundle = json.loads(path.read_text())
            result = freeze(bundle, path.parent, sha256_file(path), datetime.now(timezone.utc))  # noqa: UP017
        else:
            frozen_path, training_path, holdout_path = map(Path, (args.frozen, args.calibration, args.holdout))
            frozen = json.loads(frozen_path.read_text())
            training, holdout = (json.loads(path.read_text()) for path in (training_path, holdout_path))
            result = score(frozen, training, training_path.parent, sha256_file(training_path),
                           holdout, holdout_path.parent)
            result["source_sha256"] = {"frozen": sha256_file(frozen_path), "holdout": sha256_file(holdout_path),
                                       "validator": sha256_file(Path(__file__))}
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"Cannot validate evidence: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
