#!/usr/bin/env python3
"""Convert a serving-profile run into versioned, provenance-linked evidence."""

import argparse
import hashlib
import json
import sys
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


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def convert(directory, output, concurrency, model_revision, tokenizer_revision, policy, allocation_audit=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    result_path, raw_path = directory / "result.json", directory / "raw.jsonl"
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete" or sha256_file(raw_path) != result.get("raw_sha256"):
        raise ValueError("serving profile is incomplete or the raw hash differs")
    meta, cells = result["metadata"], result["cells"]
    config = meta["config"]
    route = config.get("pd_route", "manual")
    router = meta.get("router")
    if (route not in {"manual", "epp"} or (route == "epp" and
            (not isinstance(router, dict) or any(not router.get(k) for k in ("name", "uid", "node", "image_ids", "restart_counts"))))):
        raise ValueError("P/D routing or router identity is missing")
    if concurrency not in config["concurrencies"]:
        raise ValueError("concurrency was not measured")
    if not model_revision or not tokenizer_revision:
        raise ValueError("model and tokenizer revision declarations are required")
    raw = [json.loads(line) for line in raw_path.read_bytes().splitlines()]
    if not raw or any(not isinstance(event, dict) for event in raw) or raw[-1].get("event") != "complete":
        raise ValueError("serving profile has no completed worker record")
    prompt_hashes = {}
    for event in raw:
        if event.get("event") != "inputs" or event.get("concurrency") != concurrency:
            continue
        block, prompts = event.get("block"), event.get("tokens")
        if (type(block) is not int or block in prompt_hashes or not isinstance(prompts, list)
                or len(prompts) != config["requests"]
                or any(not isinstance(p, list) or len(p) != config["input_tokens"]
                       or any(type(token) is not int or token < 0 for token in p) for p in prompts)):
            raise ValueError("invalid or duplicated recorded prompt set")
        prompt_hashes[block] = [hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
                                for prompt in prompts]
    if set(prompt_hashes) != set(range(config["blocks"])):
        raise ValueError("recorded prompt blocks are incomplete")
    for row in raw:
        if (row.get("event") != "request" or row.get("warmup") is not False
                or row.get("concurrency") != concurrency):
            continue
        block, number = row.get("block"), row.get("number")
        if (type(block) is not int or block not in prompt_hashes or type(number) is not int
                or not 0 <= number < config["requests"]
                or row.get("prompt_sha256") != prompt_hashes[block][number]
                or type(row.get("ok")) is not bool):
            raise ValueError("request identity or paired prompt does not match the recorded inputs")
        if row["ok"] and (row.get("status") != 200 or row.get("done") is not True
                or row.get("prompt_tokens") != config["input_tokens"]
                or row.get("completion_tokens") != config["output_tokens"]):
            raise ValueError("successful request is missing completion or exact token accounting")
    identity = {"model": config["model"], "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision, "input_tokens": config["input_tokens"],
                "output_tokens": config["output_tokens"], "concurrency": concurrency,
                "sampling": {"temperature": 0, "ignore_eos": True},
                "routing": "manual sidecar" if route == "manual" else "epp",
                "cache_settings": "operator verification required"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    workload = {"model_revision": model_revision, "tokenizer_revision": tokenizer_revision,
                "input_tokens": config["input_tokens"], "output_tokens": config["output_tokens"],
                "load": {"kind": "closed_loop", "concurrency": concurrency},
                "sampling": identity["sampling"], "comparison_fingerprint": fingerprint}
    output.mkdir(parents=True, exist_ok=False)
    sources = [{"path": str(result_path), "sha256": sha256_file(result_path)},
               {"path": str(raw_path), "sha256": sha256_file(raw_path)}]
    audited_pods = {}
    if allocation_audit:
        audit_path = Path(allocation_audit).resolve()
        audit = json.loads(audit_path.read_text())
        if audit.get("verified_dra_allocations") is not True:
            raise ValueError("DRA allocation audit did not pass")
        audited_pods = {row["pod"]: row for row in audit["devices"]}
        if len(audited_pods) != len(audit["devices"]):
            raise ValueError("duplicate pod in DRA allocation audit")
        sources.append({"path": str(audit_path), "sha256": sha256_file(audit_path)})
    candidates = []
    for arm, topology in (("MONO", "aggregated"), ("PD", "disaggregated")):
        names = meta["roles"]["mono"] if arm == "MONO" else [meta["roles"]["prefill"], meta["roles"]["decode"]]
        pod_info = [{key: meta["pods"][name].get(key) for key in ("uid", "node", "image_ids", "restart_counts")}
                    for name in names]
        if arm == "PD" and route == "epp":
            pod_info.append({key: meta["router"].get(key) for key in ("uid", "node", "image_ids", "restart_counts")})
        allocated = bool(audited_pods) and all(
            name in audited_pods and audited_pods[name].get("pod_uid") == meta["pods"][name].get("uid")
            and audited_pods[name].get("node") == meta["pods"][name].get("node")
            for name in names)
        deployment = hashlib.sha256(json.dumps(pod_info, sort_keys=True).encode()).hexdigest()
        candidate = {"id": arm, "topology": topology, "comparison_fingerprint": fingerprint,
                     "deployment_fingerprint": deployment,
                     "allocation": {"verified": allocated, "exclusive_verified": False,
                                    "active_gpus": meta["gpu_counts"][arm],
                                    "reserved_gpus": sum(meta["gpu_counts"].values())},
                     "quality_status": "not_verified", "blocks": []}
        for number in range(config["blocks"]):
            matches = [c for c in cells if c["arm"] == arm and c["concurrency"] == concurrency and c["block"] == number]
            rows = [r for r in raw if r.get("event") == "request" and not r.get("warmup")
                    and r.get("arm") == arm and r.get("concurrency") == concurrency and r.get("block") == number]
            if len(matches) != 1 or len(rows) != config["requests"] or len({r["number"] for r in rows}) != len(rows):
                raise ValueError(f"incomplete cell {arm} block {number}")
            requests = []
            for row in sorted(rows, key=lambda r: r["number"]):
                outcome = "completed" if row.get("ok") else ("failed" if row.get("status") and row["status"] != 200 else "unknown")
                requests.append({"id": str(row["number"]), "prompt_sha256": row["prompt_sha256"],
                                 "outcome": outcome, "ttft_ms": row.get("ttft_ms"), "tpot_ms": row.get("tpot_ms"),
                                 "e2e_ms": row.get("elapsed_ms")})
            bid, wall = f"block-{number}", matches[0]["wall_seconds"]
            content = {"candidate_id": arm, "comparison_fingerprint": fingerprint,
                       "block_id": bid, "wall_seconds": wall, "requests": requests}
            path = output / f"{arm.lower()}-{bid}.json"
            write_json(path, content)
            digest = sha256_file(path)
            sources.append({"path": path.name, "sha256": digest})
            candidate["blocks"].append({"id": bid, "wall_seconds": wall, "requests": requests,
                                        "source_sha256": digest})
        candidates.append(candidate)
    bundle = {"schema_version": VERSION, "comparison_verified": False, "workload": workload, "policy": policy,
              "candidates": candidates, "sources": sources,
              "serving_provenance": {"started_at": meta.get("created_at"), "cluster": meta.get("cluster"),
                                     "namespace": meta.get("namespace"), "worker_sha256": meta.get("worker_sha256"),
                                     "controller_sha256": meta.get("controller_sha256"),
                                     "result_sha256": sha256_file(result_path)},
              "provenance_limitations": ["model and tokenizer revisions are operator declarations",
                  "DRA allocation audit is an external assertion, not independently replayed by this importer",
                  "GPU isolation is not independently verified", f"{route} routing selection is not independently verified",
                  "historical profiles may not contain request-mean TPOT"]}
    write_json(output / "evidence.json", bundle)
    decision = compare(bundle, artifacts_verified=verify_bundle_sources(bundle, output))
    write_json(output / "decision.json", decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serving_dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--policy", required=True, help="Explicit policy JSON")
    parser.add_argument("--allocation-audit", help="Optional saved DRA claim and pod UID audit")
    args = parser.parse_args()
    try:
        decision = convert(args.serving_dir, args.output_dir, args.concurrency, args.model_revision,
                           args.tokenizer_revision, json.loads(Path(args.policy).read_text()), args.allocation_audit)
        print(json.dumps(decision, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot import serving profile: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
