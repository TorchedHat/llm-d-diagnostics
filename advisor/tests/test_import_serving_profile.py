import hashlib
import json

import pytest

from advisor.import_serving_profile import convert


def test_epp_import_tracks_router_and_keeps_deployment_unverified(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    def pod(name):
        return {"uid": name + "-uid", "node": "node", "image_ids": {"vllm": "sha"},
                "restart_counts": {"vllm": 0}}
    config = {"model": "model", "input_tokens": 64, "output_tokens": 8, "concurrencies": [1],
              "requests": 2, "blocks": 2, "pd_route": "epp"}
    metadata = {"config": config, "roles": {"mono": ["m1", "m2"], "prefill": "p", "decode": "d"},
                "pods": {name: pod(name) for name in ("m1", "m2", "p", "d")},
                "router": {"name": "r", **pod("r")}, "gpu_counts": {"MONO": 2, "PD": 2}}
    rows, cells = [], []
    prompts = [[number] * 64 for number in range(2)]
    for block in range(2):
        rows.append({"event": "inputs", "concurrency": 1, "block": block, "tokens": prompts})
    for arm in ("MONO", "PD"):
        for block in range(2):
            cells.append({"arm": arm, "concurrency": 1, "block": block, "wall_seconds": 1})
            for number in range(2):
                rows.append({"event": "request", "arm": arm, "concurrency": 1, "block": block,
                             "number": number, "warmup": False, "ok": True,
                             "prompt_sha256": hashlib.sha256(json.dumps(prompts[number]).encode()).hexdigest(),
                             "status": 200, "done": True, "prompt_tokens": 64, "completion_tokens": 8,
                             "ttft_ms": 20, "tpot_ms": 2, "elapsed_ms": 40})
    rows.append({"event": "complete"})
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    (source / "raw.jsonl").write_bytes(raw)
    result = {"status": "complete", "raw_sha256": hashlib.sha256(raw).hexdigest(),
              "metadata": metadata, "cells": cells}
    (source / "result.json").write_text(json.dumps(result))
    policy = {"objective": "ttft_p50", "ttft_slo_ms": 200, "tpot_slo_ms": 50,
              "min_observed_attainment": 1, "practical_margin": .1, "min_blocks": 2,
              "min_requests_per_block": 2, "require_quality": False}
    output = tmp_path / "epp"
    decision = convert(source, output, 1, "model-rev", "tokenizer-rev", policy)
    assert decision["status"] == "insufficient_evidence" and decision["choice"] is None
    attachment = json.loads((output / "evidence.json").read_text())
    assert attachment["comparison_verified"] is False
    assert "epp routing selection is not independently verified" in attachment["provenance_limitations"]
    metadata["router"]["uid"] = "other-router"
    (source / "result.json").write_text(json.dumps(result))
    second = tmp_path / "changed-router"
    convert(source, second, 1, "model-rev", "tokenizer-rev", policy)
    other = json.loads((second / "evidence.json").read_text())
    assert attachment["candidates"][1]["deployment_fingerprint"] != other["candidates"][1]["deployment_fingerprint"]
    metadata["router"] = None
    (source / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="router identity"):
        convert(source, tmp_path / "missing-router", 1, "model-rev", "tokenizer-rev", policy)

    metadata["router"] = {"name": "r", **pod("r")}
    next(row for row in rows if row.get("event") == "request")["completion_tokens"] = 7
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    (source / "raw.jsonl").write_bytes(raw)
    result["raw_sha256"] = hashlib.sha256(raw).hexdigest()
    (source / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="token accounting"):
        convert(source, tmp_path / "invalid-tokens", 1, "model-rev", "tokenizer-rev", policy)
