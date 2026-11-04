import copy
import hashlib
import json

from advisor.evidence import VERSION, compare, verify_bundle_sources, verify_sources


def bundle():
    candidates = []
    for name, topology, ttft in (("mono", "aggregated", 100), ("pd", "disaggregated", 60)):
        blocks = []
        for block in ("b0", "b1"):
            rows = [{"id": str(i), "prompt_sha256": str(i) * 64,
                     "outcome": "completed", "ttft_ms": ttft, "tpot_ms": 5}
                    for i in (1, 2)]
            blocks.append({"id": block, "wall_seconds": 2, "requests": rows})
        candidates.append({"id": name, "topology": topology, "comparison_fingerprint": "same",
                           "deployment_fingerprint": name + "-deployment",
                           "allocation": {"verified": True, "exclusive_verified": True,
                                          "active_gpus": 2, "reserved_gpus": 4},
                           "blocks": blocks})
    return {"schema_version": VERSION, "comparison_verified": True,
            "workload": {"model_revision": "m", "tokenizer_revision": "t", "input_tokens": 4096,
                         "output_tokens": 256, "load": "c8", "sampling": {"temperature": 0},
                         "comparison_fingerprint": "same"},
            "policy": {"objective": "ttft_p50", "min_blocks": 2, "min_requests_per_block": 2,
                       "practical_margin": .1, "min_observed_attainment": 1,
                       "ttft_slo_ms": 200, "tpot_slo_ms": 50, "require_quality": False},
            "candidates": candidates}


def test_paired_measured_decision_and_service_gate():
    data = bundle()
    result = compare(data, artifacts_verified=True)
    assert result["status"] == "measured" and result["choice"] == "pd"
    data["policy"]["ttft_slo_ms"] = 50
    assert compare(data, artifacts_verified=True)["status"] == "no_candidate_meets_slo"


def test_per_cell_metrics_distinguish_unknown_and_missing_end_to_end():
    data = bundle()
    for candidate in data["candidates"]:
        for block in candidate["blocks"]:
            block["requests"][0]["e2e_ms"] = 200
            block["requests"][1]["outcome"] = "unknown"
    result = compare(data, artifacts_verified=True)
    cell = result["candidates"]["mono"]["blocks"][0]
    assert cell["unknown"] == 1 and cell["failed"] == 0
    assert cell["e2e_p50"] == 200 and cell["tpot_p90"] == 5
    assert cell["joint_goodput_per_reserved_gpu"] == .125
    assert cell["reserved_gpu_seconds_during_cell"] == 8
    data = bundle()
    result = compare(data, artifacts_verified=True)
    assert result["candidates"]["mono"]["blocks"][0]["e2e_p50"] is None


def test_mismatch_unknown_and_unverified_sources_abstain():
    data = bundle()
    assert compare(data)["status"] == "insufficient_evidence"
    changed = copy.deepcopy(data)
    changed["candidates"][1]["blocks"][0]["requests"][0]["prompt_sha256"] = "b" * 64
    assert compare(changed, artifacts_verified=True)["status"] == "insufficient_evidence"
    changed = copy.deepcopy(data)
    changed["candidates"][0]["allocation"]["verified"] = False
    assert compare(changed, artifacts_verified=True)["status"] == "insufficient_evidence"


def test_small_or_inconsistent_advantage_is_inconclusive():
    data = bundle()
    for row in data["candidates"][1]["blocks"][1]["requests"]:
        row["ttft_ms"] = 95
    assert compare(data, artifacts_verified=True)["status"] == "inconclusive"


def test_quality_and_equal_gpu_budget_are_explicit():
    data = bundle()
    data["policy"]["require_quality"] = True
    assert compare(data, artifacts_verified=True)["status"] == "insufficient_evidence"
    data["policy"]["require_quality"] = False
    data["candidates"][1]["allocation"]["active_gpus"] = 1
    assert compare(data, artifacts_verified=True)["status"] == "insufficient_evidence"


def test_invalid_quality_policy_and_duplicate_deployment_abstain():
    data = bundle()
    data["policy"]["require_quality"] = None
    assert compare(data, artifacts_verified=True)["status"] == "insufficient_evidence"
    data = bundle()
    data["candidates"][1]["deployment_fingerprint"] = data["candidates"][0]["deployment_fingerprint"]
    assert compare(data, artifacts_verified=True)["status"] == "insufficient_evidence"


def test_hash_verification(tmp_path):
    (tmp_path / "source").write_bytes(b"evidence")
    items = [{"path": "source", "sha256": hashlib.sha256(b"evidence").hexdigest()}]
    assert verify_sources(items, tmp_path)
    (tmp_path / "source").write_bytes(b"changed")
    assert not verify_sources(items, tmp_path)


def test_normalized_cells_are_bound_to_their_sources(tmp_path):
    data = bundle()
    data["sources"] = []
    for candidate in data["candidates"]:
        for block in candidate["blocks"]:
            content = {"candidate_id": candidate["id"], "block_id": block["id"],
                       "comparison_fingerprint": candidate["comparison_fingerprint"],
                       "wall_seconds": block["wall_seconds"], "requests": block["requests"]}
            path = tmp_path / f"{candidate['id']}-{block['id']}.json"
            path.write_text(json.dumps(content))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            block["source_sha256"] = digest
            data["sources"].append({"path": path.name, "sha256": digest})
    assert verify_bundle_sources(data, tmp_path)
    data["candidates"][0]["blocks"][0]["requests"][0]["ttft_ms"] = 1
    assert not verify_bundle_sources(data, tmp_path)
