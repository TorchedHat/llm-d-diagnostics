"""Provider verifies local artifacts and never accepts a caller verification flag."""

import copy
import json
from pathlib import Path

from advisor.evidence import sha256_file
from advisor.provider import EvidenceTopologyProvider
from advisor.tests.test_evidence import bundle


def verified_bundle(tmp_path):
    evidence = bundle()
    sources = []
    for candidate in evidence["candidates"]:
        for block in candidate["blocks"]:
            path = tmp_path / (candidate["id"] + "-" + block["id"] + ".json")
            path.write_text(json.dumps({"candidate_id": candidate["id"],
                "comparison_fingerprint": candidate["comparison_fingerprint"], "block_id": block["id"],
                "wall_seconds": block["wall_seconds"], "requests": block["requests"]}))
            block["source_sha256"] = sha256_file(path)
            sources.append({"path": path.name, "sha256": block["source_sha256"]})
    evidence["sources"] = sources
    return evidence


def test_provider_verifies_sources_and_candidate_context(tmp_path):
    evidence = verified_bundle(tmp_path)
    context = {"workload": evidence["workload"], "policy": evidence["policy"]}
    provider = EvidenceTopologyProvider(tmp_path)
    assert provider.recommend(context, evidence["candidates"], evidence)["choice"] == "pd"
    candidates = copy.deepcopy(evidence["candidates"])
    candidates[0]["deployment_fingerprint"] = "different"
    assert provider.recommend(context, candidates, evidence)["status"] == "insufficient_evidence"
    Path(tmp_path / evidence["sources"][0]["path"]).write_text("tampered")
    result = provider.recommend(context, evidence["candidates"], evidence)
    assert result["status"] == "insufficient_evidence" and result["choice"] is None


def test_provider_rejects_paths_outside_allowed_root(tmp_path):
    evidence = verified_bundle(tmp_path)
    evidence["sources"][0]["path"] = "../outside"
    context = {"workload": evidence["workload"], "policy": evidence["policy"]}
    result = EvidenceTopologyProvider(tmp_path).recommend(context, evidence["candidates"], evidence)
    assert result["status"] == "insufficient_evidence" and "outside" in result["reasons"][0]
