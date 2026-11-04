"""Frozen choices use independent inputs and matched serving provenance."""

import copy
import json
from datetime import datetime, timezone

import pytest

from advisor.evidence import sha256_file
from advisor.evidence_holdout import freeze, score
from advisor.tests.test_evidence import bundle


def make_evidence(root, started, prompt_digit, *, pd_ms=60, verified=True):
    root.mkdir()
    value = bundle()
    value["comparison_verified"] = verified
    for candidate in value["candidates"]:
        for block in candidate["blocks"]:
            for i, row in enumerate(block["requests"]):
                row["prompt_sha256"] = format(prompt_digit + i + 2 * int(block["id"][-1]), "x") * 64
                if candidate["id"] == "pd":
                    row["ttft_ms"] = pd_ms
    sources = []
    for candidate in value["candidates"]:
        for block in candidate["blocks"]:
            path = root / (candidate["id"] + "-" + block["id"] + ".json")
            path.write_text(json.dumps({"candidate_id": candidate["id"],
                "comparison_fingerprint": candidate["comparison_fingerprint"], "block_id": block["id"],
                "wall_seconds": block["wall_seconds"], "requests": block["requests"]}))
            block["source_sha256"] = sha256_file(path)
            sources.append({"path": path.name, "sha256": block["source_sha256"]})
    metadata = {"created_at": started, "cluster": "cluster", "namespace": "ns",
                "worker_sha256": "a" * 64, "controller_sha256": "b" * 64}
    result = root / "result.json"
    result.write_text(json.dumps({"metadata": metadata}))
    result_sha = sha256_file(result)
    sources.append({"path": result.name, "sha256": result_sha})
    value["sources"] = sources
    value["serving_provenance"] = {"started_at": started, "cluster": "cluster", "namespace": "ns",
        "worker_sha256": "a" * 64, "controller_sha256": "b" * 64, "result_sha256": result_sha}
    path = root / "evidence.json"
    path.write_text(json.dumps(value))
    return value, sha256_file(path)


NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)  # noqa: UP017


def test_frozen_choice_is_checked_on_disjoint_holdout(tmp_path):
    training, digest = make_evidence(tmp_path / "train", "2026-01-01T00:00:00+00:00", 1)
    holdout, _ = make_evidence(tmp_path / "holdout", "2026-01-03T00:00:00+00:00", 5)
    advice = freeze(training, tmp_path / "train", digest, NOW)
    assert advice["decision"]["choice"] == "pd"
    outcome = score(advice, training, tmp_path / "train", digest, holdout, tmp_path / "holdout")
    assert outcome["verdict"] == "confirmed" and outcome["frozen_choice"] == "pd"
    assert outcome["calibration_active_gpu_seconds_in_cells"] == 16
    assert outcome["holdout_constant_candidate_regret"]["pd"] == 0
    assert outcome["holdout_constant_candidate_regret"]["mono"] == pytest.approx(2 / 3)
    different, _ = make_evidence(tmp_path / "changed", "2026-01-03T00:00:00+00:00", 5, pd_ms=120)
    outcome = score(advice, training, tmp_path / "train", digest, different, tmp_path / "changed")
    assert outcome["verdict"] == "not_confirmed"


def test_overlap_and_unfrozen_chronology_are_rejected(tmp_path):
    training, digest = make_evidence(tmp_path / "train", "2026-01-01T00:00:00+00:00", 1)
    advice = freeze(training, tmp_path / "train", digest, NOW)
    overlap, _ = make_evidence(tmp_path / "overlap", "2026-01-03T00:00:00+00:00", 1)
    with pytest.raises(ValueError, match="prompts overlap"):
        score(advice, training, tmp_path / "train", digest, overlap, tmp_path / "overlap")
    early, _ = make_evidence(tmp_path / "early", "2026-01-01T12:00:00+00:00", 3)
    with pytest.raises(ValueError, match="frozen between"):
        score(advice, training, tmp_path / "train", digest, early, tmp_path / "early")
    changed = copy.deepcopy(advice)
    changed["decision"]["choice"] = "mono"
    holdout, _ = make_evidence(tmp_path / "holdout", "2026-01-03T00:00:00+00:00", 5)
    with pytest.raises(ValueError, match="differs from the calibration"):
        score(changed, training, tmp_path / "train", digest, holdout, tmp_path / "holdout")


def test_missing_equivalence_stays_abstained(tmp_path):
    training, digest = make_evidence(tmp_path / "train", "2026-01-01T00:00:00+00:00", 1, verified=False)
    holdout, _ = make_evidence(tmp_path / "holdout", "2026-01-03T00:00:00+00:00", 5, verified=False)
    advice = freeze(training, tmp_path / "train", digest, NOW)
    outcome = score(advice, training, tmp_path / "train", digest, holdout, tmp_path / "holdout")
    assert advice["decision"]["choice"] is None and outcome["verdict"] == "abstained"
