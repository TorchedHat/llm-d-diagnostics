"""Operating-map decisions preserve absolute attainment and incomplete evidence."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import operating_map as om


def sample():
    config = {"schema_version": om.VERSION, "arms": ["MONO", "PD"], "repeats": [0, 1],
              "declared_gpus_per_arm": 2, "practical_margin": .05,
              "min_observed_attainment": .9}
    rows = []
    for arm in config["arms"]:
        for repeat in config["repeats"]:
            good = 1 if arm == "PD" else 0
            rows.append({"model": "example", "revision": "rev", "arm": arm, "phase": "main",
                         "repeat": str(repeat), "prompt_tokens": "100", "rate_key": "low",
                         "requests": "100", "successes": "100", "failures": "0",
                         "slo_passes": str(good), "wall_seconds": "1", "valid": "true",
                         "goodput_per_gpu": str(good / 2)})
    return rows, config


def test_relative_win_does_not_imply_acceptable_attainment():
    rows, config = sample()
    result = om.summarize(rows, config)
    position = result["positions"][0]
    assert position["relative_winner"] == "PD"
    assert position["gated_winner"] == "no_candidate_meets_slo"
    assert position["arms"]["PD"][0]["observed_slo_attainment"] == .01
    assert "allocation" in result["scope"]


def test_requires_complete_unique_cells_and_recomputed_goodput():
    rows, config = sample()
    with pytest.raises(ValueError, match="incomplete"):
        om.summarize(rows[:-1], config)
    with pytest.raises(ValueError, match="duplicate"):
        om.summarize([*rows, rows[0]], config)
    wrong = copy.deepcopy(rows)
    wrong[-1]["goodput_per_gpu"] = "2"
    with pytest.raises(ValueError, match="goodput differs"):
        om.summarize(wrong, config)


def test_inconsistent_repeats_are_inconclusive():
    rows, config = sample()
    rows[0]["slo_passes"] = "2"
    rows[0]["goodput_per_gpu"] = "1"
    result = om.summarize(rows, config)
    assert result["positions"][0]["relative_winner"] == "inconclusive"
