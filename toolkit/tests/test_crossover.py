"""Test saved crossover evidence and the reported decision scope."""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import crossover
import experiment_io as io


def make_campaign(root, mismatch=False):
    root.mkdir()
    arms = [
        {"name": "mono", "kind": "monolithic", "gpu_count": 2, "prefill_backoff_tokens": 0,
         "instances": [{"label": "m", "role": "monolithic", "gpus": 2}]},
        {"name": "pd", "kind": "pd", "gpu_count": 2, "prefill_backoff_tokens": 0,
         "instances": [{"label": "p", "role": "prefill", "gpus": 1},
                       {"label": "d", "role": "decode", "gpus": 1}]},
    ]
    for arm in arms:
        directory = root / arm["name"]
        directory.mkdir()
        config = {"arms": arms}
        io.write_json(directory / "config.json", config)
        rows, prompts = [], []
        good = 1 if arm["name"] == "mono" else 2
        for i in range(4):
            prompt = [1, 2, i + (20 if mismatch and arm["name"] == "pd" and i == 0 else 3)]
            prompts.append({"request": i, "token_ids": prompt, "sha256": io.digest(prompt)})
            rows.append({"request": i, "success": i < good, "prompt_sha256": io.digest(prompt),
                         "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                         "generated_token_ids": [4, 5], "ttft_ms": 1, "itl_mean_ms": 1})
        for filename, items in (("requests.jsonl.gz", rows), ("prompts.jsonl.gz", prompts)):
            with gzip.open(directory / filename, "wt") as stream:
                for item in items:
                    stream.write(json.dumps(item) + "\n")
        before, after = {"pods": {}}, {"pods": {}}
        for instance in arm["instances"]:
            label, role = instance["label"], instance["role"]
            def lines(compute, external):
                return ["process_start_time_seconds 10",
                        f'vllm:prompt_tokens_by_source_total{{source="local_compute"}} {compute}',
                        'vllm:prompt_tokens_by_source_total{source="local_cache_hit"} 0',
                        f'vllm:prompt_tokens_by_source_total{{source="external_kv_transfer"}} {external}']
            before["pods"][label] = lines(0, 0)
            after["pods"][label] = lines(good * 3 if role != "decode" else 0,
                                           good * 3 if role == "decode" else 0)
        io.write_json(directory / "metrics-before.json", before)
        io.write_json(directory / "metrics-after.json", after)
        io.write_json(directory / "accounting.json", crossover.verify_accounting(arm, before, after, rows))
        io.write_json(directory / "summary.json", {"model": "m", "model_revision": "r", "nonce": "n",
            "arm": arm["name"], "kind": arm["kind"], "prompt_tokens": 3, "output_tokens": 2,
            "concurrency": 1, "repeat": 1, "attempted": 4, "successful": good,
            "slo_good": good, "gpu_count": 2, "wall_seconds": 10, "slo_ttft_ms": 2,
            "slo_itl_ms": 2, "goodput_rps_per_gpu": good / 20, "deployment_settings": {}})
        io.write_json(directory / "complete.json", {"requests": 4})
    io.write_json(root / "campaign-complete.json", {"cells": 2})
    io.manifest(root)
    return root


def test_relative_goodput_reports_absolute_attainment_and_unverified_allocation(tmp_path):
    root = make_campaign(tmp_path / "data")
    result = crossover.analyze([root], tmp_path / "result")
    assert len(result) == 1
    row = result[0]
    assert row["pd_wins"] and row["pd_slo_attainment"] == 0.5
    assert row["monolithic_slo_attainment"] == 0.25
    assert row["pd_attempted"] == row["monolithic_attempted"] == 4
    assert row["gpu_allocation_verified"] is False
    info = json.loads((tmp_path / "result" / "analysis-info.json").read_text())
    assert "not a qualified deployment recommendation" in info["pd_wins_scope"]


def test_unpaired_prompts_are_rejected(tmp_path):
    root = make_campaign(tmp_path / "data", mismatch=True)
    with pytest.raises(ValueError, match="paired token-ID prompts"):
        crossover.analyze([root], tmp_path / "result")


def test_manifest_tampering_is_rejected(tmp_path):
    root = make_campaign(tmp_path / "data")
    (root / "mono" / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        crossover.analyze([root], tmp_path / "result")
