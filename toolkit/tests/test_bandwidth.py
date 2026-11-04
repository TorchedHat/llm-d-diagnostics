"""Validate bandwidth geometry, rank correlation and attribution limits."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bandwidth


def test_tp_requires_per_rank_trace_and_distinct_metrics():
    config = json.loads((Path(__file__).resolve().parents[2] / "examples/bandwidth.json").read_text())
    config["tensor_parallel_size"] = 2
    config.pop("rank_trace_url", None)
    with pytest.raises(ValueError, match="TP > 1 requires rank_trace_url"):
        bandwidth.validate(config)
    config["rank_trace_url"] = "http://trace:8000/trace"
    assert bandwidth.validate(config)["tensor_parallel_size"] == 2
    config["decode"]["metrics_url"] = config["prefill"]["metrics_url"]
    with pytest.raises(ValueError, match="distinct labels and direct metrics"):
        bandwidth.validate(config)


def test_rank_critical_path_uses_correlated_clocks_and_requests():
    config = {"tensor_parallel_size": 2}
    row = {"server_request_id": "req"}
    ranks = [
        {"rank": 0, "clock_domain": "node-a", "request_id": "req-0",
         "posted_monotonic_ns": 1_000_000, "observed_done_monotonic_ns": 4_000_000,
         "bytes": 100, "nixl_xfer_us": 2000, "nixl_post_us": 10},
        {"rank": 1, "clock_domain": "node-a", "request_id": "req-1",
         "posted_monotonic_ns": 2_000_000, "observed_done_monotonic_ns": 6_000_000,
         "bytes": 100, "nixl_xfer_us": 4000, "nixl_post_us": 20},
    ]
    out = bandwidth.rank_measurement(config, row, ranks)
    assert out["nixl_critical_path_ms"] == 5
    assert out["rank_xfer_ms_sum"] == 6 and out["rank_payload_bytes"] == 200
    wrong_clock = [dict(item) for item in ranks]
    wrong_clock[1]["clock_domain"] = "node-b"
    with pytest.raises(ValueError, match="clocks differ"):
        bandwidth.rank_measurement(config, row, wrong_clock)
    wrong_request = [dict(item) for item in ranks]
    wrong_request[1]["request_id"] = "other"
    with pytest.raises(ValueError, match="different response request"):
        bandwidth.rank_measurement(config, row, wrong_request)
    duplicate = [dict(item) for item in ranks]
    duplicate[1]["rank"] = 0
    with pytest.raises(ValueError, match="missing or duplicate"):
        bandwidth.rank_measurement(config, row, duplicate)


def test_payload_fit_is_descriptive_and_rejects_unphysical_interpretation():
    assert bandwidth.fit_payload([(1, 3)])["status"] == "insufficient distinct payload sizes"
    fit = bandwidth.fit_payload([(1, 4), (2, 6)])
    assert fit["status"] == "descriptive fit" and fit["physical_parameters"]
    assert fit["ms_per_GB"] == 2 and fit["effective_payload_GB_s"] == 500
    assert not bandwidth.fit_payload([(1, 6), (2, 4)])["physical_parameters"]
