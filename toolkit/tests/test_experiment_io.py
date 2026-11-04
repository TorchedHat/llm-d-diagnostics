"""Regression coverage for the shared explicit-experiment client."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_io as io


def snapshot(start, value):
    lines = [f"vllm:test_total {value}"]
    if start is not None:
        lines.append(f"process_start_time_seconds {start}")
    return {"pods": {"server": lines}}


def test_counter_delta_requires_stable_process_identity():
    assert io.delta(snapshot(10, 1), snapshot(10, 4), "server", "vllm:test_total") == 3
    with pytest.raises(ValueError, match="missing or ambiguous process identity"):
        io.delta(snapshot(None, 1), snapshot(None, 4), "server", "vllm:test_total")
    with pytest.raises(ValueError, match="process restarted"):
        io.delta(snapshot(10, 1), snapshot(11, 4), "server", "vllm:test_total")
    with pytest.raises(ValueError, match="counter reset"):
        io.delta(snapshot(10, 4), snapshot(10, 1), "server", "vllm:test_total")
    with pytest.raises(ValueError, match="invalid process identity"):
        io.delta(snapshot("NaN", 1), snapshot("NaN", 4), "server", "vllm:test_total")
    duplicate = snapshot(10, 4)
    duplicate["pods"]["server"].append("process_start_time_seconds 10")
    with pytest.raises(ValueError, match="ambiguous"):
        io.delta(snapshot(10, 1), duplicate, "server", "vllm:test_total")


class Response:
    status = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def event(value):
    return ("data: " + json.dumps(value) + "\n").encode()


def test_incomplete_stream_is_not_retried(monkeypatch):
    calls = []

    def http(*args, **kwargs):
        calls.append(1)
        return Response([event({"choices": [{"token_ids": [2, 3]}],
                                "usage": {"prompt_tokens": 1, "completion_tokens": 2}})])

    monkeypatch.setattr(io, "http", http)
    row = io.send({"model": "model", "nonce": "x", "timeout": 1},
                  {"url": "http://example"}, [1], 0, "salt", 2)
    assert calls == [1]
    assert not row["success"] and "missing stream completion" in row["error"]
    assert row["itl_mean_ms"] is None and row["itl_p99_ms"] is None


def test_first_coalesced_chunk_cannot_provide_mean_itl(monkeypatch):
    lines = [event({"choices": [{"token_ids": [2, 3]}]}),
             event({"usage": {"prompt_tokens": 1, "completion_tokens": 2}}),
             b"data: [DONE]\n"]
    monkeypatch.setattr(io, "http", lambda *args, **kwargs: Response(lines))
    row = io.send({"model": "model", "nonce": "x", "timeout": 1},
                  {"url": "http://example"}, [1], 0, "salt", 2)
    assert row["success"]
    assert row["ttft_ms"] is not None and row["itl_mean_ms"] is None
    assert row["itl_p99_ms"] is None and not row["single_token_chunks"]
