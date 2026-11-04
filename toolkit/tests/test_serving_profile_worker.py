"""Verify the streaming worker keeps incomplete responses as failures."""

import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import serving_profile_worker as worker


class FakeResponse:
    status = 200

    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_stream_requires_done_and_exact_usage(monkeypatch):
    events = []
    monkeypatch.setattr(worker, "emit", events.append)
    chunk = b'data: {"choices":[{"text":"hello"}]}\n'
    usage = b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n'
    config = {"mono_urls": ["http://example/v1/completions"], "model": "m",
              "output_tokens": 1, "timeout_seconds": 1}
    monkeypatch.setattr(worker, "post", lambda *a, **k: FakeResponse([chunk, usage, b"data: [DONE]\n"]))
    assert worker.one(config, "MONO", 1, 0, 0, [1, 2])["ok"]
    monkeypatch.setattr(worker, "post", lambda *a, **k: FakeResponse([chunk, usage]))
    assert not worker.one(config, "MONO", 1, 0, 0, [1, 2])["ok"]
    assert len(events) == 2


def test_http_error_status_is_recorded_without_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "emit", lambda value: None)
    config = {"mono_urls": ["http://example/v1/completions"], "model": "m",
              "output_tokens": 1, "timeout_seconds": 1}

    def fail(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("http://example", 503, "unavailable", None, None)

    monkeypatch.setattr(worker, "post", fail)
    result = worker.one(config, "MONO", 1, 0, 0, [1, 2])
    assert result["status"] == 503 and not result["ok"] and len(calls) == 1


def test_prompts_are_repeatable_and_distinct_across_blocks():
    base = list(range(100))
    first = worker.make_prompts(base, 32, 8, 1, "c8-block0")
    assert first == worker.make_prompts(base, 32, 8, 1, "c8-block0")
    assert len({tuple(x) for x in first}) == 8
    second = worker.make_prompts(base, 32, 8, 1, "c8-block1")
    assert not {tuple(x) for x in first}.intersection(tuple(x) for x in second)


def test_pd_route_only_sends_manual_prefill_header_when_requested(monkeypatch):
    seen = []
    monkeypatch.setattr(worker, "emit", lambda value: None)
    def capture(url, payload, headers, timeout):
        seen.append(headers)
        return FakeResponse([b'data: {"choices":[{"text":"x"}]}\n',
                             b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1}}\n',
                             b'data: [DONE]\n'])
    monkeypatch.setattr(worker, "post", capture)
    config = {"pd_url": "http://router/v1/completions", "prefill_host": "prefill:8000",
              "model": "m", "output_tokens": 1, "timeout_seconds": 1}
    assert worker.one(config, "PD", 1, 0, 0, [1, 2])["ok"]
    config["pd_route"] = "epp"
    assert worker.one(config, "PD", 1, 0, 0, [1, 2])["ok"]
    assert seen == [{"x-prefiller-host-port": "prefill:8000"}, {}]
    config["pd_route"] = "unsupported"
    with pytest.raises(ValueError):
        worker.one(config, "PD", 1, 0, 0, [1, 2])
