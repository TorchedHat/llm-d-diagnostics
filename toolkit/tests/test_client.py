"""Tests for client.py utilities and schemas.py TypedCSVWriter.

Tests build_prompt, TypedCSVWriter, and write_run_info.
"""

import json
import os
import runpy
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from typing import TypedDict

import client
from client import RequestResult, build_prompt
from schemas import TypedCSVWriter


class _TestRow(TypedDict):
    a: str
    b: str
    c: str


class _ThreadRow(TypedDict):
    id: str
    value: str


class TestBuildPrompt(unittest.TestCase):

    def test_minimum_one_rep(self):
        prompt = build_prompt(1)
        self.assertGreater(len(prompt), 0)

    def test_scales_with_tokens(self):
        p10 = build_prompt(10)
        p100 = build_prompt(100)
        self.assertGreater(len(p100), len(p10))

    def test_zero_tokens(self):
        prompt = build_prompt(0)
        # Should produce at least 1 rep
        self.assertGreater(len(prompt), 0)

    def test_approximate_token_count(self):
        """Each BASE_SENTENCE rep is ~10 tokens. Check proportionality."""
        p50 = build_prompt(50)
        p100 = build_prompt(100)
        # 100-token prompt should be roughly 2x the 50-token prompt
        ratio = len(p100) / len(p50)
        self.assertGreater(ratio, 1.5)
        self.assertLess(ratio, 2.5)


class TestBuildPromptCacheBust(unittest.TestCase):
    """Cache busting must produce unique prompts across configs and runs.

    The exp12 bug: cache_bust=(max_tokens, run_num) omitted config_name,
    so BASELINE and DISAGG got identical prompts, causing prefix cache
    contamination. These tests encode the invariant that prevents that.
    """

    def test_different_cache_bust_produces_different_prompts(self):
        p1 = build_prompt(100, cache_bust=("BASELINE", 1))
        p2 = build_prompt(100, cache_bust=("DISAGG-1D", 1))
        self.assertNotEqual(p1, p2)

    def test_same_cache_bust_produces_same_prompt(self):
        p1 = build_prompt(100, cache_bust=("BASELINE", 1))
        p2 = build_prompt(100, cache_bust=("BASELINE", 1))
        self.assertEqual(p1, p2)

    def test_different_runs_produce_different_prompts(self):
        p1 = build_prompt(100, cache_bust=("BASELINE", 1))
        p2 = build_prompt(100, cache_bust=("BASELINE", 2))
        self.assertNotEqual(p1, p2)

    def test_cache_bust_differs_from_non_busted(self):
        p_plain = build_prompt(100)
        p_busted = build_prompt(100, cache_bust=("config", 1))
        self.assertNotEqual(p_plain, p_busted)

    def test_three_config_uniqueness(self):
        """Simulates the actual exp11/12/13 pattern: 3 configs, same run."""
        configs = ["BASELINE", "DISAGG-1D", "DISAGG-2D"]
        prompts = [build_prompt(100, cache_bust=(c, 1)) for c in configs]
        self.assertEqual(len(set(prompts)), 3)


class TestRequestResult(unittest.TestCase):

    def test_frozen(self):
        r = RequestResult(ttft_ms=1.0, total_ms=2.0, status=200,
                          prompt_tokens=10, completion_tokens=5)
        with self.assertRaises(AttributeError):
            r.status = 500

    def test_ok_success(self):
        r = RequestResult(ttft_ms=1.0, total_ms=2.0, status=200,
                          prompt_tokens=10, completion_tokens=5)
        self.assertTrue(r.ok)

    def test_ok_http_error(self):
        r = RequestResult(ttft_ms=1.0, total_ms=2.0, status=500,
                          prompt_tokens=0, completion_tokens=0)
        self.assertFalse(r.ok)

    def test_ok_exception(self):
        r = RequestResult(ttft_ms=0, total_ms=1.0, status=0,
                          prompt_tokens=0, completion_tokens=0,
                          error="connection refused")
        self.assertFalse(r.ok)

    def test_token_times_is_tuple(self):
        r = RequestResult(ttft_ms=1.0, total_ms=2.0, status=200,
                          prompt_tokens=10, completion_tokens=3,
                          token_times=(0.1, 0.2, 0.3))
        self.assertIsInstance(r.token_times, tuple)
        self.assertEqual(len(r.token_times), 3)

    def test_default_token_times_empty_tuple(self):
        r = RequestResult(ttft_ms=1.0, total_ms=2.0, status=200,
                          prompt_tokens=10, completion_tokens=0)
        self.assertEqual(r.token_times, ())


class TestTypedCSVWriter(unittest.TestCase):

    def test_write_and_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            writer = TypedCSVWriter(path, _TestRow)
            writer.write({"a": "1", "b": "2", "c": "3"})
            writer.write({"a": "4", "b": "5", "c": "6"})
            writer.close()

            with open(path) as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 3)  # header + 2 rows
            self.assertIn("a,b,c", lines[0])
            self.assertIn("1,2,3", lines[1])
        finally:
            os.unlink(path)

    def test_schema_mismatch_missing_key(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            writer = TypedCSVWriter(path, _TestRow)
            with self.assertRaises(ValueError) as cm:
                writer.write({"a": "1", "b": "2"})  # missing "c"
            self.assertIn("missing", str(cm.exception))
            writer.close()
        finally:
            os.unlink(path)

    def test_schema_mismatch_extra_key(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            writer = TypedCSVWriter(path, _TestRow)
            with self.assertRaises(ValueError) as cm:
                writer.write({"a": "1", "b": "2", "c": "3", "d": "4"})
            self.assertIn("extra", str(cm.exception))
            writer.close()
        finally:
            os.unlink(path)

    def test_thread_safety(self):
        """Multiple threads writing should not corrupt data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name

        try:
            writer = TypedCSVWriter(path, _ThreadRow)

            def write_rows(start, count):
                for i in range(start, start + count):
                    writer.write({"id": str(i), "value": str(i * 10)})

            threads = [threading.Thread(target=write_rows, args=(i * 100, 50))
                       for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            writer.close()

            with open(path) as f:
                lines = f.readlines()

            # header + 200 rows (4 threads * 50 each)
            self.assertEqual(len(lines), 201)
        finally:
            os.unlink(path)


class TestWriteRunInfo(unittest.TestCase):

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "run-info.json")

            info = {"toolkit": {"model": "test"}, "experiments": {}}
            info["experiments"]["exp1"] = {"started_at": "2026-01-01T00:00:00"}
            with open(filepath, "w") as f:
                json.dump(info, f)

            with open(filepath) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["toolkit"]["model"], "test")
            self.assertIn("exp1", loaded["experiments"])

    def test_merge_experiments(self):
        """Second experiment call should add to existing file, not overwrite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "run-info.json")

            info = {
                "toolkit": {"model": "test"},
                "experiments": {
                    "exp1": {"started_at": "2026-01-01T00:00:00"}
                }
            }
            with open(filepath, "w") as f:
                json.dump(info, f)

            with open(filepath) as f:
                existing = json.load(f)
            existing["experiments"]["exp2"] = {"started_at": "2026-01-01T01:00:00"}
            with open(filepath, "w") as f:
                json.dump(existing, f)

            with open(filepath) as f:
                loaded = json.load(f)
            self.assertIn("exp1", loaded["experiments"])
            self.assertIn("exp2", loaded["experiments"])

    def test_records_toolkit_commit(self):
        # The two tests above only exercise json; this one calls write_run_info.
        with tempfile.TemporaryDirectory() as tmpdir, \
                patch.object(client, "DATA_DIR", tmpdir), \
                patch.object(client, "detect_transport", return_value="test"):
            with patch.dict(os.environ, {"TOOLKIT_COMMIT": "abc123def456"}):
                client.write_run_info("exp1", {"arm": "A"})
            with open(os.path.join(tmpdir, "run-info.json")) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["toolkit"]["toolkit_commit"], "abc123def456")
            self.assertEqual(loaded["experiments"]["exp1"]["arm"], "A")
        # Without run.sh, from a checkout: the checkout's own commit.
        with patch.dict(os.environ, {"TOOLKIT_COMMIT": ""}):
            commit = client.toolkit_commit()
        self.assertRegex(commit, r"^([0-9a-f]{12}(-dirty)?|unknown)$")


class TestRouting(unittest.TestCase):

    def test_epp_removes_manual_target_case_insensitively(self):
        url = "http://router/v1/completions"
        extra = {"X-Prefiller-Host-Port": "prefill:8000", "Authorization": "test"}
        with patch.object(client, "EPP_URL", url):
            headers = client.request_headers(url + "/", extra)
        self.assertEqual(headers, {"Content-Type": "application/json", "Authorization": "test"})
        self.assertIn("X-Prefiller-Host-Port", extra)

    def test_manual_target_is_preserved(self):
        extra = {"x-prefiller-host-port": "prefill:8000"}
        with patch.object(client, "EPP_URL", "http://router/v1/completions"):
            headers = client.request_headers("http://decode:8000/v1/completions", extra)
        self.assertEqual(headers["x-prefiller-host-port"], "prefill:8000")

    def test_pinned_epp_merges_headers_without_manual_target(self):
        url = "http://router/v1/completions"
        conn = client.PinnedConnection(url, extra_headers={"x-prefiller-host-port": "p1:8000"})
        with patch.object(client, "EPP_URL", url):
            headers = conn._merge_headers({"X-Prefiller-Host-Port": "p2:8000"})
        self.assertEqual(headers, {"Content-Type": "application/json"})

    def test_send_disagg_uses_epp_without_manual_header(self):
        url = "http://router/v1/completions"
        with patch.object(client, "ROUTING_MODE", "epp"), patch.object(client, "EPP_URL", url), \
                patch.object(client, "send_request") as send:
            client.send_disagg("http://decode/v1/completions", "hello", 5)
        send.assert_called_once_with(url, "hello", 5, extra_headers={})

    def test_epp_mode_keeps_comparison_endpoints_manual(self):
        overrides = {"ROUTING_MODE": "epp", "EPP_URL": "http://router/v1/completions"}
        with patch.dict(os.environ, overrides):
            for name in ("DISAGG_URL", "DISAGG_D1_URL", "DISAGG_D2_URL"):
                os.environ.pop(name, None)
            config = runpy.run_path(client.__file__)
        for name in ("DISAGG_URL", "DISAGG_D1_URL", "DISAGG_D2_URL"):
            self.assertNotEqual(config[name], overrides["EPP_URL"])


if __name__ == "__main__":
    unittest.main()
