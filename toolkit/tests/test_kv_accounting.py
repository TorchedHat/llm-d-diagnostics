"""Tests for cache-busted prompts and KV accounting (kv_sources.py, analyze.py)."""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import analyze
from kv_sources import delta, summary_line

import client
from client import build_prompt


class TestPromptNonce(unittest.TestCase):
    def test_nonce_changes_busted_prompts(self):
        """Toolkit runs against the same pods must not reuse each other's prompts."""
        with patch.object(client, "PROMPT_NONCE", "run-a"):
            a = build_prompt(200, cache_bust=("exp1", 200, "BASELINE", 1))
        with patch.object(client, "PROMPT_NONCE", "run-b"):
            b = build_prompt(200, cache_bust=("exp1", 200, "BASELINE", 1))
        self.assertNotEqual(a, b)

    def test_nonce_reproduces_prompts(self):
        with patch.object(client, "PROMPT_NONCE", "fixed"):
            a = build_prompt(200, cache_bust=("exp5", 200, "D-disaggregated", 3))
            b = build_prompt(200, cache_bust=("exp5", 200, "D-disaggregated", 3))
        self.assertEqual(a, b)

    def test_unbusted_prompt_ignores_nonce(self):
        """Warm-up and cache experiments rely on the fixed prompt."""
        with patch.object(client, "PROMPT_NONCE", "x"):
            a = build_prompt(100)
        with patch.object(client, "PROMPT_NONCE", "y"):
            b = build_prompt(100)
        self.assertEqual(a, b)

    def test_each_experiment_records_its_nonce(self):
        """run-info.json must say which nonce produced each experiment's
        prompts, or PROMPT_NONCE cannot reproduce them."""
        with tempfile.TemporaryDirectory() as d, patch.object(client, "DATA_DIR", d):
            with patch.object(client, "PROMPT_NONCE", "n1"):
                client.write_run_info("exp1")
            with patch.object(client, "PROMPT_NONCE", "n2"):
                client.write_run_info("exp5")
            with open(os.path.join(d, "run-info.json")) as f:
                experiments = json.load(f)["experiments"]
        self.assertEqual(experiments["exp1"]["prompt_nonce"], "n1")
        self.assertEqual(experiments["exp5"]["prompt_nonce"], "n2")

    def test_exp5_configs_get_distinct_prompts(self):
        """The exp5 bug: one prompt per run, shared by A/B/C/D, so C and D hit
        the cache filled by A and B."""
        keys = [("exp5", 1000, cfg, 1) for cfg in
                ("A-prefill-direct", "B-decode-direct", "C-sidecar-only", "D-disaggregated")]
        prompts = {build_prompt(1000, cache_bust=k) for k in keys}
        self.assertEqual(len(prompts), 4)

    def test_no_shared_prefix_block(self):
        """Distinct keys must differ within the first KV block (16 tokens),
        or the prompts share a cached prefix block."""
        a = build_prompt(500, cache_bust=("exp1", 500, "BASELINE", 1)).split()
        b = build_prompt(500, cache_bust=("exp1", 500, "BASELINE", 2)).split()
        self.assertNotEqual(a[:8], b[:8])


def snap(pods, t="t"):
    return {"taken_at": t, "pods": pods}


def pod(role, compute=0, hit=0, ext=0, nbytes=0, transfers=0):
    return {"role": role, "node": "n",
            "sources": {"local_compute": compute, "local_cache_hit": hit,
                        "external_kv_transfer": ext},
            "nixl_bytes": nbytes, "nixl_transfers": transfers}


KIB256 = 256 * 1024


class TestDelta(unittest.TestCase):
    def test_sums_roles_and_derives_ratios(self):
        before = snap({"p": pod("prefill", compute=100),
                       "d1": pod("decode", ext=10, nbytes=10 * KIB256),
                       "d2": pod("decode")})
        after = snap({"p": pod("prefill", compute=2100),
                      "d1": pod("decode", ext=1010, nbytes=1010 * KIB256),
                      "d2": pod("decode", hit=50, ext=950, nbytes=950 * KIB256)})
        d = delta(before, after)
        self.assertEqual(d["roles"]["prefill"]["local_compute"], 2000)
        self.assertEqual(d["roles"]["prefill"]["cache_hit_fraction"], 0)
        dec = d["roles"]["decode"]
        self.assertEqual(dec["external_kv_transfer"], 1950)
        self.assertAlmostEqual(dec["cache_hit_fraction"], 50 / 2000)
        self.assertEqual(dec["nixl_bytes_per_external_token"], KIB256)
        self.assertIn("256 KiB/token", summary_line(d))

    def test_restarted_pods_are_excluded(self):
        before = snap({"old": pod("decode", ext=5), "p": pod("prefill")})
        after = snap({"new": pod("decode", ext=7), "p": pod("prefill", compute=7)})
        d = delta(before, after)
        self.assertEqual(d["skipped_pods"], ["new", "old"])
        self.assertNotIn("decode", d["roles"])

    def test_no_traffic(self):
        d = delta(snap({"p": pod("prefill")}), snap({"p": pod("prefill")}))
        self.assertIsNone(d["roles"]["prefill"]["cache_hit_fraction"])
        self.assertEqual(summary_line(d), "no prompt tokens recorded")


class TestAnalyzeAccounting(unittest.TestCase):
    def write(self, d, exp, roles):
        os.makedirs(os.path.join(d, "kv-sources"), exist_ok=True)
        with open(os.path.join(d, "kv-sources", f"{exp}.json"), "w") as f:
            json.dump({"roles": roles}, f)

    @staticmethod
    def roles(pf_hit=0.0, dc_hit=0.0, bpt=KIB256):
        return {"prefill": {"cache_hit_fraction": pf_hit},
                "decode": {"cache_hit_fraction": dc_hit, "external_kv_transfer": 1000,
                           "nixl_bytes_per_external_token": bpt}}

    def test_clean_run_has_no_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "exp1", self.roles())
            self.write(d, "exp5", self.roles())
            lines = analyze.kv_accounting(d)
        self.assertFalse([line for line in lines if "WARNING" in line])

    def test_cache_reuse_is_flagged_outside_cache_experiments(self):
        """The TCP-baseline signature: decode served most prompts from cache."""
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "exp5", self.roles(dc_hit=0.9))
            self.write(d, "exp8", self.roles(dc_hit=0.9))
            warnings = [line for line in analyze.kv_accounting(d) if "WARNING" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("exp5", warnings[0])
        self.assertIn("90% of decode prompt tokens", warnings[0])

    def test_inconsistent_bytes_per_token_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            for exp in ("exp1", "exp2", "exp3"):
                self.write(d, exp, self.roles())
            self.write(d, "exp6", self.roles(bpt=16 * 1024))
            warnings = [line for line in analyze.kv_accounting(d) if "WARNING" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("exp6: 16 KiB per NIXL token vs 256 KiB", warnings[0])

    def test_before_after_snapshots_are_not_rows(self):
        with tempfile.TemporaryDirectory() as d:
            self.write(d, "exp1", self.roles())
            self.write(d, "exp1.before", {"pods": {}})
            lines = analyze.kv_accounting(d)
        self.assertEqual(sum(1 for line in lines if line.strip().startswith("exp1")), 1)

    def test_no_accounting_data(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(analyze.kv_accounting(d), [])


if __name__ == "__main__":
    unittest.main()
