"""Tests for exp10's KV capacity reading and pressure-test sizing."""

import io
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import exp10_kv_eviction as exp10

# Abridged from a vLLM 0.31 /metrics page (Qwen3-32B-FP8 on one H100).
METRICS = (
    '# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n'
    'vllm:cache_config_info{block_size="16",cache_dtype="auto",enable_prefix_caching="True",'
    'engine="0",gpu_memory_utilization="0.9",kv_cache_size_tokens="140400",'
    'num_gpu_blocks="8775"} 1.0\n'
)


def serve(text):
    return patch.object(exp10.urllib.request, "urlopen",
                        return_value=io.BytesIO(text.encode()))


class TestKvCapacity(unittest.TestCase):
    URL = "http://10.0.0.1:8001/v1/completions"

    def test_reads_kv_cache_size_tokens(self):
        with serve(METRICS) as urlopen:
            self.assertEqual(exp10.kv_capacity_tokens(self.URL), 140400)
        self.assertEqual(urlopen.call_args[0][0], "http://10.0.0.1:8001/metrics")

    def test_falls_back_to_blocks_times_block_size(self):
        with serve(METRICS.replace('kv_cache_size_tokens="140400",', "")):
            self.assertEqual(exp10.kv_capacity_tokens(self.URL), 8775 * 16)

    def test_unknown_when_metric_missing_or_unreachable(self):
        with serve("vllm:num_requests_running 0\n"):
            self.assertIsNone(exp10.kv_capacity_tokens(self.URL))
        with patch.object(exp10.urllib.request, "urlopen", side_effect=OSError):
            self.assertIsNone(exp10.kv_capacity_tokens(self.URL))


class TestPressureCount(unittest.TestCase):
    def test_auto_exceeds_capacity(self):
        with patch.object(exp10, "PRESSURE_PROMPTS_N", "auto"):
            n = exp10.pressure_prompt_count(140400)
        self.assertGreaterEqual(n * exp10.CACHE_PROMPT_TOKENS, 1.25 * 140400)

    def test_explicit_count_is_kept(self):
        with patch.object(exp10, "PRESSURE_PROMPTS_N", "20"), \
             patch.object(exp10, "progress") as progress:
            self.assertEqual(exp10.pressure_prompt_count(140400), 20)
        self.assertIn("cannot evict", progress.call_args[0][0])

    def test_unknown_capacity_falls_back(self):
        with patch.object(exp10, "PRESSURE_PROMPTS_N", "auto"), \
             patch.object(exp10, "progress"):
            self.assertEqual(exp10.pressure_prompt_count(None), exp10.PRESSURE_FALLBACK_N)


if __name__ == "__main__":
    unittest.main()
