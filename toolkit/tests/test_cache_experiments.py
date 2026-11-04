"""Tests for exp10's KV capacity reading and pressure-test sizing."""

import io
import os
import sys
import unittest
from types import SimpleNamespace
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


class TestStreamedCacheTiming(unittest.TestCase):
    def test_distributions_use_streamed_first_token(self):
        rows = []
        writer = SimpleNamespace(write=rows.append)
        def result(ttft):
            return SimpleNamespace(ttft_ms=ttft, total_ms=500, status=200, error="",
                                   token_times=(ttft / 1000,), prompt_tokens=100, completion_tokens=1)
        samples = iter([result(200), result(200), result(100), result(210), result(200), result(110)])
        conn = SimpleNamespace(pod_name="decode", send_streaming=lambda *args: next(samples))
        with patch.object(exp10, "progress"), patch.object(exp10, "dot"):
            cold, warm, threshold = exp10.establish_distributions(conn, writer, n=2)
        self.assertEqual(cold, [200, 210])
        self.assertEqual(warm, [100, 110])
        self.assertEqual(threshold, 155)
        self.assertEqual(len(rows), 4)

    def test_missing_first_token_is_not_classified_as_a_hit(self):
        result = SimpleNamespace(status=200, error="", token_times=())
        with self.assertRaisesRegex(RuntimeError, "valid streamed first token"):
            exp10.require_streamed_ttft(result)

    def test_overlapping_cold_and_warm_samples_have_no_threshold(self):
        result = SimpleNamespace(ttft_ms=100, total_ms=150, status=200, error="",
                                 token_times=(.1,), prompt_tokens=100, completion_tokens=1)
        conn = SimpleNamespace(pod_name="decode", send_streaming=lambda *args: result)
        with patch.object(exp10, "progress"), patch.object(exp10, "dot"):
            with self.assertRaisesRegex(ValueError, "quartiles overlap"):
                exp10.establish_distributions(conn, SimpleNamespace(write=lambda _: None), n=2)


if __name__ == "__main__":
    unittest.main()
