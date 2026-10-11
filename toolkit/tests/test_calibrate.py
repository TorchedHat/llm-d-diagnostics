"""Tests for exp18 (one-GPU calibration) helpers."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exp18_calibrate import (
    fit_linear,
    fit_poly,
    overlap_theta,
    parse_kv_capacity,
    steady_decode_gaps,
)


class TestFitPoly(unittest.TestCase):

    def test_recovers_quadratic(self):
        xs = [256, 512, 1024, 2048, 4096]
        ys = [0.02 + 1e-4 * x + 3e-9 * x * x for x in xs]
        (t0, a, b), r2 = fit_poly(xs, ys, 2)
        self.assertAlmostEqual(t0, 0.02, places=6)
        self.assertAlmostEqual(a, 1e-4, places=9)
        self.assertAlmostEqual(b, 3e-9, places=12)
        self.assertAlmostEqual(r2, 1.0, places=9)

    def test_recovers_line(self):
        (w, slope), _r2 = fit_poly([1, 2, 4, 8], [0.01 + 0.002 * b for b in (1, 2, 4, 8)], 1)
        self.assertAlmostEqual(w, 0.01, places=9)
        self.assertAlmostEqual(slope, 0.002, places=9)

    def test_needs_enough_points(self):
        with self.assertRaises(ValueError):
            fit_poly([1, 1, 2], [1.0, 1.0, 2.0], 2)


class TestFitLinear(unittest.TestCase):

    def test_separates_per_request_and_kv_terms(self):
        # step = 8 ms + B * (0.5 ms + 100 ns x ctx)
        points = [(b, c) for b in (1, 4, 8, 16) for c in (512, 2048)]
        rows = [[1.0, b, b * c] for b, c in points]
        ys = [0.008 + b * (0.0005 + 1e-7 * c) for b, c in points]
        (w, c0, k), r2 = fit_linear(rows, ys)
        self.assertAlmostEqual(w, 0.008, places=9)
        self.assertAlmostEqual(c0, 0.0005, places=9)
        self.assertAlmostEqual(k, 1e-7, places=12)
        self.assertAlmostEqual(r2, 1.0, places=9)


class TestSteadyDecodeGaps(unittest.TestCase):

    def test_window_is_all_decoding_none_finished(self):
        # Request A starts at 0: first token at 0.10, then every 0.01 s.
        # Request B starts at 0: first token at 0.20 (queued prefill), then every 0.01 s.
        a = [0.10 + 0.01 * i for i in range(30)]          # last token 0.39
        b = [0.20 + 0.01 * i for i in range(30)]          # last token 0.49
        gaps = steady_decode_gaps([(0.0, a), (0.0, b)])
        # Window [0.20, 0.39]: A contributes 19 gaps, B contributes 19 gaps.
        self.assertEqual(len(gaps), 38)
        self.assertTrue(all(abs(g - 0.01) < 1e-9 for g in gaps))

    def test_multi_token_chunk_is_shared_between_its_tokens(self):
        # B's tokens arrive every 10 ms, except one chunk carrying 2 tokens
        # 20 ms after the previous one: two 10 ms steps, not one 20 ms step.
        a = [0.10 + 0.01 * i for i in range(30)]
        b = [0.10 + 0.01 * i for i in range(30) if i != 15]
        counts = [1] * 29
        counts[15] = 2
        gaps = steady_decode_gaps([(0.0, a), (0.0, b, counts)])
        self.assertEqual(len(gaps), 2 * 29)
        self.assertTrue(all(abs(g - 0.01) < 1e-9 for g in gaps))
        # Without the counts the merged chunk reads as one 20 ms step.
        self.assertAlmostEqual(max(steady_decode_gaps([(0.0, a), (0.0, b)])), 0.02)

    def test_no_window_when_a_request_failed(self):
        self.assertEqual(steady_decode_gaps([(0.0, [0.1, 0.2]), (0.0, [])]), [])


class TestOverlapTheta(unittest.TestCase):

    def test_bounds(self):
        # Decode adds fully to the chunk: mixed = chunk + decode.
        self.assertAlmostEqual(overlap_theta(0.25, 0.20, 0.05), 0.0)
        # Decode is free inside the chunk: mixed = chunk.
        self.assertAlmostEqual(overlap_theta(0.20, 0.20, 0.05), 1.0)
        with self.assertRaises(ValueError):
            overlap_theta(0.2, 0.2, 0.0)


class TestParseKvCapacity(unittest.TestCase):

    def test_simulator_format(self):
        text = ('# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig.\n'
                'vllm:cache_config_info{block_size="16",num_gpu_blocks="1024"} 1\n')
        self.assertEqual(parse_kv_capacity(text), 16384)

    def test_vllm_format_with_more_labels(self):
        text = ('vllm:cache_config_info{block_size="16",cache_dtype="auto",engine="0",'
                'gpu_memory_utilization="0.9",num_gpu_blocks="8000",prefix_caching_hash_algo="sha256"} 1.0\n')
        self.assertEqual(parse_kv_capacity(text), 128000)

    def test_missing_metric(self):
        self.assertIsNone(parse_kv_capacity("vllm:num_requests_running 0\n"))


if __name__ == "__main__":
    unittest.main()
