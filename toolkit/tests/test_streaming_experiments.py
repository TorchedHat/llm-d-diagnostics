"""Tests for the streaming measurement path: load helpers, exp12, exp16."""

import csv
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from analyze import _steady_window, analyze_exp12, analyze_exp16
from exp12_tput_outlen import mean_itl_ms
from exp16_per_token_trace import token_rows

from client import RequestResult, batched, closed_loop, route_throughput, run_load
from schemas import ConfigThroughput, Exp12Row, Exp16Row, fields_for


def _result(token_times=(), ttft_ms=0.0, total_ms=0.0, completion_tokens=0, usage=0):
    return RequestResult(ttft_ms=ttft_ms, total_ms=total_ms, status=200,
                         prompt_tokens=100, completion_tokens=completion_tokens,
                         token_times=tuple(token_times), usage_completion_tokens=usage)


def _write(path, row_type, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields_for(row_type))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _run(analyzer, tmpdir):
    buf = io.StringIO()
    with redirect_stdout(buf):
        analyzer(tmpdir)
    return buf.getvalue()


def _exp12_row(cfg, run, total, itl, tokens=100, status=200, load_mode="batch"):
    return {
        "experiment": "exp12", "config": cfg, "prompt_tokens_target": 1000,
        "max_tokens": 100, "concurrency": 4, "run": run, "ttft_ms": 300.0,
        "total_ms": total, "status_code": status, "prompt_tokens_actual": 1000,
        "completion_tokens": tokens, "target": "d1", "streaming": 1,
        "load_mode": load_mode, "inflight_at_start": 0, "itl_mean_ms": itl,
        "server_completion_tokens": tokens, "error": "",
    }


class TestHelpers(unittest.TestCase):

    def test_mean_itl(self):
        r = _result(token_times=(0.1,), ttft_ms=100.0, total_ms=1090.0, completion_tokens=100)
        self.assertAlmostEqual(mean_itl_ms(r), 10.0)

    def test_mean_itl_prefers_server_count(self):
        r = _result(token_times=(0.1,), ttft_ms=100.0, total_ms=1090.0,
                    completion_tokens=50, usage=100)
        self.assertAlmostEqual(mean_itl_ms(r), 10.0)

    def test_mean_itl_undefined_without_streaming(self):
        r = _result(ttft_ms=1000.0, total_ms=1000.0, completion_tokens=100)
        self.assertEqual(mean_itl_ms(r), 0.0)

    def test_token_rows_first_gap_is_ttft(self):
        rows = list(token_rows(_result(token_times=(0.2, 0.21, 0.25))))
        self.assertEqual(rows[0], (0, 200.0, 200.0, ""))
        self.assertEqual(rows[1], (1, 210.0, 10.0, ""))
        self.assertEqual(rows[2], (2, 250.0, 40.0, ""))

    def test_token_rows_record_tokens_per_chunk(self):
        r = _result(token_times=(0.2, 0.21, 0.25))
        r = RequestResult(**{**r.__dict__, "token_counts": (1, 1, 3)})
        self.assertEqual([row[3] for row in token_rows(r)], [1, 1, 3])

    def test_route_alternates_for_2d(self):
        tags = [route_throughput(ConfigThroughput.DISAGG_2D, s)[2] for s in (1, 2)]
        self.assertEqual(tags, ["d1", "d2"])
        _url, hdrs, _ = route_throughput(ConfigThroughput.BASELINE, 2)
        self.assertIsNone(hdrs)
        self.assertEqual(route_throughput(ConfigThroughput.DISAGG_EPP, 1)[2], "epp")


class TestLoadModes(unittest.TestCase):

    def test_closed_loop_keeps_workers_busy(self):
        lock = threading.Lock()
        state = {"now": 0, "peak": 0}

        def fn(seq, inflight):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.01 * (1 + seq % 3))
            with lock:
                state["now"] -= 1
            return (seq, inflight)

        results = closed_loop(4, 20, fn)
        self.assertEqual(sorted(s for s, _ in results), list(range(1, 21)))
        self.assertEqual(state["peak"], 4)
        # After the first wave, every request starts with the other 3 in flight.
        steady = [i for s, i in results if s > 4]
        self.assertTrue(all(i == 3 for i in steady), steady)

    def test_closed_loop_reraises_and_stops(self):
        started = []

        def fn(seq, inflight):
            started.append(seq)
            if seq == 3:
                raise ValueError("boom")
            time.sleep(0.01)
            return seq

        with self.assertRaises(ValueError):
            closed_loop(2, 20, fn)
        self.assertLess(len(started), 20)

    def test_batched_reraises(self):
        def fn(seq, inflight):
            if seq == 2:
                raise ValueError("boom")
            return seq

        with self.assertRaises(ValueError):
            batched(3, 6, fn)

    def test_run_load_reaches_requested_concurrency(self):
        _, total = run_load("steady", 8, 5, lambda s, i: s)
        self.assertEqual(total, 24)
        results, total = run_load("batch", 8, 5, lambda s, i: (s, i))
        self.assertEqual(total, 8)
        self.assertEqual([i for _, i in results], list(range(8)))
        # Whole batches only: 10 requests at batch size 4 become 12.
        _, total = run_load("batch", 4, 10, lambda s, i: s)
        self.assertEqual(total, 12)
        with self.assertRaises(ValueError):
            run_load("open", 4, 4, lambda s, i: s)


class TestSteadyWindow(unittest.TestCase):

    def test_drops_first_and_last_wave(self):
        rows = [{"load_mode": "steady", "config": "BASELINE", "concurrency": "2",
                 "run": str(run)} for run in range(1, 9)]
        rows.append({"load_mode": "batch", "config": "BASELINE", "concurrency": "2", "run": "1"})
        kept, dropped = _steady_window(rows, lambda r: (r["config"], r["concurrency"]))
        self.assertEqual([r["run"] for r in kept if r["load_mode"] == "steady"],
                         ["3", "4", "5", "6"])
        self.assertEqual(dropped, 4)
        self.assertEqual(sum(1 for r in kept if r["load_mode"] == "batch"), 1)

    def test_failed_tail_does_not_shift_window(self):
        rows = [_exp12_row("BASELINE", run, 1000.0, 10.0, load_mode="steady")
                for run in range(1, 13)]
        for r in rows[-4:]:
            r["status_code"] = 500
        rows = [{k: str(v) for k, v in r.items()} for r in rows]
        kept, _ = _steady_window(rows, lambda r: (r["config"], r["concurrency"]))
        # Window is runs 5..8 (c=4, 12 requests) regardless of the failed tail.
        self.assertEqual([r["run"] for r in kept], ["5", "6", "7", "8"])


class TestExp12Analysis(unittest.TestCase):

    def test_overhead_and_itl_ratio(self):
        rows = [_exp12_row(cfg, run, total, itl)
                for cfg, total, itl in (("BASELINE", 1000.0, 15.0), ("DISAGG-2D", 1200.0, 7.0))
                for run in range(1, 6)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write(os.path.join(tmpdir, "exp12-results.csv"), Exp12Row, rows)
            out = _run(analyze_exp12, tmpdir)
        self.assertIn("DISAGG-2D: overhead 200ms", out)
        self.assertIn("ITL ratio BASELINE/config 2.14", out)

    def test_warns_when_outputs_stop_early(self):
        rows = [_exp12_row("BASELINE", run, 1000.0, 10.0, tokens=t)
                for run, t in enumerate((100, 40, 100), start=1)]
        with tempfile.TemporaryDirectory() as tmpdir:
            _write(os.path.join(tmpdir, "exp12-results.csv"), Exp12Row, rows)
            out = _run(analyze_exp12, tmpdir)
        self.assertIn("1/3 requests stopped before max_tokens", out)

    def test_reads_old_non_streaming_csv(self):
        old_fields = ["experiment", "config", "prompt_tokens_target", "max_tokens",
                      "concurrency", "run", "ttft_ms", "total_ms", "status_code",
                      "prompt_tokens_actual", "completion_tokens", "target", "error"]
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "exp12-results.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=old_fields)
                w.writeheader()
                for run in range(1, 4):
                    w.writerow({"experiment": "exp12", "config": "BASELINE",
                                "prompt_tokens_target": 500, "max_tokens": 20,
                                "concurrency": 8, "run": run, "ttft_ms": 900.0,
                                "total_ms": 900.0, "status_code": 200,
                                "prompt_tokens_actual": 500, "completion_tokens": 20,
                                "target": "d1", "error": ""})
            out = _run(analyze_exp12, tmpdir)
        self.assertIn("Non-streaming data", out)
        self.assertIn("BASELINE", out)


class TestExp16Analysis(unittest.TestCase):

    def _rows(self, cfg, gaps_per_request, concurrency=4):
        rows = []
        for run, gaps in enumerate(gaps_per_request, start=1):
            elapsed = 0.0
            for idx, gap in enumerate(gaps):
                elapsed += gap
                rows.append({
                    "experiment": "exp16", "config": cfg, "load_mode": "batch",
                    "prompt_tokens_target": 1000, "max_tokens": len(gaps),
                    "concurrency": concurrency, "run": run, "inflight_at_start": 0,
                    "token_idx": idx, "elapsed_ms": elapsed, "gap_ms": gap,
                    "server_completion_tokens": len(gaps), "status_code": 200,
                    "target": "d1", "error": "",
                })
        return rows

    def test_multi_token_chunks_are_reported_not_counted_short(self):
        rows = self._rows("BASELINE", [[300.0] + [12.0] * 9 for _ in range(3)])
        for row in rows:
            row["tokens_in_chunk"] = 1
        # Request 1: its last two tokens arrive in one chunk (10 tokens, 9 chunks).
        rows = [r for r in rows if not (r["run"] == 1 and r["token_idx"] == 9)]
        rows[8]["tokens_in_chunk"] = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            _write(os.path.join(tmpdir, "exp16-results.csv"), Exp16Row, rows)
            out = _run(analyze_exp16, tmpdir)
        self.assertIn("1/3 requests had chunks carrying several tokens", out)
        self.assertNotIn("stopped before max_tokens", out)
        self.assertNotIn("empty text", out)

    def test_longest_pause_and_failed_requests(self):
        # Monolithic: one 400 ms stall per request; P/D: flat 12 ms gaps.
        mono = [[300.0] + [12.0] * 20 + [400.0] + [12.0] * 20 for _ in range(5)]
        pd = [[900.0] + [12.0] * 41 for _ in range(5)]
        rows = self._rows("BASELINE", mono) + self._rows("DISAGG-2D", pd)
        rows.append({
            "experiment": "exp16", "config": "DISAGG-2D", "load_mode": "batch",
            "prompt_tokens_target": 1000, "max_tokens": 42, "concurrency": 4, "run": 99,
            "inflight_at_start": 0, "token_idx": -1, "elapsed_ms": 5000.0, "gap_ms": 0,
            "server_completion_tokens": 0, "status_code": 500, "target": "d1",
            "error": "boom",
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            _write(os.path.join(tmpdir, "exp16-results.csv"), Exp16Row, rows)
            out = _run(analyze_exp16, tmpdir)
        self.assertIn("1 failed requests, 10 complete", out)
        self.assertNotIn("carrying several tokens", out)
        baseline_line = next(l for l in out.splitlines() if l.strip().startswith("BASELINE |"))
        pd_line = next(l for l in out.splitlines() if l.strip().startswith("DISAGG-2D |"))
        self.assertIn("400.0ms", baseline_line)   # longest pause per request
        self.assertNotIn("400.0ms", pd_line)


if __name__ == "__main__":
    unittest.main()
