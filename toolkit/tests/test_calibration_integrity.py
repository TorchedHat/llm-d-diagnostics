"""Calibration keeps partial results and avoids invalid single-chunk theta."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exp18_calibrate as cal


class Writer:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


def test_multi_chunk_injection_keeps_stall_without_theta(monkeypatch):
    monkeypatch.setattr(cal, "OVERLAP_DECODERS", 2)
    monkeypatch.setattr(cal, "OVERLAP_RUNS", 1)
    monkeypatch.setattr(cal, "OVERLAP_PROMPT_TOKENS", 99)
    monkeypatch.setattr(cal, "CHUNK_BUDGET", "100")
    monkeypatch.setattr(cal, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(cal, "dot", lambda: None)
    monkeypatch.setattr(cal.time, "sleep", lambda _: None)
    monkeypatch.setattr(cal.time, "monotonic", lambda: 10)
    decoder = SimpleNamespace(ok=True, token_times=(0.1, 0.19, 0.3, 0.6, 0.7),
                              token_counts=(1, 1, 1, 1, 1))
    injected = SimpleNamespace(ok=True, ttft_ms=400, prompt_tokens=99, status=200, error="")
    monkeypatch.setattr(cal, "_start_streams", lambda *args: ([], [(9.8, decoder), (9.8, decoder)]))
    monkeypatch.setattr(cal, "_prompt", lambda *args: "prompt")
    monkeypatch.setattr(cal, "_stream", lambda *args: injected)
    prefill = {"prefill_a_s_per_token": .001, "prefill_b_s_per_token2": 0,
               "prefill_t0_s": .001}
    decode = {"decode_weight_s": .01, "decode_per_request_s": .001,
              "decode_kv_s_per_token": 0}
    writer, warnings = Writer(), []
    result = cal.phase_overlap(writer, warnings, prefill, decode)
    assert result["overlap_theta"] is None
    assert result["overlap_mixed_step_s"] > 0
    assert any("theta unavailable" in message for message in warnings)
    assert {row["metric"] for row in writer.rows} == {"mixed_step", "decode_step"}


def test_skipped_phases_do_not_report_complete_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cal, "SKIP_PHASES", {"decode", "kv", "overlap"})
    monkeypatch.setattr(cal, "write_run_info", lambda *args: None)
    monkeypatch.setattr(cal, "print_config", lambda: None)
    monkeypatch.setattr(cal, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(cal, "phase_prefill", lambda *args: {"prefill_t0_s": .1})
    cal.main()
    result = json.loads((tmp_path / "calibration.json").read_text())
    assert result["status"] == "partial"
    assert result["phase_status"] == {"prefill": "ok", "decode": "skipped",
                                       "kv": "skipped", "overlap": "skipped"}
