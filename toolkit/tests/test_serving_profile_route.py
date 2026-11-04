import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import serving_profile as profile


def test_router_url_requires_an_uncredentialed_completions_endpoint():
    assert profile.valid_epp_url("http://router:8080/v1/completions")
    for invalid in ("http://user:secret@router/v1/completions", "http://router/v1/completions?token=x",
                    "http://router/v1/chat/completions", "file:///v1/completions", "http://router:99999/v1/completions"):
        assert not profile.valid_epp_url(invalid)


def test_epp_controller_tracks_router_and_detects_restart(tmp_path, monkeypatch):
    def info(name):
        return {"name": name, "uid": name + "-uid", "node": "node", "ip": "10.0.0.1", "ready": True,
                "image_ids": {"vllm": "sha"}, "restart_counts": {"vllm": 0}}
    pods = []
    for n in ("m1", "m2", "p", "d"):
        pod = info(n)
        pod["inventory"] = {"gpus": {"status": 0, "stdout": f"0, GPU-{n}, H100"}}
        pods.append(pod)
    monkeypatch.setattr(profile, "collect", lambda *args: {"pods": pods})
    monkeypatch.setattr(profile, "cluster_server", lambda *args: "cluster")
    monkeypatch.setattr(profile, "pod_exec", lambda *args: ["fake-oc"])
    observed = []
    def worker(command, stdout, stderr, check):
        observed.append(json.loads(command[-1]))
        prompts = [[1, i] for i in range(8)]
        for arm in ("MONO", "PD"):
            for number in (0, 1):
                stdout.write(json.dumps({"event": "request", "arm": arm, "number": number,
                                         "warmup": True, "ok": True}) + "\n")
        stdout.write(json.dumps({"event": "inputs", "concurrency": 1, "block": 0,
                                 "tokens": prompts}) + "\n")
        for arm in ("MONO", "PD"):
            for number, prompt in enumerate(prompts):
                stdout.write(json.dumps({"event": "request", "arm": arm, "concurrency": 1, "block": 0,
                    "number": number, "warmup": False, "ok": True,
                    "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest()}) + "\n")
            stdout.write(json.dumps({"event": "cell", "arm": arm, "concurrency": 1, "block": 0,
                                     "requests": 8, "success": 8, "wall_seconds": 1}) + "\n")
        stdout.write('{"event":"complete"}\n')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(profile.subprocess, "run", worker)
    calls = {"router": 0}
    def pod_metadata(cli, namespace, name):
        item = info(name)
        if name == "router":
            calls["router"] += 1
            if fail_restart and calls["router"] == 2:
                item["restart_counts"]["vllm"] = 1
        return item
    monkeypatch.setattr(profile, "pod_metadata", pod_metadata)
    def invoke(output):
        monkeypatch.setattr(sys, "argv", ["serving_profile.py", "--namespace", "ns", "--client-pod", "client",
            "--mono-pods", "m1,m2", "--prefill-pod", "p", "--decode-pod", "d", "--model", "m",
            "--output-dir", str(output), "--pd-route", "epp", "--epp-url", "http://router:8080/v1/completions",
            "--router-pod", "router", "--requests", "8", "--blocks", "1", "--concurrencies", "1",
            "--input-tokens", "2"])
        return profile.main()
    fail_restart = False
    output = tmp_path / "complete"
    assert invoke(output) == 0
    result = json.loads((output / "result.json").read_text())
    assert result["metadata"]["router"]["uid"] == "router-uid"
    assert observed[0]["pd_route"] == "epp" and observed[0]["pd_url"] == "http://router:8080/v1/completions"
    events = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines()]
    missing = [event for event in events if not (event.get("event") == "request"
               and event.get("arm") == "PD" and event.get("warmup") is False and event.get("number") == 0)]
    with pytest.raises(ValueError, match="missing serving cells"):
        profile.validate_events(missing, [1], 1, 8, 2)
    corrupted = [dict(event) for event in events]
    next(event for event in corrupted if event.get("event") == "request" and event.get("warmup") is False)["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="paired input"):
        profile.validate_events(corrupted, [1], 1, 8, 2)
    calls["router"] = 0
    fail_restart = True
    output = tmp_path / "restarted"
    assert invoke(output) == 1
    assert "identity" in json.loads((output / "error.json").read_text())["error"]
    assert not (output / "result.json").exists()
