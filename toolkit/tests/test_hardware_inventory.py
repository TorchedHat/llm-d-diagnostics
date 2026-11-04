"""Exercise pod identity checks and partial hardware inventory."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hardware_profile as profile

BASE = {"name": "pod", "uid": "uid", "node": "node", "ip": "10.0.0.1", "ready": True,
        "image_ids": {"vllm": "sha256:image"}, "restart_counts": {"vllm": 0}}


def test_collect_preserves_unavailable_data_and_checks_identity(monkeypatch):
    observed = []

    def metadata(*args):
        observed.append(args)
        return copy.deepcopy(BASE)

    monkeypatch.setattr(profile, "pod_metadata", metadata)
    monkeypatch.setattr(profile, "run", lambda *args, **kwargs: json.dumps({"gpus": {"status": None},
                                                                          "rdma_ports": []}))
    monkeypatch.setattr(profile, "cluster_server", lambda _: "https://cluster")
    data = profile.collect("oc", "ns", ["pod"], "vllm")
    assert len(observed) == 2
    assert data["pods"][0]["inventory"]["gpus"]["status"] is None
    assert "not a transport witness" in data["scope"]


def test_restarted_pod_invalidates_inventory(monkeypatch):
    before, after = copy.deepcopy(BASE), copy.deepcopy(BASE)
    after["restart_counts"]["vllm"] = 1
    snapshots = iter([before, after])
    monkeypatch.setattr(profile, "pod_metadata", lambda *args: next(snapshots))
    monkeypatch.setattr(profile, "run", lambda *args: "{}")
    with pytest.raises(ValueError, match="changed while collecting"):
        profile.collect("oc", "ns", ["pod"], "vllm")


def test_incomplete_identity_and_invalid_probe_are_rejected(monkeypatch):
    missing = copy.deepcopy(BASE)
    missing["image_ids"] = {}
    monkeypatch.setattr(profile, "pod_metadata", lambda *args: missing)
    with pytest.raises(ValueError, match="complete identity"):
        profile.collect("oc", "ns", ["pod"], "vllm")
    monkeypatch.setattr(profile, "pod_metadata", lambda *args: copy.deepcopy(BASE))
    monkeypatch.setattr(profile, "run", lambda *args: "[]")
    with pytest.raises(ValueError, match="invalid inventory"):
        profile.collect("oc", "ns", ["pod"], "vllm")


def test_output_creation_never_overwrites(tmp_path):
    path = tmp_path / "profile.json"
    profile.save(path, {"original": True})
    with pytest.raises(FileExistsError):
        profile.save(path, {"original": False})
    assert json.loads(path.read_text()) == {"original": True}
