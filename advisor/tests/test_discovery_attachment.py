import json

import pytest

from advisor.discovery_attachment import summarize


def test_upstream_inventory_preserves_sidecar_status_without_native_secrets():
    source = {"timestamp": "2026-11-04", "url": "https://secret.example/?token=secret",
              "errors": [], "components": [{"metadata": {"kind": "Pod", "name": "decode",
              "namespace": "models", "annotations": {"secret": "secret"}}, "tool": "vllm",
              "native": {"role": "decode", "node": "n1", "environment": [{"value": "secret"}],
              "pod": {"metadata": {"uid": "uid1"}, "spec": {"containers": [{"name": "vllm"}],
              "initContainers": [{"name": "proxy", "restartPolicy": "Always"}]},
              "status": {"containerStatuses": [{"name": "vllm", "restartCount": 0, "ready": True}],
              "initContainerStatuses": [{"name": "proxy", "restartCount": 1, "ready": True}]}}}}]}
    result = summarize(source, "a" * 64)
    assert result["status"] == "inventory_only"
    assert result["components"][0]["containers"][1]["restartable_sidecar"] is True
    assert result["components"][0]["containers"][1]["restart_count"] == 1
    assert "secret" not in json.dumps(result)


def test_discovery_errors_remain_partial_and_missing_errors_are_rejected():
    assert summarize({"components": [], "errors": ["forbidden"]}, "a" * 64)["status"] == "partial"
    with pytest.raises(ValueError):
        summarize({"components": []}, "a" * 64)
