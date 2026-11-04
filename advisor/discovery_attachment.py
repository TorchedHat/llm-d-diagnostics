#!/usr/bin/env python3
"""Summarize an llm-d stack-discovery JSON export without copying native secrets.

Discovery is an inventory, not proof of routing, isolation, model identity or suitability.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from .evidence import sha256_file
except ImportError:
    from evidence import sha256_file  # type: ignore[no-redef]


def _containers(pod):
    spec, status = pod.get("spec", {}), pod.get("status", {})
    result = []
    for field, statuses, init in (("containers", "containerStatuses", False),
                                  ("initContainers", "initContainerStatuses", True)):
        actual = {s.get("name"): s for s in status.get(statuses, []) if isinstance(s, dict)}
        for item in spec.get(field, []):
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            observed = actual.get(item["name"], {})
            result.append({"name": item["name"], "init": init,
                           "restartable_sidecar": init and item.get("restartPolicy") == "Always",
                           "image_id": observed.get("imageID"),
                           "restart_count": observed.get("restartCount"),
                           "ready": observed.get("ready")})
    return result


def summarize(discovery, source_sha256):
    """Consume the upstream default JSON export; retain only allowlisted metadata."""
    if not isinstance(discovery, dict) or not isinstance(discovery.get("components"), list):
        raise ValueError("expected llm-d stack-discovery JSON with components")
    errors = discovery.get("errors")
    if not isinstance(errors, list):
        raise ValueError("discovery errors are missing or malformed")
    components = []
    for component in discovery["components"]:
        if not isinstance(component, dict) or not isinstance(component.get("metadata"), dict):
            raise ValueError("discovery component has invalid metadata")
        meta = component["metadata"]
        if any(not isinstance(meta.get(k), str) or not meta[k] for k in ("kind", "namespace", "name")):
            raise ValueError("discovery component identity is incomplete")
        item = {"kind": meta["kind"], "namespace": meta["namespace"], "name": meta["name"],
                "tool": component.get("tool"), "tool_version": component.get("tool_version")}
        native = component.get("native")
        if component.get("tool") == "vllm" and isinstance(native, dict):
            pod = native.get("pod") if isinstance(native.get("pod"), dict) else {}
            metadata = pod.get("metadata") if isinstance(pod.get("metadata"), dict) else {}
            item.update({"role": native.get("role"), "node": native.get("node"),
                         "pod_uid": metadata.get("uid"), "containers": _containers(pod)})
        components.append(item)
    return {"schema_version": "llmd-diagnostics-discovery/v1", "source_sha256": source_sha256,
            "discovery_timestamp": discovery.get("timestamp"), "error_count": len(errors),
            "components": components, "status": "partial" if errors else "inventory_only",
            "limitations": ["Upstream discovery output is not independently verified",
                "Routing, image compatibility, GPU allocation and isolation are not established",
                "Inventory does not authorize a topology recommendation"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("discovery_json", help="Default JSON export from llm-d stack discovery")
    parser.add_argument("--output", required=True, help="New sanitized inventory JSON")
    args = parser.parse_args()
    try:
        path = Path(args.discovery_json)
        result = summarize(json.loads(path.read_text()), sha256_file(path))
        with Path(args.output).open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Saved {result['status']} discovery inventory to {args.output}")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"Cannot summarize discovery: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
