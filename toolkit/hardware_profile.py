#!/usr/bin/env python3
"""Collect a read-only inventory from explicitly selected Kubernetes pods."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA_VERSION = 1


def run(args, timeout=30):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True).stdout


def cluster_server(cli):
    args = ([cli, "whoami", "--show-server"] if cli == "oc" else
            [cli, "config", "view", "--minify", "-o", "jsonpath={.clusters[0].cluster.server}"])
    server = run(args).strip()
    if not server:
        raise ValueError("cannot determine cluster server")
    return server


def pod_metadata(cli, namespace, name):
    pod = json.loads(run([cli, "get", "pod", name, "-n", namespace, "-o", "json"]))
    statuses = [*pod["status"].get("containerStatuses", []),
                *pod["status"].get("initContainerStatuses", [])]
    return {
        "name": name, "uid": pod["metadata"]["uid"],
        "node": pod["spec"].get("nodeName"), "ip": pod["status"].get("podIP"),
        "phase": pod["status"].get("phase"),
        "ready": any(x["type"] == "Ready" and x["status"] == "True"
                     for x in pod["status"].get("conditions", [])),
        "host_network": pod["spec"].get("hostNetwork", False),
        "claims": pod["status"].get("resourceClaimStatuses", []),
        "containers": [{"name": x["name"], "image": x["image"],
                        "resources": x.get("resources", {})} for x in pod["spec"]["containers"]],
        "init_containers": [{"name": x["name"], "image": x["image"],
                             "resources": x.get("resources", {})}
                            for x in pod["spec"].get("initContainers", [])],
        "image_ids": {x["name"]: x.get("imageID") for x in statuses},
        "restart_counts": {x["name"]: x.get("restartCount") for x in statuses},
    }


def pod_exec(cli, namespace, pod, container, code):
    args = [cli, "exec", "-n", namespace, pod]
    if container:
        args += ["-c", container]
    return [*args, "--", "python3", "-u", "-c", code]


def collect(cli, namespace, names, container):
    probe = (HERE / "hardware_probe.py").read_text()
    pods = []
    for name in names:
        before = pod_metadata(cli, namespace, name)
        if (not before["ready"] or not before["ip"] or not before["node"]
                or not before["image_ids"] or not before["restart_counts"]
                or any(not value for value in before["image_ids"].values())
                or any(type(value) is not int for value in before["restart_counts"].values())):
            raise ValueError(f"pod {name} is not ready with complete identity metadata")
        inventory = json.loads(run(pod_exec(cli, namespace, name, container, probe)))
        if not isinstance(inventory, dict):
            raise ValueError(f"pod {name} returned invalid inventory")
        after = pod_metadata(cli, namespace, name)
        keys = ("uid", "node", "ip", "image_ids", "restart_counts")
        if not after["ready"] or any(before[k] != after[k] for k in keys):
            raise ValueError(f"pod {name} changed while collecting inventory")
        before["inventory"] = inventory
        pods.append(before)
    try:
        cluster = cluster_server(cli)
    except (subprocess.SubprocessError, OSError, ValueError):
        cluster = None
    return {"schema_version": SCHEMA_VERSION, "collected_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "cluster": cluster, "namespace": namespace, "pods": pods,
            "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "probe_sha256": hashlib.sha256(probe.encode()).hexdigest(),
            "scope": "Pod-visible inventory and configuration are capability evidence, not a transport witness or proof of GPU isolation."}


def save(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pods", required=True, help="Comma-separated existing pod names")
    parser.add_argument("--container", default="")
    parser.add_argument("--cli", choices=["oc", "kubectl"], default="oc")
    parser.add_argument("--output-dir", required=True, help="New directory for profile")
    args = parser.parse_args()
    names = [x.strip() for x in args.pods.split(",") if x.strip()]
    if not names or len(set(names)) != len(names):
        parser.error("provide distinct pod names")
    outdir = Path(args.output_dir)
    try:
        outdir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"Cannot create output directory: {exc}", file=sys.stderr)
        return 1
    try:
        profile = collect(args.cli, args.namespace, names, args.container)
        save(outdir / "profile.json", profile)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        save(outdir / "error.json", {"status": "incomplete", "error": str(exc)[:1000]})
        print(f"Inventory incomplete: {exc}", file=sys.stderr)
        return 1
    print(f"Inventory written to {outdir / 'profile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
