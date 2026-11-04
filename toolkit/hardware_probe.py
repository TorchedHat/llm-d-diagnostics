#!/usr/bin/env python3
"""Collect a small allowlisted hardware inventory inside a serving pod."""

import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def command(*args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
        return {"status": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()[:1000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": None, "stdout": "", "stderr": str(exc)}


def main():
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None

    packages = {}
    for name in ("torch", "vllm", "nixl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    ports = []
    for state in sorted(Path("/sys/class/infiniband").glob("*/ports/*/state")):
        port = state.parent
        try:
            ports.append({"hca": port.parent.parent.name, "port": port.name,
                          "state": state.read_text().strip(),
                          "link_layer": (port / "link_layer").read_text().strip()})
        except OSError:
            continue
    names = ("UCX_TLS", "UCX_NET_DEVICES", "UCX_IB_GID_INDEX", "UCX_IB_ADDR_TYPE",
             "UCX_IB_ROCE_REACHABILITY_MODE", "UCX_MAX_RNDV_RAILS", "UCX_PROTO_INFO")
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = None
    result = {
        "gpus": command("nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version",
                        "--format=csv,noheader,nounits"),
        "gpu_topology": command("nvidia-smi", "topo", "-m"),
        "rdma_ports": ports,
        "rdma_devices": sorted(x.name for x in Path("/dev/infiniband").glob("*")),
        "cpu_affinity_count": affinity_count,
        "cgroup_v2": {"cpu_max": read("/sys/fs/cgroup/cpu.max"),
                      "memory_max": read("/sys/fs/cgroup/memory.max"),
                      "memory_current": read("/sys/fs/cgroup/memory.current")},
        "kernel": platform.release(),
        "packages": packages,
        "ucx_environment": {key: os.environ.get(key) for key in names},
    }
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
