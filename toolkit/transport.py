#!/usr/bin/env python3
"""Transport witness: which transport the NIXL KV-cache transfer actually uses.

Configuration (UCX_TLS, RDMA resources) says what a transfer *may* use; this
tool collects evidence in three layers without claiming per-flow attribution:

  1. capability  What each vLLM pod could use: RDMA device files, the RDMA
                 resource request, UCX_TLS / UCX_NET_DEVICES, hostNetwork,
                 HCA link layer.
  2. ucx         What UCX logged it selected (transport/device pairs). Only
                 available when the pods run with UCX_PROTO_INFO=y
                 (deploy with TRANSPORT_LOG=1).
  3. bytes       For each prefill->decode pair, a burst of cache-busted
                 disaggregated requests; NIXL payload bytes on the decode pod
                 are compared with the bytes its network interfaces received.
                 A ratio alone does not attribute the observed bytes to a flow.

Runs locally (needs `oc`); requests are sent from the test-client pod, which
must already hold the toolkit (run.sh preflight copies it).

Environment:
    NS                Namespace (required)
    DATA_DIR          Where transport.json is written (default: ./data)
    WITNESS_REQUESTS  Requests per pair (default: 5)
    WITNESS_TOKENS    Prompt length per request (default: 2000)
    POD               Test-client pod (default: test-client)
    REMOTE_DIR        Toolkit path in the test-client pod (default: /scripts/toolkit)
    SIDECAR_SCHEME    http or https (default: http)
"""

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid

NS = os.environ.get("NS", "")
DATA_DIR = os.environ.get("DATA_DIR", "data")
WITNESS_REQUESTS = int(os.environ.get("WITNESS_REQUESTS", "5"))
WITNESS_TOKENS = int(os.environ.get("WITNESS_TOKENS", "2000"))
POD = os.environ.get("POD", "test-client")
REMOTE_DIR = os.environ.get("REMOTE_DIR", "/scripts/toolkit")
SIDECAR_SCHEME = os.environ.get("SIDECAR_SCHEME", "http")

ROLE_LABEL = "llm-d.ai/role"
VLLM_CONTAINER = "vllm"
# vLLM's own port: prefill serves on 8000; decode vLLM sits behind the
# sidecar on 8001.
VLLM_PORT = {"prefill": 8000, "decode": 8001}

# Ratio of interface receive bytes to NIXL payload bytes on the decode pod.
ON_NETDEV_MIN = 0.9    # >= : the KV crossed this interface
OFF_NETDEV_MAX = 0.2   # <= : the KV did not cross any pod interface

UCX_TRANSPORT_RE = re.compile(
    r"\b(tcp|rc_mlx5|rc_verbs|dc_mlx5|ud_mlx5|ud_verbs|cuda_ipc|gdr_copy)/([\w.:-]+)")

# Executed inside each vLLM container. Reads only local state.
PROBE = r'''
import glob, json, os, re, urllib.request
keys = ("UCX_TLS", "UCX_NET_DEVICES", "UCX_IB_GID_INDEX", "UCX_IB_ADDR_TYPE",
        "UCX_IB_ROCE_REACHABILITY_MODE", "UCX_MAX_RNDV_RAILS", "UCX_PROTO_INFO")
out = {"env": {k: os.environ[k] for k in keys if k in os.environ}}
out["dev_infiniband"] = sorted(os.listdir("/dev/infiniband")) if os.path.isdir("/dev/infiniband") else []
links = {}
for d in sorted(glob.glob("/sys/class/infiniband/*")):
    try:
        links[os.path.basename(d)] = open(d + "/ports/1/link_layer").read().strip()
    except OSError:
        pass
out["hca_link_layers"] = links
netdev = {}
for d in sorted(glob.glob("/sys/class/net/*")):
    name = os.path.basename(d)
    if name == "lo":
        continue
    try:
        netdev[name] = {f: int(open(f"{d}/statistics/{f}").read()) for f in ("rx_bytes", "tx_bytes")}
    except OSError:
        pass
out["netdev"] = netdev
text = ""
try:
    port = int(os.environ["PROBE_PORT"])
    text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5).read().decode()
except Exception as e:
    out["metrics_error"] = str(e)[:200]
def total(name, label=None):
    values = []
    for m in re.finditer(r"^" + re.escape(name) + r"(\{[^}]*\})? (\S+)$", text, re.M):
        if label and label not in (m.group(1) or ""):
            continue
        values.append(float(m.group(2)))
    return sum(values) if values else None
out["nixl"] = {
    "bytes": total("vllm:nixl_bytes_transferred_sum"),
    "transfers": total("vllm:nixl_bytes_transferred_count"),
    "xfer_seconds": total("vllm:nixl_xfer_time_seconds_sum"),
    "failed": total("vllm:nixl_num_failed_transfers_total"),
}
out["external_kv_tokens"] = total("vllm:prompt_tokens_by_source_total", 'source="external_kv_transfer"')
print(json.dumps(out))
'''

# Executed in the test-client pod: the witness burst for one pair.
BURST = r'''
import os, sys
sys.path.insert(0, os.environ["REMOTE_DIR"])
from client import build_prompt, decode_pod_url_by_ip, prefill_pod_host_port, send_request
url = decode_pod_url_by_ip(os.environ["DECODE_IP"])
headers = {"x-prefiller-host-port": prefill_pod_host_port(os.environ["PREFILL_IP"])}
ok, errors = 0, []
for i in range(int(os.environ["N"])):
    prompt = build_prompt(int(os.environ["TOKENS"]), cache_bust=f"{os.environ['TAG']}-{i}")
    r = send_request(url, prompt, 1, extra_headers=headers)
    if r.ok:
        ok += 1
    else:
        errors.append(r.error or f"HTTP {r.status}")
print(ok, "|".join(errors[:3]))
'''


def oc(*args, timeout=120):
    r = subprocess.run(["oc", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"oc {' '.join(args[:3])}...: {r.stderr.strip()[:300]}")
    return r.stdout


def discover_pods():
    """Running, ready vLLM pods with their placement and RDMA request."""
    items = json.loads(oc("get", "pods", "-n", NS, "-l",
                          f"{ROLE_LABEL} in (prefill,decode)", "-o", "json"))["items"]
    pods = []
    for p in items:
        status = p.get("status", {})
        ready = any(c.get("type") == "Ready" and c.get("status") == "True"
                    for c in status.get("conditions", []))
        if status.get("phase") != "Running" or not ready:
            continue
        spec = p["spec"]
        rdma_resources = sorted({
            k for c in spec.get("containers", [])
            for section in ("requests", "limits")
            for k in (c.get("resources", {}).get(section) or {})
            if k.startswith("rdma/")})
        pods.append({
            "name": p["metadata"]["name"],
            "role": p["metadata"]["labels"][ROLE_LABEL],
            "node": spec.get("nodeName", ""),
            "ip": status.get("podIP", ""),
            "host_network": bool(spec.get("hostNetwork", False)),
            "rdma_resources": rdma_resources,
        })
    return sorted(pods, key=lambda p: (p["role"] != "prefill", p["node"], p["name"]))


def probe(pod):
    out = oc("exec", "-n", NS, pod["name"], "-c", VLLM_CONTAINER, "--",
             "env", f"PROBE_PORT={VLLM_PORT[pod['role']]}", "python3", "-c", PROBE)
    return json.loads(out.strip().splitlines()[-1])


def ucx_report(pod):
    """Transport/device pairs UCX logged, or None when UCX_PROTO_INFO is off."""
    if pod["probe"]["env"].get("UCX_PROTO_INFO", "").lower() not in ("y", "yes", "1", "on"):
        return None
    logs = oc("logs", "-n", NS, pod["name"], "-c", VLLM_CONTAINER, "--tail=50000")
    counts = {}
    for m in UCX_TRANSPORT_RE.finditer(logs):
        key = f"{m.group(1)}/{m.group(2)}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def short_node(node):
    return node.rsplit("-", 1)[-1]


def cuda_ipc_allowed(ucx_tls):
    """Whether UCX_TLS permits cuda_ipc. "^a,b" excludes; "a,b" includes only."""
    if not ucx_tls:
        return True
    if ucx_tls.startswith("^"):
        return "cuda_ipc" not in ucx_tls[1:].split(",")
    names = ucx_tls.split(",")
    return "all" in names or "cuda" in names or "cuda_ipc" in names


def capability(pod):
    p = pod["probe"]
    ucx_tls = p["env"].get("UCX_TLS", "")
    return {
        "rdma_device": bool(p["dev_infiniband"]),
        "rdma_resources": pod["rdma_resources"],
        "host_network": pod["host_network"],
        "ucx_tls": ucx_tls or "(unset: UCX default, all transports)",
        "ucx_net_devices": p["env"].get("UCX_NET_DEVICES", "(unset)"),
        "cuda_ipc_allowed": cuda_ipc_allowed(ucx_tls),
        "hca_link_layers": p["hca_link_layers"],
    }


def choose_prefill(decode, prefills):
    """Prefer a prefill on another node, so the pair crosses the network."""
    remote = [p for p in prefills if p["node"] != decode["node"]]
    return (remote or prefills)[0]


def verdict(pair, prefill_cap, decode_cap):
    """Describe whether transfer evidence is sufficient to identify a path."""
    d = pair["delta"]
    if pair["ok"] == 0:
        return "error", "no witness request succeeded"
    if pair.get("invalidity") or d["nixl_bytes"] is None or d["external_kv_tokens"] is None:
        return "unknown", "metrics or pod continuity missing, changed or reset"
    if d["nixl_bytes"] == 0 and d["external_kv_tokens"] == 0:
        return "no-kv-transfer", ("decode received no KV over NIXL "
                                  "(recomputed or served from its own cache)")
    if d["nixl_bytes"] <= 0 or d["external_kv_tokens"] <= 0:
        return "unknown", "NIXL and external-KV counters are inconsistent"
    return "unknown", ("interface and NIXL counters are not flow attribution; "
                       "correlate application transfers with selected UCX lanes")


def counter_delta(before, after):
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in (before, after)):
        return None
    if after < before:
        return None
    return after - before


def pod_identity(pod):
    data = json.loads(oc("get", "pod", pod["name"], "-n", NS, "-o", "json"))
    status = data.get("status", {})
    containers = status.get("containerStatuses", []) + status.get("initContainerStatuses", [])
    container = next((c for c in containers if c.get("name") == VLLM_CONTAINER), {})
    return {"uid": data.get("metadata", {}).get("uid"),
            "container_id": container.get("containerID"), "restarts": container.get("restartCount")}


def witness_pair(prefill, decode, tag):
    identities = {p["name"]: pod_identity(p) for p in (prefill, decode)}
    before_d = probe(decode)
    env = [f"REMOTE_DIR={REMOTE_DIR}", f"SIDECAR_SCHEME={SIDECAR_SCHEME}",
           f"DECODE_IP={decode['ip']}", f"PREFILL_IP={prefill['ip']}",
           f"N={WITNESS_REQUESTS}", f"TOKENS={WITNESS_TOKENS}", f"TAG={tag}"]
    out = oc("exec", "-n", NS, POD, "--", "env", *env, "python3", "-c", BURST,
             timeout=60 + 30 * WITNESS_REQUESTS).strip().splitlines()[-1]
    ok_str, _, errors = out.partition(" ")
    after_d = probe(decode)
    identities_after = {p["name"]: pod_identity(p) for p in (prefill, decode)}
    invalidity = []
    if (identities != identities_after or
            any(any(value is None for value in identity.values()) for identity in identities.values())):
        invalidity.append("pod or container changed or identity missing")
    nixl = {k: counter_delta(before_d["nixl"].get(k), after_d["nixl"].get(k))
            for k in after_d["nixl"]}
    external = counter_delta(before_d.get("external_kv_tokens"), after_d.get("external_kv_tokens"))
    if any(v is None for v in nixl.values()) or external is None:
        invalidity.append("NIXL/source metrics missing or reset")
    netdev = {
        name: {f: counter_delta(before_d["netdev"].get(name, {}).get(f), after_d["netdev"][name][f])
               for f in ("rx_bytes", "tx_bytes")}
        for name in after_d["netdev"]}
    if not netdev or any(v is None for values in netdev.values() for v in values.values()):
        invalidity.append("interface counters missing or reset")
    rx_total = sum(v["rx_bytes"] for v in netdev.values()) if not invalidity else None
    busiest = max(netdev, key=lambda n: netdev[n]["rx_bytes"]) if rx_total is not None else ""
    pair = {
        "prefill": prefill["name"], "prefill_node": prefill["node"],
        "decode": decode["name"], "decode_node": decode["node"],
        "same_node": prefill["node"] == decode["node"],
        "requests": WITNESS_REQUESTS, "prompt_tokens": WITNESS_TOKENS,
        "ok": int(ok_str or 0), "errors": errors,
        "pod_identities": identities, "invalidity": invalidity,
        "delta": {
            "nixl_bytes": nixl["bytes"], "nixl_transfers": nixl["transfers"],
            "nixl_xfer_seconds": nixl["xfer_seconds"], "nixl_failed": nixl["failed"],
            "external_kv_tokens": external,
            "netdev": netdev, "netdev_rx_total": rx_total,
        },
        "busiest_netdev": busiest,
        "netdev_ratio": rx_total / nixl["bytes"] if rx_total is not None and nixl["bytes"] else None,
        "nixl_gbps": (nixl["bytes"] / nixl["xfer_seconds"] / 1e9
                      if nixl["bytes"] is not None and nixl["xfer_seconds"] else None),
    }
    return pair


def overall(pairs):
    """One verdict for the deployment, preferring cross-node evidence."""
    cross = [p for p in pairs if not p["same_node"]]
    basis = cross or pairs
    kinds = sorted({p["transport"] for p in basis})
    scope = "cross-node" if cross else "same-node only"
    if len(kinds) == 1:
        return kinds[0], scope
    return "inconsistent: " + ", ".join(kinds), scope


def main():
    if not NS:
        print("ERROR: NS is not set", file=sys.stderr)
        return 2
    print("=== Transport Witness ===")
    pods = discover_pods()
    prefills = [p for p in pods if p["role"] == "prefill"]
    decodes = [p for p in pods if p["role"] == "decode"]
    if not prefills or not decodes:
        print("  FAIL: need at least one ready prefill and one ready decode pod")
        return 1

    for p in pods:
        p["probe"] = probe(p)
        p["capability"] = capability(p)
        p["ucx"] = ucx_report(p)

    print("  Layer 1, capability (what each pod could use):")
    for p in pods:
        c = p["capability"]
        rdma = "RDMA device present" if c["rdma_device"] else "no RDMA device"
        print(f"    {p['role']:7s} {p['name']:34s} {short_node(p['node']):6s} "
              f"{rdma}; UCX_TLS={c['ucx_tls']}; hostNetwork={c['host_network']}")

    print("  Layer 2, UCX self-report:")
    if all(p["ucx"] is None for p in pods):
        print("    not enabled (redeploy with TRANSPORT_LOG=1 to log UCX_PROTO_INFO)")
    for p in pods:
        if p["ucx"] is not None:
            seen = ", ".join(f"{k} x{v}" for k, v in sorted(p["ucx"].items())) or "no transport lines yet"
            print(f"    {p['name']}: {seen}")

    print(f"  Layer 3, byte accounting ({WITNESS_REQUESTS} requests x "
          f"{WITNESS_TOKENS} tokens, cache-busted, per pair):")
    run_tag = f"transport-{uuid.uuid4().hex[:8]}"
    by_name = {p["name"]: p for p in pods}
    pairs = []
    for decode in decodes:
        prefill = choose_prefill(decode, prefills)
        pair = witness_pair(prefill, decode, f"{run_tag}-{decode['name']}")
        pair["transport"], pair["reason"] = verdict(
            pair, by_name[prefill["name"]]["capability"], decode["capability"])
        pairs.append(pair)
        d = pair["delta"]
        ratio = f"{pair['netdev_ratio']:.2f}" if pair["netdev_ratio"] is not None else "n/a"
        gbps = f"{pair['nixl_gbps']:.2f} GB/s" if pair["nixl_gbps"] else "n/a"
        nixl_mb = f"{d['nixl_bytes'] / 1e6:.1f}" if d["nixl_bytes"] is not None else "n/a"
        rx_mb = f"{d['netdev_rx_total'] / 1e6:.1f}" if d["netdev_rx_total"] is not None else "n/a"
        where = "same-node " if pair["same_node"] else "cross-node"
        print(f"    {short_node(pair['prefill_node'])} -> {short_node(pair['decode_node'])} {where} "
              f"NIXL {nixl_mb:>8} MB  netdev rx {rx_mb:>8} MB  "
              f"ratio {ratio}  {gbps}  => {pair['transport']}")
        if pair["errors"]:
            print(f"      errors: {pair['errors']}")

    transport, scope = overall(pairs)
    reasons = sorted({p["reason"] for p in pairs if not p["same_node"]} or {p["reason"] for p in pairs})
    print("")
    print(f"  TRANSPORT: {transport} ({scope}, witnessed by byte accounting)")
    for r in reasons:
        print(f"    {r}")
    print("")

    result = {
        "classifier_version": 2,
        "transport": transport,
        "scope": scope,
        "witnessed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": {"requests_per_pair": WITNESS_REQUESTS, "prompt_tokens": WITNESS_TOKENS,
                   "on_netdev_min_ratio": ON_NETDEV_MIN, "off_netdev_max_ratio": OFF_NETDEV_MAX},
        "pods": [{k: p[k] for k in ("name", "role", "node", "ip", "capability", "ucx")}
                 for p in pods],
        "pairs": pairs,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "transport.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Written: {path}")
    return 0 if not transport.startswith(("error", "inconsistent")) else 1


if __name__ == "__main__":
    sys.exit(main())
