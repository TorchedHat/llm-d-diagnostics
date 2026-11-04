#!/usr/bin/env python3
"""KV accounting around an experiment: where each prompt token came from.

vLLM counts every prompt token by source in
``vllm:prompt_tokens_by_source_total{source=...}``: computed locally, served
from the local prefix cache, or received over NIXL (``external_kv_transfer``).
Snapshotting every vLLM pod before and after an experiment shows whether the
experiment measured what it claims:

  - cache-busted prompts should give a prefix-cache share near zero; a large
    share means later requests reused KV from earlier ones and skipped
    prefill and transfer;
  - NIXL payload bytes per transferred token should equal the model's KV size
    per token, the same in every experiment.

Runs locally (needs ``oc``), so it sees every pod rather than one behind a
Service.

Usage:
    kv_sources.py snapshot OUT.json
    kv_sources.py delta BEFORE.json AFTER.json OUT.json

Environment:
    NS  Namespace (required)
"""

import json
import math
import os
import subprocess
import sys
import time

NS = os.environ.get("NS", "")
ROLE_LABEL = "llm-d.ai/role"
VLLM_CONTAINER = "vllm"
VLLM_PORT = {"prefill": 8000, "decode": 8001}
SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")

# Executed inside each vLLM container; reads its own /metrics.
PROBE = r'''
import json, os, re, urllib.request
text = urllib.request.urlopen(
    f"http://127.0.0.1:{os.environ['PROBE_PORT']}/metrics", timeout=5).read().decode()
def total(name, label=None):
    values = []
    for m in re.finditer(r"^" + re.escape(name) + r"(\{[^}]*\})? (\S+)$", text, re.M):
        if label is None or label in (m.group(1) or ""):
            values.append(float(m.group(2)))
    return sum(values) if values else None
print(json.dumps({
    "sources": {src: total("vllm:prompt_tokens_by_source_total", f'source="{src}"')
                for src in ("local_compute", "local_cache_hit", "external_kv_transfer")},
    "nixl_bytes": total("vllm:nixl_bytes_transferred_sum"),
    "nixl_transfers": total("vllm:nixl_bytes_transferred_count"),
}))
'''


def oc(*args, timeout=60):
    r = subprocess.run(["oc", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"oc {' '.join(args[:3])}...: {r.stderr.strip()[:300]}")
    return r.stdout


def snapshot():
    items = json.loads(oc("get", "pods", "-n", NS, "-l",
                          f"{ROLE_LABEL} in (prefill,decode)", "-o", "json"))["items"]
    pods = {}
    for p in items:
        if p.get("status", {}).get("phase") != "Running":
            continue
        name, role = p["metadata"]["name"], p["metadata"]["labels"][ROLE_LABEL]
        statuses = (p.get("status", {}).get("containerStatuses", []) +
                    p.get("status", {}).get("initContainerStatuses", []))
        container = next((s for s in statuses if s.get("name") == VLLM_CONTAINER), {})
        identity = {"uid": p["metadata"].get("uid"),
                    "container_id": container.get("containerID"),
                    "restart_count": container.get("restartCount")}
        try:
            out = oc("exec", "-n", NS, name, "-c", VLLM_CONTAINER, "--",
                     "env", f"PROBE_PORT={VLLM_PORT[role]}", "python3", "-c", PROBE)
            pods[name] = {"role": role, "node": p["spec"].get("nodeName", ""), **identity,
                          **json.loads(out.strip().splitlines()[-1])}
        except (RuntimeError, ValueError, IndexError, subprocess.SubprocessError) as e:
            pods[name] = {"role": role, "error": str(e)[:200]}
    return {"taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pods": pods}


def delta(before, after):
    """Per-role token sources and NIXL bytes accumulated between snapshots.

    Pods present in only one snapshot (restarted, rescheduled) are reported
    and excluded, since their counters do not cover the interval.
    """
    roles = {}
    skipped = []
    reasons = {}
    def valid(value):
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0)

    def skip(name, reason):
        skipped.append(name)
        reasons[name] = reason

    for name, a in after["pods"].items():
        b = before["pods"].get(name)
        if b is None or "error" in a or "error" in b:
            skip(name, "missing snapshot or scrape failed")
            continue
        identity = ("uid", "container_id", "restart_count", "role", "node")
        if any(b.get(key) is None or a.get(key) is None or b[key] != a[key]
               for key in identity):
            skip(name, "pod or container identity unavailable or changed")
            continue
        values = [(b.get("sources", {}).get(s), a.get("sources", {}).get(s)) for s in SOURCES]
        if any(not valid(x) or not valid(y) or y < x for x, y in values):
            skip(name, "source counter missing or reset")
            continue
        optional = [(b.get(k), a.get(k)) for k in ("nixl_bytes", "nixl_transfers")]
        if any((x is not None and not valid(x)) or (y is not None and not valid(y))
               or (x is not None and y is not None and y < x) for x, y in optional):
            skip(name, "NIXL counter reset")
            continue
        r = roles.setdefault(a["role"], {**{s: 0.0 for s in SOURCES},
                                         "nixl_bytes": 0.0, "nixl_transfers": 0.0,
                                         "nixl_complete": True})
        for s in SOURCES:
            r[s] += a["sources"][s] - b["sources"][s]
        if any(x is None or y is None for x, y in optional):
            r["nixl_complete"] = False
        else:
            r["nixl_bytes"] += a["nixl_bytes"] - b["nixl_bytes"]
            r["nixl_transfers"] += a["nixl_transfers"] - b["nixl_transfers"]
    for n in before["pods"]:
        if n not in after["pods"]:
            skip(n, "missing after snapshot")
    for r in roles.values():
        tokens = sum(r[s] for s in SOURCES)
        r["prompt_tokens"] = tokens
        r["cache_hit_fraction"] = r["local_cache_hit"] / tokens if tokens else None
        ext = r["external_kv_transfer"]
        if not r["nixl_complete"]:
            r["nixl_bytes"] = r["nixl_transfers"] = None
        r["nixl_bytes_per_external_token"] = (r["nixl_bytes"] / ext
                                                if ext and r["nixl_complete"] else None)
    return {"from": before["taken_at"], "to": after["taken_at"],
            "roles": roles, "skipped_pods": sorted(set(skipped)), "skip_reasons": reasons,
            "includes_warmup": True}


def summary_line(d):
    parts = []
    for role in ("prefill", "decode"):
        r = d["roles"].get(role)
        if not r or not r["prompt_tokens"]:
            continue
        part = (f"{role}: {r['prompt_tokens']:.0f} prompt tokens, "
                f"{100 * r['cache_hit_fraction']:.1f}% prefix-cache hits")
        if role == "decode" and r["external_kv_transfer"] and r["nixl_bytes_per_external_token"] is not None:
            part += (f", {r['external_kv_transfer']:.0f} via NIXL "
                     f"({r['nixl_bytes_per_external_token'] / 1024:.0f} KiB/token)")
        parts.append(part)
    return "; ".join(parts) or "no prompt tokens recorded"


def main(argv):
    if not NS:
        print("ERROR: NS is not set", file=sys.stderr)
        return 2
    if len(argv) == 3 and argv[1] == "snapshot":
        with open(argv[2], "w") as f:
            json.dump(snapshot(), f, indent=2)
        return 0
    if len(argv) == 5 and argv[1] == "delta":
        with open(argv[2]) as f:
            before = json.load(f)
        with open(argv[3]) as f:
            after = json.load(f)
        d = delta(before, after)
        with open(argv[4], "w") as f:
            json.dump(d, f, indent=2)
        print(f"  KV accounting: {summary_line(d)}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
