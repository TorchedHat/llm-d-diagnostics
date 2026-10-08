"""Single bridge between advisor/ and toolkit/.

Every advisor module imports cluster operations and toolkit functions
through this file.  It is the ONLY place that manipulates sys.path to
reach the toolkit package, keeping the rest of the advisor code clean.

Provides:
    oc, oc_safe        — re-exported from toolkit/client.py
    SCALING_GPUS       — re-exported from toolkit/scaling_model.py
    get_pods_full      — advisor-specific full-metadata pod discovery
    scrape_pod_metrics — shared Prometheus scraping (HTTP + oc exec fallback)
"""

import http.client
import json
import os
import sys

# ── Toolkit imports (single sys.path entry point) ────────────────────────
_toolkit_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "toolkit")
if _toolkit_dir not in sys.path:
    sys.path.insert(0, _toolkit_dir)


from scaling_model import GPUS as SCALING_GPUS  # noqa: E402, F401

from client import oc, oc_safe  # noqa: E402

# ── Pod discovery ────────────────────────────────────────────────────────

def get_pods_full(namespace, label=None):
    """Get full pod metadata: name, ip, ready status, labels, image, args.

    Returns a list of dicts matching the schema used by diagnose/health/rebalance.
    """
    cmd = ["get", "pods", "-n", namespace, "-o", "json"]
    if label:
        cmd.extend(["-l", label])
    out = oc(*cmd)
    items = json.loads(out).get("items", [])
    result = []
    for p in items:
        phase = p.get("status", {}).get("phase", "")
        if phase not in ("Running", "Succeeded"):
            continue
        name = p["metadata"]["name"]
        labels = p["metadata"].get("labels", {})
        ip = p.get("status", {}).get("podIP", "")
        ready = any(
            c.get("type") == "Ready" and c.get("status") == "True"
            for c in p.get("status", {}).get("conditions", [])
        )
        containers = p["spec"].get("containers", [{}])
        init_containers = p["spec"].get("initContainers", [])
        images = [c.get("image", "") for c in containers]
        init_images = [c.get("image", "") for c in init_containers]
        role = labels.get("llm-d.ai/role", labels.get("app", ""))
        result.append({
            "name": name,
            "ip": ip,
            "ready": ready,
            "labels": labels,
            "role": role,
            "image": images[0] if images else "",
            "images": images,
            "init_images": init_images,
            "args": str(containers[0].get("args", [])),
        })
    return result


# ── Shared metrics scraping ──────────────────────────────────────────────

def scrape_pod_metrics(pod, namespace):
    """Scrape Prometheus /metrics from a pod.

    Tries direct HTTP to pod IP first (fast, works when pod network is
    reachable).  Falls back to ``oc exec`` with a Python one-liner when
    the pod IP is unreachable (e.g. running from a laptop).

    Returns the raw metrics text, or empty string on failure.
    """
    ip = pod.get("ip", "")
    is_prefill = "prefill" in pod.get("role", pod.get("name", ""))
    images = pod.get("images") or [pod.get("image", "")]
    is_sim = any("llm-d-inference-sim" in image for image in images)
    port = (8100 if is_sim else 8000) if is_prefill else 8001

    if ip:
        try:
            conn = http.client.HTTPConnection(ip, port, timeout=3)
            conn.request("GET", "/metrics")
            body = conn.getresponse().read().decode(errors="replace")
            conn.close()
            return body
        except Exception:
            pass

    out, _ = oc_safe(
        "exec", pod["name"], "-n", namespace, "--",
        "python3", "-c",
        f"import urllib.request;print(urllib.request.urlopen("
        f"'http://localhost:{port}/metrics',timeout=5).read().decode())",
        timeout=10,
    )
    return out
