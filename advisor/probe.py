"""
Synthetic disaggregated request prober.

Sends a real disagg request through the full P/D pipeline (client -> sidecar
-> prefill -> NIXL KV transfer -> decode -> response) and tracks TTFT
against a rolling baseline for anomaly detection.
"""

import http.client
import json
import os
import ssl
import statistics
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    from ._cluster import oc_safe  # noqa: F401
except ImportError:
    pass


@dataclass
class ProbeResult:
    timestamp: float
    ttft_ms: float
    status: int
    healthy: bool
    baseline_ms: float
    ratio: float  # ttft / baseline
    error: str = ""


def _send_probe(url, model, prompt="Hello", max_tokens=5, extra_headers=None):
    """Send a single streaming probe request with explicit model parameter.

    Avoids relying on toolkit's module-global MODEL variable.
    Returns dict with ttft_ms, status, error.
    """
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port
    path = parsed.path or "/"
    use_tls = parsed.scheme == "https"

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    })
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)

    start = time.monotonic()
    try:
        if use_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=20)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=20)

        conn.request("POST", path, body=payload, headers=headers)
        resp = conn.getresponse()
        token_times = []

        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    if chunk.get("choices", [{}])[0].get("text", ""):
                        token_times.append(time.monotonic() - start)
                except (json.JSONDecodeError, ValueError):
                    pass

        total = time.monotonic() - start
        conn.close()
        ttft = token_times[0] * 1000 if token_times else total * 1000
        return {"ttft_ms": round(ttft, 1), "status": resp.status, "error": ""}

    except Exception as e:
        return {"ttft_ms": 0, "status": 0, "error": str(e)[:200]}


class DisaggProber:
    """Sends synthetic disagg requests and tracks TTFT against baseline."""

    def __init__(self, namespace: str, model: str, window_size: int = 20,
                 decode_url: str = "", prefill_host: str = ""):
        self.namespace = namespace
        self.model = model
        self.window_size = window_size
        self._baseline_window: deque[float] = deque(maxlen=window_size)
        self._baseline_ms: float = 0.0
        scheme = os.environ.get("SIDECAR_SCHEME", "http")
        self.disagg_url = decode_url or f"{scheme}://vllm-decode-svc:8000/v1/completions"
        self.prefill_host = prefill_host or f"vllm-prefill-svc.{namespace}.svc.cluster.local:8100"

    def probe(self) -> ProbeResult:
        """Send one synthetic disagg request and evaluate health."""
        t = time.time()
        r = _send_probe(
            self.disagg_url, self.model,
            extra_headers={"x-prefiller-host-port": self.prefill_host},
        )
        ttft = r["ttft_ms"]
        status = r["status"]
        error = r["error"]

        if status == 200 and ttft > 0:
            self._baseline_window.append(ttft)
            if len(self._baseline_window) >= 3:
                self._baseline_ms = statistics.median(self._baseline_window)

        baseline = self._baseline_ms or ttft or 1
        ratio = ttft / baseline if baseline > 0 else 0

        healthy = status == 200 and ratio < 3.0

        return ProbeResult(
            timestamp=t, ttft_ms=ttft, status=status,
            healthy=healthy, baseline_ms=round(baseline, 1),
            ratio=round(ratio, 2), error=error,
        )

    def warmup(self, n: int = 5):
        """Send warmup probes to establish baseline."""
        for _ in range(n):
            self.probe()
