"""
llm-d Diagnostics Toolkit — HTTP client and shared utilities.

Zero external dependencies. Uses http.client for precise TTFT measurement.

All configuration via environment variables:
    SIM             Set to 1 for inference-sim mode (HTTP, fake model)
    MODEL           Model name (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    NS              Kubernetes namespace (default: default)
    BASELINE_URL    Prefill direct URL (default: vLLM :8000, SIM :8100)
    SIDECAR_SCHEME  Sidecar listener scheme (default: http)
    DISAGG_URL      Decode via sidecar (default: http://vllm-decode-svc:8000/v1/completions)
    EPP_URL         Optional OpenAI completions URL exposed by an HTTPRoute
    ROUTING_MODE    manual or epp; selects the route for send_disagg and fault probes
    DECODE_DIRECT_URL  Decode bypass sidecar (default: http://vllm-decode-direct-svc:8001/v1/completions)

Per-pod URLs (by pod IP):
    Use discover_pod_ips(label, namespace) to get pod IPs via `oc get pods`.
    Use decode_pod_url_by_ip(ip) to build a URL from a discovered IP.
    Works with Deployments (no stable DNS needed).

    PREFILL_HOST    Prefill host:port for routing header
    DATA_DIR        Output directory (default: data)
    WARMUP          Warmup requests (default: 3)
    RUNS            Measured runs per config (default: 20)
    MAX_TOKENS      Max completion tokens (default: 20)

Simulation mode (SIM=1):
    Uses inference-sim (llm-d-inference-sim) as a GPU-free test target.
    Uses HTTP by default. Deploy with: ./scripts/deploy.sh <cluster-dir> sim
"""

import http.client
import json
import os
import ssl
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

# ── Configuration ────────────────────────────────────────────────────────────

def env(name, default):
    return os.environ.get(name, default)


SIM = env("SIM", "") == "1"
NS = env("NS", "default")
_PREFILL_POD_PORT = 8100 if SIM else 8000
SIDECAR_SCHEME = env("SIDECAR_SCHEME", "http")
ROUTING_MODE = env("ROUTING_MODE", "manual").lower()
EPP_URL = env("EPP_URL", "").strip()
if ROUTING_MODE not in {"manual", "epp"}:
    raise ValueError("ROUTING_MODE must be 'manual' or 'epp'")
if SIDECAR_SCHEME not in {"http", "https"}:
    raise ValueError("SIDECAR_SCHEME must be 'http' or 'https'")

if SIM:
    MODEL = env("MODEL", env("MODEL_NAME", "sim-model"))
    _DECODE_PORT = 8000
else:
    MODEL = env("MODEL", env("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    _DECODE_PORT = 8000

BASELINE_URL = env("BASELINE_URL", f"http://vllm-prefill-svc:{_PREFILL_POD_PORT}/v1/completions")
DISAGG_URL = env("DISAGG_URL", f"{SIDECAR_SCHEME}://vllm-decode-svc:{_DECODE_PORT}/v1/completions")
DECODE_DIRECT_URL = env("DECODE_DIRECT_URL", "http://vllm-decode-direct-svc:8001/v1/completions")

# Per-pod URL construction by IP (for experiments that need per-pod targeting).
# Use discover_pod_ips() to get IPs, then build URLs with these helpers.

def decode_pod_url_by_ip(ip):
    """URL for a specific decode pod by IP (via sidecar, port 8000)."""
    return f"{SIDECAR_SCHEME}://{ip}:{_DECODE_PORT}/v1/completions"

def decode_direct_url_by_ip(ip):
    """URL for a specific decode pod by IP (bypass sidecar, port 8001)."""
    return f"http://{ip}:8001/v1/completions"

def prefill_pod_url_by_ip(ip):
    """URL for a specific prefill pod (GPU deployments use port 8000)."""
    return f"http://{ip}:{_PREFILL_POD_PORT}/v1/completions"

def prefill_pod_host_port(ip):
    """Routing-header target for a specific prefill pod."""
    return f"{ip}:{_PREFILL_POD_PORT}"

def epp_enabled():
    """Whether an EPP Gateway endpoint has been configured."""
    return bool(EPP_URL)

def manual_disagg_headers(prefill_host=None):
    """Build the explicit prefill header used by the manual-routing control."""
    return {"x-prefiller-host-port": prefill_host or PREFILL_HOST}


def request_headers(url, extra_headers=None):
    """Keep manual prefill targets out of requests scheduled by EPP."""
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    if EPP_URL and url.rstrip("/") == EPP_URL.rstrip("/"):
        headers = {key: value for key, value in headers.items()
                   if key.lower() != "x-prefiller-host-port"}
    return headers


def http_error(status, body):
    """Preserve JSON and plain-text errors from model servers and Gateways."""
    if status < 400:
        return ""
    detail = body.get("error") or body.get("raw", "")
    if isinstance(detail, dict):
        detail = detail.get("message", detail)
    return f"HTTP {status}: {str(detail)[:500]}".rstrip(": ")

def detect_transport():
    """Auto-detect the interconnect transport type.

    Checks UCX_TLS env var and probes for RDMA device files.
    Returns one of: 'tcp', 'roce', 'infiniband', 'unknown'.
    """
    ucx_tls = os.environ.get("UCX_TLS", "")
    # ^cuda_ipc means TCP-only (exclude CUDA IPC, no RDMA)
    if ucx_tls == "^cuda_ipc":
        return "tcp"
    # Check for explicit RDMA transports in UCX_TLS
    if "rc" in ucx_tls or "ud" in ucx_tls or "dc" in ucx_tls:
        # Distinguish RoCE from InfiniBand via device type
        try:
            import subprocess
            result = subprocess.run(
                ["ls", "/sys/class/infiniband/"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                devices = result.stdout.strip().split()
                # Check link layer: IB vs Ethernet (RoCE)
                for dev in devices:
                    try:
                        with open(f"/sys/class/infiniband/{dev}/ports/1/link_layer") as f:
                            layer = f.read().strip()
                            if layer == "InfiniBand":
                                return "infiniband"
                            elif layer == "Ethernet":
                                return "roce"
                    except OSError:
                        continue
                return "infiniband"  # default if we can't read link_layer
        except Exception:
            pass
        return "unknown"
    if not ucx_tls:
        return "unknown"
    return "tcp"


def _discover_via_oc(label, ns):
    """Discover pods using `oc` CLI (works outside the cluster)."""
    import subprocess
    result = subprocess.run(
        ["oc", "get", "pods", "-l", label, "-n", ns,
         "-o", "jsonpath={range .items[?(@.status.phase=='Running')]}"
               "{.metadata.name}{' '}{.status.podIP}{'\\n'}{end}"],
        capture_output=True, text=True, timeout=10,
    )
    pods = []
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            pods.append((parts[0], parts[1]))
    return pods


def _discover_via_k8s_api(label, ns):
    """Discover pods using the Kubernetes API (works inside a pod).

    Uses the service account token mounted at
    /var/run/secrets/kubernetes.io/serviceaccount/ — standard in-cluster
    client pattern, zero external dependencies.
    """
    import json
    import ssl
    import urllib.request

    sa_dir = "/var/run/secrets/kubernetes.io/serviceaccount"
    with open(os.path.join(sa_dir, "token")) as f:
        token = f.read().strip()

    ctx = ssl.create_default_context(cafile=os.path.join(sa_dir, "ca.crt"))
    url = (f"https://kubernetes.default.svc/api/v1"
           f"/namespaces/{ns}/pods?labelSelector={label}")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        data = json.loads(resp.read())

    pods = []
    for item in data.get("items", []):
        phase = item.get("status", {}).get("phase")
        pod_ip = item.get("status", {}).get("podIP")
        name = item.get("metadata", {}).get("name")
        if phase == "Running" and pod_ip and name:
            pods.append((name, pod_ip))
    return pods


def _discover_via_env(label):
    """Discover pods from env vars (works anywhere, highest priority).

    Env var format: PODS_<ROLE>=name1:ip1,name2:ip2
    where <ROLE> is derived from the label value, uppercased with hyphens
    replaced by underscores. Examples:
        label "app=vllm-prefill" → PODS_VLLM_PREFILL
        label "app=vllm-decode"  → PODS_VLLM_DECODE

    This lets the workstation discover pods once and pass them to in-pod
    experiments without requiring RBAC or CLI tools inside the pod.
    """
    # Extract value from "key=value" label selector
    if "=" in label:
        role = label.split("=", 1)[1]
    else:
        role = label
    env_key = "PODS_" + role.upper().replace("-", "_")
    val = os.environ.get(env_key, "")
    if not val:
        return []
    pods = []
    for entry in val.split(","):
        entry = entry.strip()
        if ":" in entry:
            name, ip = entry.split(":", 1)
            pods.append((name.strip(), ip.strip()))
    return pods


def discover_pod_ips(label, namespace=None):
    """Discover pod IPs by label selector.

    Returns list of (pod_name, pod_ip) tuples for Running pods, sorted by name.

    Resolution order:
      1. Env vars (PODS_VLLM_PREFILL, PODS_VLLM_DECODE) — works anywhere
      2. `oc` CLI — works on developer workstations
      3. Kubernetes API via service account token — works inside pods with RBAC
    """
    ns = namespace or NS

    # 1. Env vars (highest priority — explicit always wins)
    pods = _discover_via_env(label)
    if pods:
        return sorted(pods, key=lambda x: x[0])

    # 2. Try oc CLI (available on developer workstations)
    try:
        pods = _discover_via_oc(label, ns)
        if pods:
            return sorted(pods, key=lambda x: x[0])
    except Exception:
        pass

    # 3. Fall back to in-cluster Kubernetes API
    try:
        pods = _discover_via_k8s_api(label, ns)
        if pods:
            return sorted(pods, key=lambda x: x[0])
    except Exception:
        pass

    return []

# Backward compat: D1/D2 URLs default to the aggregate service.
# Experiments that need per-pod targeting should use discover_pod_ips().
DISAGG_D1_URL = env("DISAGG_D1_URL", DISAGG_URL)
DISAGG_D2_URL = env("DISAGG_D2_URL", DISAGG_URL)

PREFILL_HOST = env("PREFILL_HOST", f"vllm-prefill-svc.{NS}.svc.cluster.local:{_PREFILL_POD_PORT}")
PREFILL_HEADER = f"x-prefiller-host-port: {PREFILL_HOST}"
DATA_DIR = env("DATA_DIR", "data")
WARMUP = int(env("WARMUP", "3"))
RUNS = int(env("RUNS", "20"))
MAX_TOKENS = int(env("MAX_TOKENS", "20"))


# ── Prompt builder ───────────────────────────────────────────────────────────

BASE_SENTENCE = "The quick brown fox jumps over the lazy dog again"

_WORD_POOL = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
    "kilo lima mike november oscar papa quebec romeo sierra tango "
    "uniform victor whiskey xray yankee zulu red green blue yellow "
    "orange purple black white silver golden bright dark fast slow "
    "large small heavy light warm cool deep wide long short"
).split()


def build_prompt(target_tokens, cache_bust=None):
    """Build a prompt targeting approximately `target_tokens` tokens.

    Each repetition of BASE_SENTENCE is ~10 tokens.  When *cache_bust* is
    set (any hashable value — typically a run counter), the entire prompt
    body is generated from a seeded RNG so every token block is unique,
    defeating vLLM's hash-based prefix cache.
    """
    reps = max(1, target_tokens // 10)
    if cache_bust is None:
        return " ".join([BASE_SENTENCE] * reps)
    import hashlib
    import random
    seed = int(hashlib.sha256(str(cache_bust).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    words = [rng.choice(_WORD_POOL) for _ in range(target_tokens)]
    return " ".join(words)


# ── HTTP client with precise timing ──────────────────────────────────────────

@dataclass(frozen=True)
class RequestResult:
    """Immutable result of a single HTTP request.

    Frozen: constructed once by send_request/send_streaming, never mutated.
    The ``body`` dict is shallow-mutable (frozen prevents field reassignment,
    not interior mutation), but nothing in the codebase mutates it post-construction.
    """
    ttft_ms: float        # time to first token (streaming) or first byte (non-streaming)
    total_ms: float       # total request time
    status: int           # HTTP status code (0 if exception before HTTP response)
    prompt_tokens: int    # from response usage
    completion_tokens: int  # from response usage
    body: dict[str, Any] = field(default_factory=dict)  # full response JSON
    error: str = ""       # error message if request failed
    token_times: tuple[float, ...] = ()  # monotonic per-token timestamps (s from request start)

    @property
    def ok(self) -> bool:
        """True if HTTP 200 and no exception."""
        return self.status == 200 and not self.error


def send_request(url, prompt, max_tokens=MAX_TOKENS, extra_headers=None):
    """Send a completion request with precise timing.

    Uses http.client directly (not urllib/requests) for precise timing.

    MEASUREMENT LIMITATION: For non-streaming requests, `ttft_ms` measures
    time to first byte of the HTTP response body, NOT time to first
    generated token. vLLM buffers the complete response before sending,
    so ttft_ms ≈ total_ms. This is valid for RELATIVE comparisons across
    configs (the buffering overhead cancels), but is NOT true TTFT.
    For true TTFT measurement, use `send_streaming()` instead.

    Args:
        url: Full URL (http:// or https://)
        prompt: Prompt string
        max_tokens: Maximum completion tokens
        extra_headers: Dict of additional headers (e.g., prefill routing)

    Returns:
        RequestResult with timing, status, and parsed response.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    use_tls = parsed.scheme == "https"

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
    })

    headers = request_headers(url, extra_headers)

    start = time.monotonic()
    try:
        if use_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=30)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=30)

        conn.request("POST", path, body=payload, headers=headers)
        response = conn.getresponse()

        # Read first byte — this is the true TTFT
        first_byte = response.read(1)
        ttft = time.monotonic() - start

        # Read the rest
        rest = response.read()
        total = time.monotonic() - start

        body_bytes = first_byte + rest
        conn.close()

        # Parse response
        try:
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, ValueError):
            body = {"raw": body_bytes.decode("utf-8", errors="replace")}

        usage = body.get("usage", {})

        return RequestResult(
            ttft_ms=round(ttft * 1000, 2),
            total_ms=round(total * 1000, 2),
            status=response.status,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            body=body,
            error=http_error(response.status, body),
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            ttft_ms=0,
            total_ms=round(elapsed * 1000, 2),
            status=0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(e),
        )


def send_disagg(url, prompt, max_tokens=MAX_TOKENS):
    """Send via the configured manual or EPP disaggregation route."""
    target = EPP_URL if ROUTING_MODE == "epp" else url
    headers = {} if ROUTING_MODE == "epp" else manual_disagg_headers()
    if ROUTING_MODE == "epp" and not EPP_URL:
        raise ValueError("ROUTING_MODE=epp requires EPP_URL")
    return send_request(target, prompt, max_tokens, extra_headers=headers)


def send_streaming(url, prompt, max_tokens=MAX_TOKENS, extra_headers=None):
    """Send a streaming completion request with true TTFT measurement.

    Uses SSE (server-sent events) to measure the actual time to first
    generated token, and records per-token arrival times for ITL.

    Returns:
        RequestResult with accurate ttft_ms, per-token token_times,
        and completion_tokens counted from the stream.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    use_tls = parsed.scheme == "https"

    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
    })

    headers = request_headers(url, extra_headers)

    start = time.monotonic()
    try:
        if use_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn = http.client.HTTPSConnection(host, port, context=ctx, timeout=30)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=30)

        conn.request("POST", path, body=payload, headers=headers)
        response = conn.getresponse()

        status = response.status
        if status != 200:
            body = {"raw": response.read().decode("utf-8", errors="replace")}
            conn.close()
            return RequestResult(
                ttft_ms=0, total_ms=round((time.monotonic() - start) * 1000, 2),
                status=status, prompt_tokens=0, completion_tokens=0,
                body=body, error=http_error(status, body),
            )
        token_times = []
        prompt_tokens = 0
        last_chunk = {}

        # Parse SSE stream line by line
        while True:
            line = response.readline()
            if not line:
                break
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line == "data: [DONE]":
                break
            if line.startswith("data: "):
                try:
                    chunk = json.loads(line[6:])
                    choices = chunk.get("choices", [])
                    if choices and choices[0].get("text", ""):
                        token_times.append(time.monotonic() - start)
                    # Capture usage from last chunk (vLLM includes it there)
                    usage = chunk.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", 0)
                    last_chunk = chunk
                except (json.JSONDecodeError, ValueError):
                    pass

        total = time.monotonic() - start
        conn.close()

        ct = len(token_times)
        ttft = token_times[0] if token_times else total

        return RequestResult(
            ttft_ms=round(ttft * 1000, 2),
            total_ms=round(total * 1000, 2),
            status=status,
            prompt_tokens=prompt_tokens,
            completion_tokens=ct,
            body=last_chunk,
            token_times=tuple(token_times),
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        return RequestResult(
            ttft_ms=0,
            total_ms=round(elapsed * 1000, 2),
            status=0,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(e),
        )


# ── Pinned connection (protocol hygiene) ────────────────────────────────────

class PinnedConnection:
    """Persistent HTTP(S) connection to a specific pod.

    Enforces protocol hygiene rules:
      R2 — Pod pinning: fixed target IP, no service load balancing.
      R3 — Pod identity: pod_name stored, available for CSV recording.
      R6 — Connection reuse: TLS/TCP handshake amortized across requests.
      R7 — Warmup on measurement connection: TLS happens in warmup, not measurement.

    NOT thread-safe (http.client isn't). Use send_request() for threaded
    experiments that intentionally measure service-level behavior.

    Usage:
        conn = PinnedConnection(decode_direct_url_by_ip(ip), pod_name="decode-abc")
        conn.warmup(prompt)       # TLS handshake happens here
        r = conn.send(prompt, 20) # reuses connection
        conn.close()
    """

    def __init__(self, url, pod_name="unknown", extra_headers=None, timeout=30):
        self.url = url
        self.pod_name = pod_name
        self.extra_headers = extra_headers or {}
        self._timeout = timeout
        self._parsed = urlparse(url)
        self._host = self._parsed.hostname
        self._port = self._parsed.port
        self._path = self._parsed.path or "/"
        if self._parsed.query:
            self._path = f"{self._path}?{self._parsed.query}"
        self._use_tls = self._parsed.scheme == "https"
        self._ssl_ctx = None
        if self._use_tls:
            self._ssl_ctx = ssl.create_default_context()
            self._ssl_ctx.check_hostname = False
            self._ssl_ctx.verify_mode = ssl.CERT_NONE
        self._conn = None

    def _ensure_conn(self):
        """Create or reconnect the underlying HTTP connection."""
        if self._conn is not None:
            return
        if self._use_tls:
            self._conn = http.client.HTTPSConnection(
                self._host, self._port, context=self._ssl_ctx,
                timeout=self._timeout)
        else:
            self._conn = http.client.HTTPConnection(
                self._host, self._port, timeout=self._timeout)

    def _reconnect(self):
        """Force close and reconnect."""
        try:
            if self._conn:
                self._conn.close()
        except Exception:
            pass
        self._conn = None
        self._ensure_conn()

    def _merge_headers(self, extra_headers=None):
        """Merge constructor headers with per-call headers."""
        headers = dict(self.extra_headers)
        if extra_headers:
            headers.update(extra_headers)
        return request_headers(self.url, headers)

    def send(self, prompt, max_tokens=MAX_TOKENS, extra_headers=None):
        """Send a completion request, reusing the persistent connection.

        On connection reset, reconnects once and retries. Returns RequestResult.
        """
        self._ensure_conn()
        payload = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
        })
        headers = self._merge_headers(extra_headers)

        for attempt in range(2):
            start = time.monotonic()
            try:
                self._conn.request("POST", self._path, body=payload,
                                   headers=headers)
                response = self._conn.getresponse()

                first_byte = response.read(1)
                ttft = time.monotonic() - start
                rest = response.read()
                total = time.monotonic() - start

                body_bytes = first_byte + rest

                try:
                    body = json.loads(body_bytes)
                except (json.JSONDecodeError, ValueError):
                    body = {"raw": body_bytes.decode("utf-8", errors="replace")}

                usage = body.get("usage", {})
                return RequestResult(
                    ttft_ms=round(ttft * 1000, 2),
                    total_ms=round(total * 1000, 2),
                    status=response.status,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    body=body,
                    error=http_error(response.status, body),
                )

            except (ConnectionResetError, BrokenPipeError,
                    http.client.RemoteDisconnected, OSError) as e:
                if attempt == 0:
                    self._reconnect()
                    continue
                elapsed = time.monotonic() - start
                return RequestResult(
                    ttft_ms=0, total_ms=round(elapsed * 1000, 2),
                    status=0, prompt_tokens=0, completion_tokens=0,
                    error=str(e),
                )
            except Exception as e:
                elapsed = time.monotonic() - start
                return RequestResult(
                    ttft_ms=0, total_ms=round(elapsed * 1000, 2),
                    status=0, prompt_tokens=0, completion_tokens=0,
                    error=str(e),
                )

    def send_streaming(self, prompt, max_tokens=MAX_TOKENS, extra_headers=None):
        """Send a streaming completion request, reusing the persistent connection.

        Returns RequestResult with accurate ttft_ms and per-token token_times.
        """
        self._ensure_conn()
        payload = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
        })
        headers = self._merge_headers(extra_headers)

        for attempt in range(2):
            start = time.monotonic()
            try:
                self._conn.request("POST", self._path, body=payload,
                                   headers=headers)
                response = self._conn.getresponse()

                status = response.status
                if status != 200:
                    body = {"raw": response.read().decode("utf-8", errors="replace")}
                    return RequestResult(
                        ttft_ms=0, total_ms=round((time.monotonic() - start) * 1000, 2),
                        status=status, prompt_tokens=0, completion_tokens=0,
                        body=body, error=http_error(status, body),
                    )
                token_times = []
                prompt_tokens = 0
                last_chunk = {}

                while True:
                    line = response.readline()
                    if not line:
                        break
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line == "data: [DONE]":
                        # Drain remaining chunked response data so the
                        # connection is reusable for the next request.
                        response.read()
                        break
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            choices = chunk.get("choices", [])
                            if choices and choices[0].get("text", ""):
                                token_times.append(time.monotonic() - start)
                            usage = chunk.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", 0)
                            last_chunk = chunk
                        except (json.JSONDecodeError, ValueError):
                            pass

                total = time.monotonic() - start
                ct = len(token_times)
                ttft = token_times[0] if token_times else total

                return RequestResult(
                    ttft_ms=round(ttft * 1000, 2),
                    total_ms=round(total * 1000, 2),
                    status=status,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=ct,
                    body=last_chunk,
                    token_times=tuple(token_times),
                )

            except (ConnectionResetError, BrokenPipeError,
                    http.client.RemoteDisconnected, OSError) as e:
                if attempt == 0:
                    self._reconnect()
                    continue
                elapsed = time.monotonic() - start
                return RequestResult(
                    ttft_ms=0, total_ms=round(elapsed * 1000, 2),
                    status=0, prompt_tokens=0, completion_tokens=0,
                    error=str(e),
                )
            except Exception as e:
                elapsed = time.monotonic() - start
                return RequestResult(
                    ttft_ms=0, total_ms=round(elapsed * 1000, 2),
                    status=0, prompt_tokens=0, completion_tokens=0,
                    error=str(e),
                )

    def warmup(self, prompt, max_tokens=MAX_TOKENS, n=None, extra_headers=None):
        """Send warmup requests. First request triggers TLS handshake."""
        for _ in range(n if n is not None else WARMUP):
            self.send(prompt, max_tokens, extra_headers=extra_headers)

    def close(self):
        """Close the underlying connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        tls = "TLS" if self._use_tls else "plain"
        return f"PinnedConnection({self.pod_name}, {self._host}:{self._port}, {tls})"


# ── oc/kubectl subprocess wrapper ────────────────────────────────────────────

def oc(*args, timeout=60):
    """Run an ``oc`` CLI command, raising RuntimeError on non-zero exit.

    Used by advisor/_cluster.py and anywhere else that needs direct ``oc``
    access.  Consolidates the subprocess pattern that was previously
    duplicated across advisor modules and _discover_via_oc above.
    """
    import subprocess
    r = subprocess.run(
        ["oc", *list(args)], capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"oc {' '.join(args)}: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def oc_safe(*args, timeout=60):
    """Run an ``oc`` CLI command, returning (stdout, stderr) without raising."""
    import subprocess
    r = subprocess.run(
        ["oc", *list(args)], capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout.strip(), r.stderr.strip()


# ── Progress output ──────────────────────────────────────────────────────────

def write_run_info(experiment, extra=None):
    """Write or update DATA_DIR/run-info.json with toolkit config and timestamps.

    Each experiment call adds its entry under the experiment key. This makes
    every dataset self-describing: you can always tell what config, model, and
    environment produced the data.

    Args:
        experiment: Experiment identifier (e.g., "exp1", "exp4").
        extra: Optional dict of additional metadata (e.g., cluster info from oc).
    """
    import platform
    filepath = os.path.join(DATA_DIR, "run-info.json")

    # Load existing if present (multiple experiments append to same file)
    existing = {}
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass

    # Toolkit-level config (shared across all experiments)
    existing["toolkit"] = {
        "model": MODEL,
        "namespace": NS,
        "sim_mode": SIM,
        "transport": detect_transport(),
        "baseline_url": BASELINE_URL,
        "disagg_d1_url": DISAGG_D1_URL,
        "disagg_d2_url": DISAGG_D2_URL,
        "prefill_host": PREFILL_HOST,
        "routing_mode": ROUTING_MODE,
        "epp_url": EPP_URL,
        "warmup": WARMUP,
        "runs": RUNS,
        "max_tokens": MAX_TOKENS,
        "data_dir": DATA_DIR,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # Per-experiment entry
    entry = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        entry.update(extra)

    if "experiments" not in existing:
        existing["experiments"] = {}
    existing["experiments"][experiment] = entry

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")


def progress(msg, end="\n"):
    """Print progress to stderr so stdout stays clean for piping."""
    print(msg, end=end, file=sys.stderr, flush=True)


def dot():
    """Print a progress dot."""
    print(".", end="", file=sys.stderr, flush=True)


def print_config():
    """Print current configuration."""
    if SIM:
        progress("  Mode:     SIM (inference-sim, no GPU)")
    progress(f"  Model:    {MODEL}")
    progress(f"  NS:       {NS}")
    progress(f"  Baseline: {BASELINE_URL}")
    progress(f"  Decode-1: {DISAGG_D1_URL}")
    progress(f"  Decode-2: {DISAGG_D2_URL}")
    if EPP_URL:
        progress(f"  EPP:      {EPP_URL} ({ROUTING_MODE} default route)")
    progress(f"  Warmup:   {WARMUP}  Runs: {RUNS}  Max tokens: {MAX_TOKENS}")
    progress(f"  Data dir: {DATA_DIR}")
    progress("")
