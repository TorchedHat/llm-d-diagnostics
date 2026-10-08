"""
Root-cause diagnosis with fix commands for disaggregated inference.

Each check maps to a REAL production bug from llm-d, NVIDIA Dynamo,
or SGLang. Every issue includes evidence from YOUR cluster, a cause
explanation with a reference link, and a copy-pasteable fix command.

Part of the llm-d-diagnostics advisory layer.
"""

import argparse
import os
import sys
from dataclasses import dataclass

try:
    from ._cluster import get_pods_full as get_pods
    from ._cluster import oc_safe, scrape_pod_metrics
    from .plan import ModelProfile, _estimate_kv_bytes, _estimate_nixl_ms
    from .probe import _send_probe
except ImportError:
    from _cluster import get_pods_full as get_pods  # type: ignore[no-redef]
    from _cluster import oc_safe, scrape_pod_metrics  # type: ignore[no-redef]
    from plan import ModelProfile, _estimate_kv_bytes, _estimate_nixl_ms  # type: ignore[no-redef]
    from probe import _send_probe  # type: ignore[no-redef]


@dataclass
class Issue:
    severity: str       # "critical" | "warning" | "info"
    title: str
    evidence: str
    cause: str
    reference: str      # URL to the bug report / doc
    fix: str            # copy-pasteable command
    auto_fixable: bool = False


def diagnose(namespace: str, model: str = "") -> list[Issue]:
    """Run all diagnostic checks and return found issues."""
    issues = []
    pods = get_pods(namespace, "app.kubernetes.io/part-of=vllm-disagg")
    prefill = [p for p in pods if "prefill" in p.get("role", "")]
    decode = [p for p in pods if "decode" in p.get("role", "")]

    if not pods:
        issues.append(Issue(
            "critical", "No P/D pods found",
            f"Namespace {namespace} has no pods with label app.kubernetes.io/part-of=vllm-disagg",
            "Deployment may have failed or wrong namespace specified",
            "", f"oc get pods -n {namespace} -o wide",
        ))
        return issues

    issues.extend(_check_pod_health(prefill, decode, namespace))
    issues.extend(_check_image_mismatch(pods, namespace))
    issues.extend(_check_prefill_spof(prefill, namespace))
    issues.extend(_check_kv_roles(pods, namespace))
    issues.extend(_check_nixl_failures(pods, namespace))
    issues.extend(_check_kv_cache_pressure(pods, namespace))
    issues.extend(_check_stale_kv_timeout(pods, namespace))
    issues.extend(_check_nixl_config(prefill, namespace))
    issues.extend(_check_kv_expiration(pods, namespace))
    issues.extend(_check_transfer_duration(pods, namespace, model=model))

    if model:
        issues.extend(_check_probe_health(namespace, model))

    return issues


def _check_pod_health(prefill, decode, ns) -> list[Issue]:
    issues = []
    not_ready = [p["name"] for p in prefill + decode if not p["ready"]]
    for name in not_ready:
        issues.append(Issue(
            "critical", f"Pod {name} not ready",
            f"Pod {name} is not in Ready state",
            "Pod may be in CrashLoopBackOff, pending scheduling, or failing probes",
            "",
            f"oc describe pod {name} -n {ns} | tail -20",
        ))
    if not prefill:
        issues.append(Issue(
            "critical", "No prefill pods found",
            "Zero pods with app=vllm-prefill label",
            "Prefill deployment may be scaled to 0 or not created",
            "",
            f"oc get deployment -n {ns} -l app.kubernetes.io/part-of=vllm-disagg",
        ))
    if not decode:
        issues.append(Issue(
            "critical", "No decode pods found",
            "Zero pods with app=vllm-decode label",
            "Decode deployment may be scaled to 0 or not created",
            "",
            f"oc get deployment -n {ns} -l app.kubernetes.io/part-of=vllm-disagg",
        ))
    return issues


def _check_image_mismatch(pods, ns) -> list[Issue]:
    """Based on ai-dynamo#6671: NIXL version conflicts from different images."""
    vllm_pods = [p for p in pods if "prefill" in p.get("role", "") or "decode" in p.get("role", "")]
    if not vllm_pods:
        vllm_pods = pods
    images: dict[str, set[str]] = {}
    for p in vllm_pods:
        role = p.get("role", p["labels"].get("app", "unknown"))
        images.setdefault(role, set()).add(p["image"])

    all_images = set()
    for imgs in images.values():
        all_images.update(imgs)

    if len(all_images) > 1:
        img_list = ", ".join(f"{role}: {next(iter(imgs))}" for role, imgs in images.items())
        target_image = sorted(all_images)[0]
        return [Issue(
            "critical",
            "vLLM image version mismatch across pods",
            f"Different images detected: {img_list}",
            "Mismatched vLLM/NIXL versions cause handshake failures, "
            "transfer errors, or silent data corruption. "
            "Three competing NIXL installations in the same container "
            "caused hours of debugging (ai-dynamo#6671).",
            "https://github.com/ai-dynamo/dynamo/issues/6671",
            f"oc set image deployment/vllm-decode vllm={target_image} -n {ns} && "
            f"oc set image deployment/vllm-prefill vllm={target_image} -n {ns}",
            auto_fixable=True,
        )]
    return []


def _check_prefill_spof(prefill, ns) -> list[Issue]:
    """Based on llm-d ops doc: prefill is a single point of failure."""
    if len(prefill) == 1:
        return [Issue(
            "warning",
            "Prefill is a single point of failure (SPOF)",
            f"Only 1 prefill pod running ({prefill[0]['name']}). "
            f"If it dies, ALL disaggregated requests fail.",
            "llm-d operations documentation explicitly warns: "
            "'prefill needs redundancy.' Our experiments confirmed "
            "100% request failure during prefill outage with ~330s recovery.",
            "https://github.com/llm-d/llm-d/blob/main/docs/wip-docs-new/architecture/advanced/disaggregation/operations-vllm.md",
            f"oc scale deployment/vllm-prefill --replicas=2 -n {ns}",
            auto_fixable=True,
        )]
    return []


def _check_kv_roles(pods, ns) -> list[Issue]:
    producers = [p for p in pods if "kv_producer" in p["args"] or "kv_both" in p["args"]]

    if not producers:
        return [Issue(
            "critical",
            "No KV producer configured",
            "No pods have kv_role=kv_producer or kv_both in their args",
            "Without a KV producer, NIXL KV cache transfer cannot work. "
            "Disaggregated requests will fail silently.",
            "",
            f"Check --kv-transfer-config in deployment args: "
            f"oc get deployment -n {ns} -o jsonpath='{{.items[*].spec.template.spec.containers[0].args}}'",
        )]
    return []


def _check_nixl_failures(pods, ns="default") -> list[Issue]:
    """Check for active NIXL transfer failures via metrics."""
    for p in pods:
        body = scrape_pod_metrics(p, ns)
        if not body:
            continue
        for line in body.split("\n"):
            if line.startswith("vllm:nixl_num_failed_transfers"):
                try:
                    val = float(line.split()[-1])
                    if val > 0:
                        return [Issue(
                            "critical",
                            f"NIXL transfer failures detected on {p['name']}",
                            f"{int(val)} failed NIXL KV transfers",
                            "NIXL transfer failures indicate network issues between "
                            "prefill and decode pods, version incompatibility, or "
                            "NIXL handshake corruption (llm-d#759).",
                            "https://github.com/llm-d/llm-d/issues/759",
                            f"oc logs {p['name']} -n {ns} -c vllm | grep -i 'nixl\\|transfer\\|error' | tail -20",
                        )]
                except (ValueError, IndexError):
                    pass
    return []


def _check_kv_cache_pressure(pods, ns="default") -> list[Issue]:
    """Check for KV cache nearing capacity -- possible leak (ai-dynamo#6071)."""
    for p in pods:
        body = scrape_pod_metrics(p, ns)
        if not body:
            continue
        for line in body.split("\n"):
            if line.startswith("vllm:kv_cache_usage_perc"):
                try:
                    val = float(line.split()[-1])
                    if val > 0.9:
                        return [Issue(
                            "warning",
                            f"KV cache pressure on {p['name']} ({val*100:.0f}%)",
                            f"KV cache at {val*100:.0f}% utilization",
                            "High KV cache usage can cause preemptions and latency spikes. "
                            "If KV usage grows without corresponding request load, it may be "
                            "a KV cache leak from cancelled requests (ai-dynamo#6071).",
                            "https://github.com/ai-dynamo/dynamo/issues/6071",
                            f"oc rollout restart deployment/vllm-prefill -n {ns}",
                            auto_fixable=True,
                        )]
                except (ValueError, IndexError):
                    pass
    return []


def _check_stale_kv_timeout(pods, ns) -> list[Issue]:
    """Check if NIXL abort timeout is too long (llm-d ops doc)."""
    for p in pods:
        if "prefill" not in p.get("role", ""):
            continue
        out, _ = oc_safe("exec", p["name"], "-n", ns, "-c", "vllm",
                          "--", "printenv", "VLLM_NIXL_ABORT_REQUEST_TIMEOUT", timeout=10)
        timeout_val = out.strip()
        if timeout_val:
            try:
                t = int(timeout_val)
                if t > 300:
                    return [Issue(
                        "info",
                        f"NIXL abort timeout is high ({t}s)",
                        f"VLLM_NIXL_ABORT_REQUEST_TIMEOUT={t}s on {p['name']}",
                        "When a decode pod crashes, KV blocks on prefill are stranded "
                        "until this timeout expires. The default 480s means ~8 minutes "
                        "of wasted GPU memory per crash.",
                        "https://github.com/llm-d/llm-d/blob/main/docs/wip-docs-new/architecture/advanced/disaggregation/operations-vllm.md",
                        f"oc set env deployment/vllm-prefill VLLM_NIXL_ABORT_REQUEST_TIMEOUT=120 -n {ns}",
                        auto_fixable=True,
                    )]
            except ValueError:
                pass
    return []


def _check_nixl_config(prefill, ns) -> list[Issue]:
    """Check NIXL side-channel host is properly configured."""
    for p in prefill:
        out, _ = oc_safe("get", "pod", p["name"], "-n", ns,
                          "-o", "jsonpath={.spec.containers[0].env}", timeout=10)
        if "VLLM_NIXL_SIDE_CHANNEL_HOST" not in out:
            return [Issue(
                "warning",
                "NIXL side-channel host not configured on prefill",
                f"Pod {p['name']} missing VLLM_NIXL_SIDE_CHANNEL_HOST env var",
                "Without this, NIXL uses localhost which prevents cross-pod "
                "KV cache transfer. Set it to status.podIP via fieldRef.",
                "",
                "# Add to prefill deployment env:\n"
                "# - name: VLLM_NIXL_SIDE_CHANNEL_HOST\n"
                "#   valueFrom:\n"
                "#     fieldRef:\n"
                "#       fieldPath: status.podIP",
            )]
    return []


def _check_kv_expiration(pods, ns) -> list[Issue]:
    """Check for KV block expiration -- vital deployment health indicator (vLLM PR #32340)."""
    issues = []
    for p in pods:
        body = scrape_pod_metrics(p, ns)
        if not body:
            continue
        for line in body.split("\n"):
            if "nixl_num_kv_expired_reqs_total" in line and not line.startswith("#"):
                try:
                    val = float(line.split()[-1])
                    if val > 0:
                        issues.append(Issue(
                            "critical",
                            f"KV block expiration on {p['name']}",
                            f"{int(val)} requests had KV blocks expire before decode consumed them",
                            "Stranded KV blocks waste GPU memory and cause request failures. "
                            "The NIXL abort timeout is too low or decode is overloaded.",
                            "https://github.com/vllm-project/vllm/pull/32340",
                            f"oc set env deployment/vllm-decode -n {ns} VLLM_NIXL_ABORT_REQUEST_TIMEOUT=300",
                            auto_fixable=True,
                        ))
                except (ValueError, IndexError):
                    pass
    return issues


def _check_transfer_duration(pods, ns="default", model: str = "",
                             gpu_type: str = "t4") -> list[Issue]:
    """Check NIXL transfer duration for anomalies against expected model."""
    threshold_ms: float = 500
    try:
        profile = ModelProfile(model_id=model) if model else ModelProfile(model_id="")
        kv_bytes = _estimate_kv_bytes(profile)
        expected_ms = _estimate_nixl_ms(kv_bytes, seq_len=128, gpu_type=gpu_type)
        threshold_ms = expected_ms * 3
    except Exception:
        pass

    issues = []
    for p in pods:
        body = scrape_pod_metrics(p, ns)
        if not body:
            continue
        transfer_sum = 0.0
        transfer_count = 0.0
        for line in body.split("\n"):
            if "nixl_transfer_duration_seconds_sum" in line and not line.startswith("#"):
                try:
                    transfer_sum = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
            if "nixl_transfer_duration_seconds_count" in line and not line.startswith("#"):
                try:
                    transfer_count = float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
        if transfer_count > 0:
            avg_ms = transfer_sum / transfer_count * 1000
            if avg_ms > threshold_ms:
                issues.append(Issue(
                    "warning",
                    f"High NIXL transfer duration on {p['name']}",
                    f"Avg transfer: {avg_ms:.0f}ms over {int(transfer_count)} transfers "
                    f"(threshold: {threshold_ms:.0f}ms)",
                    "Network congestion or misconfigured NIXL buffer size. "
                    "Consider RDMA if available, or check for NIC bandwidth saturation.",
                    "https://github.com/ai-dynamo/nixl/blob/main/benchmark/nixlbench/README.md",
                    f"# Check network: oc exec {p['name']} -n {ns} -- ip link show",
                ))
    return issues


def _check_probe_health(ns, model, decode_url="", prefill_host="") -> list[Issue]:
    """Send a test disagg request to verify end-to-end pipeline."""
    scheme = os.environ.get("SIDECAR_SCHEME", "http")
    url = decode_url or f"{scheme}://vllm-decode-svc:8000/v1/completions"
    prefill_port = 8100 if os.environ.get("SIM") == "1" else 8000
    host = prefill_host or f"vllm-prefill-svc.{ns}.svc.cluster.local:{prefill_port}"
    headers = {"x-prefiller-host-port": host}
    r = _send_probe(url, model, extra_headers=headers)
    if r["status"] != 200:
        return [Issue(
            "critical",
            "Disaggregated inference pipeline is broken",
            f"Test request returned status={r['status']}, error={r['error'][:100]}",
            "The full pipeline (client -> sidecar -> prefill -> NIXL -> decode) is not working. "
            "Check pod health, NIXL handshake, and sidecar routing.",
            "",
            f"oc logs -l app=vllm-decode -c routing-sidecar -n {ns} --tail=20",
        )]
    return []


def print_diagnosis(issues: list[Issue]):
    """Print diagnosis results to console."""
    if not issues:
        print("\n  ALL CHECKS PASSED -- no issues detected.\n")
        return

    print(f"\n{'='*60}")
    print(f"  DIAGNOSIS: {len(issues)} issue(s) found")
    print(f"{'='*60}")

    for _i, issue in enumerate(issues, 1):
        sev = {"critical": "CRIT", "warning": "WARN", "info": "INFO"}[issue.severity]
        print(f"\n  [{sev}] {issue.title}")
        print(f"  Evidence: {issue.evidence}")
        if issue.cause:
            print(f"  Cause:    {issue.cause[:120]}")
        if issue.reference:
            print(f"  Ref:      {issue.reference}")
        print(f"  Fix:      {issue.fix}")
        if issue.auto_fixable:
            print("            (auto-fixable: copy-paste the command above)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Root-cause diagnosis for disaggregated inference clusters")
    parser.add_argument("--namespace", "-n", required=True, help="Kubernetes namespace")
    parser.add_argument("--model", "-m", default="", help="Model name (enables probe check)")
    args = parser.parse_args()

    issues = diagnose(args.namespace, args.model)
    print_diagnosis(issues)
    sys.exit(1 if any(i.severity == "critical" for i in issues) else 0)
