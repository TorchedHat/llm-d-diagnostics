#!/bin/bash
# Run toolkit experiments in-cluster via oc exec.
#
# Usage:
#   ./toolkit/run.sh <cluster-dir> <command>
#
# Commands:
#   characterize    Run all non-destructive experiments + analysis (recommended)
#   fault-test      Run fault tolerance experiments + analysis (destructive)
#
#   Individual experiments:
#     latency       Single-request latency sweep
#     decompose     Latency decomposition (sidecar vs NIXL)
#     throughput    Throughput under concurrent load
#     isolation     Prefill isolation (head-of-line blocking)
#     seqlen        Sequence length sweep (KV transfer scaling)
#     saturation    QPS saturation profiling
#     mixed         Mixed workload (realistic traffic)
#     prefix-cache  KV cache hit rates across requests and pods
#     kv-eviction   KV cache persistence under delay and pressure
#     tput-seqlen   Throughput vs prompt length
#     tput-outlen   Throughput vs output length
#     tput-sat      Saturation ceiling (high concurrency)
#     overhead-load Overhead decomposition under concurrent load
#     fault         Fault tolerance (kills pods — destructive)
#     model-load    Cold start time (kills pods — destructive)
#
#   Advisory (advisor/):
#     diagnose      Root-cause diagnosis with fix commands
#     health        Continuous health monitoring + trend detection
#     plan          GPU capacity planning + cost model
#     rebalance     P/D ratio recommendation + watch mode
#
#   Infrastructure:
#     deploy        Deploy P/D topology from env.sh
#     undeploy      Tear down deployment (--keep-pvc to keep model cache)
#     sweep         Run experiments across multiple models (JSON config)
#                   Usage: ./toolkit/run.sh sweeps/kv-ratio.json sweep
#
#   Utilities:
#     analyze       Statistical analysis of collected data (local)
#     metrics       Standalone Prometheus metrics collection
#     preflight     Verify cluster is ready for experiments
#     transport     Witness which transport the KV transfer uses (TCP, RDMA, CUDA IPC)
#
# Prerequisites:
#   1. test-client pod is running (oc apply -f manifests/ -n <namespace>)
#   2. Cluster env is at <cluster-dir>/env.sh
#
# Examples:
#   ./toolkit/run.sh clusters/my-cluster characterize
#   ./toolkit/run.sh clusters/my-cluster latency
#   ./toolkit/run.sh clusters/my-cluster fault-test
#   ./toolkit/run.sh clusters/my-cluster fault 4a 4h --skip-control
#   ./toolkit/run.sh clusters/my-cluster analyze

set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <cluster-dir> <command>}"
COMMAND="${2:-characterize}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Sweep takes a JSON config as first arg, not a cluster dir
if [ "$COMMAND" = "sweep" ]; then
    python3 "$REPO_ROOT/toolkit/sweep.py" "$CLUSTER_DIR" "${@:3}"
    exit 0
fi

if [ ! -f "$CLUSTER_DIR/env.sh" ]; then
    echo "ERROR: $CLUSTER_DIR/env.sh not found"
    exit 1
fi

source "$CLUSTER_DIR/env.sh"
SIDECAR_SCHEME="${SIDECAR_SCHEME:-http}"
if [[ "$SIDECAR_SCHEME" != "http" && "$SIDECAR_SCHEME" != "https" ]]; then
    echo "ERROR: SIDECAR_SCHEME must be 'http' or 'https'"
    exit 1
fi
export SIDECAR_SCHEME
ROUTING_MODE="${ROUTING_MODE:-manual}"
EPP_URL="${EPP_URL:-}"
EPP_RELEASE_NAME="${EPP_RELEASE_NAME:-llm-d-epp}"
EPP_SERVICE_NAME="${EPP_SERVICE_NAME:-${EPP_RELEASE_NAME}-epp}"
EPP_METRICS_URL="${EPP_METRICS_URL:-http://${EPP_SERVICE_NAME}:9090}"
if [[ "$ROUTING_MODE" != "manual" && "$ROUTING_MODE" != "epp" ]]; then
    echo "ERROR: ROUTING_MODE must be 'manual' or 'epp'"
    exit 1
fi
if [[ "$ROUTING_MODE" == "epp" && -z "$EPP_URL" ]]; then
    echo "ERROR: ROUTING_MODE=epp requires EPP_URL in $CLUSTER_DIR/env.sh"
    exit 1
fi
if [[ -n "$EPP_URL" && "$EPP_URL" != http://* && "$EPP_URL" != https://* ]]; then
    echo "ERROR: EPP_URL must start with http:// or https://"
    exit 1
fi
export ROUTING_MODE EPP_URL EPP_RELEASE_NAME EPP_SERVICE_NAME EPP_METRICS_URL
PREFILL_PORT=8000
if [[ "${SIM:-}" == "1" ]]; then
    PREFILL_PORT=8100
fi
PREFILL_HOST="${PREFILL_HOST:-vllm-prefill-svc.${NS}.svc.cluster.local:${PREFILL_PORT}}"
BASELINE_URL="${BASELINE_URL:-http://vllm-prefill-svc:${PREFILL_PORT}/v1/completions}"
DISAGG_URL="${DISAGG_URL:-${SIDECAR_SCHEME}://vllm-decode-svc:8000/v1/completions}"
DISAGG_D1_URL="${DISAGG_D1_URL:-$DISAGG_URL}"
DISAGG_D2_URL="${DISAGG_D2_URL:-$DISAGG_URL}"
export PREFILL_HOST BASELINE_URL DISAGG_URL DISAGG_D1_URL DISAGG_D2_URL
mkdir -p "$DATA_DIR"

POD="${TEST_CLIENT:-test-client}"
REMOTE_DIR="/scripts/toolkit"
# One data directory per cluster config: the test-client pod outlives
# redeployments, and a shared directory would mix results from different
# deployments (e.g. TCP and RDMA) when they are copied back.
REMOTE_DATA="$REMOTE_DIR/data/$(basename "$CLUSTER_DIR")"

# ── Preflight check ───────────────────────────────────────────────────────
check_network_policies() {
    local policies
    policies=$(oc get networkpolicy -n "$NS" -o json 2>/dev/null | python3 -c '
import json, sys
def selector(sel):
    parts = [f"{k}={v}" for k, v in sorted(sel.get("matchLabels", {}).items())]
    for e in sel.get("matchExpressions", []):
        key, vals = e["key"], ",".join(e.get("values", []))
        parts.append({"In": f"{key} in ({vals})", "NotIn": f"{key} notin ({vals})",
                      "Exists": key, "DoesNotExist": f"!{key}"}[e["operator"]])
    return ",".join(parts)
for p in json.load(sys.stdin).get("items", []):
    print(p["metadata"]["name"] + "|" + selector(p["spec"].get("podSelector", {})))
' 2>/dev/null) || return 0

    local found=false name sel pods
    while IFS='|' read -r name sel; do
        [ -z "$name" ] && continue
        pods=$(oc get pods -n "$NS" \
            -l "app.kubernetes.io/part-of in (vllm-disagg,vllm-disagg-sim)${sel:+,$sel}" \
            -o name 2>/dev/null | sed 's#^pod/##' | tr '\n' ' ')
        if [ -n "$pods" ]; then
            echo "  WARN: NetworkPolicy '$name' selects: $pods"
            found=true
        fi
    done <<< "$policies"
    if [ "$found" = true ]; then
        echo "        Traffic to these pods is limited to what the policy allows."
        echo "        It must allow HTTP (8000, 8001) and the NIXL side channel"
        echo "        (${NIXL_PORT:-5557}). Inspect: oc get networkpolicy -n $NS -o yaml"
    fi
}

preflight() {
    echo "=== Preflight Check ==="
    local ok=true

    # 1. Verify test-client pod is running
    if ! oc get pod "$POD" -n "$NS" -o jsonpath='{.status.phase}' 2>/dev/null | grep -q Running; then
        echo "  FAIL: Pod '$POD' is not running in namespace '$NS'"
        echo "        Deploy with: oc apply -f manifests/ -n $NS"
        return 1
    fi
    echo "  OK: test-client pod is running"

    # 2. Copy toolkit to pod
    echo "  Copying toolkit to pod..."
    oc exec "$POD" -n "$NS" -- mkdir -p "$REMOTE_DIR" "$REMOTE_DATA"
    oc cp "$SCRIPT_DIR/" "$NS/$POD:$(dirname $REMOTE_DIR)/"

    # 3. Warn about NetworkPolicies that select the model pods. A leftover
    #    policy from a partition experiment silently blocks HTTP or NIXL.
    check_network_policies

    # 4. Verify each endpoint with a single request
    local env_vars="MODEL=$MODEL NS=$NS DATA_DIR=$REMOTE_DATA PREFILL_HOST=$PREFILL_HOST SIDECAR_SCHEME=$SIDECAR_SCHEME"
    [ -n "${SIM:-}" ] && env_vars="$env_vars SIM=$SIM"

    local endpoints=(
        "baseline|$BASELINE_URL|none"
        "manual-decode|$DISAGG_D1_URL|manual"
    )
    if [[ "$DISAGG_D2_URL" != "$DISAGG_D1_URL" ]]; then
        endpoints+=("manual-decode-2|$DISAGG_D2_URL|manual")
    fi
    if [ -n "$EPP_URL" ]; then
        endpoints+=("epp|$EPP_URL|epp")
    fi
    for endpoint in "${endpoints[@]}"; do
        IFS='|' read -r name url route <<< "$endpoint"

        local result
        result=$(oc exec "$POD" -n "$NS" -- env $env_vars ENDPOINT_URL="$url" ROUTE_KIND="$route" \
            python3 -c "
import sys; sys.path.insert(0, '$REMOTE_DIR')
import os
from client import send_request, PREFILL_HOST
headers = {'x-prefiller-host-port': PREFILL_HOST} if os.environ['ROUTE_KIND'] == 'manual' else None
r = send_request(os.environ['ENDPOINT_URL'], 'hello', 5, extra_headers=headers)
print(f'{r.status}|{r.completion_tokens}|{r.error}')
" 2>/dev/null || echo "0|0|connection failed")

        local status=$(echo "$result" | tail -1 | cut -d'|' -f1)
        local tokens=$(echo "$result" | tail -1 | cut -d'|' -f2)
        local error=$(echo "$result" | tail -1 | cut -d'|' -f3)

        if [ "$status" = "200" ] && [ "$tokens" -gt 0 ] 2>/dev/null; then
            echo "  OK: $name ($tokens tokens)"
        else
            echo "  FAIL: $name (status=$status, error=$error)"
            ok=false
        fi
    done

    if [ "$ok" = false ]; then
        echo ""
        echo "  Preflight FAILED. Fix the issues above before running experiments."
        return 1
    fi

    echo "  All endpoints responding. Ready to run experiments."
    echo ""
    return 0
}

# ── Run remote experiment ──────────────────────────────────────────────────
# Discover pod IPs from host (where oc works) and pass as env vars.
# This provides pod discovery without requiring RBAC inside test-client.
PREFILL_IPS=$(oc get pods -l "${PREFILL_SELECTOR:-app=vllm-prefill}" -n "$NS" \
    -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{":"}{.status.podIP}{","}{end}' \
    2>/dev/null | sed 's/,$//')
DECODE_IPS=$(oc get pods -l "${DECODE_SELECTOR:-app=vllm-decode}" -n "$NS" \
    -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{":"}{.status.podIP}{","}{end}' \
    2>/dev/null | sed 's/,$//')

REMOTE_ENV="MODEL=$MODEL NS=$NS DATA_DIR=$REMOTE_DATA PREFILL_HOST=$PREFILL_HOST SIDECAR_SCHEME=$SIDECAR_SCHEME ROUTING_MODE=$ROUTING_MODE EPP_URL=$EPP_URL EPP_RELEASE_NAME=$EPP_RELEASE_NAME EPP_SERVICE_NAME=$EPP_SERVICE_NAME EPP_METRICS_URL=$EPP_METRICS_URL"
REMOTE_ENV="$REMOTE_ENV BASELINE_URL=$BASELINE_URL DISAGG_URL=$DISAGG_URL DISAGG_D1_URL=$DISAGG_D1_URL DISAGG_D2_URL=$DISAGG_D2_URL"
# One prompt nonce for every experiment in this invocation (client.py
# build_prompt). Set PROMPT_NONCE to reproduce a recorded run's prompts.
PROMPT_NONCE="${PROMPT_NONCE:-$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')}"
if ! [[ "$PROMPT_NONCE" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: PROMPT_NONCE may contain only letters, digits, '.', '_' and '-'"
    exit 1
fi
REMOTE_ENV="$REMOTE_ENV PROMPT_NONCE=$PROMPT_NONCE"
for variable in RUNS WARMUP MAX_TOKENS METRICS_ENDPOINTS; do
    if [[ -n "${!variable:-}" ]]; then
        REMOTE_ENV="$REMOTE_ENV $variable=${!variable}"
    fi
done
[ -n "$PREFILL_IPS" ] && REMOTE_ENV="$REMOTE_ENV PODS_VLLM_PREFILL=$PREFILL_IPS"
[ -n "$DECODE_IPS" ] && REMOTE_ENV="$REMOTE_ENV PODS_VLLM_DECODE=$DECODE_IPS"
[ -n "${SIM:-}" ] && REMOTE_ENV="$REMOTE_ENV SIM=$SIM"
[ -n "${CONFIGS:-}" ] && REMOTE_ENV="$REMOTE_ENV CONFIGS=$CONFIGS"
[ -n "${CONCURRENCY_LEVELS:-}" ] && REMOTE_ENV="$REMOTE_ENV CONCURRENCY_LEVELS=$CONCURRENCY_LEVELS"
[ -n "${SWEEP_LENGTHS:-}" ] && REMOTE_ENV="$REMOTE_ENV SWEEP_LENGTHS=$SWEEP_LENGTHS"
[ -n "${TOTAL_REQUESTS:-}" ] && REMOTE_ENV="$REMOTE_ENV TOTAL_REQUESTS=$TOTAL_REQUESTS"

run_remote() {
    local label="$1"
    local script="$2"
    # KV accounting: snapshot every vLLM pod's prompt-token sources around
    # the experiment (kv_sources.py). Not available in simulation mode.
    local exp="${script%%_*}" kv_dir="$DATA_DIR/kv-sources"
    if [ -z "${SIM:-}" ]; then
        mkdir -p "$kv_dir"
        NS="$NS" python3 "$SCRIPT_DIR/kv_sources.py" snapshot "$kv_dir/$exp.before.json" \
            || echo "  WARN: KV accounting snapshot failed"
    fi
    echo "=== $label ==="
    oc exec "$POD" -n "$NS" -- env $REMOTE_ENV python3 "$REMOTE_DIR/$script"
    if [ -z "${SIM:-}" ] && [ -f "$kv_dir/$exp.before.json" ]; then
        NS="$NS" python3 "$SCRIPT_DIR/kv_sources.py" snapshot "$kv_dir/$exp.after.json" \
            && NS="$NS" python3 "$SCRIPT_DIR/kv_sources.py" delta \
                "$kv_dir/$exp.before.json" "$kv_dir/$exp.after.json" "$kv_dir/$exp.json" \
            || echo "  WARN: KV accounting failed"
    fi
    echo ""
}

# ── Metrics collector (background, in-pod) ──────────────────────────────────
METRICS_PID_FILE="/tmp/metrics-collector-$NS.pid"

start_metrics() {
    echo "Starting metrics collector (background, in-pod)..."
    oc exec "$POD" -n "$NS" -- env $REMOTE_ENV SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-2}" \
        python3 "$REMOTE_DIR/metrics_collector.py" &
    METRICS_BG_PID=$!
    echo "$METRICS_BG_PID" > "$METRICS_PID_FILE"
}

stop_metrics() {
    if [ -f "$METRICS_PID_FILE" ]; then
        local pid
        pid=$(cat "$METRICS_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rm -f "$METRICS_PID_FILE"
    fi
    # Killing the local `oc exec` does not stop the collector inside the pod.
    oc exec "$POD" -n "$NS" -- pkill -f metrics_collector.py 2>/dev/null || true
}

# ── Copy results back ─────────────────────────────────────────────────────
copy_results() {
    # Stop writers first: copying a directory that changes mid-read is how
    # `oc cp` fails with "unexpected EOF". A tar stream over exec is also
    # more reliable than `oc cp` for large directories.
    stop_metrics
    echo "Copying results from pod..."
    mkdir -p "$DATA_DIR"
    local attempt
    for attempt in 1 2 3; do
        if oc exec "$POD" -n "$NS" -- tar czf - -C "$REMOTE_DATA" . \
                | tar xzf - -C "$DATA_DIR"; then
            echo "Results saved to $DATA_DIR/"
            return 0
        fi
        echo "  copy attempt $attempt failed; retrying..."
        sleep 5
    done
    echo "ERROR: could not copy results; they remain in $POD:$REMOTE_DATA"
    return 1
}

# ── Transport witness ─────────────────────────────────────────────────────
# Measures which transport the KV transfer actually uses (transport.py) and
# copies transport.json into the pod so run-info.json records the verdict.
# Sends a short burst of requests: run it before experiments, never during.
run_transport_witness() {
    mkdir -p "$DATA_DIR"
    local rc=0
    NS="$NS" DATA_DIR="$DATA_DIR" POD="$POD" REMOTE_DIR="$REMOTE_DIR" \
        SIDECAR_SCHEME="$SIDECAR_SCHEME" \
        python3 "$SCRIPT_DIR/transport.py" || rc=$?
    if [ -f "$DATA_DIR/transport.json" ]; then
        oc cp "$DATA_DIR/transport.json" "$NS/$POD:$REMOTE_DATA/transport.json"
    fi
    return $rc
}

# ── Run analysis ──────────────────────────────────────────────────────────
run_analyze() {
    echo ""
    echo "=== Analysis ==="
    python3 "$SCRIPT_DIR/analyze.py" "$DATA_DIR" | tee "$DATA_DIR/analysis.txt"
    echo ""
    echo "Analysis saved to $DATA_DIR/analysis.txt"
}

# ── Run individual experiment (with metrics + copy) ────────────────────────
run_single() {
    local label="$1"
    local script="$2"

    preflight || exit 1
    start_metrics
    trap stop_metrics EXIT

    run_remote "$label" "$script"

    copy_results
}

# ── Experiment name mapping ────────────────────────────────────────────────
# Maps user-facing names to (label, script) pairs.
# Internal IDs (CSV columns, filenames) are unchanged.
run_experiment() {
    case "$1" in
        latency|exp1)     run_single "Latency Sweep" exp1_latency.py ;;
        decompose|exp1b)  run_single "Latency Decomposition" exp1b_decompose.py ;;
        throughput|exp2)   run_single "Throughput Under Load" exp2_throughput.py ;;
        isolation|exp3)   run_single "Prefill Isolation" exp3_isolation.py ;;
        seqlen|exp5)      run_single "Sequence Length Sweep" exp5_seqlen_sweep.py ;;
        saturation|exp6)  run_single "Saturation Profiling" exp6_saturation.py ;;
        mixed|exp7)       run_single "Mixed Workload" exp7_mixed_workload.py ;;
        prefix-cache|exp8) run_single "Prefix Cache" exp8_prefix_cache.py ;;
        model-load|exp9)
            echo "model-load runs locally (kills pods via oc). Use: ./toolkit/run.sh <cluster> model-load"
            return 1
            ;;
        kv-eviction|exp10) run_single "KV Cache Eviction" exp10_kv_eviction.py ;;
        tput-seqlen|exp11) run_single "Throughput vs Prompt Length" exp11_tput_seqlen.py ;;
        tput-outlen|exp12) run_single "Throughput vs Output Length" exp12_tput_outlen.py ;;
        tput-sat|exp13) run_single "Saturation Ceiling" exp13_tput_sat.py ;;
        overhead-load|exp14) run_single "Overhead Under Load" exp14_overhead_load.py ;;
        *)
            echo "Unknown experiment: $1"
            return 1
            ;;
    esac
}

# ── Main dispatch ──────────────────────────────────────────────────────────

echo "Cluster:    $CLUSTER_DIR"
echo "Namespace:  $NS"
echo ""

case "$COMMAND" in
    characterize)
        # Full non-destructive characterization (~15 min with early-stop).
        # Ordered by information density: decompose first (most informative),
        # saturation last (early-stops when system collapses).
        preflight || exit 1
        [ -z "${SIM:-}" ] && { run_transport_witness || echo "  WARN: transport not established; see above"; }
        start_metrics
        trap stop_metrics EXIT

        run_remote "Latency Decomposition" exp1b_decompose.py
        run_remote "Latency Sweep" exp1_latency.py
        run_remote "Prefill Isolation" exp3_isolation.py
        run_remote "Throughput Under Load" exp2_throughput.py
        run_remote "Saturation Profiling" exp6_saturation.py
        run_remote "Sequence Length Sweep" exp5_seqlen_sweep.py
        run_remote "Mixed Workload" exp7_mixed_workload.py
        run_remote "Prefix Cache" exp8_prefix_cache.py
        run_remote "KV Cache Eviction" exp10_kv_eviction.py

        copy_results
        run_analyze
        ;;

    fault-test|fault|exp4)
        # Fault tolerance: runs locally (kills pods via oc)
        preflight || exit 1

        echo "=== Fault Tolerance Test ==="
        echo "WARNING: This will kill pods in namespace $NS"
        echo ""

        # Forward remaining args for sub-experiment selection:
        #   ./run.sh clusters/my-cluster fault 4a 4h --skip-control
        shift 2  # remove cluster-dir and command
        python3 "$SCRIPT_DIR/exp4_fault.py" "$@"

        run_analyze
        ;;

    analyze)
        run_analyze
        ;;

    preflight)
        preflight
        ;;

    transport)
        preflight || exit 1
        run_transport_witness
        ;;

    metrics)
        echo "=== Metrics Collection ==="
        echo "Collecting for ${COLLECT_DURATION:-60} seconds..."

        preflight || exit 1

        oc exec "$POD" -n "$NS" -- env $REMOTE_ENV \
            SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-2}" \
            COLLECT_DURATION="${COLLECT_DURATION:-60}" \
            python3 "$REMOTE_DIR/metrics_collector.py"

        copy_results
        ;;

    # Individual experiments (old or new names)
    latency|exp1|decompose|exp1b|throughput|exp2|isolation|exp3|\
    seqlen|exp5|saturation|exp6|mixed|exp7|\
    prefix-cache|exp8|kv-eviction|exp10|\
    tput-seqlen|exp11|tput-outlen|exp12|tput-sat|exp13|overhead-load|exp14)
        run_experiment "$COMMAND"
        ;;

    model-load|exp9)
        # Model load: runs locally (kills pods via oc), like fault-test
        preflight || exit 1

        echo "=== Model Load Time Test ==="
        echo "WARNING: This will kill decode pods in namespace $NS"
        echo ""

        python3 "$SCRIPT_DIR/exp9_model_load.py"

        run_analyze
        ;;

    all)
        # Alias for characterize (backwards compatibility)
        echo "(Note: 'all' is now 'characterize')"
        exec "$0" "$CLUSTER_DIR" characterize
        ;;

    # ── Infrastructure ────────────────────────────────────────────────────
    deploy)
        "$REPO_ROOT/scripts/deploy.sh" "$CLUSTER_DIR"
        ;;

    undeploy)
        shift 2
        "$REPO_ROOT/scripts/undeploy.sh" "$CLUSTER_DIR" "$@"
        ;;

    # ── Advisory layer commands ────────────────────────────────────────────
    diagnose)
        shift 2  # remove cluster-dir and command
        python3 "$REPO_ROOT/advisor/diagnose.py" --namespace "$NS" "$@"
        ;;

    health)
        shift 2
        python3 "$REPO_ROOT/advisor/health.py" --namespace "$NS" --model "$MODEL" "$@"
        ;;

    plan)
        shift 2
        python3 "$REPO_ROOT/advisor/plan.py" --model "$MODEL" \
            --gpu-type "${GPU_TYPE:-t4}" --data-dir "$DATA_DIR" "$@"
        ;;

    rebalance)
        shift 2
        python3 "$REPO_ROOT/advisor/rebalance.py" --namespace "$NS" "$@"
        ;;

    validate)
        shift 2
        python3 "$REPO_ROOT/advisor/validate.py" "$DATA_DIR" "$@"
        ;;

    *)
        echo "Unknown command: $COMMAND"
        echo ""
        echo "Commands:"
        echo "  characterize    Full performance characterization (recommended)"
        echo "  fault-test      Fault tolerance assessment (destructive)"
        echo ""
        echo "  latency         Single-request latency sweep"
        echo "  decompose       Latency decomposition"
        echo "  throughput      Throughput under load"
        echo "  isolation       Prefill isolation"
        echo "  seqlen          Sequence length sweep"
        echo "  saturation      QPS saturation profiling"
        echo "  mixed           Mixed workload"
        echo "  prefix-cache    KV cache hit rates"
        echo "  kv-eviction     KV cache persistence under pressure"
        echo "  tput-seqlen     Throughput vs prompt length"
        echo "  tput-outlen     Throughput vs output length"
        echo "  tput-sat        Saturation ceiling (high concurrency)"
        echo "  overhead-load   Overhead decomposition under load"
        echo "  fault           Fault tolerance (destructive)"
        echo "  model-load      Cold start time (destructive)"
        echo ""
        echo "  deploy          Deploy P/D topology from env.sh"
        echo "  undeploy        Tear down deployment (--keep-pvc to keep model cache)"
        echo "  sweep           Multi-model sweep (./toolkit/run.sh config.json sweep)"
        echo ""
        echo "  diagnose        Root-cause diagnosis with fix commands"
        echo "  health          Continuous health monitoring"
        echo "  plan            GPU capacity planning"
        echo "  rebalance       P/D ratio recommendation"
        echo "  validate        Validate advisor predictions against data"
        echo ""
        echo "  analyze         Run analysis on collected data"
        echo "  preflight       Verify cluster is ready"
        echo "  metrics         Standalone metrics collection"
        exit 1
        ;;
esac
