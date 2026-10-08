#!/bin/bash
# Tear down the P/D topology.
#
# Usage:
#   ./scripts/undeploy.sh clusters/my-cluster
#   ./scripts/undeploy.sh clusters/my-cluster --keep-pvc
#   ./scripts/undeploy.sh clusters/my-sim sim
#
# Deletes resources with the selected mode's ownership label. GPU mode also
# deletes the model-cache PVC unless --keep-pvc is set.

set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <cluster-dir> [gpu|sim] [--keep-pvc]}"
KEEP_PVC=false
MODE=gpu
for arg in "${@:2}"; do
    case "$arg" in
        --keep-pvc) KEEP_PVC=true ;;
        gpu|sim) MODE="$arg" ;;
        *) echo "ERROR: unrecognized argument '$arg'"; exit 2 ;;
    esac
done

if [ ! -f "$CLUSTER_DIR/env.sh" ]; then
    echo "ERROR: $CLUSTER_DIR/env.sh not found"
    exit 1
fi

source "$CLUSTER_DIR/env.sh"

echo "Tearing down in namespace $NS..."

if [ "$MODE" = "sim" ]; then
    oc delete deployment -l app.kubernetes.io/part-of=vllm-disagg-sim -n "$NS" --ignore-not-found
    oc delete service -l app.kubernetes.io/part-of=vllm-disagg-sim -n "$NS" --ignore-not-found
    oc delete pod -l app=test-client,app.kubernetes.io/part-of=vllm-disagg-sim -n "$NS" --ignore-not-found
    oc delete configmap -l app.kubernetes.io/part-of=vllm-disagg-sim -n "$NS" --ignore-not-found
    echo "Simulator resources deleted."
    exit 0
fi

oc delete deployment -l app.kubernetes.io/part-of=vllm-disagg -n "$NS" --ignore-not-found
oc delete service -l app.kubernetes.io/part-of=vllm-disagg -n "$NS" --ignore-not-found
oc delete pdb -l app.kubernetes.io/part-of=vllm-disagg -n "$NS" --ignore-not-found
oc delete pod -l app=test-client,app.kubernetes.io/part-of=vllm-disagg -n "$NS" --ignore-not-found
oc delete rolebinding test-client-pod-reader -n "$NS" --ignore-not-found 2>/dev/null || true
oc delete role pod-reader -n "$NS" --ignore-not-found 2>/dev/null || true
oc delete configmap vllm-model-config -n "$NS" --ignore-not-found

if [ "$KEEP_PVC" = true ]; then
    echo "PVC kept (--keep-pvc). Model cache preserved for next deploy."
else
    oc delete pvc model-cache -n "$NS" --ignore-not-found
    echo "PVC deleted. Model will re-download on next deploy."
fi

echo "Done."
