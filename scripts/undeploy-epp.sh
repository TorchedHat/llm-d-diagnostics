#!/bin/bash
# Remove the EPP Helm release installed by scripts/deploy-epp.sh.
# The cluster-owned Gateway and model deployments are left untouched.

set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <cluster-dir>}"
if [[ ! -f "$CLUSTER_DIR/env.sh" ]]; then
    echo "ERROR: $CLUSTER_DIR/env.sh not found" >&2
    exit 1
fi
source "$CLUSTER_DIR/env.sh"

NS="${NS:-default}"
EPP_RELEASE_NAME="${EPP_RELEASE_NAME:-llm-d-epp}"

if ! command -v helm >/dev/null 2>&1; then
    echo "ERROR: required command 'helm' is not installed" >&2
    exit 1
fi

helm uninstall "$EPP_RELEASE_NAME" --namespace "$NS" --wait
