#!/bin/bash
# Install the llm-d Gateway API EPP route over an existing P/D deployment.
# The Gateway and its InferencePool-capable implementation are provisioned by
# the cluster operator; this script installs only the namespaced llm-d Router
# chart resources and the HTTPRoute that attaches to that Gateway.

set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <cluster-dir>}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VALUES_FILE="$REPO_ROOT/manifests/epp/values.yaml"

if [[ ! -f "$CLUSTER_DIR/env.sh" ]]; then
    echo "ERROR: $CLUSTER_DIR/env.sh not found" >&2
    exit 1
fi
source "$CLUSTER_DIR/env.sh"

NS="${NS:-default}"
EPP_RELEASE_NAME="${EPP_RELEASE_NAME:-llm-d-epp}"
EPP_ROUTER_VERSION="${EPP_ROUTER_VERSION:-v0.11.0}"
EPP_ROUTER_REPOSITORY="${EPP_ROUTER_REPOSITORY:-https://github.com/llm-d/llm-d-router.git}"
EPP_GATEWAY_NAME="${EPP_GATEWAY_NAME:-}"
EPP_GATEWAY_NAMESPACE="${EPP_GATEWAY_NAMESPACE:-$NS}"

for tool in oc git helm; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "ERROR: required command '$tool' is not installed" >&2
        exit 1
    fi
done

if [[ -z "$EPP_GATEWAY_NAME" ]]; then
    echo "ERROR: set EPP_GATEWAY_NAME in $CLUSTER_DIR/env.sh" >&2
    exit 1
fi
if [[ "${SIM:-}" == "1" ]]; then
    echo "ERROR: EPP deployment requires the real vLLM P/D manifests, not SIM mode" >&2
    exit 1
fi
if [[ ! "$EPP_ROUTER_VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-][A-Za-z0-9.]+)?$ ]]; then
    echo "ERROR: invalid EPP_ROUTER_VERSION '$EPP_ROUTER_VERSION'" >&2
    exit 1
fi

missing=()
for role in prefill decode; do
    selector="llm-d-diagnostics.ai/model-server=true,llm-d.ai/role=$role"
    pods="$(oc get pods -n "$NS" -l "$selector" -o name 2>/dev/null || true)"
    if [[ -z "$pods" ]]; then
        missing+=("running P/D model pods labeled '$selector' (redeploy the updated manifests first)")
    elif ! oc wait --for=condition=Ready pod -n "$NS" -l "$selector" --timeout=1s >/dev/null 2>&1; then
        missing+=("Ready model pods labeled '$selector'")
    fi
done

managed_resources=(
    serviceaccounts \
    services \
    deployments \
    configmaps \
    roles.rbac.authorization.k8s.io \
    rolebindings.rbac.authorization.k8s.io \
    inferencepools.inference.networking.k8s.io \
    httproutes.gateway.networking.k8s.io
)
for resource in "${managed_resources[@]}"; do
    for verb in get create update patch delete; do
        if [[ "$(oc auth can-i "$verb" "$resource" -n "$NS" 2>/dev/null || true)" != "yes" ]]; then
            missing+=("$verb $resource in namespace $NS")
        fi
    done
done
for access in \
    "get secrets" "list secrets" "create secrets" "update secrets" "delete secrets" \
    "get pods" "list pods" "watch pods"; do
    verb="${access%% *}"
    resource="${access#* }"
    if [[ "$(oc auth can-i "$verb" "$resource" -n "$NS" 2>/dev/null || true)" != "yes" ]]; then
        missing+=("$verb $resource in namespace $NS")
    fi
done

if [[ "$(oc auth can-i get gateways.gateway.networking.k8s.io -n "$EPP_GATEWAY_NAMESPACE" 2>/dev/null || true)" != "yes" ]]; then
    missing+=("get Gateway in namespace $EPP_GATEWAY_NAMESPACE")
else
    if ! oc get gateway "$EPP_GATEWAY_NAME" -n "$EPP_GATEWAY_NAMESPACE" >/dev/null 2>&1; then
        missing+=("existing Gateway $EPP_GATEWAY_NAMESPACE/$EPP_GATEWAY_NAME")
    else
        conditions="$(oc get gateway "$EPP_GATEWAY_NAME" -n "$EPP_GATEWAY_NAMESPACE" \
            -o jsonpath='{range .status.conditions[*]}{.type}={.status}{" "}{end}')"
        if [[ "$conditions" != *"Programmed=True"* ]]; then
            missing+=("Gateway $EPP_GATEWAY_NAMESPACE/$EPP_GATEWAY_NAME with Programmed=True (status: ${conditions:-missing})")
        fi
    fi
fi

if ((${#missing[@]})); then
    echo "ERROR: EPP was not installed; preflight found unmet requirements:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "No cluster resources were changed." >&2
    exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "$tmp_dir"
}
trap cleanup EXIT

echo "Fetching llm-d-router $EPP_ROUTER_VERSION..."
git clone --depth 1 --branch "$EPP_ROUTER_VERSION" \
    "$EPP_ROUTER_REPOSITORY" "$tmp_dir/llm-d-router"
router_chart="$tmp_dir/llm-d-router/config/charts/llm-d-router-gateway"
helm dependency build "$router_chart"

helm template "$EPP_RELEASE_NAME" "$router_chart" \
    --namespace "$NS" \
    --values "$VALUES_FILE" \
    --set-string "router.epp.image.tag=$EPP_ROUTER_VERSION" \
    --set-string "httpRoute.inferenceGatewayName=$EPP_GATEWAY_NAME" \
    --set-string "httpRoute.inferenceGatewayNamespace=$EPP_GATEWAY_NAMESPACE" \
    >"$tmp_dir/rendered.yaml"

if ! oc apply --dry-run=server -n "$NS" -f "$tmp_dir/rendered.yaml"; then
    echo "ERROR: rendered EPP resources failed server validation; no resources were changed." >&2
    exit 1
fi

echo "Installing EPP release '$EPP_RELEASE_NAME' in namespace '$NS'..."
helm upgrade --install "$EPP_RELEASE_NAME" "$router_chart" \
    --namespace "$NS" \
    --values "$VALUES_FILE" \
    --set-string "router.epp.image.tag=$EPP_ROUTER_VERSION" \
    --set-string "httpRoute.inferenceGatewayName=$EPP_GATEWAY_NAME" \
    --set-string "httpRoute.inferenceGatewayNamespace=$EPP_GATEWAY_NAMESPACE" \
    --atomic --wait --timeout 5m

echo "Waiting for the Gateway to accept the HTTPRoute..."
route_conditions=""
route_deadline=$((SECONDS + 120))
while ((SECONDS < route_deadline)); do
    route_conditions="$(oc get httproute "$EPP_RELEASE_NAME" -n "$NS" \
        -o jsonpath="{range .status.parents[?(@.parentRef.name=='$EPP_GATEWAY_NAME')].conditions[*]}{.type}={.status}{\" \"}{end}" \
        2>/dev/null || true)"
    if [[ "$route_conditions" == *"Accepted=True"* && "$route_conditions" == *"ResolvedRefs=True"* ]]; then
        break
    fi
    if [[ "$route_conditions" == *"Accepted=False"* || "$route_conditions" == *"ResolvedRefs=False"* ]]; then
        echo "ERROR: Gateway rejected the HTTPRoute (conditions: $route_conditions)" >&2
        echo "Inspect oc get httproute $EPP_RELEASE_NAME -n $NS -o yaml" >&2
        exit 1
    fi
    sleep 5
done
if [[ "$route_conditions" != *"Accepted=True"* || "$route_conditions" != *"ResolvedRefs=True"* ]]; then
    echo "ERROR: HTTPRoute was not accepted within 120 seconds (conditions: ${route_conditions:-missing})" >&2
    echo "Inspect oc get httproute $EPP_RELEASE_NAME -n $NS -o yaml" >&2
    exit 1
fi

echo "EPP release installed. Check route attachment and run toolkit preflight:"
echo "  oc get httproute $EPP_RELEASE_NAME -n $NS -o yaml"
echo "  Set EPP_URL to the Gateway's OpenAI completions URL in $CLUSTER_DIR/env.sh"
echo "  ./toolkit/run.sh $CLUSTER_DIR preflight"
