#!/bin/bash
# Deploy the P/D topology to a cluster.
#
# Usage:
#   ./scripts/deploy.sh clusters/my-cluster
#   ./scripts/deploy.sh clusters/my-cluster sim    # GPU-free inference-sim stack
#   DRY_RUN=server ./scripts/deploy.sh clusters/my-cluster
#
# Reads cluster config from <cluster-dir>/env.sh. Topology is controlled by:
#   PREFILL_REPLICAS=2 DECODE_REPLICAS=3 → 2P+3D (5 GPUs)
# GPU allocation is selected with GPU_ALLOCATION_MODE=dra|classic. DRA claims
# use the existing gpu-<GPU_COUNT> ResourceClaimTemplate in the namespace.
#
# Follows the llm-d v0.10.0 P/D deployment conventions:
#   - Deployments (not StatefulSets)
#   - explicit NIXL roles: kv_producer on prefill, kv_consumer on decode
#   - Native sidecar (initContainer with restartPolicy: Always, K8s 1.29+)
#   - NIXL side channel on all pods (VLLM_NIXL_SIDE_CHANNEL_HOST/PORT)
#   - llm-d.ai/role labels for ecosystem compatibility
#   - UCX_TLS: ^cuda_ipc (exclude CUDA IPC for cross-pod TCP transfers)
#
# What we don't use (requires EPP/Gateway/full stack):
#   - EPP, InferencePool, InferenceModel CRDs
#   - Gateway API, HTTPRoute
#   - ModelService CRD
#   - Redis/LMCache (MultiConnector)
#
# Pod discovery:
#   oc get pods -l app=vllm-decode -n <namespace>
#   oc get pods -l llm-d.ai/role=decode -n <namespace>
#
# Scale with:
#   oc scale deployment vllm-prefill --replicas=N -n <namespace>
#   oc scale deployment vllm-decode  --replicas=N -n <namespace>

set -euo pipefail

CLUSTER_DIR="${1:?Usage: $0 <cluster-dir> [gpu|sim]}"
MODE="${2:-gpu}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIM_MANIFEST_DIR="$REPO_ROOT/manifests/sim"
DRY_RUN="${DRY_RUN:-}"

case "$MODE" in
    gpu|sim) ;;
    *) echo "ERROR: mode must be 'gpu' or 'sim' (got '$MODE')"; exit 2 ;;
esac

case "$DRY_RUN" in
    ""|client|server) ;;
    *) echo "ERROR: DRY_RUN must be empty, 'client', or 'server'"; exit 2 ;;
esac

if [ ! -f "$CLUSTER_DIR/env.sh" ]; then
    echo "ERROR: $CLUSTER_DIR/env.sh not found"
    exit 1
fi

source "$CLUSTER_DIR/env.sh"

# Topology (can be overridden in env.sh)
PREFILL_REPLICAS="${PREFILL_REPLICAS:-1}"
DECODE_REPLICAS="${DECODE_REPLICAS:-2}"

# Latest upstream vLLM release (CUDA 13.0 build) with llm-d Router v0.11.0.
# The llm-d v0.10.0 release matrix lists vLLM v0.30.0; override in env.sh to
# reproduce that matrix or select an image for the target accelerator.
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}"
SIDECAR_IMAGE="${SIDECAR_IMAGE:-ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.11.0}"
GPU_ALLOCATION_MODE="${GPU_ALLOCATION_MODE:-classic}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_RESOURCE_NAME="${GPU_RESOURCE_NAME:-nvidia.com/gpu}"
SIDECAR_SCHEME="${SIDECAR_SCHEME:-http}"
MODEL_CACHE_SIZE="${MODEL_CACHE_SIZE:-50Gi}"

# NIXL side channel port (upstream default: 5557)
NIXL_PORT="${NIXL_PORT:-5557}"

for var in PREFILL_REPLICAS DECODE_REPLICAS NIXL_PORT GPU_COUNT; do
    val="${!var}"
    if ! [[ "$val" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $var must be an integer (got '$val')"
        exit 1
    fi
done
if (( NIXL_PORT < 1 || NIXL_PORT > 65535 )); then
    echo "ERROR: NIXL_PORT must be between 1 and 65535"
    exit 1
fi

# Validate image names (prevent injection via env.sh)
for var in VLLM_IMAGE SIDECAR_IMAGE; do
    val="${!var}"
    if ! [[ "$val" =~ ^[a-zA-Z0-9./_:@-]+$ ]]; then
        echo "ERROR: Invalid $var: $val"
        exit 1
    fi
done

case "$GPU_ALLOCATION_MODE" in
    dra)
        GPU_CLAIM_TEMPLATE="gpu-${GPU_COUNT}"
        GPU_POD_RESOURCE_CLAIMS="      resourceClaims:
      - name: gpu
        resourceClaimTemplateName: ${GPU_CLAIM_TEMPLATE}"
        GPU_CONTAINER_RESOURCE_CLAIMS="          claims:
          - name: gpu"
        GPU_RESOURCE_REQUEST=""
        GPU_RESOURCE_LIMIT=""
        ;;
    classic)
        if ! [[ "$GPU_RESOURCE_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9./_-]*$ ]]; then
            echo "ERROR: Invalid GPU_RESOURCE_NAME: $GPU_RESOURCE_NAME"
            exit 1
        fi
        GPU_CLAIM_TEMPLATE=""
        GPU_POD_RESOURCE_CLAIMS=""
        GPU_CONTAINER_RESOURCE_CLAIMS=""
        GPU_RESOURCE_REQUEST="            ${GPU_RESOURCE_NAME}: \"${GPU_COUNT}\""
        GPU_RESOURCE_LIMIT="            ${GPU_RESOURCE_NAME}: \"${GPU_COUNT}\""
        ;;
    *)
        echo "ERROR: GPU_ALLOCATION_MODE must be 'dra' or 'classic' (got '$GPU_ALLOCATION_MODE')"
        exit 2
        ;;
esac
if [[ "$SIDECAR_SCHEME" != "http" && "$SIDECAR_SCHEME" != "https" ]]; then
    echo "ERROR: SIDECAR_SCHEME must be 'http' or 'https'"
    exit 1
fi
SIDECAR_SECURE_PROXY=false
[[ "$SIDECAR_SCHEME" == "https" ]] && SIDECAR_SECURE_PROXY=true

oc_apply() {
    if [[ -n "$DRY_RUN" ]]; then
        oc apply "--dry-run=$DRY_RUN" "$@"
    else
        oc apply "$@"
    fi
}

echo "Cluster:     $CLUSTER_DIR"
echo "Namespace:   $NS"
echo "Model:       $MODEL"
echo "Topology:    ${PREFILL_REPLICAS}P + ${DECODE_REPLICAS}D"
echo "vLLM image:  $VLLM_IMAGE"
echo "Sidecar:     $SIDECAR_IMAGE"
if [[ "$GPU_ALLOCATION_MODE" == "dra" ]]; then
    echo "GPU allocation: DRA ($GPU_CLAIM_TEMPLATE, ${GPU_COUNT} GPU(s) per vLLM pod)"
else
    echo "GPU allocation: classic ($GPU_RESOURCE_NAME, ${GPU_COUNT} GPU(s) per vLLM pod)"
fi
echo "Sidecar URL:  $SIDECAR_SCHEME"
echo "PVC size:    $MODEL_CACHE_SIZE"
echo ""

# Create namespace if it doesn't exist. Client dry-run is useful for rendering
# on a workstation without needing the target namespace to exist.
if [[ "$DRY_RUN" == "client" ]]; then
    echo "Client dry-run: skipping namespace lookup/creation."
elif ! oc get namespace "$NS" &>/dev/null; then
    if [[ "$DRY_RUN" == "server" ]]; then
        echo "ERROR: namespace '$NS' must already exist for a server dry-run"
        exit 1
    fi
    oc create namespace "$NS"
fi

if [[ "$MODE" != "sim" && "$GPU_ALLOCATION_MODE" == "dra" && "$DRY_RUN" != "client" ]]; then
    if ! oc get resourceclaimtemplate "$GPU_CLAIM_TEMPLATE" -n "$NS" &>/dev/null; then
        echo "ERROR: DRA ResourceClaimTemplate '$GPU_CLAIM_TEMPLATE' not found in namespace '$NS'."
        echo "Create the matching gpu-<GPU_COUNT> template before deploying."
        exit 1
    fi
fi

# Simulation mode must deploy the simulator manifests rather than the GPU
# topology. Keep the model name configurable through the same env.sh setting.
if [[ "$MODE" == "sim" ]]; then
    echo "Deploying GPU-free inference-sim mode..."
    oc_apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-model-config
  labels:
    app.kubernetes.io/part-of: vllm-disagg-sim
data:
  MODEL_NAME: "$MODEL"
EOF
    if [[ -n "$DRY_RUN" ]]; then
        for manifest in "$SIM_MANIFEST_DIR"/*.yaml; do
            [[ "$(basename "$manifest")" == "00-model-config.yaml" ]] && continue
            oc_apply -n "$NS" -f "$manifest"
        done
        echo "Dry-run complete; no resources were changed."
        exit 0
    fi
    for manifest in "$SIM_MANIFEST_DIR"/*.yaml; do
        [[ "$(basename "$manifest")" == "00-model-config.yaml" ]] && continue
        oc_apply -n "$NS" -f "$manifest"
    done
    for deployment in vllm-prefill vllm-decode vllm-decode-2; do
        oc rollout status "deployment/$deployment" -n "$NS" --timeout=10m
    done
    oc wait --for=condition=Ready pod/test-client -n "$NS" --timeout=2m
    echo "Simulator is ready. Run: SIM=1 ./toolkit/run.sh $CLUSTER_DIR preflight"
    exit 0
fi

# ── ConfigMap ────────────────────────────────────────────────────────────
oc_apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-model-config
  labels:
    app.kubernetes.io/part-of: vllm-disagg
data:
  MODEL_NAME: "$MODEL"
  MAX_MODEL_LEN: "${MAX_MODEL_LEN:-2048}"
  GPU_MEMORY_UTILIZATION: "${GPU_MEMORY_UTILIZATION:-0.8}"
  DTYPE: "${DTYPE:-float16}"
EOF

# ── PVC (create only — can't resize after creation) ──────────────────────
if [[ "$DRY_RUN" == "client" ]] || ! oc get pvc model-cache -n "$NS" &>/dev/null; then
    SC_LINE=""
    if [ -n "${STORAGE_CLASS:-}" ]; then
        SC_LINE="  storageClassName: $STORAGE_CLASS"
    fi
    oc_apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
  labels:
    app.kubernetes.io/part-of: vllm-disagg
spec:
${SC_LINE:+$SC_LINE
}  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: $MODEL_CACHE_SIZE
EOF
else
    echo "PVC model-cache already exists, skipping (delete and re-run to resize)"
fi

# ── Prefill headless Service ──────────────────────────────────────────────
oc_apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-prefill-svc
  labels:
    app: vllm-prefill
    llm-d.ai/role: prefill
    app.kubernetes.io/part-of: vllm-disagg
spec:
  clusterIP: None
  selector:
    app: vllm-prefill
  ports:
  - name: http
    port: 8100
    targetPort: 8100
    protocol: TCP
  - name: nixl
    port: $NIXL_PORT
    targetPort: $NIXL_PORT
    protocol: TCP
EOF

# ── Prefill Deployment ──────────────────────────────────────────────────
# No sidecar on prefill (upstream pattern: prefill is direct vLLM only).
# Clients hit prefill directly or are routed via x-prefiller-host-port header.
echo "Creating prefill deployment (replicas=$PREFILL_REPLICAS)..."
oc_apply -n "$NS" -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-prefill
  labels:
    app: vllm-prefill
    llm-d.ai/role: prefill
    app.kubernetes.io/part-of: vllm-disagg
spec:
  replicas: $PREFILL_REPLICAS
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: vllm-prefill
  template:
    metadata:
      labels:
        app: vllm-prefill
        llm-d.ai/role: prefill
        app.kubernetes.io/part-of: vllm-disagg
    spec:
${GPU_POD_RESOURCE_CLAIMS}
      containers:
      - name: vllm
        image: $VLLM_IMAGE
        command:
        - vllm
        - serve
        - \$(MODEL_NAME)
        args:
        - --host
        - "0.0.0.0"
        - --port
        - "8100"
        - --dtype
        - \$(DTYPE)
        - --gpu-memory-utilization
        - \$(GPU_MEMORY_UTILIZATION)
        - --max-model-len
        - \$(MAX_MODEL_LEN)
        - --tensor-parallel-size
        - "$GPU_COUNT"
        - --trust-remote-code
        - --kv-transfer-config
        - '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
        env:
        - name: HF_HOME
          value: /model-cache/hf-cache
        - name: VLLM_NIXL_SIDE_CHANNEL_HOST
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: VLLM_NIXL_SIDE_CHANNEL_PORT
          value: "$NIXL_PORT"
        - name: UCX_TLS
          value: "^cuda_ipc"
        envFrom:
        - configMapRef:
            name: vllm-model-config
        ports:
        - containerPort: 8100
          name: http
          protocol: TCP
        - containerPort: $NIXL_PORT
          name: nixl
          protocol: TCP
        resources:
          requests:
            cpu: "4"
            memory: 16Gi
${GPU_RESOURCE_REQUEST:+$GPU_RESOURCE_REQUEST
}          limits:
            cpu: "4"
            memory: 16Gi
${GPU_RESOURCE_LIMIT:+$GPU_RESOURCE_LIMIT
}${GPU_CONTAINER_RESOURCE_CLAIMS}
        startupProbe:
          httpGet:
            path: /health
            port: 8100
          failureThreshold: 60
          initialDelaySeconds: 15
          periodSeconds: 30
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8100
          failureThreshold: 3
          periodSeconds: 5
        livenessProbe:
          tcpSocket:
            port: 8100
          failureThreshold: 3
          periodSeconds: 5
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 5"]
        volumeMounts:
        - name: model-cache
          mountPath: /model-cache
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 4Gi
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app.kubernetes.io/part-of
                  operator: In
                  values:
                  - vllm-disagg
              topologyKey: kubernetes.io/hostname
      terminationGracePeriodSeconds: 30
EOF

# ── Decode headless Service ───────────────────────────────────────────────
oc_apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-decode-svc
  labels:
    app: vllm-decode
    llm-d.ai/role: decode
    app.kubernetes.io/part-of: vllm-disagg
spec:
  clusterIP: None
  selector:
    app: vllm-decode
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
  - name: nixl
    port: $NIXL_PORT
    targetPort: $NIXL_PORT
    protocol: TCP
EOF

# Decode direct: port 8001 (bypass sidecar — for exp1b latency decomposition)
# Not in upstream — specific to our diagnostics toolkit.
oc_apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: vllm-decode-direct-svc
  labels:
    app: vllm-decode
    llm-d.ai/role: decode
    app.kubernetes.io/part-of: vllm-disagg
spec:
  clusterIP: None
  selector:
    app: vllm-decode
  ports:
  - name: vllm
    port: 8001
    targetPort: 8001
    protocol: TCP
EOF

# ── Decode Deployment ─────────────────────────────────────────────────────
# Routing sidecar runs as native sidecar (initContainer with restartPolicy: Always).
# This means:
#   - Sidecar starts first and must pass readiness before vLLM starts
#   - If sidecar crashes, K8s restarts it without restarting the vLLM container
#   - Pod startup is ordered: sidecar → vllm
echo "Creating decode deployment (replicas=$DECODE_REPLICAS)..."
oc_apply -n "$NS" -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-decode
  labels:
    app: vllm-decode
    llm-d.ai/role: decode
    app.kubernetes.io/part-of: vllm-disagg
spec:
  replicas: $DECODE_REPLICAS
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: vllm-decode
  template:
    metadata:
      labels:
        app: vllm-decode
        llm-d.ai/role: decode
        app.kubernetes.io/part-of: vllm-disagg
    spec:
${GPU_POD_RESOURCE_CLAIMS}
      initContainers:
      # Native sidecar (restartPolicy: Always, K8s 1.29+/OCP 4.17+).
      # Proxies requests: client → sidecar:8000 → vllm:8001
      # Handles KV routing for disaggregated inference.
      - name: routing-sidecar
        image: $SIDECAR_IMAGE
        args:
        - "--port=8000"
        - "--kv-connector=nixlv2"
        - "--model-server-port=8001"
        - "--secure-proxy=$SIDECAR_SECURE_PROXY"
        - "--zap-log-level=1"
        restartPolicy: Always
        securityContext:
          capabilities:
            drop:
            - MKNOD
          allowPrivilegeEscalation: false
        ports:
        - containerPort: 8000
          protocol: TCP
        resources:
          requests:
            cpu: "100m"
            memory: 128Mi
          limits:
            cpu: "500m"
            memory: 256Mi
        livenessProbe:
          tcpSocket:
            port: 8000
          failureThreshold: 3
          periodSeconds: 5
        readinessProbe:
          tcpSocket:
            port: 8000
          failureThreshold: 3
          periodSeconds: 5
      containers:
      - name: vllm
        image: $VLLM_IMAGE
        command:
        - vllm
        - serve
        - \$(MODEL_NAME)
        args:
        - --host
        - "0.0.0.0"
        - --port
        - "8001"
        - --dtype
        - \$(DTYPE)
        - --gpu-memory-utilization
        - \$(GPU_MEMORY_UTILIZATION)
        - --max-model-len
        - \$(MAX_MODEL_LEN)
        - --tensor-parallel-size
        - "$GPU_COUNT"
        - --trust-remote-code
        - --kv-transfer-config
        - '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
        env:
        - name: HF_HOME
          value: /model-cache/hf-cache
        - name: VLLM_NIXL_SIDE_CHANNEL_HOST
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: VLLM_NIXL_SIDE_CHANNEL_PORT
          value: "$NIXL_PORT"
        - name: UCX_TLS
          value: "^cuda_ipc"
        envFrom:
        - configMapRef:
            name: vllm-model-config
        ports:
        - containerPort: 8001
          name: vllm
          protocol: TCP
        - containerPort: $NIXL_PORT
          name: nixl
          protocol: TCP
        resources:
          requests:
            cpu: "4"
            memory: 16Gi
${GPU_RESOURCE_REQUEST:+$GPU_RESOURCE_REQUEST
}          limits:
            cpu: "4"
            memory: 16Gi
${GPU_RESOURCE_LIMIT:+$GPU_RESOURCE_LIMIT
}${GPU_CONTAINER_RESOURCE_CLAIMS}
        startupProbe:
          httpGet:
            path: /health
            port: 8001
          failureThreshold: 60
          initialDelaySeconds: 15
          periodSeconds: 30
          timeoutSeconds: 5
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          failureThreshold: 3
          periodSeconds: 5
        livenessProbe:
          tcpSocket:
            port: 8001
          failureThreshold: 3
          periodSeconds: 5
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 5"]
        volumeMounts:
        - name: model-cache
          mountPath: /model-cache
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 4Gi
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app.kubernetes.io/part-of
                  operator: In
                  values:
                  - vllm-disagg
              topologyKey: kubernetes.io/hostname
      terminationGracePeriodSeconds: 30
EOF

# ── PodDisruptionBudgets ─────────────────────────────────────────────────
oc_apply -n "$NS" -f - <<EOF
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: vllm-prefill-pdb
  labels:
    app.kubernetes.io/part-of: vllm-disagg
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: vllm-prefill
EOF

oc_apply -n "$NS" -f - <<EOF
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: vllm-decode-pdb
  labels:
    app.kubernetes.io/part-of: vllm-disagg
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: vllm-decode
EOF

# ── Test client pod ──────────────────────────────────────────────────────
if [ "$MODE" != "sim" ]; then
    oc_apply -n "$NS" -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: test-client
  labels:
    app: test-client
    app.kubernetes.io/part-of: vllm-disagg
spec:
  containers:
  - name: python
    image: $VLLM_IMAGE
    command: ["sleep", "infinity"]
    envFrom:
    - configMapRef:
        name: vllm-model-config
    env:
    - name: NS
      valueFrom:
        fieldRef:
          fieldPath: metadata.namespace
    resources:
      requests:
        cpu: "500m"
        memory: 512Mi
      limits:
        cpu: "2"
        memory: 1Gi
  restartPolicy: Never
EOF

    # RBAC for in-pod pod discovery (K8s API path in client.py).
    # Best-effort: if user lacks RBAC permissions, pod discovery falls
    # back to env vars injected by run.sh at experiment time.
    echo "Creating RBAC for in-pod discovery..."
    if ! oc_apply -n "$NS" -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  labels:
    app.kubernetes.io/part-of: vllm-disagg
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: test-client-pod-reader
  labels:
    app.kubernetes.io/part-of: vllm-disagg
subjects:
- kind: ServiceAccount
  name: default
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
EOF
    then
        echo "WARNING: RBAC creation failed (insufficient permissions)."
        echo "  Pod discovery will use env vars from run.sh instead."
    fi
fi

# ── Clean up legacy resources ────────────────────────────────────────────
echo ""

if [[ -z "$DRY_RUN" ]]; then
    # Remove legacy objects only when they belong to this deployment. This
    # keeps a simulator deployment with overlapping names intact.
    delete_legacy_if_owned() {
        local kind="$1"
        local name="$2"
        local owner
        owner=$(oc get "$kind" "$name" -n "$NS" \
            -o jsonpath='{.metadata.labels.app\.kubernetes\.io/part-of}' 2>/dev/null || true)
        if [[ "$owner" == "vllm-disagg" ]]; then
            echo "Removing legacy $kind: $name"
            oc delete "$kind" "$name" -n "$NS"
        elif [[ -n "$owner" ]]; then
            echo "Keeping $kind/$name (owned by $owner)"
        fi
    }

    # Remove old StatefulSets (replaced by Deployments)
    for old in vllm-prefill vllm-decode; do
        delete_legacy_if_owned statefulset "$old"
    done

    # Remove old per-instance Deployments from the per-instance era
    for old in vllm-prefill-1 vllm-prefill-2 \
               vllm-decode-1 vllm-decode-2 vllm-decode-3; do
        delete_legacy_if_owned deployment "$old"
    done

    # Remove per-instance services from the per-instance era
    for old in vllm-prefill-1-svc vllm-prefill-2-svc \
               vllm-decode-1-svc vllm-decode-2-svc vllm-decode-3-svc; do
        delete_legacy_if_owned service "$old"
    done
fi

echo ""
echo "Topology: ${PREFILL_REPLICAS}P + ${DECODE_REPLICAS}D deployed."
echo ""
echo "  Pods (discover by label):"
echo "    oc get pods -l app=vllm-prefill -n $NS"
echo "    oc get pods -l app=vllm-decode -n $NS"
echo "    oc get pods -l llm-d.ai/role=prefill -n $NS"
echo "    oc get pods -l llm-d.ai/role=decode -n $NS"
echo ""
echo "  Services:"
echo "    vllm-prefill-svc:8100        (prefill, headless)"
echo "    vllm-decode-svc:8000         (decode via sidecar, headless)"
echo "    vllm-decode-direct-svc:8001  (decode bypass sidecar, headless)"
echo "    NIXL side channel:$NIXL_PORT (on all pods)"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "Dry-run complete; no resources were changed."
    exit 0
fi
echo "  Scale:"
echo "    oc scale deployment vllm-prefill --replicas=N -n $NS"
echo "    oc scale deployment vllm-decode  --replicas=N -n $NS"
echo ""
echo "Waiting for pods to be ready (up to 20 min for model download)..."
TIMEOUT=1200
INTERVAL=30
ELAPSED=0
EXPECTED=$((PREFILL_REPLICAS + DECODE_REPLICAS))

while [ $ELAPSED -lt $TIMEOUT ]; do
    READY=0
    for deployment in vllm-prefill vllm-decode; do
        replicas="$(oc get deployment "$deployment" -n "$NS" -o jsonpath='{.spec.replicas}')"
        # Old Ready pods can mask a failing new ReplicaSet during an image rollout.
        if oc rollout status "deployment/$deployment" -n "$NS" --timeout=1s >/dev/null 2>&1; then
            READY=$((READY + replicas))
        fi
    done
    if [ "$READY" -ge "$EXPECTED" ]; then
        echo ""
        echo "All $EXPECTED vLLM pods ready."
        oc get pods -n "$NS" -l app.kubernetes.io/part-of=vllm-disagg
        echo ""
        echo "Done."
        exit 0
    fi
    echo "  $READY/$EXPECTED pods ready (${ELAPSED}s elapsed)..."
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "WARNING: Only $READY/$EXPECTED pods ready after ${TIMEOUT}s."
oc get pods -n "$NS" -l app.kubernetes.io/part-of=vllm-disagg
exit 1
