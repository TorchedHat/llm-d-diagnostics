#!/bin/bash
# Cluster config: rdu3-t4x3 (3x T4 GPUs)
#
# Source this before running experiments:
#   source clusters/rdu3-t4x3/env.sh
#   ./toolkit/run.sh clusters/rdu3-t4x3 exp5
#
# Also used by deploy.sh to apply manifests with the right model/images.

export NS=llm-d
export MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
export MAX_MODEL_LEN=2048
export GPU_MEMORY_UTILIZATION=0.8
export DTYPE=float16
export DATA_DIR="clusters/rdu3-t4x3/data"

# Images
export VLLM_IMAGE="vllm/vllm-openai:v0.31.0"
export SIDECAR_IMAGE="ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.11.0"
export GPU_ALLOCATION_MODE="dra"
export GPU_COUNT=1
export SIDECAR_SCHEME="http"

# Model cache PVC — needs RWX for multi-node anti-affinity
export MODEL_CACHE_SIZE="50Gi"
export STORAGE_CLASS=""

# Topology: 1P+2D (3 GPUs)
export PREFILL_REPLICAS=1
export DECODE_REPLICAS=2

# In-cluster URLs are the defaults in client.py — no overrides needed.
# Experiments run inside the test-client pod via `oc exec`.
export PREFILL_HOST="vllm-prefill-svc.${NS}.svc.cluster.local:8100"
