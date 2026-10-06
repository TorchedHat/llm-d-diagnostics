#!/bin/bash
# Cluster config: rdu3-t4x3-phi3-control (Phi-3 control for GQA experiment)
#
# Same 1P+2D topology as rdu3-t4x3 but with Phi-3 instead of TinyLlama.
# Validates the apparatus before switching to GQA models.
#
# Source this before running experiments:
#   source clusters/rdu3-t4x3-phi3-control/env.sh
#   ./toolkit/run.sh clusters/rdu3-t4x3-phi3-control exp5

export NS=llm-d
export MODEL="microsoft/Phi-3-mini-4k-instruct"
export MAX_MODEL_LEN=2048
export GPU_MEMORY_UTILIZATION=0.85
export DTYPE=float16
export DATA_DIR="clusters/rdu3-t4x3-phi3-control/data"

# Images
export VLLM_IMAGE="vllm/vllm-openai:v0.31.0"
export SIDECAR_IMAGE="ghcr.io/llm-d/llm-d-router-disagg-sidecar:v0.11.0"
export GPU_ALLOCATION_MODE="dra"
export GPU_COUNT=1
export SIDECAR_SCHEME="http"

# Model cache PVC
export MODEL_CACHE_SIZE="50Gi"
export STORAGE_CLASS=""

# Topology: 1P+2D (3 GPUs)
export PREFILL_REPLICAS=1
export DECODE_REPLICAS=2

export PREFILL_HOST="vllm-prefill-svc.${NS}.svc.cluster.local:8100"
