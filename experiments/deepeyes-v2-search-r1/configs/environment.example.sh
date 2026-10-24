#!/usr/bin/env bash
# Source after replacing the example asset paths in your prepared training container.
# This file does not start Ray, a service, or a training job.
export RELAX="$(git rev-parse --show-toplevel)"
export MODEL_CONFIG_DIR="${RELAX}/scripts/models"
export MODEL_DIR="${ASSET_ROOT:?Set your asset root}/models"
export DATA_DIR="${ASSET_ROOT}/deepeyes/data/experiment"
export SAVE_DIR="${RUN_ROOT:?Set a local run root}"
export EXPERIMENT_RUN_DIR="${SAVE_DIR}/seed1234-train200"
export EXPERIMENT_OPS_DIR="${RELAX}/experiments/deepeyes-v2-search-r1/implementation"
export EXPERIMENT_DATA_MANIFEST="${DATA_DIR}/manifest.json"
export SANDBOX_CONFIG_PATH="${SANDBOX_CONFIG_FILE:?Set a verified sandbox YAML}"
export APPTAINER_REAL_BIN="${APPTAINER_BIN:?Set the verified Apptainer executable}"
export APPTAINER_IMAGE_PATH="${SANDBOX_IMAGE:?Set a verified kernel image}"
export DEEPEYES_V2_APP_ENV_ROOT="${APP_ENV_ROOT:?Set the prepared app virtualenv root}"
export DEEPEYES_V2_SEARCH_CONFIG="${SEARCH_CONFIG_FILE:?Set your retriever YAML}"
export DEEPEYES_V2_SEARCH_CACHE_PATHS="${ASSET_ROOT}/image-search-cache/converted-cache.json"
export DEEPEYES_JUDGE_BASE_URL="${JUDGE_BASE_URL:?Set your Judge service base URL}"
export DEEPEYES_JUDGE_MODELS="Qwen3-8B"
export NCCL_NVLS_ENABLE=0
