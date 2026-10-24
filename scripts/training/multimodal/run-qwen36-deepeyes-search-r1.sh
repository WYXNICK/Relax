#!/bin/bash
# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -ex
set -o pipefail

###############################################################################
#                                 ENVIRONMENT                                 #
###############################################################################

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPO_ROOT="${RELAX:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SCRIPT_DIR="${REPO_ROOT}/examples/deepeyes_v2_agentic"
: "${EXPERIMENT_RUN_DIR:?Set a unique worker-local run directory}"
: "${EXPERIMENT_OPS_DIR:?Set the experiment operations directory}"
: "${SANDBOX_CONFIG_PATH:?Set the verified sandbox YAML}"
: "${DEEPEYES_JUDGE_BASE_URL:?Start the separate Judge service first}"
: "${DEEPEYES_V2_SEARCH_CONFIG:?Set the retriever YAML}"
: "${EXPERIMENT_DATA_MANIFEST:?Set the frozen data manifest}"
: "${APPTAINER_REAL_BIN:?Set the verified Apptainer binary}"
[ -n "${RELAX_ENTRYPOINT_MODE:-}" ] || source "${REPO_ROOT}/scripts/entrypoint/ray-job.sh"
# This recipe uses one training worker. Keep local HTTP traffic on loopback.
export MASTER_ADDR=127.0.0.1 SLIME_HOST_IP=127.0.0.1
source "${MODEL_CONFIG_DIR}/qwen36-35B-A3B.sh"
export TENSORBOARD_DIR="${EXPERIMENT_RUN_DIR}/tensorboard"
mkdir -p "${EXPERIMENT_RUN_DIR}/logs"

###############################################################################
#                                    DIRS                                     #
###############################################################################

PROJECT_NAME="Relax/deepeyes-v2-search-r1"
EXP_NAME="$(basename "${EXPERIMENT_RUN_DIR}")"

if [ -z "${MODEL_DIR:-}" ] || [ -z "${DATA_DIR:-}" ] || [ -z "${SAVE_DIR:-}" ]; then
    echo "ERROR: MODEL_DIR, DATA_DIR, and SAVE_DIR must be set."
    echo "Example: MODEL_DIR=/path/to/models DATA_DIR=/path/to/data SAVE_DIR=/path/to/save bash $0"
    exit 1
fi
mkdir -p ${SAVE_DIR}

###############################################################################
#                             STARTUP CLEANUP                                 #
###############################################################################
# Sessions that die by SIGKILL / OOM / uncaught crash never run
# ApptainerJupyterSession.close(), so their /tmp/relax-apptainer-* dirs
# (kernel conn files + bind-mounted imgs/inputs holding base64-decoded
# dataset images) leak. Over 2000 rollouts × 32 batch × 8 samples ≈ 512K
# sessions, even a 1% leak fills the container disk and triggers eviction.
# mtime > 240 min is well above any single session's max wall-clock (bounded
# by max_turns × code_timeout_s ≈ tens of minutes), so live sessions are
# never touched.
# Apptainer accepts a SIF or a pre-expanded root filesystem.
export APPTAINER_IMAGE_PATH="${APPTAINER_IMAGE_PATH:-${DATA_DIR}/sif/deepeyes_v2_kernel.sif}"
if [ ! -e "${APPTAINER_IMAGE_PATH}" ]; then
    echo "ERROR: Apptainer image not found at ${APPTAINER_IMAGE_PATH}."
    echo "Run: DATA_DIR=${DATA_DIR} bash ${SCRIPT_DIR}/scripts/prepare.sh"
    exit 1
fi
DEEPEYES_V2_APP_PYTHON="${DEEPEYES_V2_APP_ENV_ROOT:-/tmp/deepeyes-v2-app-env}/.venv/bin/python"
if [ ! -x "${DEEPEYES_V2_APP_PYTHON}" ]; then
    echo "ERROR: DeepEyes V2 app environment not found at ${DEEPEYES_V2_APP_PYTHON}."
    echo "Run: bash ${SCRIPT_DIR}/scripts/prepare_app_env.sh"
    exit 1
fi

###############################################################################
#                              JUDGE MODEL API                                #
###############################################################################

# Judge is a separately managed GPU service; preserve the configured endpoint.

###############################################################################
#                                  MODEL CONFIG                               #
###############################################################################

# This backend prunes BEFORE saving: keep=1 retains previous + new checkpoint.
# A failed save still leaves the previous complete checkpoint.
CKPT_ARGS=(
    --hf-checkpoint ${MODEL_DIR}/Qwen3.6-35B-A3B
    --ref-load ${MODEL_DIR}/Qwen3.6-35B-A3B
    --warm-hf-checkpoint-page-cache
    --save ${EXPERIMENT_RUN_DIR}/checkpoints
    --megatron-to-hf-mode bridge
    --save-interval 50
    --max-actor-ckpt-to-keep 1
)

###############################################################################
#                                  DATASETS                                   #
###############################################################################

# Layout produced by scripts/prepare.sh:
#   ${DATA_DIR}/data/{perception_all_*,reason,search,vstar_test}.parquet
#   ${DATA_DIR}/sif/deepeyes_v2_kernel.sif
TRAIN_FILES=(
    "'${DATA_DIR}/search_train.parquet'"
    "'${DATA_DIR}/perception_train.parquet'"
    "'${DATA_DIR}/reason_train.parquet'"
)
TEST_FILES=("${DATA_DIR}/vstar_test.parquet")
PROMPT_SET="[$(IFS=,; echo "${TRAIN_FILES[*]}")]"

###############################################################################
#                               ROLLOUT CONFIG                                #
###############################################################################

NUM_ROLLOUT="${NUM_ROLLOUT:-200}"

# Sandbox env vars propagated into every Ray worker so the per-session
# agent process can find apptainer / search cache.
# SANDBOX_CONFIG_PATH is required — the agent reads it in _build_executor
# to find the apptainer backend YAML config (image path, bind paths, etc).
# Runtime configuration may include credentials.
set +x
export DEEPEYES_V2_APP_PYTHON TENSORBOARD_DIR EXPERIMENT_RUN_DIR EXPERIMENT_OPS_DIR EXPERIMENT_DATA_MANIFEST
export RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-}"
[ -n "${RUNTIME_ENV_JSON}" ] || RUNTIME_ENV_JSON='{}'
RUNTIME_ENV_JSON=$(python3 - <<'PYTHON'
import json, os
payload = json.loads(os.environ["RUNTIME_ENV_JSON"])
env = payload.setdefault("env_vars", {})
for name in ("SANDBOX_CONFIG_PATH", "DEEPEYES_V2_SEARCH_CACHE_PATHS", "DEEPEYES_JUDGE_BASE_URL",
             "DEEPEYES_JUDGE_MODELS", "DEEPEYES_JUDGE_API_KEY", "APPTAINER_IMAGE_PATH",
             "DEEPEYES_V2_APP_PYTHON", "TENSORBOARD_DIR", "EXPERIMENT_RUN_DIR", "EXPERIMENT_OPS_DIR",
             "EXPERIMENT_DATA_MANIFEST", "DEEPEYES_V2_SEARCH_CONFIG", "APPTAINER_REAL_BIN",
             "HF_HOME", "HF_HUB_OFFLINE", "PYTHONNOUSERSITE", "XDG_CACHE_HOME",
             "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "NCCL_SOCKET_IFNAME",
             "NCCL_SOCKET_FAMILY", "NCCL_IB_DISABLE", "NCCL_NVLS_ENABLE",
             "MASTER_ADDR", "SLIME_HOST_IP"):
    if os.environ.get(name): env[name] = os.environ[name]
env["SANDBOX_BACKEND"] = "apptainer_jupyter"
env["PYTHONPATH"] = os.pathsep.join([os.environ["EXPERIMENT_OPS_DIR"], os.path.join(os.environ["RELAX"], "examples/deepeyes_v2_agentic"), env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))])
print(json.dumps(payload))
PYTHON
)
set -x

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_SET}"
    --input-key prompt
    --label-key reward_model
    --multimodal-keys '{"image":"images"}'
    --reward-key score
    --metadata-key extra_info
    --custom-rm-path examples.deepeyes_v2_agentic.reward_deepeyes_v2.reward_func
    --use-agentic-rollout
    --agent-command "bash ${EXPERIMENT_OPS_DIR}/run_observed_agent.sh"
    --agent-cwd "${SCRIPT_DIR}"
    # Per-run agent log dir: every session's stdout/stderr is tee'd to
    # ${dir}/${session_id}.log (run_agent_app.sh). Successful AND failed
    # sessions are both kept, which is what makes hang/timeout diagnosis
    # possible — Relax's own tmpdir-based capture drops both.
    --agent-env "AGENT_DEBUG_LOG_DIR=${EXPERIMENT_RUN_DIR}/agent"
    --num-rollout ${NUM_ROLLOUT}
    --seed 1234
    --rollout-seed 1234
    --rollout-stop-token-ids 248059 248046
    --rollout-result-dir "${EXPERIMENT_RUN_DIR}/rollouts"
    --custom-rollout-log-function-path experiment_metrics.train
    --custom-eval-rollout-log-function-path experiment_metrics.evaluate
    --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-32}
    --micro-batch-size 1
    --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
    --rollout-max-context-len 16384
    --rollout-max-response-len 4096
    --rollout-max-prompt-len 4096
    --rollout-temperature 1
    --global-batch-size 256
    --rollout-shuffle
    --use-streaming-dataset
)

###############################################################################
#                                EVAL CONFIG                                  #
###############################################################################

EVAL_ARGS=(
    --eval-interval 50
    --eval-prompt-data search "${DATA_DIR}/search_holdout.parquet" vstar "${TEST_FILES[@]}"
    --n-samples-per-eval-prompt 8
    --eval-max-response-len 4096
    --eval-top-p 0.7
    --agentic-eval-concurrency ${AGENTIC_EVAL_CONCURRENCY:-8}
)

###############################################################################
#                              ALGORITHM CONFIG                               #
###############################################################################

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
    --eps-clip-c 3
    --use-tis
)

###############################################################################
#                              OPTIMIZER CONFIG                               #
###############################################################################

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --override-opt-param-scheduler
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --no-rope-fusion
)

###############################################################################
#                               SGLANG CONFIG                                 #
###############################################################################

SGLANG_ARGS=(
    --use-slime-router
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.60
    --sglang-mamba-scheduler-strategy extra_buffer
)

###############################################################################
#                               LOGGING CONFIG                                #
###############################################################################

LOG_ARGS=(
    --use-tensorboard
    --use-metrics-service
    --tb-project-name ${PROJECT_NAME}
    --tb-experiment-name ${EXP_NAME}
)

###############################################################################
#                              MEGATRON CONFIG                                #
###############################################################################

MEGATRON_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 8
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    # Native MoE communication backend used by the Qwen3.6 image recipe.
    --moe-token-dispatcher-type flex
    --moe-flex-dispatcher-backend deepep
    --max-tokens-per-gpu 16384
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-dynamic-batch-size
    --balance-data
)

###############################################################################
#                              RESOURCE CONFIG                                #
###############################################################################

RAY_RESOURCE_ARGS=(
    --resource '{"actor": [1, 8], "rollout": [1, 8]}'
    --max-staleness 0
    --num-data-storage-units 1
    # --use-health-check
    --colocate
)

###############################################################################
#                                 LAUNCH JOB                                  #
###############################################################################

mkdir -p "${EXPERIMENT_RUN_DIR}/logs"

# Pass search configuration to Ray workers without tracing credentials.
set +x
export RUNTIME_ENV_JSON
RUNTIME_ENV_JSON=$(python3 - <<'PYTHON'
import json
import os

payload = json.loads(os.environ["RUNTIME_ENV_JSON"])
for name in ("DEEPEYES_V2_SEARCH_CONFIG", "DEEPEYES_V2_SEARCH_API_KEY"):
    value = os.environ.get(name)
    if value:
        payload["env_vars"][name] = value
print(json.dumps(payload))
PYTHON
)
touch "${EXPERIMENT_RUN_DIR}/.started"
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json "${RUNTIME_ENV_JSON}" \
    -- python3 relax/entrypoints/train.py \
    "${RAY_RESOURCE_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${MEGATRON_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "$@" \
    2>&1 | tee "${EXPERIMENT_RUN_DIR}/logs/train-${TIMESTAMP}.log"
set -x
