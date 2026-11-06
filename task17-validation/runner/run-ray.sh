#!/bin/bash
set -euo pipefail
export PATH=/tmp/relax-task17/ray-venv/bin:$PATH
export RELAX=${RELAX_REPO:-/workspace/task17/Relax}
export MEGATRON=/tmp/relax-task17/dependencies/Megatron-LM
export PYTHONPATH="$RELAX:$MEGATRON"
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
export RAY_ADDRESS=auto RAY_DASHBOARD_ADDRESS=http://127.0.0.1:8265 RAY_NO_WAIT=1
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=0
export RELAX_SKIP_CLUSTER_CLEANUP=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TORCH_LIB_PATH=/tmp/relax-task17/ray-venv/lib/python3.12/site-packages/torch/lib
export NCCL_NET=Socket NCCL_NET_PLUGIN=none NCCL_SOCKET_IFNAME=lo NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export GLOO_SOCKET_IFNAME=lo CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 RAY_DEDUP_LOGS=0
export RELAX_PROPAGATE_ENV_VARS=NCCL_NET,NCCL_NET_PLUGIN,NCCL_SOCKET_IFNAME,NCCL_SOCKET_FAMILY,NCCL_IB_DISABLE,NCCL_NVLS_ENABLE,GLOO_SOCKET_IFNAME,CUBLAS_WORKSPACE_CONFIG,HF_HUB_OFFLINE,TRANSFORMERS_OFFLINE,RAY_DEDUP_LOGS
if [[ -z "${RELAX_ENTRYPOINT_MODE:-}" ]]; then
    exec bash "$RELAX/scripts/entrypoint/ray-job.sh" "$0" "$@"
fi
model=$1
export CONDITION_CACHE_BYTES=$2 STEPS=$3 DATASET_SIZE=$4
label=$5
shift 5
if [[ "$model" == qwen ]]; then
    export MODEL_PATH=/tmp/relax-task17/models/Qwen-Image-Edit
    adapter=relax.models.qwen_image.edit_sft.QwenImageEditSFTAdapter
else
    export MODEL_PATH=/tmp/relax-task17/models/SenseNova-U1.5-8B-MoT
    adapter=relax.models.sensenova.adapter.SenseNovaSFTAdapter
fi
export DATA_ROOT=/tmp/relax-task17/data METADATA_PATH=/tmp/relax-task17/data/train-$DATASET_SIZE.json
export GPUS=2 MODE=lora LORA_RANK=8 MAX_PIXELS=262144 CONDITIONING_IMAGES=1
export ATTENTION_IMPLEMENTATION=torch DETERMINISTIC_ATTENTION=1 PREFIX_GRADIENT_CHECKPOINTING=1
export CONDITION_CACHE_ENTRIES=128 IMAGE_PREFETCH_DEPTH=0
export SAVE_DIR=/tmp/relax-task17/runs/$label/checkpoints SAVE_INTERVAL=$STEPS
RUNTIME_ENV_JSON=$(python - <<'PY'
import json, os
runtime = json.loads(os.environ['RUNTIME_ENV_JSON'])
for name in os.environ['RELAX_PROPAGATE_ENV_VARS'].split(',') + ['RELAX_PROPAGATE_ENV_VARS']:
    runtime['env_vars'][name] = os.environ[name]
print(json.dumps(runtime))
PY
)
export RUNTIME_ENV_JSON
cd "$RELAX"
exec bash scripts/training/diffusion/run-sensenova-i2i-sft-8xgpu.sh \
    --model-adapter-path "$adapter" --num-gpus-per-node 2 --actor-num-gpus-per-node 2 \
    --global-batch-size 4 --rollout-batch-size 4 --lr-warmup-iters 0 --tq-rdma-mode off \
    --tb-experiment-name "$label" "$@"
