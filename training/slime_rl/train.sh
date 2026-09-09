#!/usr/bin/env bash
# One-node security coding-agent GRPO training for Qwen3.5-35B-A3B.
# Adapted from examples/orchard_swe/scripts/
# train_swe_qwen3.5_35B_A3B_multi-node_grpo.sh.

set -euo pipefail

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export AUTOBAX_SRC_DIR="${AUTOBAX_SRC_DIR:-${PUBLIC_ROOT}/dependencies/autobax_arc}"
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR to the external Slime checkout}"
python3 "${SCRIPT_DIR}/../../dependencies/orchard/check.py" "${SLIME_DIR}"
export AZURE_MODAL_PATH="${AZURE_MODAL_PATH:-}"
export ORCHARD_SANDBOX_ENDPOINT="${ORCHARD_SANDBOX_ENDPOINT:-${AZURE_CAAS_ENDPOINT:-${SANDBOX_BASE_URL:-}}}"
MODEL_CONFIG="${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"

required_vars=(
  HF_CHECKPOINT
  REF_MODEL_PATH
  JOINT_PATCHEVAL_DATA
  JOINT_AUTOBAX_DATA
  AUTOBAX_SRC_DIR
  MEGATRON_PATH
  ORCHARD_SANDBOX_ENDPOINT
  SANDBOX_API_KEY
)
for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: ${name}" >&2
    exit 1
  fi
done

for path in \
  "${HF_CHECKPOINT}" \
  "${REF_MODEL_PATH}" \
  "${JOINT_PATCHEVAL_DATA}" \
  "${JOINT_AUTOBAX_DATA}" \
  "${AUTOBAX_SRC_DIR}/in_container_runner.py" \
  "${MEGATRON_PATH}" \
  "${MODEL_CONFIG}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done

# shellcheck disable=SC1090
source "${MODEL_CONFIG}"

NVLINK_COUNT="$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)"
if (( NVLINK_COUNT > 0 )); then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} (${NVLINK_COUNT} NVLink references)"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-${HAS_NVLINK}}"

# Mini-SWE security rollout and fresh-sandbox grader settings.
export SWE_CONFIG_PATH="${SWE_CONFIG_PATH:-${SCRIPT_DIR}/patcheval.yaml}"
export JOINT_PATCHEVAL_DATA JOINT_AUTOBAX_DATA AUTOBAX_SRC_DIR
export JOINT_PATCHEVAL_WEIGHT="${JOINT_PATCHEVAL_WEIGHT:-0.5}"
export JOINT_AUTOBAX_WEIGHT="${JOINT_AUTOBAX_WEIGHT:-0.5}"
export AUTOBAX_TIMEOUT_REWARD_TOTAL="${AUTOBAX_TIMEOUT_REWARD_TOTAL:-1800}"
export BAXBENCH_TEST_TIMEOUT="${BAXBENCH_TEST_TIMEOUT:-90}"
python3 - "${JOINT_PATCHEVAL_WEIGHT}" "${JOINT_AUTOBAX_WEIGHT}" <<'PY'
import math
import sys

weights = tuple(float(value) for value in sys.argv[1:])
if any(not math.isfinite(value) or value < 0 for value in weights):
    raise ValueError(f"joint weights must be finite and nonnegative: {weights}")
if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError(f"joint weights must sum to one: {weights}")
PY
export SWE_TRAJECTORY_DIR="${SWE_TRAJECTORY_DIR:-${SCRIPT_DIR}/runs/trajectories}"
export SWE_TIMEOUT_CREATE_ENV="${SWE_TIMEOUT_CREATE_ENV:-480}"
export SWE_TIMEOUT_LLM_INFERENCE="${SWE_TIMEOUT_LLM_INFERENCE:-60}"
export SWE_TIMEOUT_GET_OBSERVATION="${SWE_TIMEOUT_GET_OBSERVATION:-45}"
export SWE_TIMEOUT_STOP_ENV="${SWE_TIMEOUT_STOP_ENV:-60}"
export SWE_TIMEOUT_REWARD_EXECUTE="${SWE_TIMEOUT_REWARD_EXECUTE:-360}"
export SWE_TIMEOUT_REWARD_TOTAL="${SWE_TIMEOUT_REWARD_TOTAL:-420}"
export SWE_MAX_ENV_RETRIES="${SWE_MAX_ENV_RETRIES:-2}"
export SWE_ENV_RETRY_WAIT="${SWE_ENV_RETRY_WAIT:-10}"
export SWE_MAX_CREATE_ENV_RETRIES="${SWE_MAX_CREATE_ENV_RETRIES:-2}"
export SWE_CREATE_ENV_RETRY_WAIT="${SWE_CREATE_ENV_RETRY_WAIT:-10}"
export SWE_ENV_CREATE_JITTER_MAX="${SWE_ENV_CREATE_JITTER_MAX:-5}"
export SWE_UNRESOLVED_REWARD="${SWE_UNRESOLVED_REWARD:-0}"
export SWE_REWARD_INVALID_PATCH="${SWE_REWARD_INVALID_PATCH:--1.0}"
export SWE_REWARD_APPLY_FAILURE="${SWE_REWARD_APPLY_FAILURE:--0.75}"
export SWE_REWARD_BOTH_FAILED="${SWE_REWARD_BOTH_FAILED:--0.5}"
export SWE_REWARD_SECURITY_ONLY="${SWE_REWARD_SECURITY_ONLY:--0.5}"
export SWE_REWARD_FUNCTIONAL_ONLY="${SWE_REWARD_FUNCTIONAL_ONLY:-0.5}"
export SWE_REWARD_FULL="${SWE_REWARD_FULL:-1.0}"

# One-node topology. Training uses CP/EP across all eight GPUs. Serving uses
# four two-GPU engines with DP-attention and EP so Qwen3.5 GDN tensors remain
# replicated during weight synchronization.
NUM_NODES=1
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-8}"
EP_SIZE="${EP_SIZE:-8}"
ETP_SIZE="${ETP_SIZE:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-8}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
SGLANG_DP_SIZE="${SGLANG_DP_SIZE:-2}"
SGLANG_EP_SIZE="${SGLANG_EP_SIZE:-2}"

if (( TP_SIZE * PP_SIZE * CP_SIZE != GPUS_PER_NODE )); then
  echo "TP_SIZE * PP_SIZE * CP_SIZE must equal GPUS_PER_NODE for this launcher" >&2
  exit 1
fi
if (( ETP_SIZE * EP_SIZE * PP_SIZE > GPUS_PER_NODE )); then
  echo "ETP_SIZE * EP_SIZE * PP_SIZE exceeds GPUS_PER_NODE" >&2
  exit 1
fi
if (( ROLLOUT_NUM_GPUS % ROLLOUT_GPUS_PER_ENGINE != 0 )); then
  echo "ROLLOUT_GPUS_PER_ENGINE must divide ROLLOUT_NUM_GPUS" >&2
  exit 1
fi

# GRPO and rollout defaults. Each prompt yields one four-sample training group;
# progressive rollout may generate a second stride to replace unusable samples.
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES="${N_SAMPLES:-4}"
N_SAMPLES_MAX="${N_SAMPLES_MAX:-8}"
N_SAMPLES_STRIDE="${N_SAMPLES_STRIDE:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES))}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
LR="${LR:-1e-6}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"

if (( GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE * N_SAMPLES )); then
  echo "GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE * N_SAMPLES" >&2
  exit 1
fi
if (( N_SAMPLES_MAX < N_SAMPLES || N_SAMPLES_MAX % N_SAMPLES_STRIDE != 0 )); then
  echo "N_SAMPLES_MAX must be >= N_SAMPLES and divisible by N_SAMPLES_STRIDE" >&2
  exit 1
fi

RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/security_qwen35_$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/checkpoints}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5}"

CKPT_ARGS=(
  --hf-checkpoint "${HF_CHECKPOINT}"
  --ref-load "${REF_MODEL_PATH}"
  --save "${SAVE_PATH}"
  --save-interval "${SAVE_INTERVAL}"
)
if [[ -n "${LOAD_PATH:-}" ]]; then
  CKPT_ARGS+=(--load "${LOAD_PATH}")
fi
if [[ -n "${CKPT_STEP:-}" ]]; then
  CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi

ROLLOUT_ARGS=(
  --prompt-data "${JOINT_PATCHEVAL_DATA}"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --multimodal-keys '{}'
  --rollout-shuffle
  --custom-generate-function-path orchard_compat.generate
  --custom-rm-path "${CUSTOM_RM_PATH:-slime_rl.joint_reward.reward_func}"
  --data-source-path slime_rl.joint_data_source.JointRolloutDataSource
  --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted_nonzero_std_and_pos_reward}"
  --num-rollout "${NUM_ROLLOUT}"
  --over-sampling-batch-size "${ROLLOUT_BATCH_SIZE}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES}"
  # --n-samples-per-prompt-max "${N_SAMPLES_MAX}"
  # --n-samples-per-prompt-stride "${N_SAMPLES_STRIDE}"
  # --progressive-rollout-pos-ratio-min 0
  # --progressive-rollout-pos-ratio-max 1
  --rollout-max-context-len "${MAX_CONTEXT_LEN}"
  --rollout-max-response-len "${MAX_RESPONSE_LEN}"
  --rollout-stop-token-ids 248046
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
  --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)
if [[ -n "${START_ROLLOUT_ID:-}" ]]; then
  ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size "${ETP_SIZE}"
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size 1024
)
if (( TP_SIZE > 1 )); then
  PERF_ARGS+=(--sequence-parallel)
fi

ALGO_ARGS=(
  --advantage-estimator "${ADVANTAGE_ESTIMATOR:-grpo}"
  --ppo-epochs "${PPO_EPOCHS}"
  --kl-loss-coef "${KL_LOSS_COEF:-0}"
  --kl-loss-type low_var_kl
  --entropy-coef "${ENTROPY_COEF:-1e-4}"
  --eps-clip "${EPS_CLIP:-0.2}"
  --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style cosine
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --sglang-moe-runner-backend triton
  --sglang-enable-dp-attention
  --sglang-data-parallel-size "${SGLANG_DP_SIZE}"
  --sglang-expert-parallel-size "${SGLANG_EP_SIZE}"
  --sglang-enable-dp-lm-head
  --sglang-moe-dense-tp-size 1
  --sglang-server-concurrency 64
  --sglang-max-running-requests 64
  # --sglang-chunked-prefill-size 8192
  --sglang-chunked-prefill-size 16384
  --sglang-mem-fraction-static 0.75
  --sglang-mamba-scheduler-strategy extra_buffer
  --sglang-tool-call-parser qwen3_coder
  --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
  --attention-dropout 0
  --hidden-dropout 0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --moe-token-dispatcher-type flex
  --moe-enable-deepep
  --colocate
)

EVAL_ARGS=()
if [[ -n "${EVAL_PROMPT_DATA:-}" ]]; then
  EVAL_ARGS+=(
    --eval-interval "${EVAL_INTERVAL:-10}"
    --skip-eval-before-train
    --eval-prompt-data security_val "${EVAL_PROMPT_DATA}"
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-8192}"
    --eval-temperature "${EVAL_TEMPERATURE:-0.0}"
    --eval-task-timeout "${EVAL_TASK_TIMEOUT:-1500}"
    --eval-max-concurrency "${EVAL_MAX_CONCURRENCY:-128}"
  )
fi

WANDB_ARGS=()
if [[ "${USE_WANDB:-1}" == 1 ]]; then
  : "${WANDB_API_KEY:?Set WANDB_API_KEY when USE_WANDB=1}"
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT:-safevibe-slime-rl}"
    --wandb-group "${WANDB_GROUP:-qwen35-1node-${ADVANTAGE_ESTIMATOR:-grpo}}"
    --wandb-dir "${WANDB_DIR:-${RUN_ROOT}/wandb}"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
  fi
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  PYTHONPATH="${AZURE_MODAL_PATH}:${MEGATRON_PATH}:${SLIME_DIR}:${TRAINING_DIR}:${PYTHONPATH:-}" \
    python3 - "${JOINT_PATCHEVAL_DATA}" "${JOINT_AUTOBAX_DATA}" <<'PY'
import json
import sys

from orchard_env.client.sandbox_client import AsyncSandboxClient  # noqa: F401
from orchard_compat import generate  # noqa: F401
from slime_rl.joint_data_source import JointRolloutDataSource  # noqa: F401
from slime_rl.joint_reward import reward_func  # noqa: F401

requirements = {
    "patcheval": {"grade_image_name", "mask_patch", "test_patch", "security_eval_cmd"},
    "autobax": {"scenario", "env", "code_dir", "image_url"},
}
for task_type, path in zip(("patcheval", "autobax"), sys.argv[1:], strict=True):
    with open(path, encoding="utf-8") as source:
        row = json.loads(next(line for line in source if line.strip()))
    if not {"prompt", "label", "metadata"} <= row.keys():
        raise ValueError(f"{task_type} data must contain prompt, label, and metadata")
    metadata = row["metadata"]
    missing = requirements[task_type] - metadata.keys()
    if missing:
        raise ValueError(f"{task_type} metadata is missing: {sorted(missing)}")
    if metadata.get("task_type") != task_type:
        raise ValueError(f"{path} is not routed as task_type={task_type}")
PY
  echo "Preflight passed"
  echo "training: nodes=${NUM_NODES} gpus=${GPUS_PER_NODE} tp=${TP_SIZE} cp=${CP_SIZE} ep=${EP_SIZE}"
  echo "rollout: gpus=${ROLLOUT_NUM_GPUS} per_engine=${ROLLOUT_GPUS_PER_ENGINE} dp=${SGLANG_DP_SIZE} ep=${SGLANG_EP_SIZE}"
  echo "patcheval_data=${JOINT_PATCHEVAL_DATA} autobax_data=${JOINT_AUTOBAX_DATA}"
  echo "weights=patcheval:${JOINT_PATCHEVAL_WEIGHT},autobax:${JOINT_AUTOBAX_WEIGHT}"
  echo "save=${SAVE_PATH} load=${LOAD_PATH:-none}"
  exit 0
fi

mkdir -p "${RUN_ROOT}/rollout_dumps" "${SWE_TRAJECTORY_DIR}" "${SAVE_PATH}"

if [[ "${RESET_RAY:-1}" == 1 ]]; then
  pkill -9 sglang || true
  ray stop --force || true
  pkill -9 ray || true
fi

if ! ray status >/dev/null 2>&1; then
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${GPUS_PER_NODE}" \
    --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
fi

cd "${SLIME_DIR}"
export MEGATRON_PATH SLIME_DIR TRAINING_DIR
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

keys = (
    "ORCHARD_SANDBOX_ENDPOINT", "AZURE_CAAS_ENDPOINT", "AZURE_MODAL_PATH", "SANDBOX_API_KEY",
    "SWE_CONFIG_PATH", "SWE_TRAJECTORY_DIR",
    "SWE_TIMEOUT_CREATE_ENV", "SWE_TIMEOUT_LLM_INFERENCE",
    "SWE_TIMEOUT_GET_OBSERVATION", "SWE_TIMEOUT_STOP_ENV",
    "SWE_TIMEOUT_REWARD_EXECUTE", "SWE_TIMEOUT_REWARD_TOTAL",
    "SWE_MAX_ENV_RETRIES", "SWE_ENV_RETRY_WAIT",
    "SWE_MAX_CREATE_ENV_RETRIES", "SWE_CREATE_ENV_RETRY_WAIT",
    "SWE_ENV_CREATE_JITTER_MAX", "SWE_UNRESOLVED_REWARD",
    "SWE_REWARD_INVALID_PATCH", "SWE_REWARD_APPLY_FAILURE",
    "SWE_REWARD_BOTH_FAILED", "SWE_REWARD_SECURITY_ONLY",
    "SWE_REWARD_FUNCTIONAL_ONLY", "SWE_REWARD_FULL",
    "JOINT_PATCHEVAL_DATA", "JOINT_AUTOBAX_DATA",
    "JOINT_PATCHEVAL_WEIGHT", "JOINT_AUTOBAX_WEIGHT",
    "PROCESS_REWARD_OUTCOME_WEIGHT", "PROCESS_REWARD_PROCESS_WEIGHT",
    "PROCESS_REWARD_MATCHED_CWE", "PROCESS_REWARD_TEST_GENERATION",
    "PROCESS_REWARD_TEST_EXECUTION", "PROCESS_REWARD_SUSPICIOUS_COMMAND",
    "PROCESS_REWARD_MAX_MATCHED_CWE", "PROCESS_REWARD_MAX_TEST_GENERATIONS",
    "PROCESS_REWARD_MAX_TEST_EXECUTIONS", "PROCESS_REWARD_MAX_SUSPICIOUS",
    "PROCESS_REWARD_OBJECTIVE", "PROCESS_REWARD_BASE_RM_PATH",
    "AUTOBAX_SRC_DIR", "AUTOBAX_TIMEOUT_REWARD_TOTAL",
    "AUTOBAX_HARNESS_URL", "BAXBENCH_TEST_TIMEOUT",
    "MASTER_ADDR", "MASTER_PORT", "no_proxy", "NO_PROXY",
)
env = {key: os.environ[key] for key in keys if key in os.environ}
env["PYTHONPATH"] = ":".join(
    (os.environ["AZURE_MODAL_PATH"], os.environ["MEGATRON_PATH"], os.environ["SLIME_DIR"], os.environ["TRAINING_DIR"])
)
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = os.environ["NCCL_NVLS_ENABLE"]
print(json.dumps({"env_vars": env}))
PY
)"

ray job submit --address=http://127.0.0.1:8265 \
  --runtime-env-json="${RUNTIME_ENV_JSON}" -- python3 -u train.py \
  --actor-num-nodes "${NUM_NODES}" --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" "${ALGO_ARGS[@]}" "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" "${EVAL_ARGS[@]}" "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/run.log"

echo "One-node security training complete"
