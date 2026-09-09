#!/usr/bin/env bash
# End-to-end SafeVibe coding-agent RL for Qwen3.5-35B-A3B on one 8-GPU node
# using slime's Mini-SWE-Agent
# v2 tool-calling rollout.  The model talks directly to the SGLang rollout
# endpoint and uses Mini-SWE's bash tool/environment interface; Claude Code,
# Anthropic credentials, Node.js tarballs, and the local custom sandbox adapter
# are not part of this launch path.
# Run from a long-lived shell / tmux session on the Ray head node; do not wrap
# in a short-lived nohup launcher or Ray child processes get cleaned up with it.

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SAFEVIBE_DIR="${SAFEVIBE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR to the patched Orchard trainer/slime checkout}"
python3 "${SCRIPT_DIR}/../../dependencies/orchard/check.py" "${SLIME_DIR}"
MEGATRON_PATH="${MEGATRON_PATH:-${SAFEVIBE_DIR}/dependencies/megatron-lm/Megatron-LM}"
EXPECTED_HEAD_HOST="${EXPECTED_HEAD_HOST:-$(hostname -s)}"

# Conda's bundled GCC 7 linker cannot read the pod's newer glibc RELR
# sections. FlashQLA/TileLang JIT compilation must use the host toolchain.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export LD="${LD:-/usr/bin/ld}"

# Do not discover or start remote workers for this single-node launcher.
HOSTFILE="${HOSTFILE:-/dev/null}"

# Load remote sandbox-service credentials without exposing values through the
# launcher's xtrace output. Explicitly exported environment variables still
# take precedence over values from this file.
SANDBOX_ENV_FILE="${SANDBOX_ENV_FILE:-${SCRIPT_DIR}/configs/.sandbox_env}"
if [[ -f "${SANDBOX_ENV_FILE}" ]]; then
  set +x
  SANDBOX_BASE_URL_OVERRIDE="${SANDBOX_BASE_URL:-}"
  SANDBOX_SECOND_URL_OVERRIDE="${SANDBOX_SECOND_URL:-}"
  SANDBOX_API_KEY_OVERRIDE="${SANDBOX_API_KEY:-}"
  # shellcheck disable=SC1090
  source "${SANDBOX_ENV_FILE}"
  [[ -n "${SANDBOX_BASE_URL_OVERRIDE}" ]] && SANDBOX_BASE_URL="${SANDBOX_BASE_URL_OVERRIDE}"
  [[ -n "${SANDBOX_SECOND_URL_OVERRIDE}" ]] && SANDBOX_SECOND_URL="${SANDBOX_SECOND_URL_OVERRIDE}"
  [[ -n "${SANDBOX_API_KEY_OVERRIDE}" ]] && SANDBOX_API_KEY="${SANDBOX_API_KEY_OVERRIDE}"
  export SANDBOX_BASE_URL SANDBOX_SECOND_URL SANDBOX_API_KEY
  unset SANDBOX_BASE_URL_OVERRIDE SANDBOX_SECOND_URL_OVERRIDE SANDBOX_API_KEY_OVERRIDE
fi

# azure_modal_docker.py uses the Azure names documented by Mini-SWE. Keep the
# private .sandbox_env file as the source of truth and translate its endpoint.
export AZURE_CAAS_ENDPOINT="${AZURE_CAAS_ENDPOINT:-${SANDBOX_BASE_URL:-${SANDBOX_SECOND_URL:-}}}"
export ORCHARD_SANDBOX_ENDPOINT="${ORCHARD_SANDBOX_ENDPOINT:-${AZURE_CAAS_ENDPOINT}}"
export AZURE_MODAL_PATH="${AZURE_MODAL_PATH:-}"

# One B200 node with eight visible GPUs.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))"

# ============ model parallelism ============
# TP=2 and CP=4 use all eight training ranks. EP spans the full node.
export TP_SIZE="${TP_SIZE:-2}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-4}"
export EP_SIZE="${EP_SIZE:-8}"
export ETP_SIZE="${ETP_SIZE:-1}"

# ============ rollout engine ============
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-2}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-2}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.75}"
PREFILL_NUM_SERVERS="${PREFILL_NUM_SERVERS:-0}"
if ! [[ "${PREFILL_NUM_SERVERS}" =~ ^[0-9]+$ ]]; then
  echo "PREFILL_NUM_SERVERS must be a non-negative integer; got ${PREFILL_NUM_SERVERS}." >&2
  exit 1
fi

# ============ Qwen3.5-35B-A3B architecture ============
NLAYERS=40
FIRST_K_DENSE_REPLACE=0

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done
printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-65536}"
MAX_GEN_LEN="${MAX_GEN_LEN:-4096}"
ROLLOUT_STOP_TOKEN_ID="${ROLLOUT_STOP_TOKEN_ID:-248046}" # <|im_end|>
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"

# One prompt group per optimizer step, with replacement samples for incomplete
# rollouts and bounded per-GPU token chunks.
NUM_ROLLOUT="${NUM_ROLLOUT:-100}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
N_SAMPLES_PER_PROMPT_MAX="${N_SAMPLES_PER_PROMPT_MAX:-$((N_SAMPLES_PER_PROMPT * 2))}"
N_SAMPLES_PER_PROMPT_STRIDE="${N_SAMPLES_PER_PROMPT_STRIDE:-${N_SAMPLES_PER_PROMPT}}"
PROGRESSIVE_ROLLOUT_POS_RATIO_MIN="${PROGRESSIVE_ROLLOUT_POS_RATIO_MIN:-0}"
PROGRESSIVE_ROLLOUT_POS_RATIO_MAX="${PROGRESSIVE_ROLLOUT_POS_RATIO_MAX:-1}"
if ! [[ "${NUM_ROLLOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_ROLLOUT must be a positive integer; got ${NUM_ROLLOUT}." >&2
  exit 1
fi
if (( GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT )); then
  echo "GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT." >&2
  exit 1
fi
if ! [[ "${MAX_TOKENS_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_TOKENS_PER_GPU must be a positive integer; got ${MAX_TOKENS_PER_GPU}." >&2
  exit 1
fi
if ! [[ "${ROLLOUT_SEED}" =~ ^[0-9]+$ ]]; then
  echo "ROLLOUT_SEED must be a non-negative integer; got ${ROLLOUT_SEED}." >&2
  exit 1
fi

PROGRESSIVE_ROLLOUT_ARGS=()
if [[ -n "${N_SAMPLES_PER_PROMPT_MAX}" ]]; then
  if ! [[ "${N_SAMPLES_PER_PROMPT_MAX}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_SAMPLES_PER_PROMPT_MAX must be a positive integer; got ${N_SAMPLES_PER_PROMPT_MAX}." >&2
    exit 1
  fi
  if ! [[ "${N_SAMPLES_PER_PROMPT_STRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "N_SAMPLES_PER_PROMPT_STRIDE must be a positive integer when progressive rollout is enabled; got ${N_SAMPLES_PER_PROMPT_STRIDE:-unset}." >&2
    exit 1
  fi
  if (( N_SAMPLES_PER_PROMPT_MAX <= N_SAMPLES_PER_PROMPT )); then
    echo "N_SAMPLES_PER_PROMPT_MAX must exceed N_SAMPLES_PER_PROMPT; got ${N_SAMPLES_PER_PROMPT_MAX} <= ${N_SAMPLES_PER_PROMPT}." >&2
    exit 1
  fi
  if (( N_SAMPLES_PER_PROMPT_MAX % N_SAMPLES_PER_PROMPT_STRIDE != 0 )); then
    echo "N_SAMPLES_PER_PROMPT_MAX must be divisible by N_SAMPLES_PER_PROMPT_STRIDE; got ${N_SAMPLES_PER_PROMPT_MAX} and ${N_SAMPLES_PER_PROMPT_STRIDE}." >&2
    exit 1
  fi
  PROGRESSIVE_ROLLOUT_ARGS+=(
    --n-samples-per-prompt-max "${N_SAMPLES_PER_PROMPT_MAX}"
    --n-samples-per-prompt-stride "${N_SAMPLES_PER_PROMPT_STRIDE}"
    --progressive-rollout-pos-ratio-min "${PROGRESSIVE_ROLLOUT_POS_RATIO_MIN}"
    --progressive-rollout-pos-ratio-max "${PROGRESSIVE_ROLLOUT_POS_RATIO_MAX}"
  )
fi

# ============ paths — override before launching ============
# Point these at your own checkpoints and dataset.
HF_CHECKPOINT="${HF_CHECKPOINT:?Set HF_CHECKPOINT to the Qwen3.5-35B-A3B Hugging Face checkpoint}"
REF_MODEL_PATH="${REF_MODEL_PATH:-${HF_CHECKPOINT%/}_torch_dist_slime-0.3.0}"
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/../../data/recipes/rl_patcheval_train_generic.jsonl}"
export SWE_CONFIG_PATH="${SWE_CONFIG_PATH:-${SCRIPT_DIR}/configs/patcheval.yaml}"
CUSTOM_RM_PATH="${CUSTOM_RM_PATH:-slime_rl.patcheval_reward.reward_func}"
DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted_and_reward_nonzero_std}"
DATA_SOURCE_PATH="${DATA_SOURCE_PATH:-slime.rollout.data_source.RolloutDataSourceWithBuffer}"

EXP_TAG="${EXP_TAG:-qwen3.5_35b_a3b_swe_rl_1node}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/runs/${EXP_TAG}_${STAMP}}"

# Resumable Megatron checkpoints live outside the source workspace.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${SAFEVIBE_DIR}/checkpoints}"
SAVE_PATH="${SAVE_PATH:-${CHECKPOINT_ROOT}/$(basename "${RUN_ROOT}")}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
if ! [[ "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SAVE_INTERVAL must be a positive integer; got ${SAVE_INTERVAL}." >&2
  exit 1
fi

# W&B credentials are supplied through WANDB_API_KEY so they never appear in
# Ray's printed train.py command.
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-safevibe-slime-rl}"
WANDB_GROUP="${WANDB_GROUP:-qwen3.5-35b-a3b-swe-1node}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"
if [[ "${USE_WANDB}" != "0" && "${USE_WANDB}" != "1" ]]; then
  echo "USE_WANDB must be 0 or 1; got ${USE_WANDB}." >&2
  exit 1
fi
if [[ "${WANDB_MODE}" != "online" && "${WANDB_MODE}" != "offline" && "${WANDB_MODE}" != "disabled" ]]; then
  echo "WANDB_MODE must be online, offline, or disabled; got ${WANDB_MODE}." >&2
  exit 1
fi

for required_path in "${SLIME_DIR}/train.py" "${SWE_CONFIG_PATH}" "${HF_CHECKPOINT}" "${REF_MODEL_PATH}" "${PROMPT_DATA}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

if [[ ! -d "${MEGATRON_PATH}/megatron" ]]; then
  echo "Megatron checkout is missing or invalid: ${MEGATRON_PATH}" >&2
  exit 1
fi

CURRENT_HOST="$(hostname -s)"
if [[ "${CURRENT_HOST}" != "${EXPECTED_HEAD_HOST}" ]]; then
  echo "WARN: expected head host ${EXPECTED_HEAD_HOST}, running on ${CURRENT_HOST}." >&2
fi

AVAILABLE_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l)"
if (( AVAILABLE_GPUS < ACTOR_NUM_GPUS_PER_NODE )); then
  echo "Need ${ACTOR_NUM_GPUS_PER_NODE} GPUs, but this node exposes ${AVAILABLE_GPUS}." >&2
  exit 1
fi

MODEL_PARALLEL_SIZE="$((TP_SIZE * CP_SIZE * PP_SIZE))"
EXPERT_PARALLEL_SIZE="$((ETP_SIZE * EP_SIZE * PP_SIZE))"
if (( ROLLOUT_NUM_GPUS % MODEL_PARALLEL_SIZE != 0 )); then
  echo "Invalid training topology: ${ROLLOUT_NUM_GPUS} is not divisible by TP*CP*PP=${MODEL_PARALLEL_SIZE}." >&2
  exit 1
fi

if (( ROLLOUT_NUM_GPUS % EXPERT_PARALLEL_SIZE != 0 )); then
  echo "Invalid expert topology: ${ROLLOUT_NUM_GPUS} is not divisible by ETP*EP*PP=${EXPERT_PARALLEL_SIZE}." >&2
  exit 1
fi

if (( ROLLOUT_NUM_GPUS % ROLLOUT_TP_SIZE != 0 )); then
  echo "Invalid rollout topology: GPUs per engine (${ROLLOUT_TP_SIZE}) must divide total rollout GPUs (${ROLLOUT_NUM_GPUS})." >&2
  exit 1
fi

if (( ROLLOUT_DP_SIZE > ROLLOUT_TP_SIZE )); then
  echo "Invalid rollout topology: SGLang DP size (${ROLLOUT_DP_SIZE}) cannot exceed GPUs per engine (${ROLLOUT_TP_SIZE})." >&2
  exit 1
fi

for required_command in python3 ray ssh; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Required command is not available: ${required_command}" >&2
    exit 1
  fi
done

# ============ logging ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps" "${SAVE_PATH}"
LOG_FILE="${LOG_DIR}/run.log"
WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  mkdir -p "${WANDB_DIR}"
  WANDB_ARGS+=(
    --use-wandb
    --wandb-mode "${WANDB_MODE}"
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-dir "${WANDB_DIR}"
    --wandb-always-use-train-step
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
  fi
fi
echo "======================================================================"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "CHECKPOINTS=path:${SAVE_PATH},interval:${SAVE_INTERVAL}"
echo "ROLLOUT_POLICY=filter:${DYNAMIC_SAMPLING_FILTER_PATH},shuffle:1,seed:${ROLLOUT_SEED},progressive_max:${N_SAMPLES_PER_PROMPT_MAX:-disabled},progressive_stride:${N_SAMPLES_PER_PROMPT_STRIDE:-disabled}"
echo "WANDB=enabled:${USE_WANDB},mode:${WANDB_MODE},project:${WANDB_PROJECT},group:${WANDB_GROUP},dir:${WANDB_DIR}"
echo "======================================================================"

MODEL_ARGS=(
   --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 2
   --kv-channels 256
   --num-layers 40
   --hidden-size 2048
   --ffn-hidden-size 512
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

   # moe
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 256
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   --moe-permute-fusion
   --moe-aux-loss-coeff 0

   # qwen3.5 specific
   --attention-output-gate
   --moe-shared-expert-gate
)

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
   --save "${SAVE_PATH}"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --custom-generate-function-path orchard_compat.generate
   --custom-rm-path "${CUSTOM_RM_PATH}"
   --dynamic-sampling-filter-path "${DYNAMIC_SAMPLING_FILTER_PATH}"
   --data-source-path "${DATA_SOURCE_PATH}"
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --multimodal-keys '{}'
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-stop-token-ids ${ROLLOUT_STOP_TOKEN_ID}
   --rollout-temperature 1.0
   --rollout-shuffle
   --rollout-seed "${ROLLOUT_SEED}"
   "${PROGRESSIVE_ROLLOUT_ARGS[@]}"
   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --micro-batch-size 1
   --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
)

SEQUENCE_PARALLEL_ARGS=()
if (( TP_SIZE > 1 )); then
  SEQUENCE_PARALLEL_ARGS+=(--sequence-parallel)
fi

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE}
   "${SEQUENCE_PARALLEL_ARGS[@]}"
   --pipeline-model-parallel-size ${PP_SIZE}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size ${EP_SIZE}
   --expert-tensor-parallel-size ${ETP_SIZE}
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   # one CP rank's slice of MAX_CONTEXT_LEN; log-probs chunked along T to
   # avoid OOM on long single trajectories.
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
   --use-dynamic-batch-size
)

ALGO_ARGS=(
   --advantage-estimator gspo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 1e-4
   --eps-clip-high 2e-4
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
   --rollout-num-gpus-per-engine ${ROLLOUT_TP_SIZE}
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION}
   --sglang-enable-dp-attention
   --sglang-dp-size ${ROLLOUT_DP_SIZE}
   --sglang-ep-size ${ROLLOUT_EP_SIZE}
   --sglang-enable-dp-lm-head
   --sglang-moe-dense-tp-size 1
   --sglang-moe-runner-backend triton
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
   --sglang-mamba-scheduler-strategy extra_buffer
)

if (( PREFILL_NUM_SERVERS > 0 )); then
  SGLANG_ARGS+=(--prefill-num-servers "${PREFILL_NUM_SERVERS}")
fi

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --moe-token-dispatcher-type flex
   --moe-enable-deepep
   --colocate
)

# ============ ray cluster network ============
# Set MASTER_ADDR before the Mini-SWE configuration block.
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ Mini-SWE-Agent rollout knobs ============
export SWE_TRAJECTORY_DIR="${SWE_TRAJECTORY_DIR:-${RUN_ROOT}/trajectories}"
export SWE_TIMEOUT_CREATE_ENV="${SWE_TIMEOUT_CREATE_ENV:-480}"
export SWE_TIMEOUT_LLM_INFERENCE="${SWE_TIMEOUT_LLM_INFERENCE:-120}"
export SWE_TIMEOUT_GET_OBSERVATION="${SWE_TIMEOUT_GET_OBSERVATION:-45}"
export SWE_TIMEOUT_STOP_ENV="${SWE_TIMEOUT_STOP_ENV:-120}"
export SWE_TIMEOUT_REWARD_EXECUTE="${SWE_TIMEOUT_REWARD_EXECUTE:-900}"
export SWE_TIMEOUT_REWARD_TOTAL="${SWE_TIMEOUT_REWARD_TOTAL:-1200}"
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
export SLIME_DISABLE_BUFFER_CPU_BACKUP="${SLIME_DISABLE_BUFFER_CPU_BACKUP:-0}"
export SGLANG_ENABLE_TORCH_INFERENCE_MODE="${SGLANG_ENABLE_TORCH_INFERENCE_MODE:-true}"

# ============ proxy bypass for in-cluster traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

if ! PYTHONPATH="${AZURE_MODAL_PATH}:${SLIME_DIR}:${SAFEVIBE_DIR}:${PYTHONPATH:-}" python3 - "${DYNAMIC_SAMPLING_FILTER_PATH}" <<'PY'
import sys

import minisweagent
import swebench
import transformers
from orchard_env.client.sandbox_client import AsyncSandboxClient
from slime.utils.misc import load_function

load_function(sys.argv[1])

required = "4.57.1"
if transformers.__version__ != required:
    raise RuntimeError(
        f"This SGLang Qwen3.5 config requires transformers=={required}; "
        f"found {transformers.__version__}"
    )
PY
then
  echo "Runtime dependency/filter check failed. Install mini-swe-agent==1.17.5, the SWE-bench harness, and transformers==4.57.1, and verify DYNAMIC_SAMPLING_FILTER_PATH in the Ray worker environment." >&2
  exit 1
fi

if [[ -z "${ORCHARD_SANDBOX_ENDPOINT:-}" ]]; then
  echo "Mini-SWE azure_docker requires AZURE_CAAS_ENDPOINT (mapped from SANDBOX_BASE_URL or SANDBOX_SECOND_URL)." >&2
  exit 1
fi
if [[ -z "${SANDBOX_API_KEY:-}" ]]; then
  echo "Mini-SWE azure_docker requires SANDBOX_API_KEY." >&2
  exit 1
fi


if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight passed on ${CURRENT_HOST}."
  echo "SLIME_DIR=${SLIME_DIR}"
  echo "MEGATRON_PATH=${MEGATRON_PATH}"
  echo "HF_CHECKPOINT=${HF_CHECKPOINT}"
  echo "REF_MODEL_PATH=${REF_MODEL_PATH}"
  echo "PROMPT_DATA=${PROMPT_DATA}"
  echo "SAVE_PATH=${SAVE_PATH}"
  echo "SAVE_INTERVAL=${SAVE_INTERVAL}"
  echo "SWE_CONFIG_PATH=${SWE_CONFIG_PATH}"
  echo "CUSTOM_RM_PATH=${CUSTOM_RM_PATH}"
  echo "DATA_SOURCE_PATH=${DATA_SOURCE_PATH}"
  echo "HOSTFILE=${HOSTFILE:-${SAFEVIBE_DIR}/hostfile}"
  echo "TRAINING_TOPOLOGY=nodes:${ACTOR_NUM_NODES},gpus_per_node:${ACTOR_NUM_GPUS_PER_NODE},tp:${TP_SIZE},cp:${CP_SIZE},pp:${PP_SIZE},ep:${EP_SIZE},etp:${ETP_SIZE}"
  echo "ROLLOUT_TOPOLOGY=gpus:${ROLLOUT_NUM_GPUS},tp:${ROLLOUT_TP_SIZE},dp:${ROLLOUT_DP_SIZE},ep:${ROLLOUT_EP_SIZE}"
  echo "BOUNDED_LOOP=num_rollout:${NUM_ROLLOUT},rollout_batch:${ROLLOUT_BATCH_SIZE},samples_per_prompt:${N_SAMPLES_PER_PROMPT},global_batch:${GLOBAL_BATCH_SIZE}"
  echo "ROLLOUT_POLICY=filter:${DYNAMIC_SAMPLING_FILTER_PATH},shuffle:1,seed:${ROLLOUT_SEED},progressive_max:${N_SAMPLES_PER_PROMPT_MAX:-disabled},progressive_stride:${N_SAMPLES_PER_PROMPT_STRIDE:-disabled},positive_ratio:${PROGRESSIVE_ROLLOUT_POS_RATIO_MIN}-${PROGRESSIVE_ROLLOUT_POS_RATIO_MAX}"
  echo "MEMORY_LIMITS=max_context:${MAX_CONTEXT_LEN},max_generation:${MAX_GEN_LEN},max_tokens_per_gpu:${MAX_TOKENS_PER_GPU},log_probs_chunk:${LOG_PROBS_CHUNK_SIZE}"
  echo "WANDB=enabled:${USE_WANDB},mode:${WANDB_MODE},project:${WANDB_PROJECT},group:${WANDB_GROUP},dir:${WANDB_DIR},entity:${WANDB_ENTITY:-default}"
  exit 0
fi

cd "${SLIME_DIR}"

# ============ bring up ray cluster ============
# Best-effort cleanup after configuration validation so a typo does not tear
# down an otherwise healthy local Ray/SGLang job.
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -f "${HOSTFILE}" ]]; then
  for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
    [[ -z "${WORKER_IP}" ]] && continue
    [[ "${WORKER_IP}" == "${MASTER_ADDR}" ]] && continue
    echo "Starting Ray worker on ${WORKER_IP}"
    ssh -o StrictHostKeyChecking=no "root@${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
       ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_NUM_GPUS_PER_NODE} \
         --node-ip-address ${WORKER_IP} --disable-usage-stats" &
  done
  wait
fi

echo "Waiting for Ray cluster to stabilize..."
sleep 30
ray status

# ============ runtime env propagated to ray workers ============
export SLIME_DIR SAFEVIBE_DIR MEGATRON_PATH
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "SWE_CONFIG_PATH", "SWE_TRAJECTORY_DIR",
    "SWE_TIMEOUT_CREATE_ENV", "SWE_TIMEOUT_LLM_INFERENCE",
    "SWE_TIMEOUT_GET_OBSERVATION", "SWE_TIMEOUT_STOP_ENV",
    "SWE_TIMEOUT_REWARD_EXECUTE", "SWE_TIMEOUT_REWARD_TOTAL",
    "AUTOBAX_TIMEOUT_REWARD_TOTAL", "BAXBENCH_TEST_TIMEOUT",
    "AUTOBAX_SRC_DIR", "AUTOBAX_HARNESS_URL",
    "JOINT_PATCHEVAL_DATA", "JOINT_AUTOBAX_DATA",
    "JOINT_PATCHEVAL_WEIGHT", "JOINT_AUTOBAX_WEIGHT",
    "DYNAMIC_FILTER_MAX_RETRY_ROUNDS",
    "SWE_MAX_ENV_RETRIES", "SWE_ENV_RETRY_WAIT",
    "SWE_MAX_CREATE_ENV_RETRIES", "SWE_CREATE_ENV_RETRY_WAIT",
    "SWE_ENV_CREATE_JITTER_MAX", "SWE_UNRESOLVED_REWARD",
    "SWE_REWARD_INVALID_PATCH", "SWE_REWARD_APPLY_FAILURE",
    "SWE_REWARD_BOTH_FAILED", "SWE_REWARD_SECURITY_ONLY",
    "SWE_REWARD_FUNCTIONAL_ONLY", "SWE_REWARD_FULL",
    "SLIME_DISABLE_BUFFER_CPU_BACKUP",
    "SGLANG_ENABLE_TORCH_INFERENCE_MODE",
    "ORCHARD_SANDBOX_ENDPOINT", "AZURE_CAAS_ENDPOINT", "AZURE_MODAL_PATH", "AZURE_SANDBOX_MANIFEST_DIR",
    "SANDBOX_API_KEY", "SANDBOX_PREFIX",
    "WANDB_API_KEY", "WANDB_MODE",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = ":".join((
    os.environ["AZURE_MODAL_PATH"], os.environ["MEGATRON_PATH"],
    os.environ["SLIME_DIR"], os.environ["SAFEVIBE_DIR"],
))
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
