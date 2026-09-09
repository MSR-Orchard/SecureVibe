#!/bin/bash
# SecureGen PatchEval + instance-specific teacher hints + OPD.
#
# Single-node 8-GPU colocated OPD + security reward run for Qwen3.5-35B-A3B.
# Student training + student rollout share the same 8 GPUs; the teacher SGLang
# server is launched separately (set TEACHER_IP / TEACHER_PORT in the shell).
#
# All the env-toggle / experiment-sweep machinery has been stripped and every
# CLI flag that this branch no longer supports has been removed. Adjust the
# hardcoded defaults inline if you need a different configuration.
#
# Prepare the data separately, then set PROMPT_DATA and EVAL_DATA to it.

set -euo pipefail

TEACHER_HINT_METADATA_KEY="${TEACHER_HINT_METADATA_KEY:-teacher_hint}"
if [[ ! "${TEACHER_HINT_METADATA_KEY}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid TEACHER_HINT_METADATA_KEY: ${TEACHER_HINT_METADATA_KEY}" >&2
    exit 2
fi
export TEACHER_HINT_METADATA_KEY

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT="${SLIME_DIR:?Set SLIME_DIR to the external Slime checkout}"
REPO_ROOT=$(cd -- "${REPO_ROOT}" && pwd)
python3 "${SCRIPT_DIR}/../../dependencies/orchard/check.py" "${REPO_ROOT}"
cd "${REPO_ROOT}"

ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true
rm -f /dev/shm/cuda.shm.* 2>/dev/null
rm -rf /tmp/ray/session_* 2>/dev/null

# Set WANDB_API_KEY / WANDB_BASE_URL in your shell before launching.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export ORCHARD_SANDBOX_ENDPOINT="${ORCHARD_SANDBOX_ENDPOINT:-${AZURE_CAAS_ENDPOINT:-}}"
export SANDBOX_API_KEY="${SANDBOX_API_KEY:-}"
export AUTOBAX_SRC_DIR="${AUTOBAX_SRC_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)/dependencies/autobax_arc}"
export AUTOBAX_HARNESS_URL="${AUTOBAX_HARNESS_URL:-}"
export AUTOBAX_HARNESS_SHA256="${AUTOBAX_HARNESS_SHA256:-}"
MEGATRON_PATH="${MEGATRON_PATH:-}"

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_NAME=Qwen3.5-35B-A3B
STUDENT_HF="${STUDENT_HF:-}"
STUDENT_TORCH_DIST="${STUDENT_TORCH_DIST:-}"
# The teacher and student use the same Qwen3.5 vocabulary. This path is used
# locally for tokenizer validation; the teacher weights live behind the server.
TEACHER_HF="${TEACHER_HF:-${STUDENT_HF}}"
PROMPT_DATA="${PROMPT_DATA:-}"
EVAL_DATA="${EVAL_DATA:-}"

for required_var in MEGATRON_PATH STUDENT_HF STUDENT_TORCH_DIST PROMPT_DATA EVAL_DATA; do
    if [[ -z "${!required_var}" ]]; then
        echo "${required_var} must be set." >&2
        exit 1
    fi
done

STUDENT_NAME=$(basename "${STUDENT_HF}")
TEACHER_NAME=$(basename "${TEACHER_HF}")

SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/outputs/security-coding-agent-opd}"
# Megatron training loads the distributed SFT checkpoint. STUDENT_HF supplies
# the matching Hugging Face config/tokenizer and conversion fallback.
LOAD_DIR="${LOAD_DIR:-${STUDENT_TORCH_DIST}}"
RESUME_OPD="${RESUME_OPD:-0}"
START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"

if [[ "${RESUME_OPD}" == "1" ]]; then
    # A true OPD continuation loads model, optimizer, scheduler, and RNG state.
    # Leave start_rollout_id unset so slime derives checkpoint iteration + 1.
    LOAD_DIR="${RESUME_LOAD_DIR:-${SAVE_DIR}}"
    if [[ ! -f "${LOAD_DIR}/latest_checkpointed_iteration.txt" ]]; then
        echo "Resume checkpoint tracker not found: ${LOAD_DIR}/latest_checkpointed_iteration.txt" >&2
        exit 1
    fi
    LOAD_STATE_ARGS=()
else
    LOAD_STATE_ARGS=(--finetune --no-load-optim --no-load-rng)
fi

START_ROLLOUT_ARGS=()
if [[ -n "${START_ROLLOUT_ID}" ]]; then
    START_ROLLOUT_ARGS=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

export PATCHEVAL_CONFIG_PATH="${PATCHEVAL_CONFIG_PATH:-${SCRIPT_DIR}/patcheval.yaml}"
export AUTOBAX_CONFIG_PATH="${AUTOBAX_CONFIG_PATH:-${SCRIPT_DIR}/autobax.yaml}"
# Kept for compatibility with helpers that expect one default config.
export SWE_CONFIG_PATH="${SWE_CONFIG_PATH:-${PATCHEVAL_CONFIG_PATH}}"

# ── Teacher server ─────────────────────────────────────────────────────────
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-8300}"

if [[ -z "${ORCHARD_SANDBOX_ENDPOINT}" || -z "${SANDBOX_API_KEY}" ]]; then
    echo "ORCHARD_SANDBOX_ENDPOINT and SANDBOX_API_KEY must be non-empty." >&2
    exit 1
fi
if [[ -n "${AUTOBAX_HARNESS_URL}" && ! "${AUTOBAX_HARNESS_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "AUTOBAX_HARNESS_SHA256 must be a 64-character hexadecimal digest when AUTOBAX_HARNESS_URL is set." >&2
    exit 1
fi
# ── SWE / reward runtime settings ──────────────────────────────────────────
export SWE_MAX_ALL_TOKENS=65536
export FLASHINFER_USE_CUDA_NORM=1

export SWE_REWARD_MODE="simple"
export SWE_UNRESOLVED_REWARD=0.0
# Keep these defaults overridable so a continuation can be tuned without
# editing the launcher.  Two environment attempts absorb a transient deleted
# container or escaped command timeout before an atomic eight-sibling group is
# discarded. The 45-second observation cap remains unchanged to avoid turning
# ordinary slow commands into longer rollout tails.
export SWE_TIMEOUT_CREATE_ENV="${SWE_TIMEOUT_CREATE_ENV:-480}"
export SWE_TIMEOUT_LLM_INFERENCE="${SWE_TIMEOUT_LLM_INFERENCE:-120}"
export SWE_TIMEOUT_GET_OBSERVATION="${SWE_TIMEOUT_GET_OBSERVATION:-45}"
export SWE_TIMEOUT_STOP_ENV="${SWE_TIMEOUT_STOP_ENV:-60}"
export SWE_TIMEOUT_REWARD_EXECUTE="${SWE_TIMEOUT_REWARD_EXECUTE:-600}"
export SWE_TIMEOUT_REWARD_TOTAL="${SWE_TIMEOUT_REWARD_TOTAL:-600}"
export SWE_MAX_ENV_RETRIES="${SWE_MAX_ENV_RETRIES:-2}"
export SWE_ENV_RETRY_WAIT="${SWE_ENV_RETRY_WAIT:-15}"
export SWE_MAX_CREATE_ENV_RETRIES="${SWE_MAX_CREATE_ENV_RETRIES:-2}"
export SWE_CREATE_ENV_RETRY_WAIT="${SWE_CREATE_ENV_RETRY_WAIT:-15}"
export SWE_ENV_CREATE_JITTER_MAX="${SWE_ENV_CREATE_JITTER_MAX:-5}"

# ── Batch-size config ──────────────────────────────────────────────────────
NUM_NODES=1
ROLL_BS="${ROLL_BS:-4}"
NPROMPT_PER_INSTANCE="${NPROMPT_PER_INSTANCE:-8}"
TRAIN_BS="${TRAIN_BS:-32}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-1}"
DYNAMIC_SAMPLING_ABORTED_BACKOFF_SECONDS="${DYNAMIC_SAMPLING_ABORTED_BACKOFF_SECONDS:-30}"
# Keep retrying through transient sandbox outages. A value of 0 disables the
# hard circuit breaker; the exponential backoff above still prevents a hot loop.
DYNAMIC_SAMPLING_ABORTED_MAX_CONSECUTIVE="${DYNAMIC_SAMPLING_ABORTED_MAX_CONSECUTIVE:-0}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-1024}"
NUM_ROLLOUT="${NUM_ROLLOUT:-300}"

# ── OPD training mode ──────────────────────────────────────────────────────
# OPD_ONLY=1 -> OPD only: zero out the GRPO task-reward advantage so only the OPD
#              KL distillation signal trains the student (--zero-train-adv-for-opd).
# OPD_ONLY=0 -> OPD + GRPO: keep the SWE task-reward advantage AND the OPD KL
#              signal. Override via env var, e.g. OPD_ONLY=1 bash <script>.
OPD_ONLY="${OPD_ONLY:-0}"
if [[ "${OPD_ONLY}" == "1" ]]; then
    zero_adv_flag="--zero-train-adv-for-opd"
else
    zero_adv_flag=""
fi
# OPD KL coefficient (--opd-kl-coef). Override via env var, e.g. OPD_KL_COEF=0.5.
OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
# OPD loss surrogate (--opd-reverse-kl-loss-type). One of: k1, k2, k3, js_skew.
# Override via env var, e.g. OPD_LOSS_TYPE=k3 bash <script>.
OPD_LOSS_TYPE="${OPD_LOSS_TYPE:-js_skew}"
# Teacher mixture weight lambda for the js_skew loss (--opd-js-mixture-weight),
# in (0, 1]. Only used when OPD_LOSS_TYPE=js_skew. Override via env var.
OPD_JS_MIXTURE_WEIGHT="${OPD_JS_MIXTURE_WEIGHT:-0.5}"
# ── OPD config (teacher endpoint + cross-vocab alignment) ──────────────────
# Ray jobs outlive the submitting launcher shell.  Keep the runtime OPD config in
# the experiment save directory instead of a launcher-owned temporary file; a
# shell exit must not silently disable teacher scoring in the still-running job.
mkdir -p "${SAVE_DIR}"
RUNTIME_OPD_CONFIG="${RUNTIME_OPD_CONFIG:-${SAVE_DIR}/runtime-opd-config.yaml}"
cat > "${RUNTIME_OPD_CONFIG}" <<EOF
teacher:
  url: "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
  tokenizer_path: "${TEACHER_HF}"
  hint_metadata_key: "${TEACHER_HINT_METADATA_KEY}"
  hint_template: |
    <teacher_instructions>
    You are a helpful assistant that can interact with a computer shell to solve programming tasks.

    Treat every case-specific security hint as a required observable behavior. Use this workflow:
    1. Extract the functional, benign-compatibility, and security requirements before editing.
    2. Inspect the relevant implementation, existing tests, and repository history first.
    3. Make the smallest targeted source edit; never replace an entire existing source file.
    4. Test through the real public API and real classes, including exact runtime defaults, values,
       names, error text/types, status codes, mutations, and platform-specific behavior named by the task.
    5. Exercise at least one benign case and one adversarial case, then run relevant existing tests.
    6. Before submission, inspect the final diff and complete an evidence-backed audit labeled
       FUNCTIONAL, SECURITY, REGRESSION, and DIFF. Fix any failed item before submitting.
    7. After two timeouts from equivalent commands, stop repeating them and use a different check.

    Case-specific security hint:
    {hint}
    </teacher_instructions>

timeout: 600
teacher_truncation_side: prefix
teacher_max_len: 75000
teacher_topk: 1
EOF

for required_path in "${MEGATRON_PATH}" "${STUDENT_HF}" "${STUDENT_TORCH_DIST}" "${PROMPT_DATA}" "${EVAL_DATA}" "${PATCHEVAL_CONFIG_PATH}" "${AUTOBAX_CONFIG_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done
if ! curl -fsS --connect-timeout 5 "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; then
    echo "Teacher is not reachable at http://${TEACHER_IP}:${TEACHER_PORT}" >&2
    exit 1
fi

# ── Argument groups ────────────────────────────────────────────────────────
WANDB_ARGS=(
   --use-wandb
   --wandb-project "${WANDB_PROJECT:-safevibe-slime-opd}"
   --wandb-group "${WANDB_GROUP:-${MODEL_NAME}-security-coding-agent-opd}"
   --disable-wandb-random-suffix
)

EVAL_ARGS=(
   --eval-interval 5
   --eval-prompt-data func_val "${EVAL_DATA}"
   --n-samples-per-eval-prompt 4
   --log-passrate
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-ep-size 2
   --sglang-moe-runner-backend triton
   --sglang-attention-backend trtllm_mha
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-mem-fraction-static 0.7
   --sglang-server-concurrency 256
   --sglang-max-running-requests 256
   --sglang-chunked-prefill-size 4096
   --sglang-data-parallel-size 2
   --sglang-enable-dp-attention
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 ${TRAIN_BS})
)

ROLLOUT_ARGS=(
   --rollout-shuffle
   --over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE}
   --dynamic-sampling-aborted-backoff-seconds ${DYNAMIC_SAMPLING_ABORTED_BACKOFF_SECONDS}
   --dynamic-sampling-aborted-max-consecutive ${DYNAMIC_SAMPLING_ABORTED_MAX_CONSECUTIVE}
   --prompt-data "${PROMPT_DATA}"
   --multimodal-keys '{}'
   --input-key prompt
   --label-key label
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLL_BS}
   --n-samples-per-prompt ${NPROMPT_PER_INSTANCE}
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --rollout-top-p 1.0
)

OPT_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-precision-aware-optimizer
   --calculate-per-token-loss
)

# Slime's eval-only path uses --num-rollout 0, but Megatron still constructs an
# optimizer scheduler while restoring the actor checkpoint. Without an explicit
# positive horizon, num_rollout=0 derives lr_decay_steps=0 and initialization
# aborts before evaluation. The scheduler is never stepped in eval-only mode.
if [[ "${NUM_ROLLOUT}" == "0" ]]; then
   # This must match the scheduler horizon stored by the training run. The
   # current checkpoint was trained with NUM_ROLLOUT=300 (9600 samples at
   # global batch 32), so use 300 unless explicitly overridden for another
   # checkpoint lineage.
   EVAL_ONLY_LR_DECAY_ITERS="${EVAL_ONLY_LR_DECAY_ITERS:-300}"
   OPT_ARGS+=(--lr-decay-iters "${EVAL_ONLY_LR_DECAY_ITERS}")
fi

REWARD_ARGS=(
   --custom-generate-function-path slime_opd.generate.generate
   --custom-rm-path slime_opd.combined_reward.reward_func
   --custom-reward-post-process-path slime_opd.combined_reward.post_process_rewards
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
   --opd-config "${RUNTIME_OPD_CONFIG}"
)

# ── Model arch ─────────────────────────────────────────────────────────────
source "scripts/models/${MODEL_NAME/Q/q}.sh"

# ── Ray (single node) ──────────────────────────────────────────────────────
export MASTER_ADDR="$(hostname -I | awk '{print $1}')"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${TEACHER_IP}"
export NO_PROXY="${no_proxy}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l) || NVLINK_COUNT=0
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

RAY_OBJECT_STORE_MEMORY=$(( 32 * 1024 * 1024 * 1024 ))
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --object-store-memory ${RAY_OBJECT_STORE_MEMORY} \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

sleep 5
mkdir -p "${SAVE_DIR}"
cp "$0" "${SAVE_DIR}/run_config.sh"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:${MEGATRON_PATH}:${TRAINING_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"FLASHINFER_USE_CUDA_NORM\": \"${FLASHINFER_USE_CUDA_NORM}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"${no_proxy}\",
    \"WANDB_API_KEY\": \"${WANDB_API_KEY}\",
    \"WANDB_BASE_URL\": \"${WANDB_BASE_URL}\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"WANDB_ENTITY\": \"${WANDB_ENTITY}\",
    \"ORCHARD_SANDBOX_ENDPOINT\": \"${ORCHARD_SANDBOX_ENDPOINT}\",
    \"SANDBOX_API_KEY\": \"${SANDBOX_API_KEY}\",
    \"AUTOBAX_SRC_DIR\": \"${AUTOBAX_SRC_DIR}\",
    \"AUTOBAX_HARNESS_URL\": \"${AUTOBAX_HARNESS_URL}\",
    \"AUTOBAX_HARNESS_SHA256\": \"${AUTOBAX_HARNESS_SHA256}\",
    \"SWE_REWARD_MODE\": \"${SWE_REWARD_MODE}\",
    \"SWE_CONFIG_PATH\": \"${SWE_CONFIG_PATH}\",
    \"PATCHEVAL_CONFIG_PATH\": \"${PATCHEVAL_CONFIG_PATH}\",
    \"AUTOBAX_CONFIG_PATH\": \"${AUTOBAX_CONFIG_PATH}\",
    \"SWE_TIMEOUT_CREATE_ENV\": \"${SWE_TIMEOUT_CREATE_ENV}\",
    \"SWE_TIMEOUT_LLM_INFERENCE\": \"${SWE_TIMEOUT_LLM_INFERENCE}\",
    \"SWE_TIMEOUT_GET_OBSERVATION\": \"${SWE_TIMEOUT_GET_OBSERVATION}\",
    \"SWE_TIMEOUT_STOP_ENV\": \"${SWE_TIMEOUT_STOP_ENV}\",
    \"SWE_TIMEOUT_REWARD_EXECUTE\": \"${SWE_TIMEOUT_REWARD_EXECUTE}\",
    \"SWE_TIMEOUT_REWARD_TOTAL\": \"${SWE_TIMEOUT_REWARD_TOTAL}\",
    \"SWE_MAX_ENV_RETRIES\": \"${SWE_MAX_ENV_RETRIES}\",
    \"SWE_ENV_RETRY_WAIT\": \"${SWE_ENV_RETRY_WAIT}\",
    \"SWE_MAX_CREATE_ENV_RETRIES\": \"${SWE_MAX_CREATE_ENV_RETRIES}\",
    \"SWE_CREATE_ENV_RETRY_WAIT\": \"${SWE_CREATE_ENV_RETRY_WAIT}\",
    \"SWE_ENV_CREATE_JITTER_MAX\": \"${SWE_ENV_CREATE_JITTER_MAX}\"
  }
}"

# ── Train (8 GPUs colocated: train + rollout share GPUs) ───────────────────
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    --actor-num-nodes ${NUM_NODES} \
    --actor-num-gpus-per-node 8 \
    --colocate \
    --megatron-config-path "${SCRIPT_DIR}/megatron_actor.yaml" \
    --distributed-timeout-minutes 60 \
    \
    "${MODEL_ARGS[@]}" \
    \
    --hf-checkpoint "${STUDENT_HF}" \
    --ref-load    "${STUDENT_TORCH_DIST}" \
    --load        "${LOAD_DIR}" \
    "${LOAD_STATE_ARGS[@]}" \
    --save        "${SAVE_DIR}" \
    --save-interval 5 \
    \
    --global-batch-size ${TRAIN_BS} \
    "${START_ROLLOUT_ARGS[@]}" \
    --update-weight-buffer-size $(( 128 * 1024 * 1024 )) \
    --balance-data \
    "${ROLLOUT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${OPT_ARGS[@]}" \
    "${REWARD_ARGS[@]}" \
    \
    --advantage-estimator grpo \
    --use-opd \
    --opd-type sglang \
    --opd-mode loss \
    --opd-kl-coef ${OPD_KL_COEF} \
    ${zero_adv_flag} \
    --opd-reverse-kl-loss-type ${OPD_LOSS_TYPE} \
    --opd-js-mixture-weight ${OPD_JS_MIXTURE_WEIGHT} \
    --use-kl-loss \
    --kl-loss-coef 0.0 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 4 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu 16384 \
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" \
    \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --moe-token-dispatcher-type flex \
    --moe-enable-deepep \
    --attention-backend flash

# ── Cleanup ────────────────────────────────────────────────────────────────
ray stop --force || true
pkill -9 ray    || true
