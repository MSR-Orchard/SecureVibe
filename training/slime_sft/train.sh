#!/usr/bin/env bash
set -euo pipefail

date=$(date +%Y-%m-%d)
echo "Current date and time: ${date}"

LAUNCH_DIR="$(pwd)"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SECUREVIBE_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"
MODEL_CONFIG="${SCRIPT_DIR}/qwen3.5-35B-A3B.sh"
TRAIN_ENTRYPOINT="${SCRIPT_DIR}/train_async.py"

SLIME_ROOT=${SLIME_ROOT:-${SECUREVIBE_DIR}/../slime}
MEGATRON_PATH=${MEGATRON_PATH:-${SECUREVIBE_DIR}/dependencies/megatron-lm/Megatron-LM}
RUNTIME_PYTHONPATH="${SLIME_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

POLICY_MODEL=Qwen3.5-35B-A3B-Base
MODEL=${MODEL:?Set MODEL to the Qwen3.5-35B-A3B Hugging Face checkpoint}
DATA=${DATA:?Set DATA to an SFT JSONL file}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR to the output checkpoint directory}
SAVE_HF=${SAVE_HF:-${OUTPUT_DIR%/}-hf/rollout_{rollout_id}}
HF_CHECKPOINT=${HF_CHECKPOINT:-${MODEL}}
REF_LOAD=${REF_LOAD:-${MODEL%/}_torch_dist_slime-0.3.0}
SCHEMES=("${SCHEME:-$(basename "${DATA}" .jsonl)}")

REMOVE_REASONING_FLAG=false

NUM_EPOCH=${NUM_EPOCH:-3}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
TP_SIZE=${TP_SIZE:-2}
CP_SIZE=${CP_SIZE:-2}
EP_SIZE=${EP_SIZE:-4}
PP_SIZE=${PP_SIZE:-1}
RUN_VARIANT=${RUN_VARIANT:-${ACTOR_GPUS_PER_NODE}gpu-tp${TP_SIZE}-cp${CP_SIZE}-ep${EP_SIZE}-pp${PP_SIZE}}

WANDB_PROJECT=${WANDB_PROJECT:-securevibe-qwen35-slime}
WANDB_NAME=${WANDB_NAME:-qwen35-35b-a3b-sft-${ACTOR_GPUS_PER_NODE}gpu}
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_MODE=${WANDB_MODE:-online}

if [[ "${DATA}" = /* ]]; then
   DEFAULT_DATA_FILE="${DATA}"
elif [ -f "${SCRIPT_DIR}/${DATA}" ]; then
   DEFAULT_DATA_FILE="${SCRIPT_DIR}/${DATA}"
elif [ -f "${LAUNCH_DIR}/${DATA}" ]; then
   DEFAULT_DATA_FILE="${LAUNCH_DIR}/${DATA}"
else
   DEFAULT_DATA_FILE="${SCRIPT_DIR}/${DATA}"
fi
DATA_DIR="$(dirname "${DEFAULT_DATA_FILE}")"
SAVE_DIR="${OUTPUT_DIR}"

for REQUIRED_CMD in python3 ray nvidia-smi wc; do
   if ! command -v "${REQUIRED_CMD}" >/dev/null 2>&1; then
      echo "Required command not found: ${REQUIRED_CMD}"
      exit 1
   fi
done

if [ ! -f "${MODEL_CONFIG}" ]; then
   echo "Model config not found: ${MODEL_CONFIG}"
   exit 1
fi

if [ ! -f "${TRAIN_ENTRYPOINT}" ]; then
   echo "Training entrypoint not found: ${TRAIN_ENTRYPOINT}"
   exit 1
fi

AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${AVAILABLE_GPUS}" -lt "${ACTOR_GPUS_PER_NODE}" ]; then
   echo "Need ${ACTOR_GPUS_PER_NODE} GPUs on this node, but nvidia-smi reports ${AVAILABLE_GPUS}."
   exit 1
fi

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
   echo "WANDB_MODE=online requires WANDB_API_KEY. Set WANDB_MODE=offline to log locally." >&2
   exit 1
fi

for SCHEME in "${SCHEMES[@]}"; do
   echo "=========================================="
   echo "Training scheme: ${SCHEME}"
   echo "=========================================="

   DATA_FILE="${DEFAULT_DATA_FILE}"
   if [ ! -f "${DATA_FILE}" ]; then
      echo "Data file not found: ${DATA_FILE}"
      exit 1
   fi
   NUM_SAMPLES=$(wc -l < "${DATA_FILE}")
   ROLLOUTS_PER_EPOCH=$(( (NUM_SAMPLES + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))
   NUM_ROLLOUT=$(( (NUM_SAMPLES * NUM_EPOCH + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))
   SAVE_INTERVAL=${SAVE_INTERVAL:-${ROLLOUTS_PER_EPOCH}}

   if [ "${NUM_ROLLOUT}" -le 0 ] || [ "${SAVE_INTERVAL}" -le 0 ]; then
      echo "Invalid rollout configuration: num_rollout=${NUM_ROLLOUT}, save_interval=${SAVE_INTERVAL} for ${DATA_FILE}."
      exit 1
   fi

   echo "Samples: ${NUM_SAMPLES}; rollouts per epoch: ${ROLLOUTS_PER_EPOCH}; total rollouts: ${NUM_ROLLOUT}; save interval: ${SAVE_INTERVAL}"

   # Stop this node's Ray runtime before starting a fresh local Ray head.
   ray stop --force || true
   sleep 3

   export PYTHONUNBUFFERED=1
   export WANDB_PROJECT
   export WANDB_NAME
   export WANDB_ENTITY
   export WANDB_MODE

   NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
   if [ "$NVLINK_COUNT" -gt 0 ]; then
       HAS_NVLINK=1
   else
       HAS_NVLINK=0
   fi

   source "${MODEL_CONFIG}"

   CKPT_ARGS=(
      --hf-checkpoint "${HF_CHECKPOINT}"
      --ref-load "${REF_LOAD}"
      --load "${SAVE_DIR}"
      --save "${SAVE_DIR}"
      --save-interval ${SAVE_INTERVAL}
      --save-hf "${SAVE_HF}"
   )

   SFT_ARGS=(
      --rollout-function-path slime.rollout.sft_rollout.generate_rollout
      --prompt-data "${DATA_FILE}"
      --input-key messages
      --tool-key tools
      --rollout-max-prompt-len 65536
      --rollout-shuffle
      --num-rollout ${NUM_ROLLOUT}
      --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
      --global-batch-size ${ROLLOUT_BATCH_SIZE}

      --loss-type sft_loss
      --loss-mask-type qwen3_5
      --calculate-per-token-loss
      --disable-compute-advantages-and-returns
      --debug-train-only
      $([ "$REMOVE_REASONING_FLAG" = true ] && echo "--remove-reasoning-content" || true)
   )

   PERF_ARGS=(
      --tensor-model-parallel-size ${TP_SIZE}
      --sequence-parallel
      --pipeline-model-parallel-size ${PP_SIZE}
      --context-parallel-size ${CP_SIZE}
      --expert-model-parallel-size ${EP_SIZE}
      --expert-tensor-parallel-size 1

      --recompute-granularity full
      --recompute-method uniform
      --recompute-num-layers 1

      --use-dynamic-batch-size
      --max-tokens-per-gpu 8192
   )

   OPTIMIZER_ARGS=(
      --optimizer adam
      --lr 1e-5
      --lr-decay-style cosine
      --min-lr 1e-6
      --lr-warmup-fraction 0.1
      --weight-decay 0.1
      --adam-beta1 0.9
      --adam-beta2 0.95

      --use-distributed-optimizer
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )

   WANDB_ARGS=()
   if [[ "${WANDB_MODE}" != "disabled" ]]; then
      WANDB_ARGS+=(
         --use-wandb
         --wandb-project "${WANDB_PROJECT}"
         --wandb-group "${WANDB_NAME}"
         --wandb-mode "${WANDB_MODE}"
      )
      if [[ -n "${WANDB_ENTITY}" ]]; then
         WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
      fi
      if [[ -n "${WANDB_API_KEY:-}" ]]; then
         WANDB_ARGS+=(--wandb-key "${WANDB_API_KEY}")
      fi
      if [[ -n "${WANDB_HOST:-}" ]]; then
         WANDB_ARGS+=(--wandb-host "${WANDB_HOST}")
      fi
      if [[ -n "${WANDB_DIR:-}" ]]; then
         WANDB_ARGS+=(--wandb-dir "${WANDB_DIR}")
      fi
   fi

   MISC_ARGS=(
      --attention-dropout 0.0
      --hidden-dropout 0.0
      --accumulate-allreduce-grads-in-fp32
      --attention-softmax-in-fp32
      --attention-backend flash

      # --moe-token-dispatcher-type flex
      # --moe-enable-deepep
      --moe-token-dispatcher-type flex
      --moe-flex-dispatcher-backend deepep
   )

   export MASTER_ADDR=${MASTER_ADDR:-${VC_MASTER_HOSTS:-"127.0.0.1"}}
   export no_proxy="127.0.0.1,${MASTER_ADDR}"
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${ACTOR_GPUS_PER_NODE} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

   if [ "${ACTOR_NUM_NODES}" -gt 1 ]; then
      if ! command -v ssh >/dev/null 2>&1; then
         echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES}, but ssh is not available."
         exit 1
      fi
      if [[ -z "${HOSTFILE:-}" || ! -f "${HOSTFILE}" ]]; then
         echo "ACTOR_NUM_NODES=${ACTOR_NUM_NODES} requires HOSTFILE to name a readable host file."
         exit 1
      fi
      # Start Ray workers on other nodes
      for WORKER_IP in $(awk '{print $1}' "${HOSTFILE}"); do
        if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
          continue
        fi
        echo "Starting Ray worker on ${WORKER_IP}"
        ssh "${SSH_USER:-root}@${WORKER_IP}" \
          "ray stop --force ; ray start --address=${MASTER_ADDR}:6379 --num-gpus ${ACTOR_GPUS_PER_NODE} --node-ip-address ${WORKER_IP} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265" &
      done
      wait
   fi

   RUNTIME_ENV_JSON="{
     \"working_dir\": \"${SCRIPT_DIR}\",
     \"env_vars\": {
       \"PYTHONPATH\": \"${RUNTIME_PYTHONPATH}\",
       \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
       \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
       \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
       \"no_proxy\": \"${no_proxy}\",
       \"MASTER_ADDR\": \"${MASTER_ADDR}\",
       \"PYTHONUNBUFFERED\": \"${PYTHONUNBUFFERED}\",
       \"WANDB_API_KEY\": \"${WANDB_API_KEY:-}\",
       \"WANDB_PROJECT\": \"${WANDB_PROJECT}\",
       \"WANDB_NAME\": \"${WANDB_NAME}\",
       \"WANDB_ENTITY\": \"${WANDB_ENTITY}\",
       \"WANDB_MODE\": \"${WANDB_MODE}\"
     }
   }"

   ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- python3 train_async.py \
      --actor-num-nodes ${ACTOR_NUM_NODES} \
      --actor-num-gpus-per-node ${ACTOR_GPUS_PER_NODE} \
      ${MODEL_ARGS[@]} \
      ${CKPT_ARGS[@]} \
      ${SFT_ARGS[@]} \
      ${OPTIMIZER_ARGS[@]} \
      ${WANDB_ARGS[@]} \
      ${PERF_ARGS[@]} \
      ${MISC_ARGS[@]}

   echo "Finished training: ${SCHEME}"
done

echo "Training completed."
