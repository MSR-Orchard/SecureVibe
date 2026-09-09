#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
RUN_SCRIPT="${RUN_SCRIPT:-${SCRIPT_DIR}/train.sh}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/../dataset/train_recipe/SFT/0713_sft}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the output checkpoint directory}"
RUN_SUFFIX="${RUN_SUFFIX:-0708}"
WANDB_NAME_SUFFIX="${WANDB_NAME_SUFFIX:-${RUN_SUFFIX}}"

runs=(
  "func:${DATA_ROOT}/func.sft.jsonl:${CHECKPOINT_ROOT}/func-qwen35-sft-slime-${RUN_SUFFIX}:func-${WANDB_NAME_SUFFIX}-qwen35-35b"
  "secu:${DATA_ROOT}/secu.sft.jsonl:${CHECKPOINT_ROOT}/secu-qwen35-sft-slime-${RUN_SUFFIX}:secu-${WANDB_NAME_SUFFIX}-qwen35-35b"
  "plan:${DATA_ROOT}/plan.sft.jsonl:${CHECKPOINT_ROOT}/plan-qwen35-sft-slime-${RUN_SUFFIX}:plan-${WANDB_NAME_SUFFIX}-qwen35-35b"
)

if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Run script not found: ${RUN_SCRIPT}" >&2
  exit 1
fi

echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Using launcher: ${RUN_SCRIPT}"
echo

for run in "${runs[@]}"; do
  IFS=: read -r name data output_dir wandb_name <<<"${run}"

  echo "=========================================="
  echo "Starting ${name}"
  echo "DATA=${data}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "WANDB_NAME=${wandb_name}"
  echo "=========================================="

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    DATA="${data}" \
    OUTPUT_DIR="${output_dir}" \
    WANDB_NAME="${wandb_name}" \
    bash "${RUN_SCRIPT}"

  echo "Finished ${name}"
  echo
done

echo "All runs finished."
