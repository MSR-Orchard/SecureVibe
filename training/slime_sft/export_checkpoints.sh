#!/usr/bin/env bash
set -euo pipefail

# Convert the Megatron torch_dist checkpoints produced by
# run_qwen35_sft_4gpu_sequence.sh back to Hugging Face safetensors.
#
# Common usage:
#   RUN_SUFFIX=0707 bash slime_sft/export_checkpoints.sh
#   SCHEMES=func bash slime_sft/export_checkpoints.sh
#   CHECKPOINT_DIR=/path/to/func-qwen35-sft-slime-0707 \
#     bash slime_sft/export_checkpoints.sh
#
# Useful overrides:
#   ORIGIN_HF_DIR=/path/to/Qwen3.5-35B-A3B-Base
#   GENERATION_CONFIG_SOURCE=/path/to/generation_config.json
#   MODEL_NAME=qwen3_5_moe                # bypass HF config auto-detection
#   PYTHON=/path/to/python                  # default: python3
#   SGLANG_PYTHONPATH=/path/to/sglang/python  # if sglang is not importable
#   OUTPUT_DIR=/path/to/hf-out              # only with CHECKPOINT_DIR
#   SAVE_HF_ROOT=/path/to/hf-root           # only with CHECKPOINT_DIR
#   ITERATION=123                           # otherwise latest tracker/newest iter_* is used
#   CONVERTER=basic                         # default: parallel
#   LOAD_MAX_WORKERS=2 SAVE_MAX_WORKERS=16  # parallel converter knobs
#   FORCE=1                                 # allow an existing output dir
#   PROGRESS=0                              # disable output-size progress monitor
#   PROGRESS_INTERVAL=30                    # seconds between progress updates
#   DRY_RUN=1                               # print commands only

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SECUREVIBE_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

SLIME_ROOT="${SLIME_ROOT:-${SECUREVIBE_DIR}/../slime}"
MEGATRON_PATH="${MEGATRON_PATH:-${SECUREVIBE_DIR}/dependencies/megatron-lm/Megatron-LM}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-}"
PYTHON="${PYTHON:-python3}"

ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-${MODEL:-}}"
: "${ORIGIN_HF_DIR:?Set ORIGIN_HF_DIR or MODEL to the original Hugging Face checkpoint}"
GENERATION_CONFIG_SOURCE="${GENERATION_CONFIG_SOURCE:-${ORIGIN_HF_DIR%/}/generation_config.json}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
RUN_SUFFIX="${RUN_SUFFIX:-0707}"
SCHEMES="${SCHEMES:-func,secu,plan}"

CONVERTER="${CONVERTER:-parallel}"
LOAD_MAX_WORKERS="${LOAD_MAX_WORKERS:-2}"
SAVE_MAX_WORKERS="${SAVE_MAX_WORKERS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
VOCAB_SIZE="${VOCAB_SIZE:-248320}"
MODEL_NAME="${MODEL_NAME:-}"
FORCE="${FORCE:-0}"
PROGRESS="${PROGRESS:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
EXPECTED_OUTPUT_BYTES="${EXPECTED_OUTPUT_BYTES:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "${label} does not exist: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "${label} does not exist: ${path}" >&2
    exit 1
  fi
}

trim_space() {
  tr -d '[:space:]'
}

strip_iter_padding() {
  local iter="$1"
  iter="${iter#iter_}"
  if [[ "${iter}" =~ ^[0-9]+$ ]]; then
    printf '%d\n' "$((10#${iter}))"
  else
    printf '%s\n' "${iter}"
  fi
}

iter_dir_name() {
  local iter="$1"
  if [[ "${iter}" == "release" ]]; then
    printf 'release\n'
  elif [[ "${iter}" =~ ^iter_[0-9]+$ ]]; then
    printf '%s\n' "${iter}"
  elif [[ "${iter}" =~ ^[0-9]+$ ]]; then
    printf 'iter_%07d\n' "${iter}"
  else
    echo "Invalid ITERATION=${iter}; expected a number, iter_XXXXXXX, or release." >&2
    exit 1
  fi
}

resolve_input_dir() {
  local checkpoint_dir="$1"

  if [[ -n "${INPUT_DIR:-}" ]]; then
    printf '%s\n' "${INPUT_DIR}"
    return 0
  fi

  if [[ -n "${ITERATION:-}" ]]; then
    printf '%s/%s\n' "${checkpoint_dir%/}" "$(iter_dir_name "${ITERATION}")"
    return 0
  fi

  local tracker="${checkpoint_dir%/}/latest_checkpointed_iteration.txt"
  if [[ -f "${tracker}" ]]; then
    local iter
    iter="$(trim_space < "${tracker}")"
    printf '%s/%s\n' "${checkpoint_dir%/}" "$(iter_dir_name "${iter}")"
    return 0
  fi

  local newest
  newest="$(find "${checkpoint_dir}" -maxdepth 1 -type d -name 'iter_*' -printf '%f\n' 2>/dev/null | sort | tail -n 1)"
  if [[ -n "${newest}" ]]; then
    printf '%s/%s\n' "${checkpoint_dir%/}" "${newest}"
    return 0
  fi

  echo "Could not find latest_checkpointed_iteration.txt or any iter_* dirs under ${checkpoint_dir}" >&2
  exit 1
}

step_for_output() {
  local input_dir="$1"
  local base
  base="$(basename "${input_dir}")"
  if [[ "${base}" == "release" ]]; then
    printf 'release\n'
  else
    strip_iter_padding "${base}"
  fi
}

default_output_dir() {
  local checkpoint_dir="$1"
  local input_dir="$2"
  local step
  step="$(step_for_output "${input_dir}")"

  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    printf '%s\n' "${OUTPUT_DIR}"
  elif [[ -n "${SAVE_HF_ROOT:-}" ]]; then
    printf '%s/rollout_%s\n' "${SAVE_HF_ROOT%/}" "${step}"
  else
    printf '%s-hf/rollout_%s\n' "${checkpoint_dir%/}" "${step}"
  fi
}

append_optional_arg() {
  local -n arr_ref="$1"
  local flag="$2"
  local value="$3"
  if [[ -n "${value}" ]]; then
    arr_ref+=("${flag}" "${value}")
  fi
}

runtime_pythonpath() {
  local parts=()
  if [[ -d "${SGLANG_PYTHONPATH}" ]]; then
    parts+=("${SGLANG_PYTHONPATH}")
  fi
  parts+=("${MEGATRON_PATH}" "${SLIME_ROOT}")
  if [[ -n "${PYTHONPATH:-}" ]]; then
    parts+=("${PYTHONPATH}")
  fi
  local IFS=:
  printf '%s\n' "${parts[*]}"
}

check_runtime_imports() {
  PYTHONPATH="$(runtime_pythonpath)" "${PYTHON}" - <<'PY'
import sys
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding
print(f"Python executable: {sys.executable}")
print("Runtime import check passed.")
PY
}

bytes_in_safetensors() {
  local dir="$1"
  find "${dir}" -maxdepth 1 -type f -name '*.safetensors' -printf '%s\n' 2>/dev/null \
    | awk '{sum += $1} END {printf "%.0f\n", sum}'
}

bytes_in_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    printf '0\n'
    return 0
  fi
  find "${dir}" -type f -printf '%s\n' 2>/dev/null \
    | awk '{sum += $1} END {printf "%.0f\n", sum}'
}

human_bytes() {
  local bytes="$1"
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec --suffix=B "${bytes}"
  else
    awk -v b="${bytes}" 'BEGIN {
      split("B KiB MiB GiB TiB", units)
      i = 1
      while (b >= 1024 && i < 5) { b /= 1024; i++ }
      printf "%.1f%s", b, units[i]
    }'
  fi
}

print_progress_bar() {
  local label="$1"
  local current="$2"
  local expected="$3"
  local width=30
  local pct=0
  local filled=0

  if [[ "${expected}" =~ ^[0-9]+$ && "${expected}" -gt 0 ]]; then
    pct=$(( current * 100 / expected ))
    (( pct > 100 )) && pct=100
    filled=$(( pct * width / 100 ))
  fi

  local bar=""
  for ((i = 0; i < width; i++)); do
    if (( i < filled )); then
      bar+="#"
    else
      bar+="."
    fi
  done

  printf '\r[%s] %3d%% %s / %s %s' \
    "${bar}" \
    "${pct}" \
    "$(human_bytes "${current}")" \
    "$(human_bytes "${expected}")" \
    "${label}"
}

monitor_progress() {
  local name="$1"
  local output_dir="$2"
  local expected_bytes="$3"
  local target_pid="$4"

  if [[ ! "${expected_bytes}" =~ ^[0-9]+$ || "${expected_bytes}" -le 0 ]]; then
    echo "Progress monitor disabled for ${name}: expected output size is unknown." >&2
    return 0
  fi

  echo "Progress monitor for ${name}: expecting about $(human_bytes "${expected_bytes}") of HF weights." >&2
  while kill -0 "${target_pid}" >/dev/null 2>&1; do
    print_progress_bar "${name}" "$(bytes_in_dir "${output_dir}")" "${expected_bytes}" >&2
    sleep "${PROGRESS_INTERVAL}"
  done
  print_progress_bar "${name}" "$(bytes_in_dir "${output_dir}")" "${expected_bytes}" >&2
  printf '\n' >&2
}

convert_one() {
  local name="$1"
  local checkpoint_dir="$2"

  require_dir "${checkpoint_dir}" "Checkpoint directory for ${name}"

  local input_dir output_dir converter_script
  input_dir="$(resolve_input_dir "${checkpoint_dir}")"
  output_dir="$(default_output_dir "${checkpoint_dir}" "${input_dir}")"

  require_dir "${input_dir}" "Input checkpoint directory for ${name}"
  require_file "${input_dir}/common.pt" "Megatron common.pt for ${name}"
  require_file "${input_dir}/.metadata" "Megatron torch_dist metadata for ${name}"

  case "${CONVERTER}" in
    parallel)
      converter_script="${SLIME_ROOT}/tools/convert_torch_dist_to_hf_parallel.py"
      ;;
    basic)
      converter_script="${SLIME_ROOT}/tools/convert_torch_dist_to_hf.py"
      ;;
    *)
      echo "Invalid CONVERTER=${CONVERTER}; expected parallel or basic." >&2
      exit 1
      ;;
  esac

  require_file "${converter_script}" "slime converter"

  echo "Checking converter imports with PYTHONPATH=$(runtime_pythonpath)"
  check_runtime_imports

  local expected_output_bytes
  if [[ -n "${EXPECTED_OUTPUT_BYTES}" ]]; then
    expected_output_bytes="${EXPECTED_OUTPUT_BYTES}"
  else
    expected_output_bytes="$(bytes_in_safetensors "${ORIGIN_HF_DIR}")"
  fi

  local cmd=(
    "${PYTHON}" "${converter_script}"
    --input-dir "${input_dir}"
    --output-dir "${output_dir}"
    --origin-hf-dir "${ORIGIN_HF_DIR}"
  )

  append_optional_arg cmd --model-name "${MODEL_NAME}"
  if [[ "${FORCE}" == "1" ]]; then
    cmd+=("--force")
  fi
  append_optional_arg cmd --chunk-size "${CHUNK_SIZE}"
  append_optional_arg cmd --vocab-size "${VOCAB_SIZE}"

  if [[ "${CONVERTER}" == "parallel" ]]; then
    cmd+=(--load-max-workers "${LOAD_MAX_WORKERS}")
    cmd+=(--save-max-workers "${SAVE_MAX_WORKERS}")
  fi

  echo "=========================================="
  echo "Converting ${name}"
  echo "CHECKPOINT_DIR=${checkpoint_dir}"
  echo "INPUT_DIR=${input_dir}"
  echo "OUTPUT_DIR=${output_dir}"
  echo "GENERATION_CONFIG_SOURCE=${GENERATION_CONFIG_SOURCE}"
  echo "CONVERTER=${CONVERTER}"
  echo "=========================================="
  printf 'Command:'
  printf ' %q' env "PYTHONPATH=$(runtime_pythonpath)" "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1; skipping ${name}"
    echo
    return 0
  fi

  (
    cd "${SLIME_ROOT}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    if [[ "${PROGRESS}" == "1" ]]; then
      PYTHONPATH="$(runtime_pythonpath)" "${cmd[@]}" &
      local convert_pid=$!
      monitor_progress "${name}" "${output_dir}" "${expected_output_bytes}" "${convert_pid}" &
      local monitor_pid=$!
      local convert_status=0
      if wait "${convert_pid}"; then
        convert_status=0
      else
        convert_status=$?
      fi
      kill "${monitor_pid}" >/dev/null 2>&1 || true
      wait "${monitor_pid}" >/dev/null 2>&1 || true
      print_progress_bar "${name}" "$(bytes_in_dir "${output_dir}")" "${expected_output_bytes}" >&2
      printf '\n' >&2
      return "${convert_status}"
    else
      PYTHONPATH="$(runtime_pythonpath)" "${cmd[@]}"
    fi
  )

  cp -f -- "${GENERATION_CONFIG_SOURCE}" "${output_dir}/generation_config.json"
  echo "Copied generation config: ${output_dir}/generation_config.json"
  echo "Finished ${name}: ${output_dir}"
  echo
}

require_dir "${SLIME_ROOT}" "SLIME_ROOT"
require_dir "${MEGATRON_PATH}" "MEGATRON_PATH"
require_dir "${ORIGIN_HF_DIR}" "ORIGIN_HF_DIR"
require_file "${GENERATION_CONFIG_SOURCE}" "generation_config.json source"

if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  convert_one "checkpoint" "${CHECKPOINT_DIR}"
else
  : "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT when CHECKPOINT_DIR is not provided}"
  for single_ckpt_var in INPUT_DIR OUTPUT_DIR SAVE_HF_ROOT; do
    if [[ -n "${!single_ckpt_var:-}" ]]; then
      echo "${single_ckpt_var} is only supported when CHECKPOINT_DIR is set." >&2
      exit 1
    fi
  done

  IFS=',' read -r -a scheme_list <<<"${SCHEMES}"
  for scheme in "${scheme_list[@]}"; do
    scheme="$(printf '%s' "${scheme}" | trim_space)"
    [[ -z "${scheme}" ]] && continue
    convert_one "${scheme}" "${CHECKPOINT_ROOT%/}/${scheme}-qwen35-sft-slime-${RUN_SUFFIX}"
  done
fi

echo "All requested conversions finished."
