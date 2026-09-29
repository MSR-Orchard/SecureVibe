#!/usr/bin/env bash
# Single-node joint PatchEval + AutoBax Qwen3.5-35B-A3B RL entrypoint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SECUREVIBE_DIR="${SECUREVIBE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SLIME_DIR="${SLIME_DIR:?Set SLIME_DIR to the patched Orchard trainer/slime checkout}"
ONE_NODE_LAUNCHER="${SCRIPT_DIR}/train_mini_swe.sh"

PATCHEVAL_CONFIG="${PATCHEVAL_CONFIG:-${SCRIPT_DIR}/configs/patcheval.yaml}"
AUTOBAX_CONFIG="${AUTOBAX_CONFIG:-${SCRIPT_DIR}/configs/autobax.yaml}"

export JOINT_PATCHEVAL_DATA="${JOINT_PATCHEVAL_DATA:-${SCRIPT_DIR}/../../data/recipes/rl_patcheval_train_generic.jsonl}"
export JOINT_AUTOBAX_DATA="${JOINT_AUTOBAX_DATA:-${SCRIPT_DIR}/../../data/recipes/rl_autobax_train_generic.jsonl}"
export JOINT_PATCHEVAL_WEIGHT="${JOINT_PATCHEVAL_WEIGHT:-0.5}"
export JOINT_AUTOBAX_WEIGHT="${JOINT_AUTOBAX_WEIGHT:-0.5}"
export CUSTOM_RM_PATH="slime_rl.joint_reward.reward_func"
export DATA_SOURCE_PATH="slime_rl.joint_data_source.JointRolloutDataSource"
PUBLIC_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export AUTOBAX_SRC_DIR="${AUTOBAX_SRC_DIR:-${PUBLIC_ROOT}/dependencies/autobax_arc}"
export EXP_TAG="${EXP_TAG:-joint_patcheval_autobax_1node}"
export WANDB_GROUP="${WANDB_GROUP:-qwen3.5-35b-a3b-joint-1node}"
export SWE_CONFIG_PATH="${SWE_CONFIG_PATH:-${PATCHEVAL_CONFIG}}"
export MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export N_SAMPLES_PER_PROMPT_MAX="${N_SAMPLES_PER_PROMPT_MAX:-8}"
export DYNAMIC_SAMPLING_FILTER_PATH="${DYNAMIC_SAMPLING_FILTER_PATH:-slime.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted_and_reward_nonzero_std}"
# The joint source reads JOINT_*_DATA. PROMPT_DATA is still required by Slime's
# standard validation and rollout-size accounting.
export PROMPT_DATA="${PROMPT_DATA:-${JOINT_PATCHEVAL_DATA}}"
export SLIME_DIR SECUREVIBE_DIR

if [[ ! -f "${ONE_NODE_LAUNCHER}" ]]; then
  echo "One-node launcher does not exist: ${ONE_NODE_LAUNCHER}" >&2
  exit 1
fi

for readable_file in \
  "${JOINT_PATCHEVAL_DATA}" \
  "${JOINT_AUTOBAX_DATA}" \
  "${PATCHEVAL_CONFIG}" \
  "${AUTOBAX_CONFIG}"; do
  if [[ ! -f "${readable_file}" || ! -r "${readable_file}" ]]; then
    echo "Required joint-training file is missing or unreadable: ${readable_file}" >&2
    exit 1
  fi
done

if [[ ! -f "${AUTOBAX_SRC_DIR}/in_container_runner.py" ]]; then
  echo "AUTOBAX_SRC_DIR does not contain in_container_runner.py: ${AUTOBAX_SRC_DIR}" >&2
  echo "Restore the bundled dependencies/autobax_arc source or set AUTOBAX_SRC_DIR." >&2
  exit 1
fi

if ! PYTHONPATH="${SLIME_DIR}:${SECUREVIBE_DIR}:${PYTHONPATH:-}" python3 - \
  "${JOINT_PATCHEVAL_WEIGHT}" "${JOINT_AUTOBAX_WEIGHT}" \
  "${CUSTOM_RM_PATH}" "${DATA_SOURCE_PATH}" <<'PY'
import inspect
import math
import sys

from slime.utils.misc import load_function

weights = (float(sys.argv[1]), float(sys.argv[2]))
if any(not math.isfinite(weight) or weight < 0 for weight in weights):
    raise ValueError(f"Joint weights must be finite and nonnegative: {weights}")
if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError(f"Joint weights must sum to one: {weights}")

reward_func = load_function(sys.argv[3])
data_source = load_function(sys.argv[4])
if not inspect.iscoroutinefunction(reward_func):
    raise TypeError(f"Joint reward must be async: {sys.argv[3]}")
if not inspect.isclass(data_source):
    raise TypeError(f"Joint data source must be a class: {sys.argv[4]}")
PY
then
  echo "Joint weight or import preflight failed; verify weights, SLIME_DIR, and runtime dependencies." >&2
  exit 1
fi

exec bash "${ONE_NODE_LAUNCHER}" "$@"
