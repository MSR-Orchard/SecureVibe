"""
Combined OPD + SWE reward function.

Queries the teacher for log-probs (OPD distillation) AND computes the SWE-bench
task reward. ``reward_func`` returns the SWE task scalar (a plain float) and
attaches the OPD teacher log-probs / top-k fields directly onto the ``Sample``,
so downstream reward filters and training see a consistent schema.

Usage in training script:
    --custom-rm-path slime_opd.combined_reward.reward_func
    --custom-reward-post-process-path slime_opd.combined_reward.post_process_rewards

Error handling — four cases
----------------------------
Case 1 — SWE succeeds, OPD succeeds:
    reward = swe_scalar, teacher_log_probs = real values, teacher_valid_mask = real mask.
    KL penalty applied normally. Sample used for training.

Case 2 — SWE succeeds, OPD fails (timeout or exception):
    reward = swe_scalar (preserved), teacher_log_probs = zeros (length=response_length),
    teacher_valid_mask = [0, 0, ..., 0] (all-zero → KL term fully masked out).
    Sample trains on task reward only, no KL penalty.

Case 3 — SWE fails (timeout or exception):
    swe_reward_func returns 0.0. The sample trains on the (zero) task reward;
    if OPD also failed, teacher log-probs are zero-filled and the KL is masked.

Case 4 — No patch / empty model output (early exit):
    Same as Case 3: reward = 0.0.
"""

import asyncio
import json
import logging
import os
import traceback
from typing import Any

import numpy as np
import torch

from slime.rollout.on_policy_distillation import (
    _TOPK_ID_DTYPE,
    _TOPK_LP_DTYPE,
    _load_config as _load_opd_config,
    reward_func as opd_reward_func,
)
from slime.utils.types import Sample

from .patcheval_reward import reward_func as patcheval_reward_func

logger = logging.getLogger(__name__)


# Set SWE_OPD_QUIET_REWARD=1 to suppress the per-sample [SWE_OPD_REWARD] /
# [SWE_OPD_POST_PROCESS] info printouts (warnings/errors are still shown).
if os.environ.get("SWE_OPD_QUIET_REWARD", "0") == "1":
    logger.setLevel(logging.WARNING)

# Fallback timeout when no OPD config is available (legacy args path).
# When --opd-config YAML is used, cfg.timeout takes precedence.
_OPD_TIMEOUT_DEFAULT = 120


async def _task_reward_func(args, sample: Sample, **kwargs) -> float:
    task_type = (sample.metadata or {}).get("task_type", "patcheval")
    if task_type == "autobax":
        from .autobax_reward import reward_func as autobax_reward_func

        return float(await autobax_reward_func(args, sample, **kwargs))
    if task_type == "patcheval":
        return float(await patcheval_reward_func(args, sample, **kwargs))
    raise ValueError(f"Unsupported security coding task_type: {task_type!r}")


def _get_opd_timeout(args) -> float:
    """Return the OPD teacher timeout from config, falling back to _OPD_TIMEOUT_DEFAULT."""
    from slime.rollout.on_policy_distillation import _load_config
    try:
        cfg = _load_config(args)
        return cfg.timeout
    except Exception:
        return _OPD_TIMEOUT_DEFAULT


async def reward_func(args, sample: Sample, **kwargs) -> float:
    """Combined reward: OPD teacher log-probs + SWE task reward.

    Returns the SWE task reward as a plain float (so dynamic-sampling reward
    filters and GRPO advantage see a real number). The OPD teacher log-probs /
    top-k fields are attached directly onto ``sample`` for post-processing.

    During evaluation (evaluation=True in kwargs), the OPD teacher query is
    skipped — only the SWE task reward is computed.

    Error handling:
    - SWE timeout/error  → returns 0.0, teacher fields left as None
    - OPD timeout/error  → SWE reward preserved, teacher fields left as None (KL
                           skipped for this sample)
    """
    instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"
    evaluation = kwargs.get("evaluation", False)

    # --- SWE task reward (always, has its own internal timeout+error handling) ---
    # swe_reward_func returns a plain float scalar.
    swe_scalar = await _task_reward_func(args, sample, **kwargs)
    swe_reward = float(swe_scalar)
    logger.info(f"[SWE_OPD_REWARD] instance_id={instance_id} swe_reward={swe_reward} evaluation={evaluation}")

    if evaluation:
        return swe_reward

    # --- OPD teacher log-probs (training only, with independent error handling) ---
    # Attach the teacher fields directly onto the sample so the reward stays a
    # plain float. Fields are left as None when the teacher fails; post-processing
    # zero-fills and masks out the KL for those samples.
    opd_timeout = _get_opd_timeout(args)
    try:
        opd_result = await asyncio.wait_for(
            opd_reward_func(args, sample, **kwargs),
            timeout=opd_timeout,
        )
        sample.teacher_log_probs = opd_result.get("teacher_log_probs")
        sample.teacher_valid_mask = opd_result.get("valid_mask")
    except asyncio.TimeoutError:
        logger.warning(
            f"[SWE_OPD_REWARD] OPD teacher timed out after {opd_timeout}s for "
            f"instance_id={instance_id}. Proceeding without teacher log-probs."
        )
    except Exception as e:
        logger.warning(
            f"[SWE_OPD_REWARD] OPD teacher error for instance_id={instance_id}: {e}\n"
            f"{traceback.format_exc()}"
            f"Proceeding without teacher log-probs."
        )

    return swe_reward


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Finalize teacher log-probs on each sample and apply GRPO normalization.

    ``reward_func`` already set ``sample.reward`` to the SWE task scalar (float)
    and attached the OPD teacher fields onto the sample. For each sample:
    - Converts ``sample.teacher_log_probs`` (list) to a tensor, or zero-fills and
      masks out the KL when the teacher failed (teacher_log_probs is None).

    Then applies GRPO group normalization to the SWE scalars (mirroring the
    logic in _post_process_rewards that is bypassed when a custom post-process
    function is registered).

    Returns (raw_rewards, normalized_rewards).
    """
    swe_scalars = []
    for sample in samples:
        swe_scalar = float(sample.reward) if sample.reward is not None else 0.0
        instance_id = sample.metadata.get("instance_id", "unknown") if sample.metadata else "unknown"

        teacher_lps = sample.teacher_log_probs
        if teacher_lps is not None:
            if not torch.is_tensor(teacher_lps):
                teacher_lps = torch.tensor(teacher_lps, dtype=torch.float32)
            sample.teacher_log_probs = teacher_lps
        else:
            # OPD teacher failed for this sample — zero-fill so the batch stays
            # consistent. All-zero valid_mask means the KL penalty is fully masked
            # out, so this sample trains on task reward only.
            response_length = sample.response_length
            logger.warning(
                f"[SWE_OPD_POST_PROCESS] teacher_log_probs missing for "
                f"instance_id={instance_id} response_length={response_length}. "
                f"Zero-filling and masking out KL for this sample."
            )
            sample.teacher_log_probs = torch.zeros(response_length, dtype=torch.float32)
            sample.teacher_valid_mask = [0] * response_length

        swe_scalars.append(swe_scalar)
        sample.reward = swe_scalar

    # GRPO normalization — mirrors _post_process_rewards in ray/rollout.py
    if (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
    ):
        rewards = torch.tensor(swe_scalars, dtype=torch.float)
        n_samples = getattr(args, "n_samples_per_prompt", 1)
        batch_size = getattr(args, "rollout_batch_size", len(swe_scalars))
        full_batch_size = n_samples * batch_size
        is_full_batch = rewards.shape[-1] == full_batch_size
        if is_full_batch:
            rewards = rewards.reshape(-1, n_samples)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        if not is_full_batch:
            logger.warning(
                f"[SWE_OPD_POST_PROCESS] sample count mismatch: "
                f"expected n_samples_per_prompt * rollout_batch_size = {n_samples} * {batch_size} = {full_batch_size}, "
                f"got {len(swe_scalars)}. "
                f"This likely means {batch_size - len(swe_scalars) // n_samples} group(s) were filtered. "
                f"Falling back to single-group normalization which may produce near-zero rewards."
            )
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if (
            getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
            and getattr(args, "grpo_std_normalization", False)
        ):
            std = rewards.std(dim=-1, keepdim=True)
            logger.info(
                f"[SWE_OPD_POST_PROCESS] std_norm: "
                f"per_group_std={std.flatten().tolist()}"
            )
            rewards = rewards / (std + 1e-6)

        normalized = rewards.flatten().tolist()
        logger.info(
            f"[SWE_OPD_POST_PROCESS] normalized_rewards_mean={sum(normalized)/len(normalized):.6f} "
            f"normalized_rewards_min={min(normalized):.6f} normalized_rewards_max={max(normalized):.6f}"
        )
        return swe_scalars, normalized

    return swe_scalars, swe_scalars
