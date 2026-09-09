"""Reward dispatcher for routed PatchEval and AutoBax samples."""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

RewardFunction = Callable[..., Awaitable[float]]


async def _autobax_reward(args: Any, sample: Sample, **kwargs: Any) -> float:
    from slime_rl.autobax_reward import reward_func

    return await reward_func(args, sample, **kwargs)


async def _patcheval_reward(args: Any, sample: Sample, **kwargs: Any) -> float:
    from slime_rl.patcheval_reward import reward_func

    return await reward_func(args, sample, **kwargs)


_REWARD_FUNCTIONS: dict[str, RewardFunction] = {
    "autobax": _autobax_reward,
    "patcheval": _patcheval_reward,
}


def _unresolved_reward() -> float:
    return float(os.environ.get("SWE_UNRESOLVED_REWARD", "0"))


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Dispatch solely from explicit reward metadata and preserve adapter output."""
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    reward_type = metadata.get("reward_type")
    instance_id = metadata.get("instance_id", f"task_{sample.index}")
    task_type = metadata.get("task_type")
    adapter = (
        _REWARD_FUNCTIONS.get(reward_type) if isinstance(reward_type, str) else None
    )
    if adapter is None:
        metadata["reward_error"] = f"unknown_joint_reward_type: {reward_type!r}"
        logger.error(
            "Joint reward dispatch failed for instance_id=%s task_type=%r: %s",
            instance_id,
            task_type,
            metadata["reward_error"],
        )
        return _unresolved_reward()
    try:
        return float(await adapter(args, sample, **kwargs))
    except Exception as error:
        dispatch_error = (
            f"joint_reward_dispatch_failed[{reward_type}]: "
            f"{type(error).__name__}: {error}"
        )
        previous_error = metadata.get("reward_error")
        metadata["reward_error"] = (
            f"{previous_error}; {dispatch_error}" if previous_error else dispatch_error
        )
        logger.exception(
            "Joint reward adapter failed for instance_id=%s task_type=%r "
            "reward_type=%r",
            instance_id,
            task_type,
            reward_type,
        )
        return _unresolved_reward()


async def binary_reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Map the dispatched joint reward to a binary success reward.

    Positive rewards are successes (1.0); zero and negative rewards are
    failures (0.0). Preserve both values in metadata so rollout artifacts
    retain the underlying grader result.
    """
    original_reward = float(await reward_func(args, sample, **kwargs))
    binary_reward = 1.0 if original_reward > 0.0 else 0.0
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    metadata["joint_original_reward"] = original_reward
    metadata["joint_binary_reward"] = binary_reward
    return binary_reward


async def strict_binary_reward_func(
    args: Any, sample: Sample, **kwargs: Any
) -> float:
    """Reward only samples that pass both security and functionality checks."""
    original_reward = float(await reward_func(args, sample, **kwargs))
    binary_reward = 1.0 if original_reward == 1.0 else 0.0
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    metadata["joint_original_reward"] = original_reward
    metadata["joint_strict_binary_reward"] = binary_reward
    return binary_reward
