"""SafeVibe reward adapter for slime's Mini-SWE-Agent rollout.

The agent rollout comes from ``examples.orchard_swe.swe_generate_v2``. Its
stock reward targets canonical SWE-bench records; SafeVibe records instead
ship separate functional and hidden-security evaluation commands. This module
evaluates the submitted patch in a fresh environment created by Mini-SWE's own
environment factory and returns a graded reward for the two outcomes.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

from slime.utils.types import Sample

from examples.orchard_swe.swe_wrapper_v2 import (
    create_environment,
    load_config,
    stop_environment,
)

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 48_000
_OUTPUT_LIMIT = 8_000
_DIFF_GIT_RE = re.compile(r'^diff --git a/(.+?) b/(.+?)\s*$')
_PATCH_PATH_RE = re.compile(r'^\+\+\+\s+(?:b/)?([^\t\n]+)', re.MULTILINE)


def _resolve_swe_config_path(args, sample):
    from orchard_compat import resolve_swe_config_path
    return resolve_swe_config_path(args, sample)


def _reward(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _graded_reward(
    *, functional_available: bool, functional_passed: bool | None, security_passed: bool
) -> float:
    if not functional_available:
        return (
            _reward("SWE_REWARD_FULL", "1.0")
            if security_passed
            else _reward("SWE_REWARD_BOTH_FAILED", "-0.5")
        )
    if functional_passed and security_passed:
        return _reward("SWE_REWARD_FULL", "1.0")
    if security_passed:
        # A security-only result is not a valid solution when it introduces a
        # functional regression, so it receives the failed-solution reward.
        return _reward("SWE_REWARD_SECURITY_ONLY", "-0.5")
    if functional_passed:
        return _reward("SWE_REWARD_FUNCTIONAL_ONLY", "0.25")
    return _reward("SWE_REWARD_BOTH_FAILED", "-0.5")


def _set_components(metadata: dict[str, Any], **updates: Any) -> dict[str, Any]:
    components = metadata.setdefault("reward_components", {})
    if not isinstance(components, dict):
        components = {}
        metadata["reward_components"] = components
    components.update(updates)
    return components


def _reward_artifact_path(metadata: dict[str, Any]) -> Path | None:
    """Return the stock Mini-SWE sibling reward path for this trajectory."""
    trajectory_path = metadata.get("trajectory_path")
    if not isinstance(trajectory_path, str) or not trajectory_path:
        return None
    path = Path(trajectory_path)
    if path.suffix == ".json":
        return path.with_name(f"{path.stem}_rewards.json")
    return path.with_name(f"{path.name}_rewards.json")


def _persist_reward_artifact(sample: Sample, reward: float) -> Path | None:
    """Write a collector-compatible SafeVibe reward record atomically.

    The collector treats only the top-level ``resolved`` field as success.
    SafeVibe records the joint functional/security outcome explicitly because
    a functional-only result can have a positive shaped reward without being a
    complete solution.
    """
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    reward_path = _reward_artifact_path(metadata)
    if reward_path is None:
        logger.warning(
            "Cannot persist SafeVibe reward artifact for %s: trajectory_path is missing",
            metadata.get("instance_id", f"task_{sample.index}"),
        )
        return None

    components = metadata.get("reward_components")
    if not isinstance(components, dict):
        components = {}

    functional_available = components.get("functional_available")
    functional_passed = components.get("functional_passed")
    security_passed = components.get("security_passed")
    derived_resolved = security_passed is True and (
        functional_passed is True or functional_available is False
    )
    resolved = bool(metadata.get("reward_passed", derived_resolved))

    status = getattr(sample, "status", None)
    status_name = status.name if hasattr(status, "name") else str(status)
    exit_status = metadata.get("exit_status")
    record = {
        "instance_id": metadata.get("instance_id", f"task_{sample.index}"),
        "sample_index": sample.index,
        "passed": resolved,
        "resolved": resolved,
        "resolution": "RESOLVED_FULL" if resolved else "RESOLVED_NO",
        "final_reward": float(reward),
        "reward": float(reward),
        "status": status_name,
        "exit_status": exit_status,
        "truncated": status_name == "TRUNCATED"
        or exit_status in {"LimitsExceeded", "TimeExceeded"},
        "error": metadata.get("reward_error"),
        "apply_output": metadata.get("reward_apply_output", ""),
        "patch_valid": components.get("patch_valid"),
        "patch_applied": components.get("patch_applied"),
        "functional_available": functional_available,
        "functional_passed": functional_passed,
        "functional_returncode": metadata.get("functional_reward_returncode"),
        "functional_output": metadata.get("functional_reward_output", ""),
        "security_passed": security_passed,
        "security_returncode": metadata.get("security_reward_returncode"),
        "security_output": metadata.get("security_reward_output", ""),
        "reward_components": components,
    }

    reward_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=reward_path.parent,
            prefix=f".{reward_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
        os.replace(temp_path, reward_path)
        metadata["reward_artifact_path"] = str(reward_path)
        logger.info("Saved SafeVibe reward artifact to %s", reward_path)
        return reward_path
    except Exception as error:
        metadata["reward_artifact_error"] = f"{type(error).__name__}: {error}"
        logger.exception("Failed to save SafeVibe reward artifact to %s", reward_path)
        return None
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


async def _execute(env: Any, command: str) -> dict[str, Any]:
    """Run a command against either an async or synchronous Mini-SWE env."""
    execute = env.execute
    if inspect.iscoroutinefunction(execute):
        result = await execute(command)
    else:
        result = await asyncio.to_thread(execute, command)
    if not isinstance(result, dict):
        raise TypeError(
            f"Mini-SWE environment returned {type(result).__name__}, expected dict"
        )
    return {
        "output": str(result.get("output") or ""),
        "returncode": int(result.get("returncode", -1)),
    }


async def _write_file(env: Any, path: str, content: str) -> None:
    """Transfer text with the service file API, with a shell fallback."""
    native_write = getattr(env, "write_file", None)
    if native_write is not None:
        result = native_write(path, content)
        if inspect.isawaitable(result):
            await result
        return

    encoded = base64.b64encode(content.encode()).decode("ascii")
    encoded_path = f"{path}.b64"
    reset_result = await _execute(
        env,
        f"rm -f {shlex.quote(encoded_path)} {shlex.quote(path)} && "
        f": > {shlex.quote(encoded_path)}",
    )
    if reset_result["returncode"] != 0:
        raise RuntimeError(
            f"failed to prepare upload for {path}: {reset_result['output'][:500]}"
        )
    for offset in range(0, len(encoded), _CHUNK_SIZE):
        chunk = encoded[offset : offset + _CHUNK_SIZE]
        result = await _execute(
            env,
            f"printf '%s' {shlex.quote(chunk)} >> {shlex.quote(encoded_path)}",
        )
        if result["returncode"] != 0:
            raise RuntimeError(f"failed to upload {path}: {result['output'][:500]}")
    result = await _execute(
        env,
        f"base64 -d {shlex.quote(encoded_path)} > {shlex.quote(path)} && "
        f"rm -f {shlex.quote(encoded_path)}",
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"failed to materialize {path}: {result['output'][:500]}")


def _valid_patch(patch: str) -> bool:
    return bool(patch.strip()) and (
        "diff --git " in patch
        or ("--- " in patch and "+++ " in patch and "@@" in patch)
    )


def _test_file_paths(test_patch: str) -> set[str]:
    """Return repository-relative paths introduced or modified by an oracle patch."""
    paths = set()
    for path in _PATCH_PATH_RE.findall(test_patch or ""):
        path = path.strip().strip('"')
        if path != "/dev/null":
            paths.add(path)
    return paths


def _filter_diff(patch: str, drop_paths: set[str]) -> str:
    """Remove complete diff blocks that modify hidden oracle test files."""
    if not drop_paths:
        return patch
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    if not blocks:
        return patch

    kept = []
    for block in blocks:
        match = _DIFF_GIT_RE.match(block[0].rstrip("\n"))
        if match and (match.group(1) in drop_paths or match.group(2) in drop_paths):
            continue
        kept.append("".join(block))
    return "".join(kept)


async def _apply_repo_patch(
    env: Any, workdir: str, patch: str, filename: str
) -> tuple[bool, str]:
    """Apply a patch with the same compatibility fallbacks as PatchEval."""
    patch_path = f"/tmp/{filename}"
    await _write_file(env, patch_path, patch if patch.endswith("\n") else patch + "\n")
    quoted_workdir = shlex.quote(workdir)
    quoted_patch = shlex.quote(patch_path)
    outputs = []
    commands = (
        f"git apply --whitespace=nowarn {quoted_patch}",
        f"git apply --whitespace=nowarn -p0 {quoted_patch}",
        f"git apply --3way --whitespace=nowarn {quoted_patch}",
        f"patch -p1 --fuzz=3 -i {quoted_patch}",
    )
    try:
        for command in commands:
            result = await _execute(env, f"cd {quoted_workdir} && {command}")
            if result["returncode"] == 0:
                return True, result["output"]
            outputs.append(f"$ {command}\n{result['output']}")
        return False, "\n".join(outputs)[-_OUTPUT_LIMIT:]
    finally:
        await _execute(env, f"rm -f {quoted_patch}")


async def _evaluate(args: Any, sample: Sample) -> float:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    patch = str(metadata.get("final_output") or "")
    if not _valid_patch(patch):
        reward = _reward("SWE_REWARD_INVALID_PATCH", "-1.0")
        metadata["reward_error"] = "missing_or_invalid_submitted_patch"
        _set_components(
            metadata,
            patch_valid=False,
            patch_applied=False,
            functional_available=isinstance(metadata.get("functional_eval_cmd"), str),
            functional_passed=None,
            security_passed=None,
            final_reward=reward,
        )
        return reward

    workdir = str(metadata.get("workdir") or "/testbed")
    unresolved_reward = _reward("SWE_UNRESOLVED_REWARD", "0")
    mask_patch = str(metadata.get("mask_patch") or "")
    test_patch = str(metadata.get("test_patch") or "")
    if not _valid_patch(mask_patch):
        metadata["reward_error"] = "missing_or_invalid_mask_patch"
        return unresolved_reward
    patch = _filter_diff(patch, _test_file_paths(test_patch))
    if not _valid_patch(patch):
        reward = _reward("SWE_REWARD_INVALID_PATCH", "-1.0")
        metadata["reward_error"] = "model_patch_only_modified_hidden_tests"
        _set_components(
            metadata,
            patch_valid=False,
            patch_applied=False,
            functional_available=isinstance(metadata.get("functional_eval_cmd"), str),
            functional_passed=None,
            security_passed=None,
            final_reward=reward,
        )
        return reward
    grade_image_name = metadata.get("grade_image_name")
    image_url = grade_image_name
    functional_eval_cmd = metadata.get("functional_eval_cmd")
    security_eval_cmd = metadata.get("security_eval_cmd")
    legacy_eval_cmd = metadata.get("eval_cmd")
    instance_id = str(metadata.get("instance_id") or f"task_{sample.index}")
    if not isinstance(image_url, str) or not image_url:
        metadata["reward_error"] = "missing_grade_image_name"
        return unresolved_reward

    separate_commands = isinstance(security_eval_cmd, str) and bool(
        security_eval_cmd.strip()
    )
    if separate_commands:
        if functional_eval_cmd is not None and (
            not isinstance(functional_eval_cmd, str) or not functional_eval_cmd.strip()
        ):
            metadata["reward_error"] = "invalid_functional_eval_cmd"
            return unresolved_reward
    elif not isinstance(legacy_eval_cmd, str) or not legacy_eval_cmd.strip():
        metadata["reward_error"] = "missing_security_eval_cmd"
        return unresolved_reward

    config_path = _resolve_swe_config_path(args, sample)
    config = load_config(config_path)
    env_config = dict(config.get("environment") or {})
    env_config["cwd"] = workdir
    instance = {
        **metadata,
        "instance_id": instance_id,
        # Override the rollout image for this fresh grading sandbox.
        "image": image_url,
        "image_url": image_url,
        "workdir": workdir,
    }
    metadata["reward_image"] = image_url

    env = None
    try:
        env = await create_environment(
            env_config,
            instance_id=f"eval-{instance_id}:{sample.index}",
            instance=instance,
            startup_command=config.get("run", {}).get("env_startup_command"),
        )
        # The grading image starts from the original vulnerable repository,
        # whereas the model edited a deterministically masked agent image.
        # Reconstruct that exact baseline before applying the model submission.
        mask_applied, mask_output = await _apply_repo_patch(
            env, workdir, mask_patch, "safevibe_mask.patch"
        )
        if not mask_applied:
            metadata["reward_error"] = "mask_patch_did_not_apply"
            metadata["reward_apply_output"] = mask_output
            _set_components(
                metadata,
                patch_valid=True,
                patch_applied=False,
                functional_available=isinstance(functional_eval_cmd, str),
                functional_passed=None,
                security_passed=None,
                final_reward=unresolved_reward,
            )
            return unresolved_reward

        patch_applied, apply_output = await _apply_repo_patch(
            env, workdir, patch, "safevibe_model.patch"
        )
        if not patch_applied:
            reward = _reward("SWE_REWARD_APPLY_FAILURE", "-0.75")
            metadata["reward_error"] = "model_patch_did_not_apply"
            metadata["reward_apply_output"] = apply_output
            _set_components(
                metadata,
                patch_valid=True,
                patch_applied=False,
                functional_available=isinstance(functional_eval_cmd, str),
                functional_passed=None,
                security_passed=None,
                final_reward=reward,
            )
            return reward

        if not separate_commands:
            await _write_file(env, "/tmp/safevibe_eval.sh", legacy_eval_cmd)
            eval_result = await _execute(env, "bash /tmp/safevibe_eval.sh")
            passed = eval_result["returncode"] == 0
            reward = (
                _reward("SWE_REWARD_FULL", "1.0")
                if passed
                else _reward("SWE_REWARD_BOTH_FAILED", "-0.5")
            )
            metadata["reward_passed"] = passed
            metadata["reward_returncode"] = eval_result["returncode"]
            metadata["reward_output"] = eval_result["output"][-_OUTPUT_LIMIT:]
            _set_components(
                metadata,
                patch_valid=True,
                patch_applied=True,
                functional_available=False,
                functional_passed=None,
                security_passed=passed,
                legacy_eval=True,
                final_reward=reward,
            )
            return reward

        functional_available = isinstance(functional_eval_cmd, str)
        functional_result = None
        if functional_available:
            await _write_file(
                env, "/tmp/safevibe_functional_eval.sh", functional_eval_cmd
            )
            functional_result = await _execute(
                env, "bash /tmp/safevibe_functional_eval.sh"
            )

        await _write_file(env, "/tmp/safevibe_security_eval.sh", security_eval_cmd)
        security_result = await _execute(env, "bash /tmp/safevibe_security_eval.sh")
        functional_passed = (
            functional_result["returncode"] == 0
            if functional_result is not None
            else None
        )
        security_passed = security_result["returncode"] == 0
        reward = _graded_reward(
            functional_available=functional_available,
            functional_passed=functional_passed,
            security_passed=security_passed,
        )
        metadata["functional_reward_returncode"] = (
            functional_result["returncode"] if functional_result is not None else None
        )
        metadata["functional_reward_output"] = (
            functional_result["output"][-_OUTPUT_LIMIT:]
            if functional_result is not None
            else ""
        )
        metadata["security_reward_returncode"] = security_result["returncode"]
        metadata["security_reward_output"] = security_result["output"][-_OUTPUT_LIMIT:]
        metadata["reward_passed"] = bool(
            security_passed and (functional_passed is True or not functional_available)
        )
        _set_components(
            metadata,
            patch_valid=True,
            patch_applied=True,
            functional_available=functional_available,
            functional_passed=functional_passed,
            security_passed=security_passed,
            final_reward=reward,
        )
        return reward
    finally:
        if env is not None:
            await stop_environment(env)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Evaluate one Mini-SWE submission with a bounded fresh environment."""
    del kwargs
    timeout = int(os.environ.get("SWE_TIMEOUT_REWARD_TOTAL", "900"))
    fallback_reward = float(os.environ.get("SWE_UNRESOLVED_REWARD", "0"))
    try:
        reward = float(await asyncio.wait_for(_evaluate(args, sample), timeout=timeout))
    except asyncio.TimeoutError:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = metadata
        metadata["reward_error"] = "reward_timeout"
        logger.error("SafeVibe reward timed out for %s", metadata.get("instance_id"))
        reward = fallback_reward
    except Exception as error:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = metadata
        metadata["reward_error"] = f"{type(error).__name__}: {error}"
        logger.exception("SafeVibe reward failed for %s", metadata.get("instance_id"))
        reward = fallback_reward

    _set_components(sample.metadata, final_reward=reward)
    _persist_reward_artifact(sample, reward)
    return reward
