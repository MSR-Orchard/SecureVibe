"""AutoBax reward adapter for slime's Mini-SWE-Agent rollout.

The submitted patch is applied in a fresh copy of the task image. The adapter
then uploads the same patched AutoBaxBuilder source tree used by
``grade_sandbox.py`` and runs ``in_container_runner.py`` against the generated
code directory. Rewards use the same functional/security buckets as
``patcheval_reward.py``.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
import os
import re
import shlex
import shutil
import tarfile
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from orchard_compat import resolve_swe_config_path
from slime.utils.types import Sample

from examples.orchard_swe.swe_wrapper_v2 import (
    create_environment,
    load_config,
    stop_environment,
)

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 48_000
_OUTPUT_LIMIT = 8_000
_RESULT_MARKER = "__BAXBENCH_RESULT__"
_ARCHIVE_HEADER = "AUTOBAX_TAR_GZ_BASE64"
_IGNORED_SUBMISSION_DIRS = {
    ".cache", ".cargo", ".git", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".venv", "__pycache__", "build", "coverage", "dist",
    "log", "node_modules", "storage", "target", "tmp", "vendor", "venv",
}
_IGNORED_SUBMISSION_FILES = {".coverage", "db.sqlite3", "server.log"}
_HARNESS_DEPS = "requests pyyaml docker pdfplumber imageio Pillow numpy"
_PY312_TYPE_ALIAS_RE = re.compile(
    r"^(?P<indent>\s*)type (?P<name>[A-Za-z_][A-Za-z0-9_]*) = .*$",
    re.MULTILINE,
)


def _reward(name: str, default: str) -> float:
    return float(os.environ.get(name, default))


def _graded_reward(*, functional_passed: bool, security_passed: bool) -> float:
    if functional_passed and security_passed:
        return _reward("SWE_REWARD_FULL", "1.0")
    if functional_passed:
        return _reward("SWE_REWARD_FUNCTIONAL_ONLY", "0.25")
    if security_passed:
        return _reward("SWE_REWARD_SECURITY_ONLY", "-0.5")
    return _reward("SWE_REWARD_BOTH_FAILED", "-0.5")


def _set_components(metadata: dict[str, Any], **updates: Any) -> None:
    components = metadata.setdefault("reward_components", {})
    if not isinstance(components, dict):
        components = {}
        metadata["reward_components"] = components
    components.update(updates)


async def _execute(env: Any, command: str) -> dict[str, Any]:
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


async def _write_bytes(env: Any, path: str, content: bytes) -> None:
    encoded = base64.b64encode(content).decode("ascii")
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


async def _write_text(env: Any, path: str, content: str) -> None:
    await _write_bytes(env, path, content.encode("utf-8"))


async def _prepare_harness(env: Any) -> str:
    existing_result = await _execute(
        env, "test -f /opt/autobax/src/in_container_runner.py"
    )
    if existing_result["returncode"] == 0:
        return "image"

    harness_url = os.environ.get("AUTOBAX_HARNESS_URL")
    if harness_url:
        download_result = await _execute(
            env,
            "rm -rf /opt/autobax/src && mkdir -p /opt/autobax/src && "
            f"(curl -fsSL {shlex.quote(harness_url)} "
            f"|| wget -qO- {shlex.quote(harness_url)}) "
            "| tar -C /opt/autobax/src -xzf -",
        )
        if download_result["returncode"] != 0:
            raise RuntimeError(
                "failed to download AutoBax harness: "
                f"{download_result['output'][-500:]}"
            )
        return "url"

    await _write_bytes(env, "/tmp/autobax_src.tgz", _patched_src_tar_bytes())
    extract_result = await _execute(
        env,
        "rm -rf /opt/autobax/src && mkdir -p /opt/autobax/src && "
        "tar -C /opt/autobax/src -xzf /tmp/autobax_src.tgz",
    )
    if extract_result["returncode"] != 0:
        raise RuntimeError(
            f"failed to extract AutoBax harness: {extract_result['output'][-500:]}"
        )
    return "worker_upload"


def _valid_patch(patch: str) -> bool:
    return bool(patch.strip()) and (
        "diff --git " in patch
        or ("--- " in patch and "+++ " in patch and "@@" in patch)
    )


def _decode_submission_archive(submission: str) -> bytes | None:
    lines = submission.strip().splitlines()
    if not lines or lines[0].strip() != _ARCHIVE_HEADER:
        return None
    encoded = "".join(line.strip() for line in lines[1:])
    if not encoded:
        raise ValueError("AutoBax submission archive is empty")
    archive = base64.b64decode(encoded, validate=True)
    max_compressed = int(os.environ.get("AUTOBAX_MAX_ARCHIVE_BYTES", "16777216"))
    max_uncompressed = int(
        os.environ.get("AUTOBAX_MAX_UNCOMPRESSED_BYTES", "67108864")
    )
    max_raw_members = int(
        os.environ.get("AUTOBAX_MAX_RAW_ARCHIVE_MEMBERS", "20000")
    )
    max_retained_members = int(
        os.environ.get("AUTOBAX_MAX_ARCHIVE_MEMBERS", "2000")
    )
    if len(archive) > max_compressed:
        raise ValueError(
            f"AutoBax submission archive exceeds {max_compressed} compressed bytes"
        )

    total_size = 0
    retained: list[tuple[tarfile.TarInfo, bytes | None]] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise ValueError("AutoBax submission archive contains no files")
        if len(members) > max_raw_members:
            raise ValueError(
                f"AutoBax submission archive contains more than {max_raw_members} raw entries"
            )
        for member in members:
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(
                    f"unsafe path in AutoBax submission archive: {member.name!r}"
                )
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(
                    f"unsupported entry in AutoBax submission archive: {member.name!r}"
                )
            total_size += member.size
            if total_size > max_uncompressed:
                raise ValueError(
                    "AutoBax submission archive exceeds "
                    f"{max_uncompressed} uncompressed bytes"
                )

            # Match the canonical AutoBax grader: dependencies, generated artifacts,
            # and binary/non-UTF-8 files are not part of the submitted application.
            normalized_parts = tuple(part for part in member_path.parts if part not in ("", "."))
            if any(part in _IGNORED_SUBMISSION_DIRS for part in normalized_parts):
                continue
            if normalized_parts and normalized_parts[-1] in _IGNORED_SUBMISSION_FILES:
                continue

            data: bytes | None = None
            if member.isfile():
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"could not read archive entry: {member.name!r}")
                data = extracted.read()
                if b"\x00" in data[:4096]:
                    continue
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            elif not member.isdir():
                raise ValueError(
                    f"unsupported entry in AutoBax submission archive: {member.name!r}"
                )
            retained.append((member, data))

    retained_files = sum(member.isfile() for member, _ in retained)
    if retained_files == 0:
        raise ValueError("AutoBax submission archive contains no application source files")
    if len(retained) > max_retained_members:
        raise ValueError(
            f"AutoBax submission archive contains more than {max_retained_members} retained entries"
        )

    filtered = io.BytesIO()
    with tarfile.open(fileobj=filtered, mode="w:gz") as output:
        for original, data in retained:
            member = tarfile.TarInfo(original.name)
            member.mode = original.mode
            member.mtime = original.mtime
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            if original.isdir():
                member.type = tarfile.DIRTYPE
                member.size = 0
                output.addfile(member)
            else:
                assert data is not None
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
    return filtered.getvalue()


def _autobax_src_dir() -> Path:
    configured = os.environ.get("AUTOBAX_SRC_DIR")
    source_dir = (
        Path(configured).expanduser() if configured else
        Path(__file__).resolve().parents[2] / "dependencies/autobax_arc"
    )
    if not (source_dir / "in_container_runner.py").is_file():
        raise FileNotFoundError(
            f"AutoBaxBuilder src not found at {source_dir}; set AUTOBAX_SRC_DIR"
        )
    return source_dir


def _patch_py312_type_aliases(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    patched = _PY312_TYPE_ALIAS_RE.sub(r"\g<indent>\g<name> = Any", text)
    if "from __future__ import annotations" not in patched:
        lines = patched.splitlines(keepends=True)
        insert_at = 0
        if lines and lines[0].startswith("#!"):
            insert_at = 1
        if len(lines) > insert_at and "coding" in lines[insert_at]:
            insert_at += 1
        lines.insert(insert_at, "from __future__ import annotations\n")
        patched = "".join(lines)
    if patched != text:
        path.write_text(patched, encoding="utf-8")


@lru_cache(maxsize=1)
def _patched_src_tar_bytes() -> bytes:
    source_dir = _autobax_src_dir()
    with tempfile.TemporaryDirectory(prefix="autobax_reward_src_") as temporary_dir:
        copied_src = Path(temporary_dir) / "src"
        shutil.copytree(
            source_dir,
            copied_src,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
        for path in copied_src.rglob("*.py"):
            _patch_py312_type_aliases(path)
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            tar.add(copied_src, arcname=".")
        return archive.getvalue()


def _parse_result(output: str) -> dict[str, Any]:
    marker_line = next(
        (line for line in output.splitlines() if line.startswith(_RESULT_MARKER)),
        None,
    )
    if marker_line is None:
        raise RuntimeError("AutoBax runner produced no result marker")
    import json

    payload = json.loads(marker_line[len(_RESULT_MARKER) :])
    if not isinstance(payload, dict):
        raise TypeError("AutoBax runner result is not an object")
    return payload


async def _evaluate(args: Any, sample: Sample) -> float:
    del args
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    sample.metadata = metadata
    submission = str(metadata.get("final_output") or "")
    try:
        submission_archive = _decode_submission_archive(submission)
    except (ValueError, tarfile.TarError) as error:
        reward = _reward("SWE_REWARD_INVALID_PATCH", "-1.0")
        metadata["reward_error"] = f"invalid_submission_archive: {error}"
        _set_components(
            metadata,
            submission_type="archive",
            patch_valid=False,
            patch_applied=False,
            functional_passed=None,
            security_passed=None,
            final_reward=reward,
        )
        return reward
    patch = submission if submission_archive is None else ""
    if submission_archive is None and not _valid_patch(patch):
        reward = _reward("SWE_REWARD_INVALID_PATCH", "-1.0")
        metadata["reward_error"] = "missing_or_invalid_autobax_submission"
        _set_components(
            metadata,
            submission_type=None,
            patch_valid=False,
            patch_applied=False,
            functional_passed=None,
            security_passed=None,
            final_reward=reward,
        )
        return reward

    instance_id = str(metadata.get("instance_id") or f"task_{sample.index}")
    image_url = metadata.get("image_url") or metadata.get("image")
    workdir = str(metadata.get("workdir") or "/app")
    code_dir = str(metadata.get("code_dir") or "/app/code")
    scenario = metadata.get("scenario")
    env_id = metadata.get("env")
    unresolved_reward = _reward("SWE_UNRESOLVED_REWARD", "0")
    if not isinstance(image_url, str) or not image_url:
        metadata["reward_error"] = "missing_image_url"
        return unresolved_reward
    if not isinstance(scenario, str) or not scenario:
        metadata["reward_error"] = "missing_autobax_scenario"
        return unresolved_reward
    if not isinstance(env_id, str) or not env_id:
        metadata["reward_error"] = "missing_autobax_env"
        return unresolved_reward

    config = load_config(
        resolve_swe_config_path(
            None,
            sample,
            used_metadata_key="reward_swe_config_path_used",
            purpose="Mini-SWE reward config",
        )
    )
    env_config = dict(config.get("environment") or {})
    env_config["cwd"] = workdir
    if scenario == "FrameExtract":
        env_config["memory"] = os.environ.get(
            "BAXBENCH_HEAVY_SANDBOX_MEMORY", "12Gi"
        )
    instance = {
        **metadata,
        "instance_id": instance_id,
        "image_url": image_url,
        "workdir": workdir,
    }

    environment = None
    try:
        environment = await create_environment(
            env_config,
            instance_id=f"eval-{instance_id}:{sample.index}",
            instance=instance,
            startup_command=config.get("run", {}).get("env_startup_command"),
        )
        if submission_archive is not None:
            code_dir = "/tmp/autobax_submission"
            await _write_bytes(
                environment, "/tmp/autobax_submission.tgz", submission_archive
            )
            extract_submission = await _execute(
                environment,
                f"rm -rf {shlex.quote(code_dir)} && mkdir -p {shlex.quote(code_dir)} "
                f"&& tar -C {shlex.quote(code_dir)} -xzf /tmp/autobax_submission.tgz",
            )
            if extract_submission["returncode"] != 0:
                reward = _reward("SWE_REWARD_APPLY_FAILURE", "-0.75")
                metadata["reward_error"] = "submission_archive_did_not_extract"
                metadata["reward_apply_output"] = extract_submission["output"][
                    -_OUTPUT_LIMIT:
                ]
                _set_components(
                    metadata,
                    submission_type="archive",
                    patch_valid=None,
                    patch_applied=False,
                    functional_passed=None,
                    security_passed=None,
                    final_reward=reward,
                )
                return reward
        else:
            await _write_text(environment, "/tmp/autobax_model.patch", patch)
            apply_result = await _execute(
                environment,
                f"cd {shlex.quote(code_dir)} && "
                "git apply --whitespace=nowarn /tmp/autobax_model.patch",
            )
            if apply_result["returncode"] != 0:
                reward = _reward("SWE_REWARD_APPLY_FAILURE", "-0.75")
                metadata["reward_error"] = "model_patch_did_not_apply"
                metadata["reward_apply_output"] = apply_result["output"][
                    -_OUTPUT_LIMIT:
                ]
                _set_components(
                    metadata,
                    submission_type="patch",
                    patch_valid=True,
                    patch_applied=False,
                    functional_passed=None,
                    security_passed=None,
                    final_reward=reward,
                )
                return reward

        metadata["autobax_harness_source"] = await _prepare_harness(environment)

        dependency_result = await _execute(
            environment,
            "command -v python3 >/dev/null 2>&1 || "
            "(apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "python3 python3-pip); "
            "python3 -m pip --version >/dev/null 2>&1 || "
            "(apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "python3-pip); "
            "python3 -c 'import docker,imageio,numpy,pdfplumber,requests,yaml; "
            "from PIL import Image' >/dev/null 2>&1 || "
            f"(python3 -m pip install --quiet {_HARNESS_DEPS} "
            f"|| python3 -m pip install {_HARNESS_DEPS})",
        )
        metadata["reward_dependency_returncode"] = dependency_result["returncode"]
        metadata["reward_dependency_output"] = dependency_result["output"][-_OUTPUT_LIMIT:]

        test_timeout = int(os.environ.get("BAXBENCH_TEST_TIMEOUT", "90"))
        skip_extra_packages = bool(metadata.get("autobax_skip_extra_pkg_cmds", True))
        runner_command = (
            "cd /opt/autobax/src && PYTHONPATH=/opt/autobax/src "
            "python3 in_container_runner.py "
            f"--scenario {shlex.quote(scenario)} --env {shlex.quote(env_id)} "
            f"--code-dir {shlex.quote(code_dir)} --timeout {test_timeout}"
        )
        if skip_extra_packages:
            runner_command += " --skip-extra-pkg-cmds"
        runner_result = await _execute(environment, runner_command)
        payload = _parse_result(runner_result["output"])

        functional_total = int(payload.get("num_total_ft", 0))
        functional_passed_count = int(payload.get("num_passed_ft", 0))
        functional_exceptions = int(payload.get("num_ft_exceptions", 0))
        security_total = int(payload.get("num_total_st", 0))
        security_exceptions = int(payload.get("num_st_exceptions", 0))
        cwes = payload.get("cwes")
        if not isinstance(cwes, list):
            raise TypeError("AutoBax runner cwes field is not a list")

        functional_passed = (
            functional_total > 0 and functional_passed_count == functional_total
        )
        security_passed = len(cwes) == 0
        reward = _graded_reward(
            functional_passed=functional_passed,
            security_passed=security_passed,
        )
        metadata["reward_passed"] = functional_passed and security_passed
        metadata["reward_returncode"] = runner_result["returncode"]
        metadata["reward_output"] = runner_result["output"][-_OUTPUT_LIMIT:]
        metadata["autobax_test_result"] = payload
        _set_components(
            metadata,
            submission_type="archive" if submission_archive is not None else "patch",
            patch_valid=True if submission_archive is None else None,
            patch_applied=True,
            functional_passed=functional_passed,
            functional_passed_count=functional_passed_count,
            functional_total=functional_total,
            functional_exceptions=functional_exceptions,
            security_passed=security_passed,
            security_total=security_total,
            security_exceptions=security_exceptions,
            cwes=cwes,
            final_reward=reward,
        )
        return reward
    finally:
        if environment is not None:
            await stop_environment(environment)


async def reward_func(args: Any, sample: Sample, **kwargs: Any) -> float:
    """Evaluate one AutoBax submission with a bounded fresh environment."""
    del kwargs
    timeout = int(
        os.environ.get(
            "AUTOBAX_TIMEOUT_REWARD_TOTAL",
            os.environ.get("SWE_TIMEOUT_REWARD_TOTAL", "1800"),
        )
    )
    try:
        return await asyncio.wait_for(_evaluate(args, sample), timeout=timeout)
    except asyncio.TimeoutError:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = metadata
        metadata["reward_error"] = "reward_timeout"
        logger.error("AutoBax reward timed out for %s", metadata.get("instance_id"))
    except Exception as error:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        sample.metadata = metadata
        metadata["reward_error"] = f"{type(error).__name__}: {error}"
        logger.exception("AutoBax reward failed for %s", metadata.get("instance_id"))
    return float(os.environ.get("SWE_UNRESOLVED_REWARD", "0"))
