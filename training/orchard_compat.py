"""SecureVibe configuration routing for the public Orchard-SWE runtime."""
import copy
import os


def resolve_swe_config_path(args, sample, *, used_metadata_key=None, purpose="SWE config"):
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_type = metadata.get("task_type", "patcheval")
    env_key = {"patcheval": "PATCHEVAL_CONFIG_PATH", "autobax": "AUTOBAX_CONFIG_PATH"}.get(task_type)
    if env_key is None:
        raise ValueError(f"Unsupported task_type: {task_type!r}")
    path = (metadata.get("swe_config_path") or getattr(args, "swe_config_path", None)
            or os.environ.get(env_key) or os.environ.get("SWE_CONFIG_PATH"))
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{purpose}: set {env_key} or SWE_CONFIG_PATH")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{purpose}: {path}")
    if used_metadata_key:
        sample.metadata = metadata
        metadata[used_metadata_key] = path
    return path


async def generate(args, sample, sampling_params):
    from examples.orchard_swe.swe_generate_v2 import generate as orchard_generate
    local_args = copy.copy(args)
    local_args.swe_config_path = resolve_swe_config_path(args, sample)
    return await orchard_generate(local_args, sample, sampling_params)
