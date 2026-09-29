# Security Coding-Agent RL

This directory implements joint PatchEval-Gen + AutoBax GRPO training.
Imports and launcher paths use the `training/slime_rl` package.

## Runtime files

- `train.sh`: one-node training launcher.
- `joint_data_source.py`: weighted sampling with checkpoint resume.
- `joint_reward.py`: dispatches rewards by task type.
- `patcheval_reward.py`: PatchEval functional and security reward.
- `autobax_reward.py`: AutoBax grading adapter.
- `patcheval.yaml`, `autobax.yaml`: agent configuration profiles.
- `requirements.txt`: example dependencies.

## Setup and launch

Use the public [Microsoft Orchard](https://github.com/microsoft/Orchard)
repository for the sandbox service, Python SDK, and pinned Slime trainer.
Orchard revision `3d7d7e992f56e3fec98f80f52afd7bc2e90af0f4` pins its
`trainer/slime` submodule to `331efaaeef75c98aad6ba1a2f5f50bd8149f6ab9`.

From the repository root:

```bash
python3 dependencies/orchard/setup.py /path/to/Orchard
python -m pip install -e /path/to/Orchard/orchard_env
export SLIME_DIR=/path/to/Orchard/trainer/slime
```

The setup script applies the checksum-verified compatibility patch. See
[dependency setup](../../dependencies/orchard/README.md) for details and checks.

Follow Orchard's `orchard_env/README.md` for sandbox deployment and SDK setup.
The launchers verify the pinned revision and compatibility patch before starting Ray.


Install the GPU training dependencies and this directory's `requirements.txt`
in the Ray worker environment.

**Compatibility:** the SecureVibe patch adds teacher-only hint support,
evaluation-mode propagation, and aborted-group refill controls. Training uses
local config routing and the public Orchard-SWE sandbox wrappers. Run the dependency checks and a short sandbox/GPU job to validate your
deployment before a full training run.

Run from the repository’s `training/` directory and supply absolute paths:

```bash
export SLIME_DIR=/path/to/Orchard/trainer/slime
export HF_CHECKPOINT=/path/to/Qwen3.5-35B-A3B
export REF_MODEL_PATH=/path/to/Qwen3.5-35B-A3B_torch_dist
export JOINT_PATCHEVAL_DATA=/path/to/rl_patcheval.train.jsonl
export JOINT_AUTOBAX_DATA=/path/to/rl_autobax.train.jsonl
# AUTOBAX_SRC_DIR defaults to the bundled dependencies/autobax_arc source.
export MEGATRON_PATH=/path/to/Megatron-LM
export ORCHARD_SANDBOX_ENDPOINT=https://your-sandbox-service
# Export SANDBOX_API_KEY from your credential environment.

PREFLIGHT_ONLY=1 bash slime_rl/train.sh
bash slime_rl/train.sh
```

Prepared JSONL records must include the runtime routing metadata and
`swe_config_path` pointing to the appropriate local YAML profile. Use absolute
config paths accessible to every Ray worker. The `training` directory must
also be accessible at the same path on workers; the launcher adds it to their
`PYTHONPATH`.

The default mixture weights are equal. Override `JOINT_PATCHEVAL_WEIGHT` and
`JOINT_AUTOBAX_WEIGHT` with finite, nonnegative values summing to one.

See the [data guide](../../data/README.md) for prepared training inputs.
Supply model checkpoints and deploy the Orchard sandbox service before training.

The AutoBax reward source is bundled at `../../dependencies/autobax_arc`.
No archive extraction is needed. Set `AUTOBAX_SRC_DIR` only to override it;
see the [snapshot provenance](../../dependencies/autobax_arc.README.md).
