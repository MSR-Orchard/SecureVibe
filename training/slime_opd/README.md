# Security Coding-Agent OPD

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


Imports and launcher paths use the `training/slime_opd` package.

**Compatibility:** the SecureVibe patch adds teacher-only hint support,
evaluation-mode propagation, and aborted-group refill controls. Training uses
local config routing and the public Orchard-SWE sandbox wrappers. Run the dependency checks and a short sandbox/GPU job to validate your
deployment before a full training run.

The `training` directory must be accessible at the same absolute path on Ray
workers; the launcher includes it in their `PYTHONPATH`.

See the [data guide](../../data/README.md) for prepared training and validation inputs.
Install this directory’s `requirements.txt` alongside the Slime GPU stack.

This example extends two existing slime examples:

- [On-policy distillation](https://github.com/MSR-Orchard/slime/blob/331efaaeef75c98aad6ba1a2f5f50bd8149f6ab9/examples/on_policy_distillation/README.md) documents OPD,
  teacher modes, loss options, checkpoint preparation, and standard OPD flags.
- [Orchard-SWE OPD](https://github.com/MSR-Orchard/slime/blob/331efaaeef75c98aad6ba1a2f5f50bd8149f6ab9/examples/orchard_swe/scripts/run-qwen3.5-35B-orchard-swe-opd.sh)
  provides the Qwen3.5 single-node rollout, Ray, Megatron, and external SGLang
  teacher setup used as this launcher's baseline.

Read those references for shared OPD and infrastructure concepts. This document
covers only what the security coding-agent example adds or changes.

## What is different

| Area | Orchard-SWE OPD baseline | Security coding-agent OPD |
|---|---|---|
| Tasks | One SWE task format | PatchEval, AutoBax, or both in one dataset |
| Dataset keys | `problem_statement`, `patch` | `prompt`, `label`, plus task metadata |
| Rollout | Direct `swe_generate_v2.generate` | Per-sample config routing, then delegates to the same Orchard generator |
| Reward | SWE reward plus OPD | Task-specific functional/security reward plus OPD |
| Teacher context | Student prompt and response | Optionally inserts a per-instance hint visible only to the teacher |
| Tokenizers | Supports normal cross-vocabulary OPD alignment | Hinted samples require identical teacher/student vocabularies |
| Truncation | Drops response suffixes | Drops prompt prefixes to preserve the response and its hint alignment |
| Failure handling | Filters aborted groups | Adds exponential refill backoff and an optional abort circuit breaker |
| Configuration | One SWE agent configuration | Independent PatchEval and AutoBax configurations |
| Continuation | Initial fine-tuning flow | Adds full optimizer/RNG resume and rollout-ID controls |

The general OPD example supports both SGLang and Megatron teachers. This
security launcher uses only an external SGLang teacher because teacher hints
are assembled during rollout.

## Reused implementation

The example intentionally keeps common behavior in existing modules:

- `generate.py` selects a task configuration on a copy of `args`, then calls
  `examples.orchard_swe.swe_generate_v2.generate`. Copying avoids races when a
  rollout batch mixes task types.
- `patcheval_reward.py` and `autobax_reward.py` use Orchard's environment creation and
  teardown wrappers.
- `combined_reward.py` calls
  `slime.rollout.on_policy_distillation.reward_func` for teacher log
  probabilities instead of implementing OPD again.
- Core model, optimizer, SGLang, and Ray arguments follow the Orchard-SWE
  launcher.

The security launcher remains a separate script because the Orchard launcher
is executable rather than a sourceable library: loading it would immediately
clean processes, start Ray, and submit training.

## Security-specific files

- `generate.py`: routes each sample to its task-specific agent config.
- `combined_reward.py`: runs task grading and OPD independently, preserving the
  task reward if the teacher request fails.
- `patcheval_reward.py`: grades PatchEval patches with separate functional and hidden
  security checks.
- `autobax_reward.py`: grades generated applications with the AutoBax harness.
- `patcheval.yaml`, `autobax.yaml`: task-specific agent/sandbox configs.
- `train.sh`: main launcher.
- `megatron_actor.yaml`: Megatron actor configuration.
- `requirements.txt`: runtime dependencies.

## Dataset contract

Every converted row has `prompt`, `label`, and `metadata`. The routing field is:

```json
{"metadata": {"task_type": "patcheval"}}
```

or:

```json
{"metadata": {"task_type": "autobax"}}
```

`task_type` defaults to `patcheval` when omitted. Labels must remain unique in
mixed files.

PatchEval metadata must identify the rollout image, grading image, instance,
mask, and sanitized functional/security evaluation commands. AutoBax metadata
must identify its image, working/code paths, scenario, environment, and harness
inputs. Supply prepared training and evaluation JSONL files separately.

An optional teacher-only hint is stored as a string in
`metadata.teacher_hint`. The field name can be changed with
`TEACHER_HINT_METADATA_KEY`. Teacher-only content must be absent from the student `prompt`. Only use trusted evaluation
commands because they execute inside grading sandboxes.

Before training, verify that hints are absent from `prompt`, hint values are
strings, labels are unique, referenced images exist, and train/eval records do
not overlap unintentionally.

## Teacher-only hints

For an unhinted sample, shared OPD behavior is unchanged. For a hinted sample:

1. The student rolls out from its original prompt.
2. The hint template is inserted only in the teacher request, immediately
   before the student response.
3. Teacher log probabilities are aligned back to the unchanged response.

Because pre-existing student token IDs and newly encoded hint IDs share one
teacher request, both tokenizers must have exactly the same vocabulary. Point
`TEACHER_HF` at compatible tokenizer files available on the training node.
Different teacher weights are allowed; only the token vocabulary must match.

The launcher uses prefix truncation so an over-length request discards old
prompt context rather than response tokens. This differs from the baseline
Orchard launcher and preserves the complete OPD target.

## Task grading differences

PatchEval grading applies the submitted patch in a fresh environment, restores
the task mask, prevents changes to hidden tests, then runs functional and
security checks. A security-only result is treated as a failed solution when it
introduces a functional regression.

AutoBax grading needs `in_container_runner.py`. It searches in this order:

1. the runner already installed in the grading image;
2. an archive supplied through `AUTOBAX_HARNESS_URL` and verified against
   `AUTOBAX_HARNESS_SHA256`; or
3. the bundled `dependencies/autobax_arc` source, uploaded from the worker
   to the sandbox (`AUTOBAX_SRC_DIR` overrides that path).

The combined reward queries grading and the teacher independently. If the
teacher times out or fails, its KL contribution is masked and the task reward
is retained. Evaluation skips the teacher request and reports only task reward.

## Sandbox resilience defaults

The launcher makes two attempts for environment creation, rollout generation,
and environment-backed grading before marking a sample aborted. A generation
retry starts in a fresh sandbox at a higher resource level. This matters because
the dynamic filter treats each eight-sibling prompt group atomically: one
aborted sibling discards the group.

The default inference and sandbox-observation limits are 120 and 45 seconds,
respectively. The sandbox command timeout is intentionally unchanged. All
retry and timeout settings remain environment-variable
overrides, so unusually stable or constrained deployments can select stricter
limits without changing the launcher. Increasing retries reduces group loss
but can repeat a trajectory after a transient failure and therefore increases
rollout cost.

## Required security-specific setup

Configure the sandbox service:

```bash
export ORCHARD_SANDBOX_ENDPOINT=http://sandbox-host:port
export SANDBOX_API_KEY=your-key
```

For AutoBax, the bundled source is the default fallback when the grading image
does not contain the runner. No extraction or environment override is required.
To use an alternative source or verified archive:

```bash
export AUTOBAX_SRC_DIR=/path/to/autobax/src
# or
export AUTOBAX_HARNESS_URL=https://host.example/autobax-src.tar.gz
export AUTOBAX_HARNESS_SHA256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

In addition to the checkpoint and teacher settings described by the baseline
examples, set the security data and output paths explicitly:

```bash
export SLIME_DIR=/path/to/Orchard/trainer/slime
export MEGATRON_PATH=/path/to/Megatron-LM
export STUDENT_HF=/path/to/student-hf-checkpoint
export STUDENT_TORCH_DIST=/path/to/student-distributed-checkpoint
export TEACHER_HF=/path/to/compatible-teacher-tokenizer
export TEACHER_IP=teacher.example.internal
export TEACHER_PORT=30002
export PROMPT_DATA=/path/to/opd_hint.train.jsonl
export EVAL_DATA=/path/to/opd.val.jsonl
export SAVE_DIR=/path/to/output-checkpoints

# Run from the repository’s training/ directory.
bash slime_opd/train.sh
```

The launcher validates required paths, sandbox credentials, and teacher health
before training. It also stops existing local Ray processes and clears stale
Ray state, so do not run it beside another Ray workload that must remain alive.
Unlike the baseline Orchard script, it does not kill the separately managed
teacher process.

## Security-specific controls

These are additions or meaningful deviations from the baseline launcher:

| Variable | Purpose |
|---|---|
| `PATCHEVAL_CONFIG_PATH` | PatchEval agent/sandbox configuration |
| `AUTOBAX_CONFIG_PATH` | AutoBax agent/sandbox configuration |
| `TEACHER_HINT_METADATA_KEY` | Optional per-sample hint field |
| `AUTOBAX_SRC_DIR` | Worker-visible AutoBax harness source |
| `AUTOBAX_HARNESS_URL` | Sandbox-downloadable AutoBax harness archive |
| `AUTOBAX_HARNESS_SHA256` | Required SHA-256 digest for the harness URL |
| `DYNAMIC_SAMPLING_ABORTED_BACKOFF_SECONDS` | Exponential delay before replacing aborted groups |
| `DYNAMIC_SAMPLING_ABORTED_MAX_CONSECUTIVE` | Abort threshold; zero disables the threshold |
| `RESUME_OPD` | Restore a full OPD training state when set to `1` |
| `RESUME_LOAD_DIR` | Checkpoint directory used for a full resume |
| `START_ROLLOUT_ID` | Optional explicit rollout-ID override |

The security defaults use multiple samples per prompt and a replacement batch
smaller than the rollout batch. This is intentional for sparse sandbox rewards
and differs from the baseline's one-sample, full-batch refill behavior.

For a full continuation:

```bash
export RESUME_OPD=1
export RESUME_LOAD_DIR=/path/to/existing-checkpoints
```

The directory must contain `latest_checkpointed_iteration.txt`. Normally slime
derives the next rollout ID; set `START_ROLLOUT_ID` only when overriding that
behavior deliberately.

See the [AutoBax snapshot provenance](../../dependencies/autobax_arc.README.md)
for source and archive checksums.
