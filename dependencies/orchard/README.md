# Orchard dependency

This bundle pins the public Microsoft Orchard repository and its Slime trainer.
`versions.json` records both revisions and the SHA-256 of `safevibe.patch`.
Checkouts, model weights, datasets, credentials, and generated runs stay outside
this directory.

From the repository root:

```bash
python3 dependencies/orchard/setup.py /path/to/Orchard
export SLIME_DIR=/path/to/Orchard/trainer/slime
python -m pip install -e /path/to/Orchard/orchard_env
```

Install Slime's CUDA/PyTorch/Megatron/SGLang dependencies and the relevant
`training/slime_rl/requirements.txt` or `training/slime_opd/requirements.txt`
in the training environment. Deploy the sandbox service using Orchard's
`orchard_env/README.md`. Export `ORCHARD_SANDBOX_ENDPOINT` and `SANDBOX_API_KEY`.
The SDK is installed as `orchard_env`.

Setup verifies the patch checksum, checks both Git revisions, and applies the
patch only to a clean Slime checkout. Re-running setup recognizes an already
applied patch. Existing checkouts on different revisions are rejected rather
than reset. To upgrade, use a new checkout, regenerate and review the patch,
update its checksum and revision pins, and repeat validation.

## Patch scope

- `on_policy_distillation.py` and `opd_config.py`: optional teacher-only hints,
  exact vocabulary matching, and response-suffix alignment. Student tokens are
  unchanged; existing teacher truncation and valid-mask accounting remain in use.
- `sglang_rollout.py`: propagate evaluation mode to custom rewards so evaluation
  skips the teacher; back off or stop after repeated aborted prompt groups.
- `arguments.py`: expose the refill controls and permit positive refill batches
  smaller than the target rollout batch.

SafeVibe reward functions, task routing, and launchers remain in `training/`.
`training/orchard_compat.py` supplies per-sample config routing without changing
shared rollout arguments.

## Validation

```bash
python3 dependencies/orchard/check.py "$SLIME_DIR"
python3 dependencies/orchard/test_compatibility.py "$SLIME_DIR"
# In the fully installed training environment:
python dependencies/orchard/check.py "$SLIME_DIR" --runtime
```

The CPU tests exercise hint isolation, vocabulary mismatch rejection, response
truncation accounting, config resolution, concurrent routing, and evaluation
propagation. They compile pure functions from the actual patched Slime source
without importing GPU dependencies. They do not replace runtime integration
checks. Launcher revision checks run before Ray cleanup/startup.

Before a full experiment, run one prepared PatchEval and one AutoBax task
against your deployed sandbox, verify their grades, then run a short GPU job
and checkpoint/resume cycle. OPD additionally requires a reachable teacher
with the same tokenizer vocabulary for hinted samples. No end-to-end GPU or
sandbox validation is claimed by this bundle.

## Real CPU checks

With CPU PyTorch, Transformers, Ray, NumPy, Pillow, PyYAML, aiohttp, and
OmegaConf installed, and downloaded tokenizer files:

```bash
python dependencies/orchard/test_real_cpu.py \
  --slime-dir "$SLIME_DIR" \
  --student-tokenizer /path/to/student-tokenizer \
  --teacher-tokenizer /path/to/teacher-tokenizer
```

This imports the actual Slime Dataset, Sample, tokenizer loaders, and joint
sampler. It uses synthetic JSONL rows and real `torch.save`/`torch.load`;
there are no mocked sampler or serialization interfaces. It compares 75
post-resume groups, final sampler state, and buffered samples across epoch
rollover, and verifies rejection of changed weights and dataset fingerprints.
Tokenizer tests use actual encoders for four hints, including multilingual
text, Unicode, braces, and a special-token string.

Run these checks with the tokenizer files from your actual student and teacher
checkpoints. Passing CPU tests does not establish sandbox or GPU compatibility.
Validate remote sandbox creation, task grading, and checkpoint/resume behavior
in your deployment before a full experiment.
