# SecureVibe training datasets

Prepared SFT, RL, and OPD inputs live under `recipes/`.
Dataset access and use remain subject to upstream terms.

The prepared recipes are published under `recipes/` in the public
[SecureVibe dataset on Hugging Face](https://huggingface.co/datasets/dqwang122/SafeVibe/tree/main/recipes).

## Integrity

Run `shasum -a 256 -c SHA256SUMS` from this directory to verify the prepared
training files. 

## Selected prepared training inputs

The flat `recipes/` directory contains:

- `sft_security_suite.jsonl`: combined functional, security, and planning SFT mixture.
- `rl_patcheval.train.jsonl`: default RL PatchEval input.
- `rl_autobax.train.jsonl`: default RL AutoBax input.
- `rl.val.jsonl`: RL functional validation input.
- `opd_hint.train.jsonl`: OPD training input with hint level 4.
- `opd.val.jsonl`: OPD functional validation input.

Filenames identify the training method, task family, and split. The source
manifest preserves upstream names independently of the repository's directory name.

From the repository’s `training/` directory, select the inputs with absolute paths:

```bash
RECIPE_DIR="$(cd ../data/recipes && pwd)"

# SFT
export DATA="${RECIPE_DIR}/sft_security_suite.jsonl"

# Joint RL
export JOINT_PATCHEVAL_DATA="${RECIPE_DIR}/rl_patcheval.train.jsonl"
export JOINT_AUTOBAX_DATA="${RECIPE_DIR}/rl_autobax.train.jsonl"

# OPD
export PROMPT_DATA="${RECIPE_DIR}/opd_hint.train.jsonl"
export EVAL_DATA="${RECIPE_DIR}/opd.val.jsonl"
```

The RL/OPD records contain `prompt`, `label`, and `metadata`. Original task and
configuration metadata are preserved; configure local runtime paths using the
training guides before launching. The RL validation file is available separately
for the chosen runner's validation configuration.
