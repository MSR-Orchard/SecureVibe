# SafeVibe SFT

Run repository-relative commands from `training/` unless stated otherwise.
Set external dependency paths explicitly for your installation.

This directory contains the SafeVibe supervised fine-tuning workflow for
Qwen3.5-35B-A3B using Slime and Megatron-LM.

## Layout

```text
slime_sft/
├── README.md
├── train.sh                              # single-dataset launcher
├── train_profiles.sh                              # sequential func, secu, and plan runs
├── qwen3.5-35B-A3B.sh                       # Megatron model arguments
├── train_async.py                          # asynchronous Slime training loop
└── export_checkpoints.sh   # checkpoint export
```

See the [data guide](../../data/README.md) for the prepared SFT mixture.
Model weights and converted checkpoints must be supplied separately.

## Requirements

The launcher expects:

- a Slime checkout configured with `SLIME_ROOT`;
- a Megatron-LM checkout configured with `MEGATRON_PATH`;
- a Qwen3.5-35B-A3B Hugging Face model and its converted Megatron reference
  checkpoint;
- Ray, Python, and the requested number of visible NVIDIA GPUs;
- `WANDB_API_KEY` when `WANDB_MODE=online`.

Use the exact SFT Slime revision:

```bash
git clone https://github.com/THUDM/slime.git slime
git -C slime checkout --detach 0988f0f4a0ab55d1bb3ce6285a597d912144fa80
```

`SLIME_ROOT` defaults to a sibling `slime` checkout. `MEGATRON_PATH` defaults
to `dependencies/megatron-lm/Megatron-LM`. Set `MEGATRON_PATH` explicitly to
a compatible checkout. The reference configuration used revision
`3714d81d418c9f1bca4594fc35f9e8289f652862` with additional SafeVibe changes.
The reference patch is included in `dependencies/megatron-lm/`. Follow the
[Megatron dependency guide](../../dependencies/megatron-lm/README.md) to verify
its checksum and apply it to the pinned revision before launching training.

## Run one dataset

The primary launcher defaults to four GPUs with TP=2, CP=2, EP=4, and PP=1.
All paths and topology values are environment-variable overrides.

```bash
cd <repository-root>/training

DATA="$(cd ../data/recipes && pwd)/sft_security_suite.jsonl" \
MODEL=/path/to/Qwen3.5-35B-A3B \
REF_LOAD=/path/to/Qwen3.5-35B-A3B_torch_dist_slime-0.3.0 \
OUTPUT_DIR=/path/to/checkpoints/func-qwen35-sft \
WANDB_MODE=offline \
bash slime_sft/train.sh
```

For all eight GPUs, retain the tested per-replica topology and select eight
actor GPUs:

```bash
ACTOR_GPUS_PER_NODE=8 bash slime_sft/train.sh
```

Frequently used overrides include `NUM_EPOCH`, `ROLLOUT_BATCH_SIZE`,
`ACTOR_NUM_NODES`, `ACTOR_GPUS_PER_NODE`, `TP_SIZE`, `CP_SIZE`, `EP_SIZE`,
`PP_SIZE`, `SAVE_INTERVAL`, `SAVE_HF`, `WANDB_PROJECT`, `WANDB_NAME`, and
`WANDB_ENTITY`.

The launcher stops the local Ray runtime before starting a fresh head node.
Run it from a long-lived shell or `tmux` session.

## Run all training profiles

`train_profiles.sh` trains the functional, security, and planning datasets
sequentially. Its default data root is
`dataset/train_recipe/SFT/0713_sft`, and it defaults to GPUs 4–7. Set
`CHECKPOINT_ROOT` explicitly so generated checkpoints remain outside the
repository.

```bash
cd <repository-root>/training
DATA_ROOT=/path/to/sft-datasets \
MODEL=/path/to/Qwen3.5-35B-A3B \
REF_LOAD=/path/to/Qwen3.5-35B-A3B_torch_dist \
CHECKPOINT_ROOT=/path/to/checkpoints \
WANDB_MODE=offline \
bash slime_sft/train_profiles.sh
```

Override `DATA_ROOT`, `CHECKPOINT_ROOT`, `RUN_SUFFIX`,
`CUDA_VISIBLE_DEVICES`, or `RUN_SCRIPT` as needed.

## Convert checkpoints to Hugging Face

The conversion tool selects the latest Megatron iteration unless `ITERATION`
is supplied:

```bash
cd <repository-root>/training
CHECKPOINT_DIR=/path/to/megatron-checkpoint \
ORIGIN_HF_DIR=/path/to/Qwen3.5-35B-A3B \
bash slime_sft/export_checkpoints.sh
```

Use `DRY_RUN=1` to inspect the conversion command. Run the tool with `--help`
for conversion and output-path overrides.
