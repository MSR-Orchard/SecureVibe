# SecureVibe

SecureVibe provides training workflows for security-aware coding agents:
supervised fine-tuning (SFT), reinforcement learning (RL), and on-policy
distillation (OPD). Evaluation on SecureGen, AutoBax, BaxBench, and SusVibes
lives in the companion [SafeVibeEval project](https://github.com/MSR-Orchard/SafeVibeEval).

## Repository layout

- [`training/slime_sft/`](training/slime_sft/README.md): supervised fine-tuning
  and checkpoint export.
- [`training/slime_rl/`](training/slime_rl/README.md): joint PatchEval and
  AutoBax reinforcement learning.
- [`training/slime_opd/`](training/slime_opd/README.md): distillation with
  optional teacher-only security hints.
- [`data/`](data/README.md): training input inventories,
  source mappings, and checksums.
- [`dependencies/`](dependencies/orchard/README.md): pinned Orchard integration,
  bundled mini-swe-agent, and AutoBax training harness.

## Evaluate a model with SafeVibeEval

Use [SafeVibeEval](https://github.com/MSR-Orchard/SafeVibeEval) for model execution, multi-CLI agents,
and benchmark grading. SafeVibeEval has its own setup, dependencies, tests, and
raw-data manifests; it does not require the SecureVibe training stack.

Clone SafeVibeEval and follow its evaluation setup:

```bash
git clone https://github.com/MSR-Orchard/SafeVibeEval.git
cd SafeVibeEval/evaluation
./setup.sh
export MODEL=openai/your-model
export LOCAL_BASE=http://127.0.0.1:8200/v1
export OUTPUT_PREFIX=your-model
DRY_RUN=1 ./evaluate.sh
```

Follow the [SafeVibeEval quickstart](https://github.com/MSR-Orchard/SafeVibeEval) to configure model
and sandbox services, run evaluation, and grade the outputs. Evaluation inputs
now belong in `SafeVibeEval/data/raw/`; prepared training inputs remain here in
`data/recipes/`. Each project can be checked out independently.

## Train a model

Choose the [SFT](training/slime_sft/README.md),
[RL](training/slime_rl/README.md), or [OPD](training/slime_opd/README.md) guide.
Each workflow provides a `train.sh` launcher. Training requires NVIDIA GPUs,
compatible model checkpoints, prepared data, and the dependencies specified
in that workflow's guide.

RL and OPD use the [pinned Orchard integration](dependencies/orchard/README.md)
and a deployed sandbox service. OPD also requires a teacher endpoint; hinted
samples require matching student and teacher tokenizer vocabularies.
The [Megatron dependency guide](dependencies/megatron-lm/README.md) provides
the pinned revision, reference patch, checksum, and installation commands.

## Data and dependencies

Training recipes and SafeVibeEval evaluation inputs are published in the
[SecureVibe dataset on Hugging Face](https://huggingface.co/datasets/dqwang122/SafeVibe):
[`recipes/`](https://huggingface.co/datasets/dqwang122/SafeVibe/tree/main/recipes)
for SecureVibe and [`raw/`](https://huggingface.co/datasets/dqwang122/SafeVibe/tree/main/raw)
for SafeVibeEval.

Dataset files, model weights, credentials, and generated results are excluded
from version control. Data manifests and documentation are included so inputs
can be identified and verified. Access to source datasets and benchmark images
is subject to their respective access requirements and terms.

The [mini-swe-agent dependency guide](dependencies/vendor/README.md) and
[AutoBax harness guide](dependencies/autobax_arc.README.md) document bundled
sources and checksum verification. See [third-party notices](THIRD_PARTY_NOTICES.md)
for bundled licenses and attribution, including AutoBaxBuilder's MIT license.

## Development checks

Training dependency checks and deployment validation are documented in the
individual workflow guides. Evaluation tests belong to
[SafeVibeEval](https://github.com/MSR-Orchard/SafeVibeEval#development-checks).

## License

SecureVibe's original code is released under the [MIT License](LICENSE).
Third-party components retain their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md). The code license does not grant
access to or license benchmark datasets, model weights, or external services.
