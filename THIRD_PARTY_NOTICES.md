# Third-party components

The root MIT license applies to SafeVibe's original code. It does not replace
the licenses or notices applicable to third-party code and assets.

- **mini-swe-agent:** bundled under `dependencies/vendor/mini-swe-agent/`.
  Its license is preserved in
  [LICENSE.md](dependencies/vendor/mini-swe-agent/LICENSE.md).
- **Megatron-LM:** the dependency bundle is documented in
  [dependencies/megatron-lm/README.md](dependencies/megatron-lm/README.md).
  The upstream [LICENSE](dependencies/megatron-lm/LICENSE) at the pinned
  revision is included. Retain that license and source notices when obtaining
  or redistributing the patched checkout.
- **Orchard and Slime:** obtained separately at the revisions recorded in
  [dependencies/orchard/versions.json](dependencies/orchard/versions.json).
  Retain their upstream license files and source notices. SafeVibe's
  compatibility patch does not replace those terms.
- **AutoBax harness:** bundled under `dependencies/autobax_arc/`, with snapshot
  provenance in [the dependency guide](dependencies/autobax_arc.README.md).
  [AutoBaxBuilder](https://github.com/eth-sri/AutoBaxBuilder) is MIT-licensed.
  Its copyright and permission notice is preserved in
  [LICENSE](dependencies/autobax_arc/LICENSE), copied verbatim from upstream
  revision `017319ac217a65632ad580bac1523715221f3dbd` on 2026-09-09.
  Copyright (c) 2025 Tobias von Arx, Mark Vero, Niels Mündler, Maximilian
  Baader, Martin Vechev. Preserve this notice when redistributing upstream
  code or substantial portions of it, including modified copies.
  This revision identifies the license source, not the original bundled
  harness revision; the local harness includes adaptations and its original
  upstream commit is not recorded. The license applies to AutoBaxBuilder
  code, not automatically to separately sourced assets or dependencies.

Benchmark datasets, model weights, and external services are governed by
their respective source terms and access requirements.
