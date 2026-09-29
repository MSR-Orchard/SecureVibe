# AutoBax training harness

This directory provides `in_container_runner.py`, registered scenarios, fixtures,
and `autobax-harness.tar.gz` for SecureVibe RL and OPD rewards. Extracted source is
ready to use. The reward modules upload it to grading sandboxes when a runner
is not already available in the grading image.

## Usage

Follow the [RL](../../training/slime_rl/README.md) or
[OPD](../../training/slime_opd/README.md) training guide. Both workflows default
to this directory. To use another source tree, set `AUTOBAX_SRC_DIR` to its
absolute path on every training worker.

OPD also supports an archive served at a URL reachable from grading sandboxes:

```bash
export AUTOBAX_HARNESS_URL=https://artifacts.example.org/autobax-harness.tar.gz
export AUTOBAX_HARNESS_SHA256=5b9069a833924ce252ffa1d67e700764e897486d58ff456eba15b1509ea8f9e1
```

Use that digest only with the archive included here. A replacement archive
requires its own verified digest.

## Integrity

The archive's SHA-256 is its version identifier; no upstream Git commit is
recorded. From this directory:

```bash
shasum -a 256 autobax-harness.tar.gz
```

Expected digest:
`5b9069a833924ce252ffa1d67e700764e897486d58ff456eba15b1509ea8f9e1`.
To verify all distributed files, run from `dependencies/`:

```bash
shasum -a 256 -c autobax_arc.SHA256SUMS
```

## Archive format

Files are rooted at the top of the archive, including `in_container_runner.py`.
The packaged source includes scenario fixtures and compatibility adaptations
for Python type aliases and deferred annotations. Generated caches and
credentials are excluded from the archive.

When replacing the harness, package a clean source directory, verify the
in-container runner against representative functional and security tests,
and update the archive digest and source manifest together.
