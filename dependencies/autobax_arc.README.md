# AutoBax training dependency

`autobax_arc/` provides extracted grading source and a versioned harness archive
for RL and OPD training. Both reward modules use this source by default;
`AUTOBAX_SRC_DIR` selects an alternative directory accessible to every worker.
No extraction is required.

See the [harness guide](autobax_arc/README.md),
[RL guide](../training/slime_rl/README.md), and
[OPD guide](../training/slime_opd/README.md) for usage.
Standalone evaluation uses `evaluation/grader/autobax/`.

Verify the extracted source and archive from `dependencies/`:

```bash
shasum -a 256 -c autobax_arc.SHA256SUMS
```

The archive SHA-256 is
`5b9069a833924ce252ffa1d67e700764e897486d58ff456eba15b1509ea8f9e1`.
The checksum identifies the archive; no upstream Git revision is recorded.
