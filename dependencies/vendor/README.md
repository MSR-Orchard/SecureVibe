# mini-swe-agent dependency

`mini-swe-agent/` contains version 2.3.0 with SecureVibe sandbox integration.
Evaluation setup installs this source in editable mode. See the
[evaluation guide](../../evaluation/README.md) for installation and usage.
The [upstream README](mini-swe-agent/README.md) and
[license](mini-swe-agent/LICENSE.md) retain upstream documentation and attribution;
SecureVibe-specific entrypoints are documented in the evaluation guide.

Verify the source from this directory:

```bash
shasum -a 256 -c mini-swe-agent.SHA256SUMS
```

When updating the dependency, review the source changes and regenerate the
checksum manifest.
