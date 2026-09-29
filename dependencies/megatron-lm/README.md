# Megatron-LM dependency

This bundle is copied from the main SecureVibe repository. It captures the
tracked local modifications from the reference Megatron checkout, originally
exported for the security coding-agent OPD experiment. The SFT workflow also
references this base revision and its SecureVibe changes.

## Version

- Upstream repository: `https://github.com/NVIDIA/Megatron-LM.git`
- Base commit: `3714d81d418c9f1bca4594fc35f9e8289f652862`
- Git description before export: `core_v0.15.0rc7-548-g3714d81d4-dirty`
- Installed package: `megatron-core==0.16.0rc0`
- Patch: `megatron-3714d81d-local.patch`
- Patch SHA-256:
  `0f7295e5ea53275a4d0a1be4d4132c89b5707300d81aafc3b587c25bccc41048`

The patch modifies 18 tracked files with 275 insertions and 60 deletions. It
contains the complete tracked difference between the reference checkout and
the base commit. Untracked files and environment-level package changes are not
included.

## Create a matching checkout

Run from the SecureVibe repository root:

```bash
export MEGATRON_BUNDLE="$PWD/dependencies/megatron-lm"
(cd "$MEGATRON_BUNDLE" && shasum -a 256 -c SHA256SUMS)

git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_BUNDLE/Megatron-LM"
export MEGATRON_PATH="$MEGATRON_BUNDLE/Megatron-LM"
git -C "$MEGATRON_PATH" checkout --detach 3714d81d418c9f1bca4594fc35f9e8289f652862

git -C "$MEGATRON_PATH" apply --check "$MEGATRON_BUNDLE/megatron-3714d81d-local.patch"
git -C "$MEGATRON_PATH" apply "$MEGATRON_BUNDLE/megatron-3714d81d-local.patch"
python -m pip install -e "$MEGATRON_PATH"
```

For an existing checkout, set `MEGATRON_PATH` to its absolute path instead of
cloning. Apply the patch once, on a clean checkout of the pinned revision.

## Verify

```bash
test "$(git -C "$MEGATRON_PATH" rev-parse HEAD)" = \
  3714d81d418c9f1bca4594fc35f9e8289f652862
git -C "$MEGATRON_PATH" apply --reverse --check "$MEGATRON_BUNDLE/megatron-3714d81d-local.patch"
python -c 'from importlib.metadata import version; print(version("megatron-core"))'
```

The package command should print `0.16.0rc0`. A dirty Git status is expected.
The original patch is preserved byte-for-byte, including six added lines with
trailing whitespace; applying it may report whitespace warnings.

The patch was checked against the 18 affected files at the pinned upstream
revision, and the patched Python files pass syntax parsing. This does not
validate GPU dependencies or establish that a full training run reproduces
the paper results. Untracked reference files and environment-level changes
are not part of the exported patch.

## License

The upstream Megatron-LM license at the pinned revision is preserved in
[LICENSE](LICENSE). Keep upstream source notices when using the patched checkout.

## Refreshing the bundle

If the reference Megatron checkout changes, regenerate the patch and update the
base commit, statistics, and checksum together:

```bash
export MEGATRON_SOURCE=/path/to/Megatron-LM
git -C "${MEGATRON_SOURCE}" diff HEAD --binary \
  --output="$PWD/dependencies/megatron-lm/megatron-3714d81d-local.patch"
sha256sum dependencies/megatron-lm/megatron-3714d81d-local.patch
```

Never clean or reset the reference checkout until the refreshed patch has been
validated against a clean clone.
