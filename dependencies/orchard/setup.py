#!/usr/bin/env python3
"""Fetch pinned public sources and apply the verified SecureVibe patch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def git(path, *args, check=True):
    return subprocess.run(['git', '-C', str(path), *args], check=check,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def setup(destination):
    versions = json.loads((HERE / 'versions.json').read_text())
    patch = HERE / 'securevibe.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != versions['patch_sha256']:
        raise RuntimeError('Compatibility patch checksum mismatch')
    destination = destination.resolve()
    if not destination.exists():
        subprocess.run(['git', 'clone', '--no-checkout', versions['orchard_url'], str(destination)], check=True)
        git(destination, 'checkout', '--detach', versions['orchard_revision'])
    if git(destination, 'rev-parse', 'HEAD').stdout.strip() != versions['orchard_revision']:
        raise RuntimeError('Existing Orchard checkout has a different revision; choose a new destination')
    slime = destination / 'trainer/slime'
    if not (slime / '.git').exists():
        git(destination, 'submodule', 'update', '--init', 'trainer/slime')
    if git(slime, 'rev-parse', 'HEAD').stdout.strip() != versions['slime_revision']:
        raise RuntimeError('Slime revision differs from versions.json')
    if git(slime, 'apply', '--reverse', '--check', str(patch), check=False).returncode == 0:
        print('Compatibility patch is already applied')
    else:
        if git(slime, 'status', '--porcelain').stdout.strip():
            raise RuntimeError('Slime has local changes; refusing to patch it')
        git(slime, 'apply', '--check', str(patch))
        git(slime, 'apply', str(patch))
    print(f'SLIME_DIR={slime}')
    print(f'Install the SDK in your training environment: python -m pip install -e {destination / "orchard_env"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='External Orchard checkout directory')
    args = parser.parse_args()
    setup(args.destination)
