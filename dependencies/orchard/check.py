#!/usr/bin/env python3
"""Verify the pinned, patched Slime checkout before starting training."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def check(slime, runtime=False):
    slime = slime.resolve()
    versions = json.loads((HERE / 'versions.json').read_text())
    patch = HERE / 'safevibe.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest() != versions['patch_sha256']:
        raise RuntimeError('SafeVibe patch checksum mismatch')
    revision = subprocess.check_output(['git','-C',str(slime),'rev-parse','HEAD'], text=True).strip()
    if revision != versions['slime_revision']:
        raise RuntimeError('Unexpected Slime revision; run dependencies/orchard/setup.py')
    subprocess.run(['git','-C',str(slime),'apply','--reverse','--check',str(patch)],check=True)
    if runtime:
        sys.path[:0] = [str(HERE.parents[1] / 'training'), str(slime)]
        for module in ('orchard_env.client.sandbox_client', 'slime_rl.joint_data_source',
                       'slime_rl.patcheval_reward', 'slime_rl.autobax_reward',
                       'slime_opd.generate', 'slime_opd.combined_reward'):
            importlib.import_module(module)
    print('Orchard compatibility check passed' + (' (runtime imports included)' if runtime else ''))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('slime',type=Path)
    parser.add_argument('--runtime',action='store_true',help='Also import training modules in the GPU environment')
    args=parser.parse_args()
    check(args.slime,args.runtime)
