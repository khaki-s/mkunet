#!/usr/bin/env python3
"""Standalone optional AtlasSegFM post-fusion ablation (official nnInteractive env)."""
from pathlib import Path
import importlib.util
import sys


def main(argv=None):
    repo = Path(__file__).resolve().parents[3]
    # Do not let this older research checkout shadow the environment's nnunetv2.
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != repo]
    source = repo / 'nnunetv2/utilities/brats2021_audit_v2/interactive/foundation_prompt_v2/atlas_adapter.py'
    name = '_brats_foundation_atlas_adapter'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.main(argv)


if __name__ == '__main__':
    main()
