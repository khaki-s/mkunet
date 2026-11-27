#!/usr/bin/env python3
"""Private adapter package; preserve installed official nnunetv2 for nnInteractive."""
import importlib
import importlib.util
from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[3]
sys.path[:] = [x for x in sys.path if Path(x or '.').resolve() != repository]
folder = repository / 'nnunetv2/utilities/brats2021_audit_v2/interactive'
name = 'brats_foundation_prompt_runtime'
spec = importlib.util.spec_from_file_location(name, folder / '__init__.py', submodule_search_locations=[str(folder)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)

if __name__ == '__main__':
    raise SystemExit(importlib.import_module(name + '.foundation_prompt_v2.cli').main())
