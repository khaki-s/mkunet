#!/usr/bin/env python3
"""Load only our adapters, leaving official nnInteractive's nnunetv2 import intact."""
import importlib
import importlib.util
from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[3]
sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != repository]
package = repository / 'nnunetv2/utilities/brats2021_audit_v2/interactive'
name = 'brats_region_evaluation_runtime'
spec = importlib.util.spec_from_file_location(name, package / '__init__.py',
                                            submodule_search_locations=[str(package)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)

if __name__ == '__main__':
    raise SystemExit(importlib.import_module(name + '.evaluation').main())
