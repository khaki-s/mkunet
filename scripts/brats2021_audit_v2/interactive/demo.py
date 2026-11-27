#!/usr/bin/env python
"""Run the standalone prompt demo without shadowing official nnInteractive dependencies."""
import importlib
import importlib.util
from pathlib import Path
import sys


def main():
    repository = Path(__file__).resolve().parents[3]
    # The dedicated demo environment owns its upstream nnunetv2 dependency.
    # Import only our prompt package under a private name, not the old training tree.
    sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != repository]
    package = repository / 'nnunetv2/utilities/brats2021_audit_v2/interactive'
    name = 'brats_prompt_demo_runtime'
    spec = importlib.util.spec_from_file_location(name, package / '__init__.py',
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    importlib.import_module(name + '.demo').main()


if __name__ == '__main__':
    main()
