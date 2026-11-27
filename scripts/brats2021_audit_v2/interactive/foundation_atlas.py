#!/usr/bin/env python3
"""Independent Atlas-style dense post-processing without old-nnunet shadowing."""
import importlib
import importlib.util
from pathlib import Path
import sys

repository = Path(__file__).resolve().parents[3]
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != repository]
folder = repository / 'nnunetv2/utilities/brats2021_audit_v2/interactive'
name = 'brats_foundation_atlas_runtime'
spec = importlib.util.spec_from_file_location(name, folder/'__init__.py', submodule_search_locations=[str(folder)])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)

if __name__ == '__main__':
    importlib.import_module(name+'.foundation_prompt_v2.atlas_pipeline').main()
