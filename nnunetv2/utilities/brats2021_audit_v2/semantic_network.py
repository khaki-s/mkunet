"""Compatibility import; canonical implementation: semantic_moe.network."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module(".semantic_moe.network", __package__)
