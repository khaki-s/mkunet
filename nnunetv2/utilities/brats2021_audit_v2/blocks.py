"""Compatibility import; canonical implementation: baseline.blocks."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module(".baseline.blocks", __package__)
