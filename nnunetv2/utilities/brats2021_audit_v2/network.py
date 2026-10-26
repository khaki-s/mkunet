"""Compatibility import; canonical implementation: baseline.network."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module(".baseline.network", __package__)
