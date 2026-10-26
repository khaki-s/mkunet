"""Compatibility import; canonical implementation: agent_v1.diagnostics."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module(".agent_v1.diagnostics", __package__)
