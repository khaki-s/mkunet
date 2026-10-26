"""Compatibility import; canonical implementation: agent_v1.trainer."""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module(".agent_v1.trainer", __package__)
