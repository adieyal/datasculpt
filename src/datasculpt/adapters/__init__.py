"""Optional profiler adapters for Datasculpt.

This module provides graceful degradation for optional dependencies.
Imports never fail - each adapter exposes an AVAILABLE flag to check
if the dependency is installed.

Usage:
    from datasculpt.adapters import frictionless

    if frictionless.AVAILABLE:
        schema = frictionless.infer_schema(df)
    else:
        # Fallback to core inference
        ...
"""

from datasculpt.adapters import dataprofiler, frictionless, reporting
from datasculpt.adapters.base import (
    AdapterRegistry,
    AdapterResult,
    BaseAdapter,
    get_registry,
    safe_import,
)

# Register available adapters
_registry = get_registry()
_registry.register(frictionless.get_adapter())
_registry.register(dataprofiler.get_adapter())
_registry.register(reporting.get_adapter())

__all__ = [
    # Base classes
    "AdapterRegistry",
    "AdapterResult",
    "BaseAdapter",
    # Registry access
    "get_registry",
    "safe_import",
    # Adapter modules (safe to import, check AVAILABLE flag)
    "dataprofiler",
    "frictionless",
    "reporting",
]
