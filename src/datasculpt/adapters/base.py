"""Base adapter interface and registry for optional profilers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from datasculpt.core.types import ColumnEvidence


@dataclass
class AdapterResult:
    """Result from an adapter's profile operation."""

    column_annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_annotations: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class BaseAdapter(ABC):
    """Base interface for optional profiler adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this adapter."""
        ...

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether the adapter's dependencies are installed."""
        ...

    @abstractmethod
    def profile(self, df: pd.DataFrame) -> AdapterResult:
        """Profile a DataFrame and return annotations.

        Args:
            df: The DataFrame to profile.

        Returns:
            AdapterResult with column and dataset annotations.
        """
        ...


class AdapterRegistry:
    """Registry for available adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}

    def register(self, adapter: BaseAdapter) -> None:
        """Register an adapter instance."""
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> BaseAdapter | None:
        """Get an adapter by name, or None if not registered."""
        return self._adapters.get(name)

    def available(self) -> list[str]:
        """List names of adapters that are available (deps installed)."""
        return [name for name, adapter in self._adapters.items() if adapter.available]

    def all(self) -> list[str]:
        """List all registered adapter names."""
        return list(self._adapters.keys())


# Global registry instance
_registry = AdapterRegistry()


def get_registry() -> AdapterRegistry:
    """Get the global adapter registry."""
    return _registry


def safe_import(module_name: str) -> Any | None:
    """Safely import a module, returning None if not available.

    Args:
        module_name: The fully qualified module name to import.

    Returns:
        The imported module, or None if import fails.
    """
    try:
        import importlib

        return importlib.import_module(module_name)
    except ImportError:
        return None


def enrich_evidence(
    evidence: ColumnEvidence,
    df: pd.DataFrame,
    adapters: list[str] | None = None,
) -> ColumnEvidence:
    """Enrich ColumnEvidence with data from available adapters.

    Only runs profiling if the adapter is available (dependencies installed).

    Args:
        evidence: The ColumnEvidence to enrich.
        df: The DataFrame containing the column data.
        adapters: List of adapter names to use. If None, uses all available adapters.

    Returns:
        The enriched ColumnEvidence (modified in place).
    """
    registry = get_registry()

    # Determine which adapters to use
    if adapters is None:
        adapter_names = registry.available()
    else:
        adapter_names = [name for name in adapters if name in registry.available()]

    for adapter_name in adapter_names:
        adapter = registry.get(adapter_name)
        if adapter is None or not adapter.available:
            continue

        _enrich_from_adapter(evidence, df, adapter_name)

    return evidence


def _enrich_from_adapter(
    evidence: ColumnEvidence,
    df: pd.DataFrame,
    adapter_name: str,
) -> None:
    """Enrich evidence using a specific adapter.

    Args:
        evidence: The ColumnEvidence to enrich.
        df: The DataFrame containing the column data.
        adapter_name: Name of the adapter to use.
    """
    if adapter_name == "frictionless":
        _enrich_from_frictionless(evidence, df)
    elif adapter_name == "dataprofiler":
        _enrich_from_dataprofiler(evidence, df)


def _enrich_from_frictionless(evidence: ColumnEvidence, df: pd.DataFrame) -> None:
    """Enrich evidence using the frictionless adapter.

    Args:
        evidence: The ColumnEvidence to enrich.
        df: The DataFrame containing the column data.
    """
    from datasculpt.adapters.frictionless import (
        AVAILABLE as FRICTIONLESS_AVAILABLE,
    )
    from datasculpt.adapters.frictionless import (
        enrich_evidence_from_schema,
        infer_schema,
    )

    if not FRICTIONLESS_AVAILABLE:
        return

    schema = infer_schema(df)
    if schema:
        enrich_evidence_from_schema(evidence, schema)


def _enrich_from_dataprofiler(evidence: ColumnEvidence, df: pd.DataFrame) -> None:
    """Enrich evidence using the dataprofiler adapter.

    Args:
        evidence: The ColumnEvidence to enrich.
        df: The DataFrame containing the column data.
    """
    from datasculpt.adapters.dataprofiler import (
        AVAILABLE as DATAPROFILER_AVAILABLE,
    )
    from datasculpt.adapters.dataprofiler import (
        enrich_evidence_from_statistics,
        get_column_statistics,
    )

    if not DATAPROFILER_AVAILABLE:
        return

    stats = get_column_statistics(df, evidence.name)
    if stats:
        enrich_evidence_from_statistics(evidence, stats)
