"""DataProfiler adapter for statistical profiling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasculpt.adapters.base import AdapterResult, BaseAdapter, safe_import
from datasculpt.core.types import ColumnEvidence, PrimitiveType

if TYPE_CHECKING:
    import pandas as pd

# Attempt to import dataprofiler
_dataprofiler = safe_import("dataprofiler")

AVAILABLE: bool = _dataprofiler is not None


class DataProfilerAdapter(BaseAdapter):
    """Adapter for DataProfiler statistical analysis."""

    @property
    def name(self) -> str:
        return "dataprofiler"

    @property
    def available(self) -> bool:
        return AVAILABLE

    def profile(self, df: pd.DataFrame) -> AdapterResult:
        """Profile a DataFrame using DataProfiler.

        Args:
            df: The DataFrame to profile.

        Returns:
            AdapterResult with statistical annotations.
        """
        if not AVAILABLE:
            return AdapterResult(
                warnings=["dataprofiler not installed; skipping adapter"]
            )

        from dataprofiler import Data, Profiler

        return _profile_with_dataprofiler(df, Data, Profiler)


def _profile_with_dataprofiler(
    df: pd.DataFrame,
    Data: type,
    Profiler: type,
) -> AdapterResult:
    """Internal profiling implementation using dataprofiler.

    Args:
        df: The DataFrame to profile.
        Data: The dataprofiler Data class.
        Profiler: The dataprofiler Profiler class.

    Returns:
        AdapterResult with statistical annotations.
    """
    column_annotations: dict[str, dict[str, Any]] = {}
    dataset_annotations: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        data = Data(df)
        profiler = Profiler(data)
        report = profiler.report(report_options={"output_format": "serializable"})

        # Extract global statistics
        if "global_stats" in report:
            dataset_annotations["dataprofiler_global"] = report["global_stats"]

        # Extract per-column statistics
        if "data_stats" in report:
            for col_stats in report["data_stats"]:
                col_name = col_stats.get("column_name")
                if col_name:
                    column_annotations[col_name] = {
                        "dataprofiler_type": col_stats.get("data_type"),
                        "dataprofiler_statistics": col_stats.get("statistics", {}),
                        "dataprofiler_null_count": col_stats.get("null_count"),
                        "dataprofiler_sample_size": col_stats.get("sample_size"),
                    }

    except Exception as e:
        warnings.append(f"dataprofiler profiling failed: {e}")

    return AdapterResult(
        column_annotations=column_annotations,
        dataset_annotations=dataset_annotations,
        warnings=warnings,
    )


def get_column_statistics(df: pd.DataFrame, column: str) -> dict[str, Any] | None:
    """Get detailed statistics for a single column.

    Args:
        df: The DataFrame containing the column.
        column: The column name to analyze.

    Returns:
        Statistics dict if available, None otherwise.
    """
    if not AVAILABLE:
        return None

    if column not in df.columns:
        return None

    from dataprofiler import Data, Profiler

    try:
        data = Data(df[[column]])
        profiler = Profiler(data)
        report = profiler.report(report_options={"output_format": "serializable"})

        if "data_stats" in report and report["data_stats"]:
            return report["data_stats"][0]
        return None
    except Exception:
        return None


def detect_data_type(df: pd.DataFrame, column: str) -> str | None:
    """Detect the data type for a column using DataProfiler.

    Args:
        df: The DataFrame containing the column.
        column: The column name to analyze.

    Returns:
        Detected type string if available, None otherwise.
    """
    if not AVAILABLE:
        return None

    stats = get_column_statistics(df, column)
    if stats:
        return stats.get("data_type")
    return None


# Singleton adapter instance
_adapter = DataProfilerAdapter()


def get_adapter() -> DataProfilerAdapter:
    """Get the dataprofiler adapter instance."""
    return _adapter


# Mapping from DataProfiler types to PrimitiveType
DATAPROFILER_TYPE_MAP: dict[str, PrimitiveType] = {
    "int": PrimitiveType.INTEGER,
    "float": PrimitiveType.NUMBER,
    "string": PrimitiveType.STRING,
    "datetime": PrimitiveType.DATETIME,
    "bool": PrimitiveType.BOOLEAN,
    "text": PrimitiveType.STRING,
    "categorical": PrimitiveType.STRING,
    "unknown": PrimitiveType.UNKNOWN,
}


def map_dataprofiler_type(dp_type: str | None) -> PrimitiveType:
    """Map a DataProfiler type to a PrimitiveType.

    Args:
        dp_type: The DataProfiler type string.

    Returns:
        Corresponding PrimitiveType.
    """
    if dp_type is None:
        return PrimitiveType.UNKNOWN
    return DATAPROFILER_TYPE_MAP.get(dp_type.lower(), PrimitiveType.UNKNOWN)


def enrich_evidence_from_statistics(
    evidence: ColumnEvidence,
    stats: dict[str, Any],
) -> ColumnEvidence:
    """Enrich ColumnEvidence with DataProfiler statistics.

    Args:
        evidence: The existing ColumnEvidence to enrich.
        stats: The DataProfiler statistics dict for this column.

    Returns:
        The enriched ColumnEvidence (modified in place).
    """
    if not stats:
        return evidence

    dp_type = stats.get("data_type")
    statistics = stats.get("statistics", {})
    null_count = stats.get("null_count")
    sample_size = stats.get("sample_size")

    # Build external data structure
    external_data: dict[str, Any] = {
        "type": dp_type,
        "inferred_primitive": map_dataprofiler_type(dp_type).value,
    }

    # Add relevant statistics
    if statistics:
        # Common statistics
        if "min" in statistics:
            external_data["min"] = statistics["min"]
        if "max" in statistics:
            external_data["max"] = statistics["max"]
        if "mean" in statistics:
            external_data["mean"] = statistics["mean"]
        if "stddev" in statistics:
            external_data["stddev"] = statistics["stddev"]
        if "variance" in statistics:
            external_data["variance"] = statistics["variance"]
        if "median" in statistics:
            external_data["median"] = statistics["median"]

        # Cardinality info
        if "unique_count" in statistics:
            external_data["unique_count"] = statistics["unique_count"]
        if "unique_ratio" in statistics:
            external_data["unique_ratio"] = statistics["unique_ratio"]

        # String-specific stats
        if "avg_length" in statistics:
            external_data["avg_length"] = statistics["avg_length"]
        if "min_length" in statistics:
            external_data["min_length"] = statistics["min_length"]
        if "max_length" in statistics:
            external_data["max_length"] = statistics["max_length"]

        # Histogram data (summarized)
        if "histogram" in statistics and statistics["histogram"]:
            external_data["has_histogram"] = True

    # Add null info
    if null_count is not None:
        external_data["null_count"] = null_count
    if sample_size is not None:
        external_data["sample_size"] = sample_size

    evidence.external["dataprofiler"] = external_data

    # Add informative notes
    if dp_type and dp_type.lower() not in ("string", "unknown"):
        evidence.notes.append(f"dataprofiler detected type: {dp_type}")

    if statistics.get("unique_ratio", 0) > 0.95:
        evidence.notes.append("dataprofiler: high cardinality (>95% unique)")

    return evidence
