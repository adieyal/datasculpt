"""Reporting adapter for generating profiling reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasculpt.adapters.base import AdapterResult, BaseAdapter, safe_import

if TYPE_CHECKING:
    import pandas as pd

    from datasculpt.core.types import ColumnEvidence

# Attempt to import ydata-profiling (formerly pandas-profiling)
_ydata_profiling = safe_import("ydata_profiling")

AVAILABLE: bool = _ydata_profiling is not None


class ReportingAdapter(BaseAdapter):
    """Adapter for generating profiling reports with ydata-profiling."""

    @property
    def name(self) -> str:
        return "reporting"

    @property
    def available(self) -> bool:
        return AVAILABLE

    def profile(self, df: pd.DataFrame) -> AdapterResult:
        """Profile a DataFrame using ydata-profiling.

        Args:
            df: The DataFrame to profile.

        Returns:
            AdapterResult with profiling annotations.
        """
        if not AVAILABLE:
            return AdapterResult(
                warnings=["ydata-profiling not installed; skipping adapter"]
            )

        return _profile_with_ydata(df)


def _profile_with_ydata(df: pd.DataFrame) -> AdapterResult:
    """Internal profiling implementation using ydata-profiling."""
    from ydata_profiling import ProfileReport

    column_annotations: dict[str, dict[str, Any]] = {}
    dataset_annotations: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        profile = ProfileReport(df, minimal=True, progress_bar=False)
        description = profile.get_description()

        # Extract table-level statistics
        try:
            table_stats = description.table
            dataset_annotations["ydata_table_stats"] = {
                "n_rows": table_stats.get("n", 0),
                "n_columns": table_stats.get("n_var", 0),
                "n_missing_cells": table_stats.get("n_cells_missing", 0),
                "n_duplicates": table_stats.get("n_duplicates", 0),
            }
        except AttributeError:
            pass

        # Extract per-column statistics
        try:
            for col_name, col_desc in description.variables.items():
                column_annotations[str(col_name)] = {
                    "ydata_type": col_desc.get("type", "Unknown"),
                    "ydata_n_distinct": col_desc.get("n_distinct"),
                    "ydata_n_missing": col_desc.get("n_missing"),
                    "ydata_is_unique": col_desc.get("is_unique", False),
                }
        except AttributeError:
            pass

    except Exception as e:
        warnings.append(f"ydata-profiling profiling failed: {e}")

    return AdapterResult(
        column_annotations=column_annotations,
        dataset_annotations=dataset_annotations,
        warnings=warnings,
    )


def generate_html_report(df: pd.DataFrame, output_path: str) -> bool:
    """Generate an HTML profiling report.

    Args:
        df: The DataFrame to profile.
        output_path: Path to write the HTML report.

    Returns:
        True if report was generated, False otherwise.
    """
    if not AVAILABLE:
        return False

    from ydata_profiling import ProfileReport

    try:
        profile = ProfileReport(df, minimal=False)
        profile.to_file(output_path)
        return True
    except Exception:
        return False


def generate_json_report(df: pd.DataFrame) -> dict[str, Any] | None:
    """Generate a JSON profiling report.

    Args:
        df: The DataFrame to profile.

    Returns:
        Report dict if available, None otherwise.
    """
    if not AVAILABLE:
        return None

    from ydata_profiling import ProfileReport

    try:
        profile = ProfileReport(df, minimal=True, progress_bar=False)
        return profile.to_json()
    except Exception:
        return None


def get_variable_summary(df: pd.DataFrame, column: str) -> dict[str, Any] | None:
    """Get a summary for a specific variable.

    Args:
        df: The DataFrame containing the column.
        column: The column name to summarize.

    Returns:
        Summary dict if available, None otherwise.
    """
    if not AVAILABLE:
        return None

    if column not in df.columns:
        return None

    from ydata_profiling import ProfileReport

    try:
        profile = ProfileReport(df[[column]], minimal=True, progress_bar=False)
        description = profile.get_description()

        try:
            if column in description.variables:
                return dict(description.variables[column])
        except AttributeError:
            pass
        return None
    except Exception:
        return None


def enrich_evidence_from_profile(
    df: pd.DataFrame,
    column_evidence: dict[str, ColumnEvidence],
) -> dict[str, ColumnEvidence]:
    """Enrich column evidence with data from ydata-profiling.

    Extracts additional statistics and type information from a ydata-profiling
    report and merges them into existing column evidence.

    Args:
        df: The DataFrame to profile.
        column_evidence: Dictionary of existing column evidence to enrich.

    Returns:
        Updated column evidence dictionary with ydata-profiling data.
    """
    if not AVAILABLE:
        return column_evidence

    from ydata_profiling import ProfileReport

    try:
        profile = ProfileReport(df, minimal=True, progress_bar=False)
        description = profile.get_description()

        try:
            variables = description.variables
        except AttributeError:
            return column_evidence

        for col_name, col_desc in variables.items():
            col_name_str = str(col_name)
            if col_name_str not in column_evidence:
                continue

            evidence = column_evidence[col_name_str]

            # Store ydata-profiling data in external dict
            ydata_data: dict[str, Any] = {
                "type": col_desc.get("type", "Unknown"),
                "n_distinct": col_desc.get("n_distinct"),
                "n_missing": col_desc.get("n_missing"),
                "p_missing": col_desc.get("p_missing"),
                "is_unique": col_desc.get("is_unique", False),
                "n_unique": col_desc.get("n_unique"),
                "p_unique": col_desc.get("p_unique"),
            }

            # Extract numeric statistics if available
            if "mean" in col_desc:
                ydata_data["mean"] = col_desc.get("mean")
                ydata_data["std"] = col_desc.get("std")
                ydata_data["min"] = col_desc.get("min")
                ydata_data["max"] = col_desc.get("max")
                ydata_data["median"] = col_desc.get("median")

            # Extract histogram data if available
            if "histogram" in col_desc:
                ydata_data["histogram"] = col_desc.get("histogram")

            # Update evidence external data
            evidence.external["ydata_profiling"] = ydata_data

            # Update null rate if ydata-profiling provides more accurate data
            p_missing = col_desc.get("p_missing")
            if p_missing is not None:
                evidence.null_rate = float(p_missing)

            # Update distinct ratio if available
            p_unique = col_desc.get("p_unique")
            if p_unique is not None:
                evidence.distinct_ratio = float(p_unique)

            # Add a note about enrichment
            evidence.notes.append("Enriched with ydata-profiling statistics")

    except Exception:
        # Silently fail - enrichment is best-effort
        pass

    return column_evidence


# Singleton adapter instance
_adapter = ReportingAdapter()


def get_adapter() -> ReportingAdapter:
    """Get the reporting adapter instance."""
    return _adapter
