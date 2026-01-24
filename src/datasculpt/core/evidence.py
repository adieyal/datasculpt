"""Evidence extraction module for column analysis.

This module provides functions to extract evidence about dataset columns,
including type detection, null rates, cardinality, and structural analysis.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from datasculpt.core.types import ColumnEvidence, PrimitiveType, StructuralType

if TYPE_CHECKING:
    from pandas import Series


# Common date formats to try when parsing strings as dates
DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d/%m/%Y",
    "%m-%d-%Y",
    "%m/%d/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
]

# Patterns for header date detection
HEADER_DATE_PATTERNS = [
    re.compile(r"^\d{4}$"),  # 2024
    re.compile(r"^\d{4}-\d{2}$"),  # 2024-01
    re.compile(r"^\d{4}/\d{2}$"),  # 2024/01
    re.compile(r"^\d{4}-Q[1-4]$"),  # 2024-Q1
    re.compile(r"^Q[1-4]-\d{4}$"),  # Q1-2024
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),  # 2024-01-15
    re.compile(r"^\d{2}/\d{2}/\d{4}$"),  # 01/15/2024
    re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[- ]\d{4}$", re.I),
    re.compile(r"^\d{4}[- ](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)$", re.I),
]


def detect_primitive_type(series: Series) -> PrimitiveType:
    """Detect the primitive type of a pandas Series.

    Args:
        series: A pandas Series to analyze.

    Returns:
        The detected PrimitiveType for the column.
    """
    # Drop null values for type detection
    non_null = series.dropna()

    if len(non_null) == 0:
        return PrimitiveType.UNKNOWN

    dtype = series.dtype

    # Check pandas dtype first
    if pd.api.types.is_bool_dtype(dtype):
        return PrimitiveType.BOOLEAN

    if pd.api.types.is_integer_dtype(dtype):
        return PrimitiveType.INTEGER

    if pd.api.types.is_float_dtype(dtype):
        # Check if all non-null values are actually integers
        if non_null.apply(lambda x: float(x).is_integer()).all():
            return PrimitiveType.INTEGER
        return PrimitiveType.NUMBER

    if pd.api.types.is_datetime64_any_dtype(dtype):
        # Check if any values have time components
        if hasattr(non_null.dt, "time"):
            has_time = non_null.dt.time.apply(
                lambda t: t.hour != 0 or t.minute != 0 or t.second != 0
            ).any()
            if has_time:
                return PrimitiveType.DATETIME
        return PrimitiveType.DATE

    # For object dtype, inspect actual values
    if dtype == object:
        return _detect_type_from_values(non_null)

    return PrimitiveType.STRING


def _detect_type_from_values(series: Series) -> PrimitiveType:
    """Detect primitive type by inspecting actual values.

    Args:
        series: A pandas Series with object dtype.

    Returns:
        The detected PrimitiveType.
    """
    # Sample for efficiency on large datasets
    sample = series.head(1000)

    # Check for boolean strings
    bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
    str_values = sample.astype(str).str.lower().unique()
    if set(str_values).issubset(bool_values):
        return PrimitiveType.BOOLEAN

    # Check for integers
    try:
        numeric = pd.to_numeric(sample, errors="coerce")
        if numeric.notna().all():
            if numeric.apply(lambda x: float(x).is_integer()).all():
                return PrimitiveType.INTEGER
            return PrimitiveType.NUMBER
    except (ValueError, TypeError):
        pass

    # Check for dates/datetimes
    date_result = attempt_date_parse(sample)
    if date_result["success_rate"] > 0.9:
        if date_result.get("has_time", False):
            return PrimitiveType.DATETIME
        return PrimitiveType.DATE

    return PrimitiveType.STRING


def compute_null_rate(series: Series) -> float:
    """Compute the null rate for a column.

    Args:
        series: A pandas Series to analyze.

    Returns:
        The ratio of null values (0.0 to 1.0).
    """
    if len(series) == 0:
        return 0.0

    null_count = series.isna().sum()
    return float(null_count / len(series))


def compute_distinct_ratio(series: Series) -> float:
    """Compute the distinct ratio (cardinality) for a column.

    The distinct ratio is the number of unique values divided by total non-null values.

    Args:
        series: A pandas Series to analyze.

    Returns:
        The ratio of unique values to total non-null values (0.0 to 1.0).
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return 0.0

    unique_count = non_null.nunique()
    return float(unique_count / len(non_null))


def attempt_date_parse(series: Series) -> dict[str, float | bool | str | None]:
    """Attempt to parse a series as dates and record success rate.

    Tries specific known formats first for reliability and performance,
    only falling back to auto-detection if no format achieves good results.

    Args:
        series: A pandas Series to analyze.

    Returns:
        A dictionary containing:
        - success_rate: Ratio of successfully parsed values
        - has_time: Whether any parsed dates have time components
        - best_format: The format string that worked best, if any
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return {"success_rate": 0.0, "has_time": False, "best_format": None}

    # Sample for efficiency
    sample = non_null.head(1000)
    sample_str = sample.astype(str)

    best_success_rate = 0.0
    best_format: str | None = None
    has_time = False

    # Try specific formats first (preferred - explicit and faster)
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(sample_str, format=fmt, errors="coerce")
            success_count = parsed.notna().sum()
            success_rate = float(success_count / len(sample))

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_format = fmt
                has_time = "%H" in fmt or "%M" in fmt or "%S" in fmt

            # Early exit if we found a perfect match
            if success_rate >= 0.99:
                break
        except (ValueError, TypeError):
            continue

    # Only fall back to auto-detection if specific formats didn't work well
    # This avoids the "Could not infer format" warning in most cases
    if best_success_rate < 0.5:
        try:
            # Use format="mixed" (pandas 2.0+) to handle mixed formats gracefully
            parsed = pd.to_datetime(sample_str, format="mixed", errors="coerce")
            success_count = parsed.notna().sum()
            success_rate = float(success_count / len(sample))

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_format = "mixed"

                # Check for time components
                valid_times = parsed.dropna()
                if len(valid_times) > 0 and hasattr(valid_times.dt, "time"):
                    has_time = bool(
                        valid_times.dt.time.apply(
                            lambda t: t.hour != 0 or t.minute != 0 or t.second != 0
                        ).any()
                    )
        except (ValueError, TypeError):
            pass

    return {
        "success_rate": best_success_rate,
        "has_time": has_time,
        "best_format": best_format,
    }


def detect_json_array(series: Series) -> dict[str, float | bool]:
    """Detect if a column contains JSON arrays.

    Args:
        series: A pandas Series to analyze.

    Returns:
        A dictionary containing:
        - is_json_array: Whether the column appears to contain JSON arrays
        - success_rate: Ratio of values that are valid JSON arrays
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return {"is_json_array": False, "success_rate": 0.0}

    # Sample for efficiency
    sample = non_null.head(1000)

    success_count = 0
    for value in sample:
        if isinstance(value, list):
            success_count += 1
            continue

        if not isinstance(value, str):
            continue

        value_str = value.strip()
        if not value_str.startswith("["):
            continue

        try:
            parsed = json.loads(value_str)
            if isinstance(parsed, list):
                success_count += 1
        except (json.JSONDecodeError, ValueError):
            continue

    success_rate = float(success_count / len(sample))

    return {
        "is_json_array": success_rate > 0.8,
        "success_rate": success_rate,
    }


def detect_header_date(column_name: str) -> dict[str, bool | str | None]:
    """Check if a column name looks like a date.

    Args:
        column_name: The column name to check.

    Returns:
        A dictionary containing:
        - is_date: Whether the column name looks like a date
        - pattern: The pattern that matched, if any
    """
    name = str(column_name).strip()

    for pattern in HEADER_DATE_PATTERNS:
        if pattern.match(name):
            return {
                "is_date": True,
                "pattern": pattern.pattern,
            }

    return {
        "is_date": False,
        "pattern": None,
    }


def detect_structural_type(series: Series) -> StructuralType:
    """Detect the structural type of a column.

    Args:
        series: A pandas Series to analyze.

    Returns:
        The detected StructuralType.
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return StructuralType.UNKNOWN

    # Sample for efficiency
    sample = non_null.head(1000)

    array_count = 0
    object_count = 0

    for value in sample:
        # Check Python types directly
        if isinstance(value, list):
            array_count += 1
            continue
        if isinstance(value, dict):
            object_count += 1
            continue

        # Check string representations
        if isinstance(value, str):
            value_str = value.strip()
            if value_str.startswith("["):
                try:
                    parsed = json.loads(value_str)
                    if isinstance(parsed, list):
                        array_count += 1
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            elif value_str.startswith("{"):
                try:
                    parsed = json.loads(value_str)
                    if isinstance(parsed, dict):
                        object_count += 1
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass

    total = len(sample)
    array_ratio = array_count / total
    object_ratio = object_count / total

    if array_ratio > 0.8:
        return StructuralType.ARRAY
    if object_ratio > 0.8:
        return StructuralType.OBJECT

    return StructuralType.SCALAR


def extract_column_evidence(series: Series, column_name: str) -> ColumnEvidence:
    """Extract all evidence about a column.

    Args:
        series: A pandas Series containing the column data.
        column_name: The name of the column.

    Returns:
        A ColumnEvidence object containing all extracted evidence.
    """
    # Detect types
    primitive_type = detect_primitive_type(series)
    structural_type = detect_structural_type(series)

    # Compute statistics
    null_rate = compute_null_rate(series)
    distinct_ratio = compute_distinct_ratio(series)

    # Build parse results
    parse_results: dict[str, float] = {}
    notes: list[str] = []

    # Date parsing attempts (for string columns)
    if primitive_type in (PrimitiveType.STRING, PrimitiveType.DATE, PrimitiveType.DATETIME):
        date_result = attempt_date_parse(series)
        parse_results["date_parse"] = date_result["success_rate"]
        if date_result["best_format"]:
            notes.append(f"Date format: {date_result['best_format']}")

    # JSON array detection
    json_result = detect_json_array(series)
    parse_results["json_array"] = json_result["success_rate"]
    if json_result["is_json_array"]:
        notes.append("Contains JSON arrays")
        structural_type = StructuralType.ARRAY

    # Header date detection
    header_date = detect_header_date(column_name)
    if header_date["is_date"]:
        notes.append(f"Column name appears to be a date (pattern: {header_date['pattern']})")

    return ColumnEvidence(
        name=column_name,
        primitive_type=primitive_type,
        structural_type=structural_type,
        null_rate=null_rate,
        distinct_ratio=distinct_ratio,
        parse_results=parse_results,
        notes=notes,
    )


def extract_dataframe_evidence(df: pd.DataFrame) -> dict[str, ColumnEvidence]:
    """Extract evidence for all columns in a DataFrame.

    Args:
        df: A pandas DataFrame to analyze.

    Returns:
        A dictionary mapping column names to ColumnEvidence objects.
    """
    evidence: dict[str, ColumnEvidence] = {}

    for column in df.columns:
        evidence[str(column)] = extract_column_evidence(df[column], str(column))

    return evidence
