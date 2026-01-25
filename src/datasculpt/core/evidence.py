"""Evidence extraction module for column analysis.

This module provides functions to extract evidence about dataset columns,
including type detection, null rates, cardinality, and structural analysis.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from datasculpt.core.types import (
    ArrayProfile,
    ColumnEvidence,
    ParseResults,
    PrimitiveType,
    StructuralType,
    ValueProfile,
)

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

# Canonical boolean values (lowercase)
BOOL_VALUES = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}

# Sample size cap
SAMPLE_SIZE = 1000


def _sample_series(series: Series) -> Series:
    """Get a representative sample from a series.

    Samples first 500 non-null values plus 500 evenly spaced values
    to avoid head-only bias (metadata rows, sparse starts, etc.).

    Args:
        series: A pandas Series to sample.

    Returns:
        A representative sample of the series.
    """
    non_null = series.dropna()

    if len(non_null) <= SAMPLE_SIZE:
        return non_null

    # First 500 non-null values
    head_sample = non_null.head(500)

    # 500 evenly spaced from the rest
    remaining = non_null.iloc[500:]
    if len(remaining) > 500:
        indices = np.linspace(0, len(remaining) - 1, 500, dtype=int)
        spaced_sample = remaining.iloc[indices]
    else:
        spaced_sample = remaining

    return pd.concat([head_sample, spaced_sample]).drop_duplicates()


def detect_primitive_type(series: Series) -> tuple[PrimitiveType, list[str]]:
    """Detect the primitive type of a pandas Series.

    Args:
        series: A pandas Series to analyze.

    Returns:
        Tuple of (detected PrimitiveType, list of notes about the detection).
    """
    notes: list[str] = []
    non_null = series.dropna()

    if len(non_null) == 0:
        return PrimitiveType.UNKNOWN, notes

    dtype = series.dtype

    # Check pandas dtype first
    if pd.api.types.is_bool_dtype(dtype):
        return PrimitiveType.BOOLEAN, notes

    if pd.api.types.is_integer_dtype(dtype):
        return PrimitiveType.INTEGER, notes

    if pd.api.types.is_float_dtype(dtype):
        sample = _sample_series(non_null)
        # Check if all non-null values are actually integers
        is_integer = sample.apply(lambda x: float(x).is_integer()).all()
        if is_integer:
            notes.append("Originally float dtype; all sampled values are whole numbers")
            return PrimitiveType.INTEGER, notes
        return PrimitiveType.NUMBER, notes

    if pd.api.types.is_datetime64_any_dtype(dtype):
        # Check if any values have time components
        if hasattr(non_null.dt, "time"):
            has_time = non_null.dt.time.apply(
                lambda t: t.hour != 0 or t.minute != 0 or t.second != 0
            ).any()
            if has_time:
                return PrimitiveType.DATETIME, notes
        return PrimitiveType.DATE, notes

    # For object dtype, inspect actual values
    if dtype is object:
        return _detect_type_from_values(non_null)

    return PrimitiveType.STRING, notes


def _detect_type_from_values(series: Series) -> tuple[PrimitiveType, list[str]]:
    """Detect primitive type by inspecting actual values.

    Args:
        series: A pandas Series with object dtype.

    Returns:
        Tuple of (detected PrimitiveType, list of notes).
    """
    notes: list[str] = []
    sample = _sample_series(series)

    # Check for boolean strings with strict rules:
    # 1. unique values must be subset of BOOL_VALUES
    # 2. unique_count must be <= 2
    # 3. >= 90% of values must be in BOOL_VALUES
    str_values = sample.astype(str).str.lower()
    unique_values = set(str_values.unique())
    unique_count = len(unique_values)

    if unique_values.issubset(BOOL_VALUES) and unique_count <= 2:
        in_bool_count = str_values.isin(BOOL_VALUES).sum()
        bool_ratio = in_bool_count / len(sample)
        if bool_ratio >= 0.9:
            return PrimitiveType.BOOLEAN, notes
        else:
            # Not quite boolean, but note the binary-like pattern
            notes.append(f"Binary-coded values detected but only {bool_ratio:.0%} match")

    # Check for integers
    try:
        numeric = pd.to_numeric(sample, errors="coerce")
        if numeric.notna().all():
            if numeric.apply(lambda x: float(x).is_integer()).all():
                return PrimitiveType.INTEGER, notes
            return PrimitiveType.NUMBER, notes
    except (ValueError, TypeError):
        pass

    # Check for dates/datetimes
    date_result = attempt_date_parse(sample)
    success_rate = date_result["success_rate"]
    if isinstance(success_rate, float) and success_rate > 0.9:
        has_time = date_result.get("has_time", False)
        if has_time:
            return PrimitiveType.DATETIME, notes
        return PrimitiveType.DATE, notes

    return PrimitiveType.STRING, notes


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


def compute_unique_count(series: Series) -> int:
    """Compute the number of unique non-null values.

    Args:
        series: A pandas Series to analyze.

    Returns:
        The count of unique non-null values.
    """
    return int(series.dropna().nunique())


def compute_value_profile(series: Series, null_rate: float) -> ValueProfile:
    """Compute value distribution profile for a numeric column.

    Args:
        series: A pandas Series to analyze.
        null_rate: Pre-computed null rate for the series.

    Returns:
        A ValueProfile with distribution statistics.
    """
    profile = ValueProfile()
    profile.mostly_null = null_rate > 0.8

    non_null = series.dropna()
    if len(non_null) == 0:
        return profile

    # Check low cardinality
    unique_count = non_null.nunique()
    profile.low_cardinality = unique_count <= 5

    # Try to convert to numeric for numeric stats
    try:
        numeric = pd.to_numeric(non_null, errors="coerce")
        valid_numeric = numeric.dropna()

        if len(valid_numeric) == 0:
            return profile

        # Only compute if we have enough numeric values
        numeric_ratio = len(valid_numeric) / len(non_null)
        if numeric_ratio < 0.5:
            return profile

        profile.min_value = float(valid_numeric.min())
        profile.max_value = float(valid_numeric.max())
        profile.mean = float(valid_numeric.mean())

        # Integer ratio: how many values are close to integers
        is_integer = valid_numeric.apply(
            lambda x: abs(x - round(x)) < 1e-9 if pd.notna(x) else False
        )
        profile.integer_ratio = float(is_integer.sum() / len(valid_numeric))

        # Non-negative ratio
        non_negative = valid_numeric >= 0
        profile.non_negative_ratio = float(non_negative.sum() / len(valid_numeric))

        # Bounded [0, 1]
        bounded_01 = (valid_numeric >= 0) & (valid_numeric <= 1)
        profile.bounded_0_1_ratio = float(bounded_01.sum() / len(valid_numeric))

        # Bounded [0, 100]
        bounded_0100 = (valid_numeric >= 0) & (valid_numeric <= 100)
        profile.bounded_0_100_ratio = float(bounded_0100.sum() / len(valid_numeric))

    except (ValueError, TypeError):
        pass

    return profile


def attempt_date_parse(
    series: Series,
) -> dict[str, float | bool | str | None | list[str]]:
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
        - failure_examples: First 3 strings that failed to parse
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return {
            "success_rate": 0.0,
            "has_time": False,
            "best_format": None,
            "failure_examples": [],
        }

    # Sample for efficiency
    sample = _sample_series(non_null)
    sample_str = sample.astype(str)

    best_success_rate = 0.0
    best_format: str | None = None
    has_time = False
    best_failures: list[str] = []

    # Try specific formats first (preferred - explicit and faster)
    for fmt in DATE_FORMATS:
        try:
            parsed = pd.to_datetime(sample_str, format=fmt, errors="coerce")
            success_mask = parsed.notna()
            success_count = success_mask.sum()
            success_rate = float(success_count / len(sample))

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_format = fmt
                has_time = "%H" in fmt or "%M" in fmt or "%S" in fmt

                # Collect failure examples (first 3)
                failures = sample_str[~success_mask].head(3).tolist()
                best_failures = failures

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
            success_mask = parsed.notna()
            success_count = success_mask.sum()
            success_rate = float(success_count / len(sample))

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_format = "mixed"

                # Collect failure examples
                best_failures = sample_str[~success_mask].head(3).tolist()

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
        "failure_examples": best_failures,
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

    sample = _sample_series(non_null)

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


def compute_array_profile(series: Series) -> ArrayProfile | None:
    """Compute array length statistics for array-type columns.

    Args:
        series: A pandas Series containing arrays (Python lists or JSON strings).

    Returns:
        ArrayProfile with length statistics, or None if not applicable.
    """
    non_null = series.dropna()

    if len(non_null) == 0:
        return None

    sample = _sample_series(non_null)
    lengths: list[int] = []

    for value in sample:
        if isinstance(value, list):
            lengths.append(len(value))
            continue

        if isinstance(value, str):
            value_str = value.strip()
            if value_str.startswith("["):
                try:
                    parsed = json.loads(value_str)
                    if isinstance(parsed, list):
                        lengths.append(len(parsed))
                except (json.JSONDecodeError, ValueError):
                    continue

    if not lengths:
        return None

    min_len = min(lengths)
    max_len = max(lengths)

    return ArrayProfile(
        avg_length=sum(lengths) / len(lengths),
        min_length=min_len,
        max_length=max_len,
        consistent_length=(max_len - min_len) <= 1,
    )


def detect_header_date(column_name: str) -> bool:
    """Check if a column name looks like a date.

    Args:
        column_name: The column name to check.

    Returns:
        True if the column name matches a date pattern.
    """
    name = str(column_name).strip()

    return any(pattern.match(name) for pattern in HEADER_DATE_PATTERNS)


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

    sample = _sample_series(non_null)

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
    primitive_type, type_notes = detect_primitive_type(series)
    structural_type = detect_structural_type(series)

    # Compute statistics
    null_rate = compute_null_rate(series)
    distinct_ratio = compute_distinct_ratio(series)
    unique_count = compute_unique_count(series)

    # Compute value profile
    value_profile = compute_value_profile(series, null_rate)

    # Build parse results
    parse_results = ParseResults()
    parse_results_dict: dict[str, float] = {}
    notes: list[str] = list(type_notes)

    # Date parsing attempts (for string columns)
    if primitive_type in (
        PrimitiveType.STRING,
        PrimitiveType.DATE,
        PrimitiveType.DATETIME,
    ):
        date_result = attempt_date_parse(series)
        success_rate = float(date_result["success_rate"])  # type: ignore[arg-type]
        has_time = bool(date_result.get("has_time", False))
        best_format = date_result.get("best_format")
        failure_examples = date_result.get("failure_examples", [])

        parse_results.date_parse_rate = success_rate
        parse_results.has_time = has_time
        parse_results.best_date_format = best_format if isinstance(best_format, str) else None
        parse_results.date_failure_examples = failure_examples if isinstance(failure_examples, list) else []

        # Legacy dict
        parse_results_dict["date_parse"] = success_rate

        if best_format:
            notes.append(f"Date format: {best_format}")

        # Add note about failures if success rate is moderate
        if 0.5 < success_rate < 0.99:
            if failure_examples and isinstance(failure_examples, list):
                notes.append(
                    f"{success_rate:.0%} parseable as date, "
                    f"failures include: {', '.join(repr(f) for f in failure_examples[:3])}"
                )

    # JSON array detection
    json_result = detect_json_array(series)
    json_array_rate = float(json_result["success_rate"])  # type: ignore[arg-type]
    parse_results.json_array_rate = json_array_rate
    parse_results_dict["json_array"] = json_array_rate

    if json_result["is_json_array"]:
        notes.append("Contains JSON arrays")
        structural_type = StructuralType.ARRAY

    # Array profile (for array-type columns)
    array_profile = None
    if structural_type == StructuralType.ARRAY:
        array_profile = compute_array_profile(series)
        if array_profile and array_profile.consistent_length:
            notes.append(
                f"Consistent array length (~{array_profile.avg_length:.0f} elements)"
            )

    # Header date detection
    header_date_like = detect_header_date(column_name)
    if header_date_like:
        notes.append("Column name appears to be a date")

    return ColumnEvidence(
        name=column_name,
        primitive_type=primitive_type,
        structural_type=structural_type,
        null_rate=null_rate,
        distinct_ratio=distinct_ratio,
        unique_count=unique_count,
        value_profile=value_profile,
        array_profile=array_profile,
        header_date_like=header_date_like,
        parse_results=parse_results,
        parse_results_dict=parse_results_dict,
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
