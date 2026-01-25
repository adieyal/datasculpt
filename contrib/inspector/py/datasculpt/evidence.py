"""Column evidence extraction for Datasculpt (browser bundle)."""


import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from datasculpt.types import (
    ArrayProfile,
    ColumnEvidence,
    ParseResults,
    PrimitiveType,
    StructuralType,
    ValueProfile,
)

if TYPE_CHECKING:
    pass


# Patterns for type detection
DATE_PATTERN = re.compile(
    r"^(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])$"
)
TIME_PATTERN = re.compile(r"\d{2}:\d{2}")


def detect_primitive_type(series: pd.Series) -> PrimitiveType:
    """Detect the primitive type of a pandas series.

    Args:
        series: Pandas series to analyze.

    Returns:
        Detected PrimitiveType.
    """
    dtype = series.dtype

    # Check for boolean - use == for numpy dtype comparison
    if dtype == bool or dtype.name == "bool":
        return PrimitiveType.BOOLEAN

    # Check for integer types
    if pd.api.types.is_integer_dtype(dtype):
        return PrimitiveType.INTEGER

    # Check for float types
    if pd.api.types.is_float_dtype(dtype):
        return PrimitiveType.NUMBER

    # Check for datetime types
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return PrimitiveType.DATETIME

    # Check for object/string types
    if dtype == object or pd.api.types.is_string_dtype(dtype):
        # Sample non-null values to determine type
        non_null = series.dropna()
        if len(non_null) == 0:
            return PrimitiveType.UNKNOWN

        # Check a sample of values
        sample = non_null.head(100)

        # Check for dates
        date_matches = sum(1 for v in sample if _looks_like_date(str(v)))
        if date_matches / len(sample) > 0.8:
            return PrimitiveType.DATE

        # Check for booleans
        bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
        bool_matches = sum(1 for v in sample if str(v).lower().strip() in bool_values)
        if bool_matches / len(sample) > 0.8:
            return PrimitiveType.BOOLEAN

        # Default to string
        return PrimitiveType.STRING

    return PrimitiveType.UNKNOWN


def _looks_like_date(value: str) -> bool:
    """Check if a string value looks like a date."""
    if DATE_PATTERN.match(value.strip()):
        return True

    # Try parsing with pandas
    try:
        pd.to_datetime(value)
        return True
    except (ValueError, TypeError):
        return False


def detect_structural_type(series: pd.Series) -> StructuralType:
    """Detect the structural type of a pandas series.

    Args:
        series: Pandas series to analyze.

    Returns:
        Detected StructuralType.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return StructuralType.UNKNOWN

    sample = non_null.head(20)

    # Check for arrays (lists)
    list_count = sum(1 for v in sample if isinstance(v, (list, tuple)))
    if list_count / len(sample) > 0.5:
        return StructuralType.ARRAY

    # Check for objects (dicts)
    dict_count = sum(1 for v in sample if isinstance(v, dict))
    if dict_count / len(sample) > 0.5:
        return StructuralType.OBJECT

    # Check for JSON strings
    if series.dtype == object:
        json_array_count = 0
        for v in sample:
            if isinstance(v, str) and v.strip().startswith("["):
                json_array_count += 1
        if json_array_count / len(sample) > 0.5:
            return StructuralType.ARRAY

    return StructuralType.SCALAR


def compute_value_profile(series: pd.Series) -> ValueProfile:
    """Compute value profile statistics for a numeric series.

    Args:
        series: Pandas series to analyze.

    Returns:
        ValueProfile with computed statistics.
    """
    profile = ValueProfile()

    # Get numeric values
    numeric = pd.to_numeric(series, errors="coerce")
    non_null_numeric = numeric.dropna()

    if len(non_null_numeric) == 0:
        return profile

    # Basic statistics
    profile.min_value = float(non_null_numeric.min())
    profile.max_value = float(non_null_numeric.max())
    profile.mean = float(non_null_numeric.mean())

    # Integer ratio
    is_integer = non_null_numeric.apply(lambda x: float(x).is_integer())
    profile.integer_ratio = float(is_integer.mean())

    # Non-negative ratio
    profile.non_negative_ratio = float((non_null_numeric >= 0).mean())

    # Bounded ratios
    in_0_1 = (non_null_numeric >= 0) & (non_null_numeric <= 1)
    profile.bounded_0_1_ratio = float(in_0_1.mean())

    in_0_100 = (non_null_numeric >= 0) & (non_null_numeric <= 100)
    profile.bounded_0_100_ratio = float(in_0_100.mean())

    # Low cardinality check
    profile.low_cardinality = non_null_numeric.nunique() <= 20

    # Mostly null
    profile.mostly_null = series.isna().mean() > 0.5

    return profile


def compute_array_profile(series: pd.Series) -> ArrayProfile | None:
    """Compute profile for array-type columns.

    Args:
        series: Pandas series with array values.

    Returns:
        ArrayProfile or None if not applicable.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    lengths = []
    for v in non_null:
        if isinstance(v, (list, tuple)):
            lengths.append(len(v))
        elif isinstance(v, str) and v.strip().startswith("["):
            # JSON array string - estimate length
            try:
                import json
                arr = json.loads(v)
                if isinstance(arr, list):
                    lengths.append(len(arr))
            except (ValueError, TypeError):
                pass

    if not lengths:
        return None

    return ArrayProfile(
        avg_length=sum(lengths) / len(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        consistent_length=len(set(lengths)) == 1,
    )


def compute_parse_results(series: pd.Series) -> ParseResults:
    """Compute parse attempt results for a column.

    Args:
        series: Pandas series to analyze.

    Returns:
        ParseResults with parsing statistics.
    """
    results = ParseResults()

    non_null = series.dropna()
    if len(non_null) == 0:
        return results

    # Date parsing
    parsed_dates = pd.to_datetime(non_null, errors="coerce")
    valid_dates = parsed_dates.notna()
    results.date_parse_rate = float(valid_dates.mean())

    # Check for time component
    if results.date_parse_rate > 0:
        valid_datetime = parsed_dates[valid_dates]
        if len(valid_datetime) > 0:
            has_nonzero_time = valid_datetime.dt.time.apply(
                lambda t: t.hour != 0 or t.minute != 0 or t.second != 0
            )
            results.has_time = bool(has_nonzero_time.any())

    # JSON array detection
    json_count = 0
    for v in non_null.head(100):
        if isinstance(v, str) and v.strip().startswith("["):
            json_count += 1
    results.json_array_rate = json_count / min(len(non_null), 100)

    return results


def is_header_date_like(column_name: str) -> bool:
    """Check if a column header looks like a date/time period.

    Args:
        column_name: Column name to check.

    Returns:
        True if name looks like a date.
    """
    name = str(column_name).strip()

    # Year patterns
    if re.match(r"^(19|20)\d{2}$", name):
        return True

    # Month-year patterns
    if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[- ]?\d{2,4}$", name, re.IGNORECASE):
        return True

    # Quarter patterns
    if re.match(r"^Q[1-4][- ]?(19|20)?\d{2}$", name, re.IGNORECASE):
        return True

    # Full date patterns
    if re.match(r"^(19|20)\d{2}[-/](0?[1-9]|1[0-2])$", name):
        return True

    return False


def extract_column_evidence(
    series: pd.Series,
    column_name: str,
) -> ColumnEvidence:
    """Extract evidence for a single column.

    Args:
        series: Pandas series to analyze.
        column_name: Name of the column.

    Returns:
        ColumnEvidence with all extracted information.
    """
    primitive_type = detect_primitive_type(series)
    structural_type = detect_structural_type(series)

    # Basic statistics
    null_rate = float(series.isna().mean())
    non_null = series.dropna()
    unique_count = non_null.nunique()
    distinct_ratio = unique_count / len(non_null) if len(non_null) > 0 else 0.0

    # Value profile for numeric columns
    value_profile = ValueProfile()
    if primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        value_profile = compute_value_profile(series)

    # Array profile for array columns
    array_profile = None
    if structural_type == StructuralType.ARRAY:
        array_profile = compute_array_profile(series)

    # Parse results
    parse_results = compute_parse_results(series)

    # Header date check
    header_date_like = is_header_date_like(column_name)

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
    )


def extract_dataframe_evidence(df: pd.DataFrame) -> dict[str, ColumnEvidence]:
    """Extract evidence for all columns in a DataFrame.

    Args:
        df: DataFrame to analyze.

    Returns:
        Dictionary mapping column names to ColumnEvidence.
    """
    evidence = {}
    for col in df.columns:
        evidence[col] = extract_column_evidence(df[col], col)
    return evidence
