"""Time axis interpretation module for Datasculpt.

This module provides functions for detecting time granularity, parsing time periods
from column headers, inferring series frequencies, and extracting time ranges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


class TimeGranularity(str, Enum):
    """Time granularity levels."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    UNKNOWN = "unknown"


@dataclass
class GranularityResult:
    """Result of time granularity detection."""

    granularity: TimeGranularity
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class ParsedTimeHeader:
    """Result of parsing a time period from a column header."""

    column_name: str
    parsed_date: date | None
    granularity: TimeGranularity
    original_format: str | None = None


@dataclass
class SeriesFrequencyResult:
    """Result of series frequency inference."""

    frequency: TimeGranularity
    array_length: int
    start_date: date | None
    end_date: date | None
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class TimeRangeResult:
    """Result of time range extraction."""

    min_date: date | None
    max_date: date | None
    granularity: TimeGranularity
    column_name: str
    row_count: int


# Month name mappings
MONTH_ABBREV_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

MONTH_FULL_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Quarter to month mapping (first month of quarter)
QUARTER_START_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}


# Time granularity patterns for column names
GRANULARITY_NAME_PATTERNS = {
    TimeGranularity.DAILY: [
        re.compile(r"\bday\b", re.I),
        re.compile(r"\bdate\b", re.I),
        re.compile(r"\bdaily\b", re.I),
    ],
    TimeGranularity.WEEKLY: [
        re.compile(r"\bweek\b", re.I),
        re.compile(r"\bweekly\b", re.I),
        re.compile(r"\bwk\b", re.I),
    ],
    TimeGranularity.MONTHLY: [
        re.compile(r"\bmonth\b", re.I),
        re.compile(r"\bmonthly\b", re.I),
        re.compile(r"\bmon\b", re.I),
    ],
    TimeGranularity.QUARTERLY: [
        re.compile(r"\bquarter\b", re.I),
        re.compile(r"\bquarterly\b", re.I),
        re.compile(r"\bqtr\b", re.I),
    ],
    TimeGranularity.ANNUAL: [
        re.compile(r"\byear\b", re.I),
        re.compile(r"\byearly\b", re.I),
        re.compile(r"\bannual\b", re.I),
        re.compile(r"\bfy\b", re.I),
    ],
}


def detect_granularity_from_values(
    series: pd.Series,
    *,
    sample_size: int = 1000,
) -> GranularityResult:
    """Detect time granularity from column values by analyzing date differences.

    Args:
        series: A pandas Series containing date/datetime values.
        sample_size: Maximum number of values to sample for analysis.

    Returns:
        GranularityResult with detected granularity, confidence, and evidence.
    """
    evidence: list[str] = []

    # Convert to datetime if needed
    try:
        dates = pd.to_datetime(series.dropna(), errors="coerce")
        dates = dates.dropna()
    except (ValueError, TypeError):
        return GranularityResult(
            granularity=TimeGranularity.UNKNOWN,
            confidence=0.0,
            evidence=["Could not parse values as dates"],
        )

    if len(dates) < 2:
        return GranularityResult(
            granularity=TimeGranularity.UNKNOWN,
            confidence=0.0,
            evidence=["Insufficient date values for analysis"],
        )

    # Sample if too large
    if len(dates) > sample_size:
        dates = dates.sample(sample_size, random_state=42)
        evidence.append(f"Sampled {sample_size} values from {len(series)} total")

    # Sort and compute differences
    sorted_dates = dates.sort_values()
    diffs = sorted_dates.diff().dropna()

    if len(diffs) == 0:
        return GranularityResult(
            granularity=TimeGranularity.UNKNOWN,
            confidence=0.0,
            evidence=["No date differences to analyze"],
        )

    # Convert to days
    diff_days = diffs.dt.days

    # Compute statistics
    median_diff = diff_days.median()
    mean_diff = diff_days.mean()
    std_diff = diff_days.std() if len(diff_days) > 1 else 0.0

    evidence.append(f"Median date difference: {median_diff:.1f} days")
    evidence.append(f"Mean date difference: {mean_diff:.1f} days")

    # Classify based on median difference
    granularity = _classify_granularity_from_diff(median_diff)

    # Calculate confidence based on consistency
    expected_diff = _expected_diff_for_granularity(granularity)
    if expected_diff > 0 and std_diff is not None:
        # Confidence is higher when std is lower relative to expected
        consistency = 1.0 - min(1.0, std_diff / expected_diff)
        confidence = consistency * 0.8 + 0.2  # Base confidence of 0.2
    else:
        confidence = 0.5

    if granularity == TimeGranularity.UNKNOWN:
        confidence = 0.0
        evidence.append("Could not classify granularity from date differences")
    else:
        evidence.append(f"Detected {granularity.value} granularity")

    return GranularityResult(
        granularity=granularity,
        confidence=confidence,
        evidence=evidence,
    )


def _classify_granularity_from_diff(median_days: float) -> TimeGranularity:
    """Classify time granularity based on median day difference."""
    if median_days < 0:
        return TimeGranularity.UNKNOWN

    # Daily: 1 day (allow 0-2)
    if median_days <= 2:
        return TimeGranularity.DAILY

    # Weekly: 7 days (allow 5-10)
    if 5 <= median_days <= 10:
        return TimeGranularity.WEEKLY

    # Monthly: ~30 days (allow 25-35)
    if 25 <= median_days <= 35:
        return TimeGranularity.MONTHLY

    # Quarterly: ~91 days (allow 80-100)
    if 80 <= median_days <= 100:
        return TimeGranularity.QUARTERLY

    # Annual: ~365 days (allow 350-380)
    if 350 <= median_days <= 380:
        return TimeGranularity.ANNUAL

    return TimeGranularity.UNKNOWN


def _expected_diff_for_granularity(granularity: TimeGranularity) -> float:
    """Return expected day difference for a granularity."""
    mapping = {
        TimeGranularity.DAILY: 1.0,
        TimeGranularity.WEEKLY: 7.0,
        TimeGranularity.MONTHLY: 30.0,
        TimeGranularity.QUARTERLY: 91.0,
        TimeGranularity.ANNUAL: 365.0,
        TimeGranularity.UNKNOWN: 0.0,
    }
    return mapping.get(granularity, 0.0)


def detect_granularity_from_name(column_name: str) -> GranularityResult:
    """Detect time granularity from column name patterns.

    Args:
        column_name: The column name to analyze.

    Returns:
        GranularityResult with detected granularity, confidence, and evidence.
    """
    evidence: list[str] = []
    name = str(column_name)

    for granularity, patterns in GRANULARITY_NAME_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(name):
                evidence.append(f"Column name '{name}' matches pattern '{pattern.pattern}'")
                return GranularityResult(
                    granularity=granularity,
                    confidence=0.7,
                    evidence=evidence,
                )

    return GranularityResult(
        granularity=TimeGranularity.UNKNOWN,
        confidence=0.0,
        evidence=[f"No granularity pattern found in column name '{name}'"],
    )


def detect_granularity(
    series: pd.Series,
    column_name: str,
) -> GranularityResult:
    """Detect time granularity from both values and column name.

    This is the main entry point that combines value-based and name-based detection.

    Args:
        series: A pandas Series containing the column data.
        column_name: The name of the column.

    Returns:
        GranularityResult with the best detected granularity.
    """
    # Try value-based detection first
    value_result = detect_granularity_from_values(series)

    # Try name-based detection
    name_result = detect_granularity_from_name(column_name)

    # Combine results
    combined_evidence = value_result.evidence + name_result.evidence

    # If value-based detection succeeded with reasonable confidence, prefer it
    if value_result.granularity != TimeGranularity.UNKNOWN and value_result.confidence >= 0.5:
        # Boost confidence if name matches
        confidence = value_result.confidence
        if name_result.granularity == value_result.granularity:
            confidence = min(1.0, confidence + 0.15)
            combined_evidence.append("Name-based detection confirms value-based detection")

        return GranularityResult(
            granularity=value_result.granularity,
            confidence=confidence,
            evidence=combined_evidence,
        )

    # Fall back to name-based detection
    if name_result.granularity != TimeGranularity.UNKNOWN:
        return GranularityResult(
            granularity=name_result.granularity,
            confidence=name_result.confidence,
            evidence=combined_evidence,
        )

    # No detection succeeded
    return GranularityResult(
        granularity=TimeGranularity.UNKNOWN,
        confidence=0.0,
        evidence=combined_evidence,
    )


def parse_time_header(column_name: str) -> ParsedTimeHeader:
    """Parse a time period from a column header.

    Handles formats like:
    - "2024-01" (YYYY-MM)
    - "2024-Q1" or "Q1-2024" (quarterly)
    - "Jan 2024" or "2024-Jan" (month-year)
    - "FY2024" or "2024" (fiscal/calendar year)

    Args:
        column_name: The column header to parse.

    Returns:
        ParsedTimeHeader with parsed date, granularity, and original format.
    """
    name = str(column_name).strip()

    # Try ISO year-month: 2024-01
    match = re.match(r"^(\d{4})[-/](\d{2})$", name)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return ParsedTimeHeader(
                column_name=column_name,
                parsed_date=date(year, month, 1),
                granularity=TimeGranularity.MONTHLY,
                original_format="YYYY-MM",
            )

    # Try quarter: 2024-Q1, 2024Q1, Q1-2024, Q1 2024
    match = re.match(r"^(\d{4})[-_\s]?[Qq]([1-4])$", name)
    if match:
        year, quarter = int(match.group(1)), int(match.group(2))
        month = QUARTER_START_MONTH[quarter]
        return ParsedTimeHeader(
            column_name=column_name,
            parsed_date=date(year, month, 1),
            granularity=TimeGranularity.QUARTERLY,
            original_format="YYYY-Qn",
        )

    match = re.match(r"^[Qq]([1-4])[-_\s]?(\d{4})$", name)
    if match:
        quarter, year = int(match.group(1)), int(match.group(2))
        month = QUARTER_START_MONTH[quarter]
        return ParsedTimeHeader(
            column_name=column_name,
            parsed_date=date(year, month, 1),
            granularity=TimeGranularity.QUARTERLY,
            original_format="Qn-YYYY",
        )

    # Try month name + year: Jan 2024, January 2024, 2024-Jan
    for month_map in [MONTH_ABBREV_MAP, MONTH_FULL_MAP]:
        for month_name, month_num in month_map.items():
            # Month Year pattern
            pattern = rf"^{month_name}[-_\s]?(\d{{4}})$"
            match = re.match(pattern, name, re.I)
            if match:
                year = int(match.group(1))
                return ParsedTimeHeader(
                    column_name=column_name,
                    parsed_date=date(year, month_num, 1),
                    granularity=TimeGranularity.MONTHLY,
                    original_format="Mon YYYY",
                )

            # Year Month pattern
            pattern = rf"^(\d{{4}})[-_\s]?{month_name}$"
            match = re.match(pattern, name, re.I)
            if match:
                year = int(match.group(1))
                return ParsedTimeHeader(
                    column_name=column_name,
                    parsed_date=date(year, month_num, 1),
                    granularity=TimeGranularity.MONTHLY,
                    original_format="YYYY Mon",
                )

    # Try fiscal year: FY2024, FY 2024
    match = re.match(r"^FY[-_\s]?(\d{4})$", name, re.I)
    if match:
        year = int(match.group(1))
        return ParsedTimeHeader(
            column_name=column_name,
            parsed_date=date(year, 1, 1),
            granularity=TimeGranularity.ANNUAL,
            original_format="FYnnnn",
        )

    # Try plain year: 2024
    match = re.match(r"^(\d{4})$", name)
    if match:
        year = int(match.group(1))
        # Validate reasonable year range (1900-2100)
        if 1900 <= year <= 2100:
            return ParsedTimeHeader(
                column_name=column_name,
                parsed_date=date(year, 1, 1),
                granularity=TimeGranularity.ANNUAL,
                original_format="YYYY",
            )

    # Try ISO date: 2024-01-15
    match = re.match(r"^(\d{4})[-/](\d{2})[-/](\d{2})$", name)
    if match:
        year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
        try:
            return ParsedTimeHeader(
                column_name=column_name,
                parsed_date=date(year, month, day),
                granularity=TimeGranularity.DAILY,
                original_format="YYYY-MM-DD",
            )
        except ValueError:
            pass  # Invalid date

    # Could not parse
    return ParsedTimeHeader(
        column_name=column_name,
        parsed_date=None,
        granularity=TimeGranularity.UNKNOWN,
        original_format=None,
    )


def parse_time_headers(
    column_names: Sequence[str],
) -> list[ParsedTimeHeader]:
    """Parse time periods from multiple column headers.

    Args:
        column_names: List of column names to parse.

    Returns:
        List of ParsedTimeHeader objects for columns that could be parsed.
    """
    results: list[ParsedTimeHeader] = []

    for name in column_names:
        parsed = parse_time_header(name)
        if parsed.parsed_date is not None:
            results.append(parsed)

    return results


def infer_series_frequency(
    df: pd.DataFrame,
    array_column: str,
    *,
    metadata_columns: Sequence[str] | None = None,
) -> SeriesFrequencyResult:
    """Infer frequency for JSON array columns by examining metadata.

    Looks for companion columns like "frequency", "start_date", "end_date"
    to determine the time frequency of array data.

    Args:
        df: DataFrame containing the array column.
        array_column: Name of the column containing JSON arrays.
        metadata_columns: Optional list of columns to search for metadata.

    Returns:
        SeriesFrequencyResult with inferred frequency and evidence.
    """
    evidence: list[str] = []

    # Check array column exists
    if array_column not in df.columns:
        return SeriesFrequencyResult(
            frequency=TimeGranularity.UNKNOWN,
            array_length=0,
            start_date=None,
            end_date=None,
            confidence=0.0,
            evidence=[f"Column '{array_column}' not found in DataFrame"],
        )

    # Get array lengths
    series = df[array_column].dropna()
    if len(series) == 0:
        return SeriesFrequencyResult(
            frequency=TimeGranularity.UNKNOWN,
            array_length=0,
            start_date=None,
            end_date=None,
            confidence=0.0,
            evidence=["No non-null values in array column"],
        )

    # Calculate array lengths
    array_lengths = _get_array_lengths(series)
    if not array_lengths:
        return SeriesFrequencyResult(
            frequency=TimeGranularity.UNKNOWN,
            array_length=0,
            start_date=None,
            end_date=None,
            confidence=0.0,
            evidence=["Could not determine array lengths"],
        )

    # Use most common length
    median_length = int(pd.Series(array_lengths).median())
    evidence.append(f"Median array length: {median_length}")

    # Search for metadata columns
    if metadata_columns is None:
        metadata_columns = list(df.columns)

    frequency_patterns = [
        re.compile(r"frequency", re.I),
        re.compile(r"freq", re.I),
        re.compile(r"periodicity", re.I),
        re.compile(r"granularity", re.I),
    ]

    start_date_patterns = [
        re.compile(r"start.?date", re.I),
        re.compile(r"begin.?date", re.I),
        re.compile(r"from.?date", re.I),
        re.compile(r"start.?period", re.I),
    ]

    end_date_patterns = [
        re.compile(r"end.?date", re.I),
        re.compile(r"to.?date", re.I),
        re.compile(r"through.?date", re.I),
        re.compile(r"end.?period", re.I),
    ]

    # Find frequency column
    frequency = TimeGranularity.UNKNOWN
    confidence = 0.0

    for col in metadata_columns:
        if col == array_column:
            continue

        for pattern in frequency_patterns:
            if pattern.search(str(col)):
                freq_value = _extract_frequency_from_column(df[col])
                if freq_value != TimeGranularity.UNKNOWN:
                    frequency = freq_value
                    confidence = 0.8
                    evidence.append(f"Found frequency from column '{col}': {frequency.value}")
                    break

    # Find start date
    start_date: date | None = None
    for col in metadata_columns:
        if col == array_column:
            continue

        for pattern in start_date_patterns:
            if pattern.search(str(col)):
                start_date = _extract_date_from_column(df[col])
                if start_date:
                    evidence.append(f"Found start date from column '{col}': {start_date}")
                    break

    # Find end date
    end_date: date | None = None
    for col in metadata_columns:
        if col == array_column:
            continue

        for pattern in end_date_patterns:
            if pattern.search(str(col)):
                end_date = _extract_date_from_column(df[col])
                if end_date:
                    evidence.append(f"Found end date from column '{col}': {end_date}")
                    break

    # If we have start and end dates, try to infer frequency from array length
    if frequency == TimeGranularity.UNKNOWN and start_date and end_date and median_length > 0:
        inferred = _infer_frequency_from_dates_and_length(start_date, end_date, median_length)
        if inferred != TimeGranularity.UNKNOWN:
            frequency = inferred
            confidence = 0.6
            evidence.append(f"Inferred frequency from date range and array length: {frequency.value}")

    return SeriesFrequencyResult(
        frequency=frequency,
        array_length=median_length,
        start_date=start_date,
        end_date=end_date,
        confidence=confidence,
        evidence=evidence,
    )


def _get_array_lengths(series: pd.Series) -> list[int]:
    """Extract lengths from array values."""
    import json

    lengths: list[int] = []
    for value in series.head(1000):
        if isinstance(value, list):
            lengths.append(len(value))
        elif isinstance(value, str):
            try:
                parsed = json.loads(value.strip())
                if isinstance(parsed, list):
                    lengths.append(len(parsed))
            except (json.JSONDecodeError, ValueError):
                pass

    return lengths


def _extract_frequency_from_column(series: pd.Series) -> TimeGranularity:
    """Extract frequency value from a metadata column."""
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return TimeGranularity.UNKNOWN

    # Get most common value
    mode = sample.mode()
    if len(mode) == 0:
        return TimeGranularity.UNKNOWN

    value = str(mode.iloc[0]).lower().strip()

    # Map common frequency strings
    freq_mapping = {
        "d": TimeGranularity.DAILY,
        "day": TimeGranularity.DAILY,
        "daily": TimeGranularity.DAILY,
        "w": TimeGranularity.WEEKLY,
        "week": TimeGranularity.WEEKLY,
        "weekly": TimeGranularity.WEEKLY,
        "m": TimeGranularity.MONTHLY,
        "month": TimeGranularity.MONTHLY,
        "monthly": TimeGranularity.MONTHLY,
        "q": TimeGranularity.QUARTERLY,
        "quarter": TimeGranularity.QUARTERLY,
        "quarterly": TimeGranularity.QUARTERLY,
        "a": TimeGranularity.ANNUAL,
        "y": TimeGranularity.ANNUAL,
        "year": TimeGranularity.ANNUAL,
        "yearly": TimeGranularity.ANNUAL,
        "annual": TimeGranularity.ANNUAL,
        "annually": TimeGranularity.ANNUAL,
    }

    return freq_mapping.get(value, TimeGranularity.UNKNOWN)


def _extract_date_from_column(series: pd.Series) -> date | None:
    """Extract a representative date from a column."""
    sample = series.dropna().head(100)
    if len(sample) == 0:
        return None

    # Get most common value
    mode = sample.mode()
    if len(mode) == 0:
        return None

    value = mode.iloc[0]

    # If already a date/datetime
    if isinstance(value, (date, datetime)):
        return value if isinstance(value, date) else value.date()

    # Try to parse
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.date()
    except (ValueError, TypeError):
        pass

    return None


def _infer_frequency_from_dates_and_length(
    start_date: date,
    end_date: date,
    array_length: int,
) -> TimeGranularity:
    """Infer frequency from date range and array length."""
    if array_length <= 0:
        return TimeGranularity.UNKNOWN

    total_days = (end_date - start_date).days
    if total_days <= 0:
        return TimeGranularity.UNKNOWN

    # Calculate expected periods for each granularity
    expected_daily = total_days + 1
    expected_weekly = (total_days // 7) + 1
    expected_monthly = ((end_date.year - start_date.year) * 12 +
                        (end_date.month - start_date.month)) + 1
    expected_quarterly = ((end_date.year - start_date.year) * 4 +
                          ((end_date.month - 1) // 3 - (start_date.month - 1) // 3)) + 1
    expected_annual = (end_date.year - start_date.year) + 1

    # Find closest match (with 10% tolerance)
    candidates = [
        (TimeGranularity.DAILY, expected_daily),
        (TimeGranularity.WEEKLY, expected_weekly),
        (TimeGranularity.MONTHLY, expected_monthly),
        (TimeGranularity.QUARTERLY, expected_quarterly),
        (TimeGranularity.ANNUAL, expected_annual),
    ]

    for granularity, expected in candidates:
        if expected > 0:
            ratio = array_length / expected
            if 0.9 <= ratio <= 1.1:  # Within 10%
                return granularity

    return TimeGranularity.UNKNOWN


def extract_time_range(
    series: pd.Series,
    column_name: str,
) -> TimeRangeResult:
    """Extract min/max dates from a time column.

    Args:
        series: A pandas Series containing date/datetime values.
        column_name: The name of the column.

    Returns:
        TimeRangeResult with min/max dates and granularity.
    """
    # Convert to datetime
    try:
        dates = pd.to_datetime(series.dropna(), errors="coerce")
        dates = dates.dropna()
    except (ValueError, TypeError):
        return TimeRangeResult(
            min_date=None,
            max_date=None,
            granularity=TimeGranularity.UNKNOWN,
            column_name=column_name,
            row_count=0,
        )

    if len(dates) == 0:
        return TimeRangeResult(
            min_date=None,
            max_date=None,
            granularity=TimeGranularity.UNKNOWN,
            column_name=column_name,
            row_count=0,
        )

    min_dt = dates.min()
    max_dt = dates.max()

    # Detect granularity
    granularity_result = detect_granularity_from_values(series)

    return TimeRangeResult(
        min_date=min_dt.date() if pd.notna(min_dt) else None,
        max_date=max_dt.date() if pd.notna(max_dt) else None,
        granularity=granularity_result.granularity,
        column_name=column_name,
        row_count=len(dates),
    )


def extract_time_ranges(
    df: pd.DataFrame,
    time_columns: Sequence[str] | None = None,
) -> list[TimeRangeResult]:
    """Extract time ranges from multiple columns in a DataFrame.

    Args:
        df: DataFrame to analyze.
        time_columns: Optional list of columns to analyze. If None, attempts to
            detect time columns automatically.

    Returns:
        List of TimeRangeResult for each time column.
    """
    results: list[TimeRangeResult] = []

    if time_columns is None:
        # Auto-detect time columns by trying to parse each column
        time_columns = []
        for col in df.columns:
            # Check if column looks like a date column
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue

            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().sum() / len(sample)
                if success_rate >= 0.8:
                    time_columns.append(str(col))
            except (ValueError, TypeError):
                continue

    for col in time_columns:
        if col in df.columns:
            result = extract_time_range(df[col], col)
            if result.min_date is not None:
                results.append(result)

    return results
