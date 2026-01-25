"""Time axis interpretation for Datasculpt (browser bundle).

This module is renamed from 'time.py' to avoid conflict with Python stdlib.
"""


import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datasculpt.types import ColumnEvidence


class TimeGranularity(str, Enum):
    """Time granularity levels."""

    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"
    HOURLY = "hourly"
    MINUTE = "minute"
    SECOND = "second"
    UNKNOWN = "unknown"


@dataclass
class GranularityResult:
    """Result of time granularity detection."""

    granularity: TimeGranularity
    confidence: float
    evidence: list[str]


# Patterns for granularity detection from column names
ANNUAL_PATTERNS = [
    re.compile(r"^(19|20)\d{2}$"),  # Just year: 2020
    re.compile(r"year", re.IGNORECASE),
    re.compile(r"annual", re.IGNORECASE),
    re.compile(r"yearly", re.IGNORECASE),
]

QUARTERLY_PATTERNS = [
    re.compile(r"^Q[1-4][- ]?(19|20)?\d{2}$", re.IGNORECASE),  # Q1 2020
    re.compile(r"quarter", re.IGNORECASE),
    re.compile(r"qtr", re.IGNORECASE),
]

MONTHLY_PATTERNS = [
    re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.IGNORECASE),
    re.compile(r"month", re.IGNORECASE),
    re.compile(r"^(19|20)\d{2}[-/](0?[1-9]|1[0-2])$"),  # 2020-01
]

WEEKLY_PATTERNS = [
    re.compile(r"week", re.IGNORECASE),
    re.compile(r"wk", re.IGNORECASE),
]

DAILY_PATTERNS = [
    re.compile(r"^(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])$"),
    re.compile(r"daily", re.IGNORECASE),
    re.compile(r"day", re.IGNORECASE),
    re.compile(r"date", re.IGNORECASE),
]


def detect_granularity_from_name(column_name: str) -> GranularityResult:
    """Detect time granularity from column name patterns.

    Args:
        column_name: Column name to analyze.

    Returns:
        GranularityResult with detected granularity.
    """
    name = str(column_name)
    evidence: list[str] = []

    # Check patterns in order of specificity
    for pattern in QUARTERLY_PATTERNS:
        if pattern.search(name):
            evidence.append(f"Name matches quarterly pattern: {pattern.pattern}")
            return GranularityResult(
                granularity=TimeGranularity.QUARTERLY,
                confidence=0.8,
                evidence=evidence,
            )

    for pattern in MONTHLY_PATTERNS:
        if pattern.search(name):
            evidence.append(f"Name matches monthly pattern: {pattern.pattern}")
            return GranularityResult(
                granularity=TimeGranularity.MONTHLY,
                confidence=0.7,
                evidence=evidence,
            )

    for pattern in ANNUAL_PATTERNS:
        if pattern.search(name):
            evidence.append(f"Name matches annual pattern: {pattern.pattern}")
            return GranularityResult(
                granularity=TimeGranularity.ANNUAL,
                confidence=0.8,
                evidence=evidence,
            )

    for pattern in WEEKLY_PATTERNS:
        if pattern.search(name):
            evidence.append(f"Name matches weekly pattern: {pattern.pattern}")
            return GranularityResult(
                granularity=TimeGranularity.WEEKLY,
                confidence=0.6,
                evidence=evidence,
            )

    for pattern in DAILY_PATTERNS:
        if pattern.search(name):
            evidence.append(f"Name matches daily pattern: {pattern.pattern}")
            return GranularityResult(
                granularity=TimeGranularity.DAILY,
                confidence=0.6,
                evidence=evidence,
            )

    return GranularityResult(
        granularity=TimeGranularity.UNKNOWN,
        confidence=0.0,
        evidence=["No recognizable time pattern in name"],
    )


def detect_granularity_from_values(series: pd.Series) -> GranularityResult:
    """Detect time granularity from datetime values.

    Args:
        series: Pandas series with datetime-like values.

    Returns:
        GranularityResult with detected granularity.
    """
    evidence: list[str] = []

    # Convert to datetime
    try:
        dates = pd.to_datetime(series, errors="coerce")
    except Exception:
        return GranularityResult(
            granularity=TimeGranularity.UNKNOWN,
            confidence=0.0,
            evidence=["Could not parse as datetime"],
        )

    dates = dates.dropna().sort_values()

    if len(dates) < 2:
        return GranularityResult(
            granularity=TimeGranularity.UNKNOWN,
            confidence=0.0,
            evidence=["Not enough values to detect granularity"],
        )

    # Calculate time differences
    diffs = dates.diff().dropna()
    median_diff = diffs.median()

    # Check time components
    has_time = (dates.dt.hour != 0).any() or (dates.dt.minute != 0).any()
    all_first_of_month = (dates.dt.day == 1).all()
    all_first_of_quarter = all_first_of_month and dates.dt.month.isin([1, 4, 7, 10]).all()
    all_jan_first = all_first_of_month and (dates.dt.month == 1).all()

    # Determine granularity based on patterns
    if all_jan_first:
        evidence.append("All dates are January 1st (annual)")
        return GranularityResult(
            granularity=TimeGranularity.ANNUAL,
            confidence=0.9,
            evidence=evidence,
        )

    if all_first_of_quarter:
        evidence.append("All dates are first of quarter months")
        return GranularityResult(
            granularity=TimeGranularity.QUARTERLY,
            confidence=0.85,
            evidence=evidence,
        )

    if all_first_of_month:
        evidence.append("All dates are first of month")
        return GranularityResult(
            granularity=TimeGranularity.MONTHLY,
            confidence=0.85,
            evidence=evidence,
        )

    # Check median difference
    days = median_diff.days if hasattr(median_diff, "days") else median_diff / pd.Timedelta(days=1)

    if days >= 350 and days <= 380:
        evidence.append(f"Median difference ~{days:.0f} days (annual)")
        return GranularityResult(
            granularity=TimeGranularity.ANNUAL,
            confidence=0.7,
            evidence=evidence,
        )

    if days >= 85 and days <= 100:
        evidence.append(f"Median difference ~{days:.0f} days (quarterly)")
        return GranularityResult(
            granularity=TimeGranularity.QUARTERLY,
            confidence=0.7,
            evidence=evidence,
        )

    if days >= 27 and days <= 32:
        evidence.append(f"Median difference ~{days:.0f} days (monthly)")
        return GranularityResult(
            granularity=TimeGranularity.MONTHLY,
            confidence=0.7,
            evidence=evidence,
        )

    if days >= 6 and days <= 8:
        evidence.append(f"Median difference ~{days:.0f} days (weekly)")
        return GranularityResult(
            granularity=TimeGranularity.WEEKLY,
            confidence=0.7,
            evidence=evidence,
        )

    if days >= 0.9 and days <= 1.1:
        evidence.append(f"Median difference ~{days:.2f} days (daily)")
        return GranularityResult(
            granularity=TimeGranularity.DAILY,
            confidence=0.7,
            evidence=evidence,
        )

    # Sub-daily granularity
    if has_time:
        hours = days * 24
        if hours < 1:
            evidence.append(f"Sub-hourly granularity detected")
            return GranularityResult(
                granularity=TimeGranularity.MINUTE,
                confidence=0.5,
                evidence=evidence,
            )
        evidence.append(f"Sub-daily granularity with time component")
        return GranularityResult(
            granularity=TimeGranularity.HOURLY,
            confidence=0.5,
            evidence=evidence,
        )

    evidence.append(f"Could not determine granularity (median diff: {days:.1f} days)")
    return GranularityResult(
        granularity=TimeGranularity.UNKNOWN,
        confidence=0.0,
        evidence=evidence,
    )


def detect_time_granularity(
    series: pd.Series,
    column_name: str,
) -> GranularityResult:
    """Detect time granularity using both name and value analysis.

    Args:
        series: Pandas series with time values.
        column_name: Name of the column.

    Returns:
        GranularityResult with best detected granularity.
    """
    # Try name-based detection
    name_result = detect_granularity_from_name(column_name)

    # Try value-based detection
    value_result = detect_granularity_from_values(series)

    # Combine results
    if name_result.granularity != TimeGranularity.UNKNOWN and value_result.granularity != TimeGranularity.UNKNOWN:
        # Both detected something
        if name_result.granularity == value_result.granularity:
            # Agreement - high confidence
            return GranularityResult(
                granularity=name_result.granularity,
                confidence=max(name_result.confidence, value_result.confidence) + 0.1,
                evidence=name_result.evidence + value_result.evidence,
            )
        else:
            # Disagreement - prefer value-based
            return GranularityResult(
                granularity=value_result.granularity,
                confidence=value_result.confidence * 0.8,
                evidence=value_result.evidence + [f"Name suggested {name_result.granularity.value} but values disagree"],
            )

    # Return whichever one detected something
    if value_result.granularity != TimeGranularity.UNKNOWN:
        return value_result
    if name_result.granularity != TimeGranularity.UNKNOWN:
        return name_result

    # Neither detected anything
    return GranularityResult(
        granularity=TimeGranularity.UNKNOWN,
        confidence=0.0,
        evidence=["Could not detect time granularity from name or values"],
    )
