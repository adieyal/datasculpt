"""Time axis interpretation module for Datasculpt.

This module provides functions for detecting time granularity, parsing time periods
from column headers, inferring series frequencies, and extracting time ranges.

The granularity detection uses a "grid fit" approach that checks whether timestamps
align to calendar period buckets (day/week/month/quarter/year) rather than relying
on median day differences. This handles gaps and irregular reporting dates correctly.
"""

from __future__ import annotations

import json
import math
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
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"
    UNKNOWN = "unknown"


@dataclass
class GridFitEvidence:
    """Evidence from grid-fit scoring for a specific granularity."""

    bucket_coverage: float  # Fraction of unique buckets vs total timestamps
    gap_rate: float  # Fraction of period steps that skip buckets
    alignment_score: float  # How well timestamps align to canonical positions
    monotonic: bool  # Whether buckets are non-decreasing
    dominant_alignment: str | None = None  # Most common alignment type (e.g., "month_start")
    alignment_consistency: float = 0.0  # Fraction of timestamps with dominant alignment
    records_per_bucket: float = 1.0  # Average records per bucket (for duplicate detection)


@dataclass
class GranularityResult:
    """Result of time granularity detection."""

    granularity: TimeGranularity
    confidence: float
    evidence: list[str] = field(default_factory=list)
    grid_fit: dict[str, GridFitEvidence] = field(default_factory=dict)


@dataclass
class ParsedTimeHeader:
    """Result of parsing a time period from a column header."""

    column_name: str
    parsed_date: date | None
    granularity: TimeGranularity
    original_format: str | None = None
    is_fiscal: bool = False  # True for FY patterns where fiscal year start is unknown


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


@dataclass
class TimeAxisResult:
    """Structured time axis artifact for downstream consumption.

    This represents the detected time axis in a format suitable for
    Invariant compatibility/comparability checks.
    """

    kind: str  # "column", "wide_headers", or "series_metadata"
    granularity: TimeGranularity
    start: date | None
    end: date | None
    has_gaps: bool
    confidence: float
    evidence: dict[str, float | str | bool] = field(default_factory=dict)
    alignment: str | None = None  # e.g., "month_start", "month_end", "week_monday"


@dataclass
class HeaderGridResult:
    """Result of header grid coherence check for wide time columns."""

    parsed_headers: list[ParsedTimeHeader]
    dominant_granularity: TimeGranularity
    is_monotonic: bool
    has_consistent_format: bool
    gap_rate: float
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class SeriesMetadataConsistency:
    """Result of checking whether series metadata is constant or per-row."""

    is_constant: bool
    unique_values: int
    mode_coverage: float  # Fraction of rows matching the mode
    evidence: list[str] = field(default_factory=list)


@dataclass
class TimeAxisCandidate:
    """A candidate time axis column with ranking metrics.

    Used when multiple columns could serve as the time axis to help
    select the most likely primary time axis.
    """

    column_name: str
    granularity: TimeGranularity
    confidence: float
    unique_bucket_count: int
    null_rate: float
    name_score: float  # How time-like the name is
    is_parseable: bool  # Whether values parse as dates
    evidence: list[str] = field(default_factory=list)


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
# Note: Generic terms like "date" are intentionally excluded because they
# don't imply a specific granularity - a column named "date" could be
# daily, monthly, or any other frequency. Only include patterns that
# clearly indicate a specific granularity.
GRANULARITY_NAME_PATTERNS: dict[TimeGranularity, list[re.Pattern[str]]] = {
    TimeGranularity.DAILY: [
        re.compile(r"\bdaily\b", re.I),
        re.compile(r"\bday_of\b", re.I),
        re.compile(r"\bper_day\b", re.I),
    ],
    TimeGranularity.WEEKLY: [
        re.compile(r"\bweek\b", re.I),
        re.compile(r"\bweekly\b", re.I),
        re.compile(r"\bwk\b", re.I),
    ],
    TimeGranularity.BIWEEKLY: [
        re.compile(r"\bbiweekly\b", re.I),
        re.compile(r"\bbi-weekly\b", re.I),
        re.compile(r"\bfortnightly\b", re.I),
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
    TimeGranularity.SEMIANNUAL: [
        re.compile(r"\bsemi-?annual\b", re.I),
        re.compile(r"\bhalf-?yearly\b", re.I),
        re.compile(r"\bh[12]\b", re.I),
    ],
    TimeGranularity.ANNUAL: [
        re.compile(r"\byear\b", re.I),
        re.compile(r"\byearly\b", re.I),
        re.compile(r"\bannual\b", re.I),
        re.compile(r"\bfy\b", re.I),
    ],
}


def _to_period_bucket(dt: datetime, granularity: TimeGranularity) -> str:
    """Convert a datetime to a period bucket key for the given granularity."""
    if granularity == TimeGranularity.DAILY:
        return dt.strftime("%Y-%m-%d")
    elif granularity == TimeGranularity.WEEKLY:
        # ISO week: YYYY-Www
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
    elif granularity == TimeGranularity.BIWEEKLY:
        # Biweekly: use ISO week divided by 2
        iso_year, iso_week, _ = dt.isocalendar()
        biweek = (iso_week - 1) // 2 + 1
        return f"{iso_year}-BW{biweek:02d}"
    elif granularity == TimeGranularity.MONTHLY:
        return dt.strftime("%Y-%m")
    elif granularity == TimeGranularity.QUARTERLY:
        quarter = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{quarter}"
    elif granularity == TimeGranularity.SEMIANNUAL:
        half = 1 if dt.month <= 6 else 2
        return f"{dt.year}-H{half}"
    elif granularity == TimeGranularity.ANNUAL:
        return str(dt.year)
    else:
        return dt.strftime("%Y-%m-%d")


def _bucket_to_index(bucket: str, granularity: TimeGranularity) -> int:
    """Convert a bucket key to a numeric index for computing steps."""
    if granularity == TimeGranularity.DAILY:
        # Parse YYYY-MM-DD and return days since epoch
        parts = bucket.split("-")
        d = date(int(parts[0]), int(parts[1]), int(parts[2]))
        return d.toordinal()
    elif granularity == TimeGranularity.WEEKLY:
        # Parse YYYY-Www
        parts = bucket.split("-W")
        year, week = int(parts[0]), int(parts[1])
        return year * 53 + week
    elif granularity == TimeGranularity.BIWEEKLY:
        # Parse YYYY-BWnn
        parts = bucket.split("-BW")
        year, biweek = int(parts[0]), int(parts[1])
        return year * 27 + biweek
    elif granularity == TimeGranularity.MONTHLY:
        # Parse YYYY-MM
        parts = bucket.split("-")
        year, month = int(parts[0]), int(parts[1])
        return year * 12 + month
    elif granularity == TimeGranularity.QUARTERLY:
        # Parse YYYY-Qn
        parts = bucket.split("-Q")
        year, quarter = int(parts[0]), int(parts[1])
        return year * 4 + quarter
    elif granularity == TimeGranularity.SEMIANNUAL:
        # Parse YYYY-Hn
        parts = bucket.split("-H")
        year, half = int(parts[0]), int(parts[1])
        return year * 2 + half
    elif granularity == TimeGranularity.ANNUAL:
        return int(bucket)
    else:
        return 0


def _check_alignment(dt: datetime, granularity: TimeGranularity) -> tuple[bool, str]:
    """Check if a datetime aligns to a canonical position for the granularity.

    Returns (is_aligned, alignment_type).
    """
    if granularity == TimeGranularity.DAILY:
        # For daily, any date is aligned
        return True, "any"
    elif granularity in (TimeGranularity.WEEKLY, TimeGranularity.BIWEEKLY):
        # Check if it's a Monday (common week start)
        if dt.weekday() == 0:
            return True, "week_monday"
        # Or Sunday (US week start)
        elif dt.weekday() == 6:
            return True, "week_sunday"
        return False, "mid_week"
    elif granularity == TimeGranularity.MONTHLY:
        if dt.day == 1:
            return True, "month_start"
        # Check for end of month (28-31)
        if dt.day >= 28:
            # Check if next day is new month
            next_day = dt + pd.Timedelta(days=1)
            if next_day.month != dt.month:
                return True, "month_end"
        return False, "mid_month"
    elif granularity == TimeGranularity.QUARTERLY:
        quarter_starts = {1, 4, 7, 10}
        if dt.day == 1 and dt.month in quarter_starts:
            return True, "quarter_start"
        # Check quarter end
        quarter_ends = {3, 6, 9, 12}
        if dt.month in quarter_ends and dt.day >= 28:
            next_day = dt + pd.Timedelta(days=1)
            if next_day.month != dt.month:
                return True, "quarter_end"
        return False, "mid_quarter"
    elif granularity == TimeGranularity.SEMIANNUAL:
        if dt.day == 1 and dt.month in {1, 7}:
            return True, "half_start"
        if dt.month in {6, 12} and dt.day >= 28:
            next_day = dt + pd.Timedelta(days=1)
            if next_day.month != dt.month:
                return True, "half_end"
        return False, "mid_half"
    elif granularity == TimeGranularity.ANNUAL:
        if dt.day == 1 and dt.month == 1:
            return True, "year_start"
        if dt.month == 12 and dt.day == 31:
            return True, "year_end"
        return False, "mid_year"
    return False, "unknown"


def _detect_business_day_pattern(sorted_dates: pd.Series) -> tuple[bool, float]:
    """Detect if dates follow a business day (Mon-Fri) pattern.

    Args:
        sorted_dates: Sorted pandas Series of datetime values.

    Returns:
        Tuple of (is_business_day_pattern, confidence).
    """
    if len(sorted_dates) < 10:
        return False, 0.0

    # Count weekday distribution
    weekdays = sorted_dates.dt.weekday  # 0=Monday, 6=Sunday
    weekday_counts = weekdays.value_counts()

    # Business days are 0-4 (Mon-Fri)
    business_day_count = sum(weekday_counts.get(d, 0) for d in range(5))
    weekend_count = sum(weekday_counts.get(d, 0) for d in [5, 6])
    total = len(sorted_dates)

    # Calculate ratios
    business_ratio = business_day_count / total if total > 0 else 0.0
    weekend_ratio = weekend_count / total if total > 0 else 0.0

    # Strong business day pattern: >90% weekdays, <10% weekends
    if business_ratio >= 0.9 and weekend_ratio <= 0.1:
        return True, 0.9

    # Moderate business day pattern: >80% weekdays
    if business_ratio >= 0.8 and weekend_ratio <= 0.2:
        return True, 0.7

    # Check for expected 5/7 ratio (71%) with some tolerance
    # This handles cases where we sample evenly
    expected_business_ratio = 5 / 7  # ~0.714
    if abs(business_ratio - expected_business_ratio) <= 0.1:
        # Could be either business days with occasional weekends or all days
        # If weekend count is significantly below expected, likely business days
        expected_weekend_ratio = 2 / 7  # ~0.286
        if weekend_ratio < expected_weekend_ratio * 0.5:
            return True, 0.6

    return False, 0.0


def _compute_grid_fit(
    sorted_dates: pd.Series,
    granularity: TimeGranularity,
) -> GridFitEvidence:
    """Compute grid-fit metrics for a candidate granularity.

    Args:
        sorted_dates: Sorted pandas Series of datetime values.
        granularity: The granularity to test.

    Returns:
        GridFitEvidence with bucket_coverage, gap_rate, alignment_score, and monotonicity.
    """
    # Convert to bucket keys
    buckets = [_to_period_bucket(dt.to_pydatetime(), granularity) for dt in sorted_dates]

    # Bucket coverage: unique buckets / total timestamps
    # High coverage means timestamps map to distinct periods (good for that granularity)
    # Low coverage means multiple records per period
    unique_buckets = list(dict.fromkeys(buckets))  # Preserve order
    bucket_coverage = len(unique_buckets) / len(buckets) if buckets else 0.0

    # Records per bucket: average records mapping to same bucket
    records_per_bucket = len(buckets) / len(unique_buckets) if unique_buckets else 1.0

    # Check monotonicity
    bucket_indices = [_bucket_to_index(b, granularity) for b in unique_buckets]
    is_monotonic = all(
        bucket_indices[i] <= bucket_indices[i + 1]
        for i in range(len(bucket_indices) - 1)
    ) if len(bucket_indices) > 1 else True

    # Gap rate: fraction of steps > 1
    if len(bucket_indices) > 1:
        steps = [
            bucket_indices[i + 1] - bucket_indices[i]
            for i in range(len(bucket_indices) - 1)
        ]
        gap_rate = sum(1 for s in steps if s > 1) / len(steps)
    else:
        gap_rate = 0.0

    # Alignment score and dominant alignment type
    alignment_types: dict[str, int] = {}
    for dt in sorted_dates:
        _, align_type = _check_alignment(dt.to_pydatetime(), granularity)
        alignment_types[align_type] = alignment_types.get(align_type, 0) + 1

    # Find dominant alignment
    if alignment_types:
        dominant_alignment = max(alignment_types, key=lambda k: alignment_types[k])
        dominant_count = alignment_types[dominant_alignment]
        alignment_consistency = dominant_count / len(sorted_dates) if sorted_dates.size > 0 else 0.0
        # Alignment score: count only "good" alignments (not mid_week, mid_month, etc.)
        good_alignments = {
            "any", "week_monday", "week_sunday", "month_start", "month_end",
            "quarter_start", "quarter_end", "half_start", "half_end",
            "year_start", "year_end",
        }
        aligned_count = sum(
            count for align_type, count in alignment_types.items()
            if align_type in good_alignments
        )
        alignment_score = aligned_count / len(sorted_dates) if sorted_dates.size > 0 else 0.0
    else:
        dominant_alignment = None
        alignment_consistency = 0.0
        alignment_score = 0.0

    return GridFitEvidence(
        bucket_coverage=bucket_coverage,
        gap_rate=gap_rate,
        alignment_score=alignment_score,
        monotonic=is_monotonic,
        dominant_alignment=dominant_alignment,
        alignment_consistency=alignment_consistency,
        records_per_bucket=records_per_bucket,
    )


def _score_grid_fit(
    grid_fit: GridFitEvidence,
    granularity: TimeGranularity,
    name_hint: float = 0.0,
) -> float:
    """Compute overall score from grid-fit evidence with shape-conditional weights.

    Different granularities have different expectations:
    - Daily/Weekly: High coverage important (few duplicates expected)
    - Monthly/Quarterly/Annual: Gaps more acceptable, alignment matters more

    Args:
        grid_fit: GridFitEvidence for a granularity.
        granularity: The granularity being scored (affects weights).
        name_hint: Score from name-based detection (0.0-1.0).

    Returns:
        Combined score (0.0-1.0).
    """
    if not grid_fit.monotonic:
        return 0.0  # Non-monotonic sequences are invalid

    # Shape-conditional weights
    # Fine granularities: coverage matters most, gaps are suspicious
    # Coarse granularities: gaps are normal, alignment and consistency matter more
    if granularity in (TimeGranularity.DAILY,):
        # Daily: high coverage critical, gaps suspicious
        w_coverage = 0.45
        w_gaps = 0.25
        w_alignment = 0.10
        w_name = 0.20
    elif granularity in (TimeGranularity.WEEKLY, TimeGranularity.BIWEEKLY):
        # Weekly: coverage important, some gaps acceptable
        w_coverage = 0.40
        w_gaps = 0.25
        w_alignment = 0.15
        w_name = 0.20
    elif granularity in (TimeGranularity.MONTHLY,):
        # Monthly: gaps common (missing months), alignment more relevant
        w_coverage = 0.30
        w_gaps = 0.20
        w_alignment = 0.25
        w_name = 0.25
    elif granularity in (TimeGranularity.QUARTERLY, TimeGranularity.SEMIANNUAL):
        # Quarterly/Semiannual: gaps very common, alignment very relevant
        w_coverage = 0.25
        w_gaps = 0.15
        w_alignment = 0.30
        w_name = 0.30
    elif granularity in (TimeGranularity.ANNUAL,):
        # Annual: gaps expected, alignment less meaningful (any day in year is fine)
        w_coverage = 0.30
        w_gaps = 0.10
        w_alignment = 0.25
        w_name = 0.35
    else:
        # Default weights
        w_coverage = 0.35
        w_gaps = 0.25
        w_alignment = 0.20
        w_name = 0.20

    # Penalize "wrong" duplicates: if records_per_bucket is high but coverage is low,
    # this granularity is likely too coarse (data is actually finer-grained)
    coverage_penalty = 0.0
    if grid_fit.records_per_bucket > 2.0 and granularity != TimeGranularity.DAILY:
        # Many records per bucket suggests data is finer than this granularity
        coverage_penalty = min(0.2, (grid_fit.records_per_bucket - 2.0) * 0.05)

    score = (
        w_coverage * grid_fit.bucket_coverage
        + w_gaps * (1.0 - grid_fit.gap_rate)
        + w_alignment * grid_fit.alignment_score
        + w_name * name_hint
        - coverage_penalty
    )
    return min(1.0, max(0.0, score))


def detect_granularity_from_values(
    series: pd.Series,
    *,
    sample_size: int = 1000,
    name_hint: TimeGranularity = TimeGranularity.UNKNOWN,
) -> GranularityResult:
    """Detect time granularity using grid-fit scoring.

    This approach checks whether timestamps align to calendar period buckets
    (day/week/month/quarter/year) rather than relying on median day differences.
    This handles gaps and irregular reporting dates correctly.

    Args:
        series: A pandas Series containing date/datetime values.
        sample_size: Maximum number of values to sample for analysis.
        name_hint: Optional hint from column name analysis.

    Returns:
        GranularityResult with detected granularity, confidence, and evidence.
    """
    evidence: list[str] = []
    grid_fit_results: dict[str, GridFitEvidence] = {}

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

    # Sample using contiguous windows if too large
    # Don't random sample - that breaks diff logic and creates artificial gaps
    if len(dates) > sample_size:
        # Sort first, then take evenly spaced indices
        sorted_dates = dates.sort_values().reset_index(drop=True)
        step = len(sorted_dates) // sample_size
        indices = range(0, len(sorted_dates), step)
        dates = sorted_dates.take(list(indices)[:sample_size])
        evidence.append(f"Sampled {len(dates)} values from {len(series)} total (evenly spaced)")
    else:
        dates = dates.sort_values()

    # Test each candidate granularity
    candidates: list[TimeGranularity] = [
        TimeGranularity.DAILY,
        TimeGranularity.WEEKLY,
        TimeGranularity.BIWEEKLY,
        TimeGranularity.MONTHLY,
        TimeGranularity.QUARTERLY,
        TimeGranularity.SEMIANNUAL,
        TimeGranularity.ANNUAL,
    ]

    scores: dict[TimeGranularity, float] = {}

    for granularity in candidates:
        grid_fit = _compute_grid_fit(dates, granularity)
        grid_fit_results[granularity.value] = grid_fit

        # Apply name hint bonus if it matches
        hint_bonus = 0.0
        if name_hint == granularity:
            hint_bonus = 1.0
        elif name_hint != TimeGranularity.UNKNOWN:
            hint_bonus = 0.0  # Different granularity hinted

        score = _score_grid_fit(grid_fit, granularity, name_hint=hint_bonus)
        scores[granularity] = score

    # Check for business day pattern (affects daily detection)
    is_business_day, business_confidence = _detect_business_day_pattern(dates)
    if is_business_day and business_confidence >= 0.6:
        evidence.append(f"Business day pattern detected (confidence: {business_confidence:.2f})")
        # Business day pattern supports daily granularity - data is daily but excludes weekends
        # Boost daily score if it's close to weekly
        if TimeGranularity.DAILY in scores and TimeGranularity.WEEKLY in scores:
            daily_score = scores[TimeGranularity.DAILY]
            weekly_score = scores[TimeGranularity.WEEKLY]
            # If daily and weekly are close, business day pattern suggests daily
            if abs(daily_score - weekly_score) < 0.15:
                scores[TimeGranularity.DAILY] += 0.1 * business_confidence

    # Find best candidate
    best_granularity = max(scores, key=lambda g: scores[g])
    best_score = scores[best_granularity]

    # Calculate confidence as margin between top 2 scores
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0] > 0:
        margin = sorted_scores[0] - sorted_scores[1]
        confidence = min(1.0, best_score * 0.6 + margin * 0.4)
    else:
        confidence = best_score

    # Threshold for valid detection
    if best_score < 0.3:
        evidence.append("No granularity scored above threshold (0.3)")
        return GranularityResult(
            granularity=TimeGranularity.UNKNOWN,
            confidence=0.0,
            evidence=evidence,
            grid_fit=grid_fit_results,
        )

    # Add evidence
    best_fit = grid_fit_results[best_granularity.value]
    evidence.append(f"Detected {best_granularity.value} granularity (score: {best_score:.2f})")
    evidence.append(f"Bucket coverage: {best_fit.bucket_coverage:.2%}")
    evidence.append(f"Gap rate: {best_fit.gap_rate:.2%}")
    evidence.append(f"Alignment score: {best_fit.alignment_score:.2%}")

    return GranularityResult(
        granularity=best_granularity,
        confidence=confidence,
        evidence=evidence,
        grid_fit=grid_fit_results,
    )


def detect_granularity_from_name(column_name: str) -> GranularityResult:
    """Detect time granularity from column name patterns.

    Note: Name-based detection should be used as a hint, not as authoritative.
    Columns named 'year' or 'month' might just be dimensions, not the primary
    time axis. Use this with lower confidence unless values confirm it.

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
                # Lower confidence (0.5) since name alone is weak signal
                return GranularityResult(
                    granularity=granularity,
                    confidence=0.5,
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

    This is the main entry point that combines grid-fit value-based detection
    with name-based hints. The name hint is passed to the grid-fit algorithm
    to boost scores for matching granularities.

    Args:
        series: A pandas Series containing the column data.
        column_name: The name of the column.

    Returns:
        GranularityResult with the best detected granularity.
    """
    # Get name hint first
    name_result = detect_granularity_from_name(column_name)
    name_hint = name_result.granularity

    # Run value-based detection with name hint
    value_result = detect_granularity_from_values(series, name_hint=name_hint)

    # Combine evidence
    combined_evidence = value_result.evidence + name_result.evidence

    # If value-based detection succeeded with reasonable confidence, use it
    if value_result.granularity != TimeGranularity.UNKNOWN and value_result.confidence >= 0.4:
        # Boost confidence if name matches
        confidence = value_result.confidence
        if name_hint == value_result.granularity:
            confidence = min(1.0, confidence + 0.1)
            combined_evidence.append("Name-based detection confirms value-based detection")

        return GranularityResult(
            granularity=value_result.granularity,
            confidence=confidence,
            evidence=combined_evidence,
            grid_fit=value_result.grid_fit,
        )

    # Fall back to name-based detection with low confidence
    if name_hint != TimeGranularity.UNKNOWN:
        combined_evidence.append("Using name-based detection as fallback (low confidence)")
        return GranularityResult(
            granularity=name_hint,
            confidence=name_result.confidence * 0.7,  # Reduce confidence for fallback
            evidence=combined_evidence,
            grid_fit=value_result.grid_fit,
        )

    # No detection succeeded
    return GranularityResult(
        granularity=TimeGranularity.UNKNOWN,
        confidence=0.0,
        evidence=combined_evidence,
        grid_fit=value_result.grid_fit,
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

    # Try ISO year-month: 2024M01, 2024_01
    match = re.match(r"^(\d{4})[Mm_](\d{2})$", name)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return ParsedTimeHeader(
                column_name=column_name,
                parsed_date=date(year, month, 1),
                granularity=TimeGranularity.MONTHLY,
                original_format="YYYYMmm",
            )

    # Try ISO week: 2024W01, 2024-W01
    match = re.match(r"^(\d{4})[-_]?[Ww](\d{2})$", name)
    if match:
        year, week = int(match.group(1)), int(match.group(2))
        if 1 <= week <= 53:
            # Convert ISO week to approximate date (Monday of that week)
            jan4 = date(year, 1, 4)  # Jan 4 is always in week 1
            week1_monday = jan4 - pd.Timedelta(days=jan4.weekday())
            target_date = week1_monday + pd.Timedelta(weeks=week - 1)
            return ParsedTimeHeader(
                column_name=column_name,
                parsed_date=target_date,
                granularity=TimeGranularity.WEEKLY,
                original_format="YYYYWww",
            )

    # Try fiscal year: FY2024, FY 2024
    # Note: We don't assume Jan 1 start - fiscal year varies by org/country
    match = re.match(r"^FY[-_\s]?(\d{4})$", name, re.I)
    if match:
        year = int(match.group(1))
        return ParsedTimeHeader(
            column_name=column_name,
            parsed_date=date(year, 1, 1),  # Placeholder date
            granularity=TimeGranularity.ANNUAL,
            original_format="FYnnnn",
            is_fiscal=True,  # Flag that fiscal year start is unknown
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


def check_header_grid_coherence(
    column_names: Sequence[str],
    min_headers: int = 3,
) -> HeaderGridResult:
    """Check whether column headers form a coherent time grid.

    This function parses all candidate time headers and checks:
    - Whether they share a dominant granularity
    - Whether they form a monotonic sequence
    - Whether they use consistent format patterns
    - Whether gaps between periods are reasonable

    Args:
        column_names: List of column names to analyze.
        min_headers: Minimum number of parseable time headers required.

    Returns:
        HeaderGridResult with coherence metrics and evidence.
    """
    evidence: list[str] = []

    # Parse all headers
    parsed = parse_time_headers(column_names)

    if len(parsed) < min_headers:
        return HeaderGridResult(
            parsed_headers=parsed,
            dominant_granularity=TimeGranularity.UNKNOWN,
            is_monotonic=False,
            has_consistent_format=False,
            gap_rate=1.0,
            confidence=0.0,
            evidence=[f"Only {len(parsed)} parseable time headers (need {min_headers}+)"],
        )

    # Find dominant granularity (mode)
    granularities = [p.granularity for p in parsed]
    granularity_counts: dict[TimeGranularity, int] = {}
    for g in granularities:
        granularity_counts[g] = granularity_counts.get(g, 0) + 1

    dominant = max(granularity_counts, key=lambda g: granularity_counts[g])
    dominant_count = granularity_counts[dominant]
    granularity_consistency = dominant_count / len(parsed)

    evidence.append(f"Dominant granularity: {dominant.value} ({dominant_count}/{len(parsed)})")

    # Check format consistency
    formats = [p.original_format for p in parsed if p.original_format]
    format_counts: dict[str, int] = {}
    for f in formats:
        format_counts[f] = format_counts.get(f, 0) + 1

    if format_counts:
        dominant_format = max(format_counts, key=lambda f: format_counts[f])
        format_consistency = format_counts[dominant_format] / len(formats)
        has_consistent_format = format_consistency >= 0.8
    else:
        format_consistency = 0.0
        has_consistent_format = False

    evidence.append(f"Format consistency: {format_consistency:.0%}")

    # Check monotonicity of dates
    dates = [p.parsed_date for p in parsed if p.parsed_date is not None]
    is_monotonic = all(dates[i] <= dates[i + 1] for i in range(len(dates) - 1))

    if not is_monotonic:
        # Check if headers are in original column order (might be out of order)
        evidence.append("Headers not in monotonic date order")

    # Calculate gap rate using grid-fit approach
    if len(dates) >= 2 and dominant != TimeGranularity.UNKNOWN:
        # Convert dates to bucket indices
        bucket_indices = [
            _bucket_to_index(
                _to_period_bucket(
                    datetime.combine(d, datetime.min.time()),
                    dominant
                ),
                dominant
            )
            for d in dates
        ]
        sorted_indices = sorted(bucket_indices)

        steps = [
            sorted_indices[i + 1] - sorted_indices[i]
            for i in range(len(sorted_indices) - 1)
        ]
        gap_rate = sum(1 for s in steps if s > 1) / len(steps) if steps else 0.0
    else:
        gap_rate = 1.0

    evidence.append(f"Gap rate: {gap_rate:.0%}")

    # Calculate confidence
    if granularity_consistency >= 0.9 and has_consistent_format and gap_rate < 0.3:
        confidence = 0.9
    elif granularity_consistency >= 0.7 and gap_rate < 0.5:
        confidence = 0.7
    elif granularity_consistency >= 0.5:
        confidence = 0.5
    else:
        confidence = 0.3

    return HeaderGridResult(
        parsed_headers=parsed,
        dominant_granularity=dominant,
        is_monotonic=is_monotonic,
        has_consistent_format=has_consistent_format,
        gap_rate=gap_rate,
        confidence=confidence,
        evidence=evidence,
    )


def _check_metadata_consistency(
    series: pd.Series,
    sample_size: int = 100,
) -> SeriesMetadataConsistency:
    """Check whether a metadata column is constant or varies per-row.

    This helps detect datasets with multiple series that have different
    start dates or frequencies per row.

    Args:
        series: A pandas Series to check.
        sample_size: Number of values to sample.

    Returns:
        SeriesMetadataConsistency with constant/varying assessment.
    """
    evidence: list[str] = []
    sample = series.dropna().head(sample_size)

    if len(sample) == 0:
        return SeriesMetadataConsistency(
            is_constant=True,
            unique_values=0,
            mode_coverage=0.0,
            evidence=["No non-null values"],
        )

    unique_values = sample.nunique()

    # Get mode coverage
    mode = sample.mode()
    mode_coverage = (sample == mode.iloc[0]).sum() / len(sample) if len(mode) > 0 else 0.0

    # Determine if constant (or mostly constant)
    is_constant = unique_values == 1 or mode_coverage >= 0.95

    if is_constant:
        evidence.append(f"Constant value (mode covers {mode_coverage:.0%})")
    else:
        evidence.append(f"Per-row variation ({unique_values} unique values)")

    return SeriesMetadataConsistency(
        is_constant=is_constant,
        unique_values=unique_values,
        mode_coverage=mode_coverage,
        evidence=evidence,
    )


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
                # Check if frequency is constant or per-row
                consistency = _check_metadata_consistency(df[col])
                freq_value = _extract_frequency_from_column(df[col])

                if freq_value != TimeGranularity.UNKNOWN:
                    frequency = freq_value
                    if consistency.is_constant:
                        confidence = 0.8
                        evidence.append(f"Found frequency from column '{col}': {frequency.value} (constant)")
                    else:
                        confidence = 0.5
                        evidence.append(
                            f"Found frequency from column '{col}': {frequency.value} "
                            f"(varies per-row, {consistency.unique_values} unique values)"
                        )
                        evidence.append("WARNING: Dataset may contain multiple series with different frequencies")
                    break

    # Find start date
    start_date: date | None = None
    for col in metadata_columns:
        if col == array_column:
            continue

        for pattern in start_date_patterns:
            if pattern.search(str(col)):
                # Check consistency
                consistency = _check_metadata_consistency(df[col])
                start_date = _extract_date_from_column(df[col])

                if start_date:
                    if consistency.is_constant:
                        evidence.append(f"Found start date from column '{col}': {start_date}")
                    else:
                        evidence.append(
                            f"Found start date from column '{col}': {start_date} "
                            f"(varies per-row, {consistency.unique_values} unique values)"
                        )
                        evidence.append("WARNING: Different rows may have different start dates")
                    break

    # Find end date
    end_date: date | None = None
    for col in metadata_columns:
        if col == array_column:
            continue

        for pattern in end_date_patterns:
            if pattern.search(str(col)):
                # Check consistency
                consistency = _check_metadata_consistency(df[col])
                end_date = _extract_date_from_column(df[col])

                if end_date:
                    if consistency.is_constant:
                        evidence.append(f"Found end date from column '{col}': {end_date}")
                    else:
                        evidence.append(
                            f"Found end date from column '{col}': {end_date} "
                            f"(varies per-row, {consistency.unique_values} unique values)"
                        )
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
        "bw": TimeGranularity.BIWEEKLY,
        "biweekly": TimeGranularity.BIWEEKLY,
        "bi-weekly": TimeGranularity.BIWEEKLY,
        "fortnightly": TimeGranularity.BIWEEKLY,
        "m": TimeGranularity.MONTHLY,
        "month": TimeGranularity.MONTHLY,
        "monthly": TimeGranularity.MONTHLY,
        "q": TimeGranularity.QUARTERLY,
        "quarter": TimeGranularity.QUARTERLY,
        "quarterly": TimeGranularity.QUARTERLY,
        "s": TimeGranularity.SEMIANNUAL,
        "h": TimeGranularity.SEMIANNUAL,
        "semi-annual": TimeGranularity.SEMIANNUAL,
        "semiannual": TimeGranularity.SEMIANNUAL,
        "half-yearly": TimeGranularity.SEMIANNUAL,
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
    expected_biweekly = (total_days // 14) + 1
    expected_monthly = ((end_date.year - start_date.year) * 12 +
                        (end_date.month - start_date.month)) + 1
    expected_quarterly = ((end_date.year - start_date.year) * 4 +
                          ((end_date.month - 1) // 3 - (start_date.month - 1) // 3)) + 1
    expected_semiannual = ((end_date.year - start_date.year) * 2 +
                           ((end_date.month - 1) // 6 - (start_date.month - 1) // 6)) + 1
    expected_annual = (end_date.year - start_date.year) + 1

    # Find closest match (with 10% tolerance)
    candidates = [
        (TimeGranularity.DAILY, expected_daily),
        (TimeGranularity.WEEKLY, expected_weekly),
        (TimeGranularity.BIWEEKLY, expected_biweekly),
        (TimeGranularity.MONTHLY, expected_monthly),
        (TimeGranularity.QUARTERLY, expected_quarterly),
        (TimeGranularity.SEMIANNUAL, expected_semiannual),
        (TimeGranularity.ANNUAL, expected_annual),
    ]

    for granularity, expected in candidates:
        if expected > 0:
            ratio = array_length / expected
            if 0.9 <= ratio <= 1.1:  # Within 10%
                return granularity

    return TimeGranularity.UNKNOWN


def rank_time_axis_candidates(
    df: pd.DataFrame,
    candidate_columns: Sequence[str] | None = None,
) -> list[TimeAxisCandidate]:
    """Rank columns by likelihood of being the primary time axis.

    When a dataset has multiple time-related columns (e.g., survey_date, month, year),
    this function helps identify which is most likely the primary time axis vs
    derived dimensions.

    Ranking criteria:
    - Higher unique bucket count (more temporal variation)
    - Lower null rate
    - Name patterns like 'date', 'period', 'time'
    - Successfully parses as dates
    - Participates in grain (if detectable)

    Args:
        df: DataFrame to analyze.
        candidate_columns: Optional list of columns to consider. If None, all columns
            are checked for date parseability.

    Returns:
        List of TimeAxisCandidate sorted by likelihood (best first).
    """
    candidates: list[TimeAxisCandidate] = []

    # Determine which columns to consider
    if candidate_columns is None:
        candidate_columns = []
        for col in df.columns:
            sample = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                success_rate = parsed.notna().sum() / len(sample)
                if success_rate >= 0.5:  # Lower threshold for discovery
                    candidate_columns.append(str(col))
            except (ValueError, TypeError):
                continue

    # Time-related name patterns (ordered by specificity)
    # Use (?:^|[_\-\s]) and (?:$|[_\-\s]) to handle compound names with underscores
    time_name_patterns = [
        (re.compile(r"(?:^|[_\-\s])(date|datetime|timestamp)(?:$|[_\-\s])", re.I), 1.0),
        (re.compile(r"(?:^|[_\-\s])(period|time)(?:$|[_\-\s])", re.I), 0.9),
        (re.compile(r"(?:^|[_\-\s])(created|updated|modified)[_\-\s]?(at|on|date)?(?:$|[_\-\s])?", re.I), 0.8),
        (re.compile(r"(?:^|[_\-\s])(start|end|from|to)[_\-\s]?(date|time)?(?:$|[_\-\s])?", re.I), 0.7),
        (re.compile(r"(?:^|[_\-\s])(year|month|quarter|week|day)(?:$|[_\-\s])", re.I), 0.5),  # Could be dimension
    ]

    for col in candidate_columns:
        if col not in df.columns:
            continue

        series = df[col]
        evidence: list[str] = []

        # Calculate null rate
        null_rate = series.isna().sum() / len(series) if len(series) > 0 else 1.0

        # Try to parse as dates
        try:
            dates = pd.to_datetime(series.dropna(), errors="coerce")
            valid_dates = dates.dropna()
            is_parseable = len(valid_dates) >= len(series.dropna()) * 0.8
        except (ValueError, TypeError):
            is_parseable = False
            valid_dates = pd.Series(dtype="datetime64[ns]")

        # Calculate name score
        name_score = 0.0
        for pattern, score in time_name_patterns:
            if pattern.search(col):
                name_score = score
                evidence.append(f"Name matches time pattern: {pattern.pattern}")
                break

        # Get granularity and unique bucket count
        if is_parseable and len(valid_dates) >= 2:
            granularity_result = detect_granularity_from_values(valid_dates)
            granularity = granularity_result.granularity
            confidence = granularity_result.confidence

            # Count unique buckets at detected granularity
            if granularity != TimeGranularity.UNKNOWN:
                sorted_dates = valid_dates.sort_values()
                grid_fit = _compute_grid_fit(sorted_dates, granularity)
                unique_bucket_count = int(grid_fit.bucket_coverage * len(sorted_dates))
                evidence.append(f"Detected granularity: {granularity.value}")
                evidence.append(f"Unique buckets: {unique_bucket_count}")
            else:
                unique_bucket_count = int(valid_dates.nunique())
        else:
            granularity = TimeGranularity.UNKNOWN
            confidence = 0.0
            unique_bucket_count = 0

        # Penalize columns that look like derived dimensions
        # (e.g., 'year' column with only 3-4 unique integer values)
        if name_score == 0.5:  # year/month/quarter/week/day pattern
            unique_values = series.nunique()
            if unique_values <= 10:
                name_score *= 0.5
                evidence.append(f"Low cardinality ({unique_values}) suggests dimension")

        candidates.append(TimeAxisCandidate(
            column_name=col,
            granularity=granularity,
            confidence=confidence,
            unique_bucket_count=unique_bucket_count,
            null_rate=null_rate,
            name_score=name_score,
            is_parseable=is_parseable,
            evidence=evidence,
        ))

    # Sort by composite score: confidence + name_score - null_rate + log(unique_buckets)
    def sort_key(c: TimeAxisCandidate) -> float:
        bucket_bonus = math.log(c.unique_bucket_count + 1) / 10  # Normalize
        return (
            c.confidence * 0.4
            + c.name_score * 0.3
            + (1 - c.null_rate) * 0.2
            + bucket_bonus * 0.1
        )

    candidates.sort(key=sort_key, reverse=True)
    return candidates


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


def extract_time_axis(
    df: pd.DataFrame,
    column_name: str | None = None,
) -> TimeAxisResult:
    """Extract a structured time axis artifact from a DataFrame.

    This function produces a standardized time axis representation that can
    be consumed by Invariant for compatibility/comparability checks.

    Args:
        df: DataFrame to analyze.
        column_name: Optional specific column to use as time axis.
            If None, auto-detects the best time column.

    Returns:
        TimeAxisResult with structured time axis information.
    """
    # If no column specified, try to auto-detect
    if column_name is None:
        time_ranges = extract_time_ranges(df)
        if not time_ranges:
            return TimeAxisResult(
                kind="column",
                granularity=TimeGranularity.UNKNOWN,
                start=None,
                end=None,
                has_gaps=True,
                confidence=0.0,
                evidence={"error": "No time columns detected"},
            )
        # Pick the one with most rows and best granularity detection
        best = max(
            time_ranges,
            key=lambda r: (r.granularity != TimeGranularity.UNKNOWN, r.row_count),
        )
        column_name = best.column_name

    if column_name not in df.columns:
        return TimeAxisResult(
            kind="column",
            granularity=TimeGranularity.UNKNOWN,
            start=None,
            end=None,
            has_gaps=True,
            confidence=0.0,
            evidence={"error": f"Column '{column_name}' not found"},
        )

    # Get granularity result with full grid-fit evidence
    series = df[column_name]
    granularity_result = detect_granularity(series, column_name)

    # Extract time range
    time_range = extract_time_range(series, column_name)

    # Determine alignment from grid-fit evidence
    alignment: str | None = None
    has_gaps = True

    if granularity_result.grid_fit and granularity_result.granularity != TimeGranularity.UNKNOWN:
        grid_fit = granularity_result.grid_fit.get(granularity_result.granularity.value)
        if grid_fit:
            has_gaps = grid_fit.gap_rate > 0.0

            # Use dominant alignment from grid-fit evidence
            # Only report alignment if it's consistent across the series
            if grid_fit.dominant_alignment and grid_fit.alignment_consistency > 0.7:
                # Filter out uninformative alignment types
                if grid_fit.dominant_alignment not in ("unknown", "any", "mid_week", "mid_month", "mid_quarter", "mid_half", "mid_year"):
                    alignment = grid_fit.dominant_alignment

    # Build evidence dict
    evidence_dict: dict[str, float | str | bool] = {
        "column": column_name,
    }
    if granularity_result.grid_fit and granularity_result.granularity != TimeGranularity.UNKNOWN:
        grid_fit = granularity_result.grid_fit.get(granularity_result.granularity.value)
        if grid_fit:
            evidence_dict["bucket_coverage"] = grid_fit.bucket_coverage
            evidence_dict["gap_rate"] = grid_fit.gap_rate
            evidence_dict["alignment_score"] = grid_fit.alignment_score
            evidence_dict["alignment_consistency"] = grid_fit.alignment_consistency
            evidence_dict["monotonic"] = grid_fit.monotonic
            if grid_fit.dominant_alignment:
                evidence_dict["dominant_alignment"] = grid_fit.dominant_alignment
            evidence_dict["records_per_bucket"] = grid_fit.records_per_bucket

    return TimeAxisResult(
        kind="column",
        granularity=granularity_result.granularity,
        start=time_range.min_date,
        end=time_range.max_date,
        has_gaps=has_gaps,
        confidence=granularity_result.confidence,
        evidence=evidence_dict,
        alignment=alignment,
    )


def extract_time_axis_from_headers(
    column_names: Sequence[str],
) -> TimeAxisResult:
    """Extract a structured time axis artifact from wide time column headers.

    This is used when the time axis is represented as column headers rather
    than as a column of values (wide format time series).

    Args:
        column_names: List of column names that may represent time periods.

    Returns:
        TimeAxisResult with structured time axis information.
    """
    # Check header grid coherence
    header_result = check_header_grid_coherence(column_names)

    if header_result.dominant_granularity == TimeGranularity.UNKNOWN:
        return TimeAxisResult(
            kind="wide_headers",
            granularity=TimeGranularity.UNKNOWN,
            start=None,
            end=None,
            has_gaps=True,
            confidence=0.0,
            evidence={"error": "No time headers detected"},
        )

    # Extract start and end dates from parsed headers
    dates = [h.parsed_date for h in header_result.parsed_headers if h.parsed_date]
    sorted_dates = sorted(dates)

    start_date = sorted_dates[0] if sorted_dates else None
    end_date = sorted_dates[-1] if sorted_dates else None

    # Build evidence
    evidence_dict: dict[str, float | str | bool] = {
        "parsed_headers": len(header_result.parsed_headers),
        "total_columns": len(column_names),
        "format_consistent": header_result.has_consistent_format,
        "monotonic": header_result.is_monotonic,
        "gap_rate": header_result.gap_rate,
    }

    return TimeAxisResult(
        kind="wide_headers",
        granularity=header_result.dominant_granularity,
        start=start_date,
        end=end_date,
        has_gaps=header_result.gap_rate > 0.0,
        confidence=header_result.confidence,
        evidence=evidence_dict,
    )
