"""Special column detection and user flagging for Datasculpt.

This module provides functionality to detect and flag special columns
such as weights, denominators, suppression flags, and quality flags.

Key improvements over naive name-matching:
- Multiple candidates per column with confidence scoring
- Value-based checks on sampled data
- Cross-column relationship detection
- Explicit ambiguity handling (requires_user_confirm)
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from datasculpt.core.types import PrimitiveType

if TYPE_CHECKING:
    from datasculpt.core.types import ColumnEvidence


class SpecialColumnType(str, Enum):
    """Types of special columns in datasets."""

    WEIGHT = "weight"
    DENOMINATOR = "denominator"
    SUPPRESSION_FLAG = "suppression_flag"
    QUALITY_FLAG = "quality_flag"


class SpecialColumnStatus(str, Enum):
    """Status of special column detection."""

    AUTO_LOCKED = "auto_locked"  # High confidence, no user confirmation needed
    NEEDS_CONFIRMATION = "needs_confirmation"  # Ambiguous, needs user input
    USER_LOCKED = "user_locked"  # User explicitly set this


@dataclass
class SpecialColumnCandidate:
    """A candidate special type for a column with scoring."""

    flag_type: SpecialColumnType
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def __lt__(self, other: SpecialColumnCandidate) -> bool:
        """Sort by confidence descending."""
        return self.confidence > other.confidence


@dataclass
class SpecialColumnResult:
    """Result of special column detection for a single column.

    Contains all candidates and the selected winner (if any).
    """

    column_name: str
    candidates: list[SpecialColumnCandidate] = field(default_factory=list)
    selected: SpecialColumnType | None = None
    status: SpecialColumnStatus = SpecialColumnStatus.NEEDS_CONFIRMATION

    @property
    def requires_user_confirm(self) -> bool:
        """Whether user confirmation is needed."""
        return self.status == SpecialColumnStatus.NEEDS_CONFIRMATION


# Legacy dataclass for backwards compatibility
@dataclass
class SpecialColumnFlag:
    """Flag indicating a column has a special role.

    Kept for backwards compatibility with existing code.
    """

    column_name: str
    flag_type: SpecialColumnType
    confidence: float
    evidence: list[str]


# -----------------------------------------------------------------------------
# Pattern definitions for name-based detection
# -----------------------------------------------------------------------------

# Strong patterns: very specific, high signal
# Note: Use (?:^|_) and (?:$|_) instead of \b to handle underscore-separated names
WEIGHT_PATTERNS_STRONG = (
    re.compile(r"(?:^|_)sample_weight(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)sampling_weight(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)person_weight(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)household_weight(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)final_weight(?:$|_)", re.IGNORECASE),
)

WEIGHT_PATTERNS_MODERATE = (
    re.compile(r"(?:^|_)weight(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)wgt(?:$|_)", re.IGNORECASE),
    re.compile(r"_wgt$", re.IGNORECASE),
    re.compile(r"_weight$", re.IGNORECASE),
    re.compile(r"^wt$", re.IGNORECASE),
    re.compile(r"_wt$", re.IGNORECASE),
)

DENOMINATOR_PATTERNS_STRONG = (
    re.compile(r"(?:^|_)denominator(?:$|_)", re.IGNORECASE),
    re.compile(r"_denominator$", re.IGNORECASE),
    re.compile(r"(?:^|_)rate_base(?:$|_)", re.IGNORECASE),
)

DENOMINATOR_PATTERNS_MODERATE = (
    re.compile(r"(?:^|_)denom(?:$|_)", re.IGNORECASE),
    re.compile(r"_denom$", re.IGNORECASE),
    re.compile(r"(?:^|_)base(?:$|_)", re.IGNORECASE),
    re.compile(r"^n$", re.IGNORECASE),
    re.compile(r"_n$", re.IGNORECASE),
)

# Weak patterns: often used for measures, need value evidence
DENOMINATOR_PATTERNS_WEAK = (
    re.compile(r"(?:^|_)total(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)population(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)count(?:$|_)", re.IGNORECASE),
)

SUPPRESSION_PATTERNS_STRONG = (
    re.compile(r"(?:^|_)suppression(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)suppressed(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)redacted(?:$|_)", re.IGNORECASE),
    re.compile(r"_suppress$", re.IGNORECASE),
    re.compile(r"_suppression$", re.IGNORECASE),
    re.compile(r"^suppression_", re.IGNORECASE),
    re.compile(r"^suppressed_", re.IGNORECASE),
)

SUPPRESSION_PATTERNS_MODERATE = (
    re.compile(r"(?:^|_)suppress(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)redact(?:$|_)", re.IGNORECASE),
    re.compile(r"_redact$", re.IGNORECASE),
    re.compile(r"(?:^|_)masked(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)censored(?:$|_)", re.IGNORECASE),
)

QUALITY_PATTERNS_STRONG = (
    re.compile(r"(?:^|_)data_quality(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)reliability_flag(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)quality_flag(?:$|_)", re.IGNORECASE),
)

QUALITY_PATTERNS_MODERATE = (
    re.compile(r"(?:^|_)quality(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)reliability(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)confidence(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)qual(?:$|_)", re.IGNORECASE),
    re.compile(r"_quality$", re.IGNORECASE),
    re.compile(r"_reliability$", re.IGNORECASE),
    re.compile(r"_conf$", re.IGNORECASE),
    re.compile(r"(?:^|_)grade(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)rating(?:$|_)", re.IGNORECASE),
)

# Confidence thresholds
LOCK_THRESHOLD = 0.75  # Auto-lock if confidence >= this
MARGIN_THRESHOLD = 0.15  # Auto-lock if winner - runner_up >= this

# Name-only confidence caps (prevent over-confidence from names alone)
NAME_ONLY_CAP_STRONG = 0.70
NAME_ONLY_CAP_MODERATE = 0.55
NAME_ONLY_CAP_WEAK = 0.40


def _matches_patterns(
    name: str, patterns: tuple[re.Pattern[str], ...]
) -> list[str]:
    """Check if name matches patterns and return matched pattern strings."""
    matches = []
    for pattern in patterns:
        if pattern.search(name):
            matches.append(pattern.pattern)
    return matches


# -----------------------------------------------------------------------------
# Value-based distribution checks
# -----------------------------------------------------------------------------


@dataclass
class ValueDistribution:
    """Summary statistics for a column's values."""

    is_numeric: bool = False
    is_non_negative: bool = False
    is_integer_like: bool = False  # All values are close to integers
    is_bounded_01: bool = False  # Values in [0, 1]
    is_bounded_0100: bool = False  # Values in [0, 100]
    cardinality: int = 0
    null_rate: float = 0.0

    # Boolean-like detection
    is_boolean_like: bool = False
    boolean_values: set[Any] = field(default_factory=set)

    # Quality/ordinal detection
    is_ordinal_codes: bool = False  # A/B/C, 1-5, etc.


def compute_value_distribution(
    sample_values: Sequence[Any],
) -> ValueDistribution:
    """Compute distribution statistics from sampled values.

    Args:
        sample_values: Sample of column values (up to 200 recommended).

    Returns:
        ValueDistribution with computed statistics.
    """
    dist = ValueDistribution()

    if not sample_values:
        return dist

    # Filter nulls
    non_null = [v for v in sample_values if v is not None and v != ""]
    dist.null_rate = 1.0 - len(non_null) / len(sample_values) if sample_values else 0.0

    if not non_null:
        return dist

    dist.cardinality = len(set(non_null))

    # Check numeric properties
    numeric_values = []
    for v in non_null:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_values.append(float(v))
        elif isinstance(v, str):
            with contextlib.suppress(ValueError, TypeError):
                numeric_values.append(float(v))

    if len(numeric_values) >= len(non_null) * 0.9:
        dist.is_numeric = True
        dist.is_non_negative = all(v >= 0 for v in numeric_values)

        # Integer-like: all values within 0.001 of an integer
        dist.is_integer_like = all(
            abs(v - round(v)) < 0.001 for v in numeric_values
        )

        # Bounded checks
        if numeric_values:
            min_val = min(numeric_values)
            max_val = max(numeric_values)
            dist.is_bounded_01 = min_val >= 0 and max_val <= 1
            dist.is_bounded_0100 = min_val >= 0 and max_val <= 100

    # Boolean-like detection
    boolean_markers = {
        True, False, "true", "false", "True", "False", "TRUE", "FALSE",
        "0", "1", "yes", "no", "Yes", "No", "YES", "NO",
        "y", "n", "Y", "N",
        "suppressed", "Suppressed", "SUPPRESSED",
        "masked", "Masked", "MASKED",
        "redacted", "Redacted", "REDACTED",
    }

    unique_values = set(non_null)
    if len(unique_values) <= 5:
        if unique_values.issubset(boolean_markers) or len(unique_values) == 2:
            dist.is_boolean_like = True
            dist.boolean_values = unique_values

    # Ordinal code detection (A/B/C, 1-5, etc.)
    if 2 <= dist.cardinality <= 10:
        string_values = [str(v).strip() for v in unique_values]
        # Check for letter grades
        if all(len(s) == 1 and s.isalpha() for s in string_values) or all(s.isdigit() and 0 <= int(s) <= 10 for s in string_values):
            dist.is_ordinal_codes = True
        # Check for quality descriptors
        quality_terms = {"good", "fair", "poor", "excellent", "high", "low", "medium"}
        if all(s.lower() in quality_terms for s in string_values):
            dist.is_ordinal_codes = True

    return dist


# -----------------------------------------------------------------------------
# Cross-column relationship detection
# -----------------------------------------------------------------------------


def detect_rate_columns(
    evidence: dict[str, ColumnEvidence],
) -> set[str]:
    """Identify columns that look like rates/percentages.

    Rate columns suggest nearby denominators.

    Args:
        evidence: Dictionary of column evidence.

    Returns:
        Set of column names that appear to be rates.
    """
    rate_patterns = (
        re.compile(r"(?:^|_)rate(?:$|_)", re.IGNORECASE),
        re.compile(r"(?:^|_)percent(?:$|_)", re.IGNORECASE),
        re.compile(r"(?:^|_)pct(?:$|_)", re.IGNORECASE),
        re.compile(r"(?:^|_)ratio(?:$|_)", re.IGNORECASE),
        re.compile(r"_rate$", re.IGNORECASE),
        re.compile(r"_pct$", re.IGNORECASE),
        re.compile(r"_%$", re.IGNORECASE),
    )

    rate_columns = set()
    for col_name in evidence:
        for pattern in rate_patterns:
            if pattern.search(col_name):
                rate_columns.add(col_name)
                break

    return rate_columns


def compute_null_correlation(
    suppression_values: Sequence[Any],
    measure_values: Sequence[Any],
) -> float:
    """Compute correlation between suppression flag and null values.

    If suppression flag correlates strongly with nulls in measure columns,
    it's more likely a real suppression flag.

    Args:
        suppression_values: Values from the suppression flag candidate.
        measure_values: Values from a measure column.

    Returns:
        Correlation score (0-1). Higher means stronger evidence.
    """
    if len(suppression_values) != len(measure_values):
        return 0.0

    if not suppression_values:
        return 0.0

    # Identify "suppressed" markers
    suppressed_markers = {
        True, "true", "True", "TRUE", "1", "yes", "Yes", "YES", "y", "Y",
        "suppressed", "Suppressed", "SUPPRESSED",
        "masked", "Masked", "MASKED",
        "redacted", "Redacted", "REDACTED",
    }

    suppressed_count = 0
    suppressed_null_count = 0
    not_suppressed_count = 0
    not_suppressed_null_count = 0

    for supp_val, meas_val in zip(suppression_values, measure_values, strict=False):
        is_suppressed = supp_val in suppressed_markers
        is_null = meas_val is None or meas_val == ""

        if is_suppressed:
            suppressed_count += 1
            if is_null:
                suppressed_null_count += 1
        else:
            not_suppressed_count += 1
            if is_null:
                not_suppressed_null_count += 1

    # Calculate P(null | suppressed) - P(null | not suppressed)
    p_null_given_suppressed = (
        suppressed_null_count / suppressed_count if suppressed_count > 0 else 0
    )
    p_null_given_not_suppressed = (
        not_suppressed_null_count / not_suppressed_count
        if not_suppressed_count > 0
        else 0
    )

    # Convert difference to a 0-1 score
    diff = p_null_given_suppressed - p_null_given_not_suppressed
    return max(0.0, min(1.0, diff))


# -----------------------------------------------------------------------------
# Individual type scoring functions
# -----------------------------------------------------------------------------


def score_weight_candidate(
    evidence: ColumnEvidence,
    value_dist: ValueDistribution | None = None,
) -> SpecialColumnCandidate | None:
    """Score a column as a potential weight.

    Args:
        evidence: Column evidence from profiling.
        value_dist: Optional value distribution statistics.

    Returns:
        SpecialColumnCandidate or None if no signal.
    """
    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    # Name-based scoring
    strong_matches = _matches_patterns(name, WEIGHT_PATTERNS_STRONG)
    moderate_matches = _matches_patterns(name, WEIGHT_PATTERNS_MODERATE)

    if strong_matches:
        score += 0.50
        evidence_list.extend([f"name:strong:{p}" for p in strong_matches])
    if moderate_matches:
        score += 0.30 if not strong_matches else 0.15
        evidence_list.extend([f"name:moderate:{p}" for p in moderate_matches])

    # Require at least some name signal
    if not (strong_matches or moderate_matches):
        return None

    # Cap name-only confidence
    if strong_matches:
        score = min(score, NAME_ONLY_CAP_STRONG)
    else:
        score = min(score, NAME_ONLY_CAP_MODERATE)

    # Type-based boost
    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        score += 0.10
        evidence_list.append("type:numeric")

    # Value-based scoring
    if value_dist is not None and value_dist.is_numeric:
        if value_dist.is_non_negative:
            score += 0.15
            evidence_list.append("value:non_negative")

        # Survey weights are often floats, not integers
        if not value_dist.is_integer_like:
            score += 0.10
            evidence_list.append("value:not_integer_like")

        # Weights typically aren't bounded to 0-1 (those are probabilities)
        if not value_dist.is_bounded_01 and value_dist.is_non_negative:
            score += 0.05
            evidence_list.append("value:not_probability_like")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.WEIGHT,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


def score_denominator_candidate(
    evidence: ColumnEvidence,
    value_dist: ValueDistribution | None = None,
    has_rate_columns: bool = False,
) -> SpecialColumnCandidate | None:
    """Score a column as a potential denominator.

    Args:
        evidence: Column evidence from profiling.
        value_dist: Optional value distribution statistics.
        has_rate_columns: Whether rate/percent columns exist in dataset.

    Returns:
        SpecialColumnCandidate or None if no signal.
    """
    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    # Name-based scoring
    strong_matches = _matches_patterns(name, DENOMINATOR_PATTERNS_STRONG)
    moderate_matches = _matches_patterns(name, DENOMINATOR_PATTERNS_MODERATE)
    weak_matches = _matches_patterns(name, DENOMINATOR_PATTERNS_WEAK)

    if strong_matches:
        score += 0.50
        evidence_list.extend([f"name:strong:{p}" for p in strong_matches])
    if moderate_matches:
        score += 0.25 if not strong_matches else 0.10
        evidence_list.extend([f"name:moderate:{p}" for p in moderate_matches])
    if weak_matches:
        # Weak patterns alone need value evidence
        score += 0.15 if (strong_matches or moderate_matches) else 0.10
        evidence_list.extend([f"name:weak:{p}" for p in weak_matches])

    # Cap name-only confidence (especially for weak patterns)
    if strong_matches:
        cap = NAME_ONLY_CAP_STRONG
    elif moderate_matches:
        cap = NAME_ONLY_CAP_MODERATE
    else:
        cap = NAME_ONLY_CAP_WEAK
    score = min(score, cap)

    # Type-based boost
    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        score += 0.10
        evidence_list.append("type:numeric")

    # Value-based scoring
    if value_dist is not None and value_dist.is_numeric:
        if value_dist.is_non_negative:
            score += 0.10
            evidence_list.append("value:non_negative")

        # Denominators are usually integers
        if value_dist.is_integer_like:
            score += 0.10
            evidence_list.append("value:integer_like")

    # No name signal at all - don't proceed
    if not (strong_matches or moderate_matches or weak_matches):
        return None

    # Cross-column relationship: rate columns exist (only if we have name evidence)
    if has_rate_columns:
        score += 0.15
        evidence_list.append("relationship:rate_columns_present")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.DENOMINATOR,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


def score_suppression_candidate(
    evidence: ColumnEvidence,
    value_dist: ValueDistribution | None = None,
    null_correlation: float | None = None,
) -> SpecialColumnCandidate | None:
    """Score a column as a potential suppression flag.

    Args:
        evidence: Column evidence from profiling.
        value_dist: Optional value distribution statistics.
        null_correlation: Optional correlation with measure nulls.

    Returns:
        SpecialColumnCandidate or None if no signal.
    """
    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    # Name-based scoring
    strong_matches = _matches_patterns(name, SUPPRESSION_PATTERNS_STRONG)
    moderate_matches = _matches_patterns(name, SUPPRESSION_PATTERNS_MODERATE)

    if strong_matches:
        score += 0.55
        evidence_list.extend([f"name:strong:{p}" for p in strong_matches])
    if moderate_matches:
        score += 0.30 if not strong_matches else 0.15
        evidence_list.extend([f"name:moderate:{p}" for p in moderate_matches])

    # Require at least some name signal
    if not (strong_matches or moderate_matches):
        return None

    # Cap name-only confidence
    if strong_matches:
        score = min(score, NAME_ONLY_CAP_STRONG)
    else:
        score = min(score, NAME_ONLY_CAP_MODERATE)

    # Type-based boost
    if evidence.primitive_type in (PrimitiveType.BOOLEAN, PrimitiveType.STRING):
        score += 0.10
        evidence_list.append("type:boolean_or_string")

    # Value-based scoring
    if value_dist is not None:
        if value_dist.is_boolean_like:
            score += 0.20
            values_str = ", ".join(str(v) for v in list(value_dist.boolean_values)[:3])
            evidence_list.append(f"value:boolean_like:[{values_str}]")
        elif value_dist.cardinality <= 5:
            score += 0.10
            evidence_list.append(f"value:low_cardinality:{value_dist.cardinality}")

    # Cross-column relationship: null correlation
    if null_correlation is not None and null_correlation > 0.3:
        boost = min(0.25, null_correlation * 0.3)
        score += boost
        evidence_list.append(f"relationship:null_correlation:{null_correlation:.2f}")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.SUPPRESSION_FLAG,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


def score_quality_candidate(
    evidence: ColumnEvidence,
    value_dist: ValueDistribution | None = None,
) -> SpecialColumnCandidate | None:
    """Score a column as a potential quality flag.

    Args:
        evidence: Column evidence from profiling.
        value_dist: Optional value distribution statistics.

    Returns:
        SpecialColumnCandidate or None if no signal.
    """
    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    # Name-based scoring
    strong_matches = _matches_patterns(name, QUALITY_PATTERNS_STRONG)
    moderate_matches = _matches_patterns(name, QUALITY_PATTERNS_MODERATE)

    if strong_matches:
        score += 0.50
        evidence_list.extend([f"name:strong:{p}" for p in strong_matches])
    if moderate_matches:
        score += 0.30 if not strong_matches else 0.15
        evidence_list.extend([f"name:moderate:{p}" for p in moderate_matches])

    # Require at least some name signal
    if not (strong_matches or moderate_matches):
        return None

    # Cap name-only confidence
    if strong_matches:
        score = min(score, NAME_ONLY_CAP_STRONG)
    else:
        score = min(score, NAME_ONLY_CAP_MODERATE)

    # Value-based scoring
    if value_dist is not None:
        # Probability-like values [0, 1]
        if value_dist.is_numeric and value_dist.is_bounded_01:
            score += 0.20
            evidence_list.append("value:probability_like:[0,1]")

        # Ordinal codes (A/B/C, 1-5)
        if value_dist.is_ordinal_codes:
            score += 0.20
            evidence_list.append("value:ordinal_codes")

        # Low cardinality is expected for quality flags
        if 2 <= value_dist.cardinality <= 10:
            score += 0.10
            evidence_list.append(f"value:low_cardinality:{value_dist.cardinality}")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.QUALITY_FLAG,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


# -----------------------------------------------------------------------------
# Legacy detection functions (for backwards compatibility)
# -----------------------------------------------------------------------------


def detect_weight_column(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a weight column based on naming patterns.

    DEPRECATED: Use score_weight_candidate for better scoring.
    """
    candidate = score_weight_candidate(evidence)
    if candidate is None:
        return None

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=candidate.flag_type,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
    )


def detect_denominator_column(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a denominator column based on naming patterns.

    DEPRECATED: Use score_denominator_candidate for better scoring.
    """
    candidate = score_denominator_candidate(evidence)
    if candidate is None:
        return None

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=candidate.flag_type,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
    )


def detect_suppression_flag(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a suppression flag based on naming patterns.

    DEPRECATED: Use score_suppression_candidate for better scoring.
    """
    candidate = score_suppression_candidate(evidence)
    if candidate is None:
        return None

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=candidate.flag_type,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
    )


def detect_quality_flag(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a quality flag based on naming patterns.

    DEPRECATED: Use score_quality_candidate for better scoring.
    """
    candidate = score_quality_candidate(evidence)
    if candidate is None:
        return None

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=candidate.flag_type,
        confidence=candidate.confidence,
        evidence=candidate.evidence,
    )


# -----------------------------------------------------------------------------
# User flagging interface
# -----------------------------------------------------------------------------


def flag_column(column_name: str, flag_type: SpecialColumnType) -> SpecialColumnFlag:
    """Manually flag a column as a special type.

    This function allows users to explicitly mark columns as having
    special roles, overriding or supplementing automatic detection.

    Args:
        column_name: Name of the column to flag.
        flag_type: The special column type to assign.

    Returns:
        SpecialColumnFlag with user-provided assignment.
    """
    return SpecialColumnFlag(
        column_name=column_name,
        flag_type=flag_type,
        confidence=1.0,
        evidence=["User-provided flag"],
    )


# -----------------------------------------------------------------------------
# Main detection API
# -----------------------------------------------------------------------------


def detect_special_columns(
    evidence: dict[str, ColumnEvidence],
    sample_values: dict[str, Sequence[Any]] | None = None,
    user_flags: list[SpecialColumnFlag] | None = None,
    lock_threshold: float = LOCK_THRESHOLD,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> list[SpecialColumnResult]:
    """Detect special columns with multi-candidate scoring.

    This is the new recommended API that returns all candidates per column
    and handles ambiguity explicitly.

    Args:
        evidence: Dictionary mapping column names to ColumnEvidence.
        sample_values: Optional dict of column name -> sample values for
            value-based scoring. Recommend 200 rows max.
        user_flags: Optional list of user-provided SpecialColumnFlags.
        lock_threshold: Minimum confidence to auto-lock without confirmation.
        margin_threshold: Minimum winner-runnerup margin to auto-lock.

    Returns:
        List of SpecialColumnResult, one per column with candidates.
    """
    if user_flags is None:
        user_flags = []
    if sample_values is None:
        sample_values = {}

    # Track user-flagged columns
    user_flagged: dict[str, SpecialColumnFlag] = {
        f.column_name: f for f in user_flags
    }

    # Detect rate columns for cross-column relationship
    rate_columns = detect_rate_columns(evidence)
    has_rate_columns = len(rate_columns) > 0

    results: list[SpecialColumnResult] = []

    # First pass: process user-flagged columns
    for col_name, user_flag in user_flagged.items():
        result = SpecialColumnResult(
            column_name=col_name,
            candidates=[
                SpecialColumnCandidate(
                    flag_type=user_flag.flag_type,
                    confidence=1.0,
                    evidence=["User-provided flag"],
                )
            ],
            selected=user_flag.flag_type,
            status=SpecialColumnStatus.USER_LOCKED,
        )
        results.append(result)

    # Second pass: detect in non-user-flagged columns
    for col_name, col_evidence in evidence.items():
        if col_name in user_flagged:
            continue

        # Compute value distribution if we have samples
        value_dist = None
        if col_name in sample_values:
            value_dist = compute_value_distribution(sample_values[col_name])

        # Score all candidate types
        candidates: list[SpecialColumnCandidate] = []

        weight_cand = score_weight_candidate(col_evidence, value_dist)
        if weight_cand:
            candidates.append(weight_cand)

        denom_cand = score_denominator_candidate(
            col_evidence, value_dist, has_rate_columns
        )
        if denom_cand:
            candidates.append(denom_cand)

        # For suppression, we'd need measure column values for null correlation
        # This is a simplification - in production, you'd pass measure values
        supp_cand = score_suppression_candidate(col_evidence, value_dist)
        if supp_cand:
            candidates.append(supp_cand)

        qual_cand = score_quality_candidate(col_evidence, value_dist)
        if qual_cand:
            candidates.append(qual_cand)

        # Skip if no candidates
        if not candidates:
            continue

        # Sort candidates by confidence (descending)
        candidates.sort()

        # Determine winner and status
        winner = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        margin = (
            winner.confidence - runner_up.confidence
            if runner_up
            else winner.confidence
        )

        # Auto-lock if confidence is high AND margin is sufficient
        if winner.confidence >= lock_threshold and margin >= margin_threshold:
            status = SpecialColumnStatus.AUTO_LOCKED
            selected = winner.flag_type
        else:
            status = SpecialColumnStatus.NEEDS_CONFIRMATION
            selected = None  # Don't select until user confirms

        result = SpecialColumnResult(
            column_name=col_name,
            candidates=candidates,
            selected=selected,
            status=status,
        )
        results.append(result)

    return results


def get_special_columns(
    evidence: dict[str, ColumnEvidence],
    flags: list[SpecialColumnFlag] | None = None,
) -> list[SpecialColumnFlag]:
    """Get all special columns from evidence and user flags.

    LEGACY API: Combines automatic detection with user-provided flags.
    For new code, prefer detect_special_columns() which provides
    multi-candidate scoring and ambiguity handling.

    User flags take precedence over automatic detection for the same column.

    Args:
        evidence: Dictionary mapping column names to ColumnEvidence.
        flags: Optional list of user-provided SpecialColumnFlags.

    Returns:
        List of all detected and flagged special columns.
    """
    # Use new API internally
    results = detect_special_columns(
        evidence=evidence,
        user_flags=flags,
    )

    # Convert to legacy format
    legacy_flags: list[SpecialColumnFlag] = []

    for result in results:
        # For legacy API, always pick the top candidate
        if result.candidates:
            top = result.candidates[0]
            legacy_flags.append(
                SpecialColumnFlag(
                    column_name=result.column_name,
                    flag_type=top.flag_type,
                    confidence=top.confidence,
                    evidence=top.evidence,
                )
            )

    # Sort to put user flags first (they have confidence=1.0 and USER_LOCKED status)
    legacy_flags.sort(key=lambda f: -f.confidence)

    return legacy_flags
