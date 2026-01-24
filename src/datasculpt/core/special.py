"""Special column detection and user flagging for Datasculpt.

This module provides functionality to detect and flag special columns
such as weights, denominators, suppression flags, and quality flags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasculpt.core.types import ColumnEvidence


class SpecialColumnType(str, Enum):
    """Types of special columns in datasets."""

    WEIGHT = "weight"
    DENOMINATOR = "denominator"
    SUPPRESSION_FLAG = "suppression_flag"
    QUALITY_FLAG = "quality_flag"


@dataclass
class SpecialColumnFlag:
    """Flag indicating a column has a special role."""

    column_name: str
    flag_type: SpecialColumnType
    confidence: float
    evidence: list[str]


# Pattern definitions for detection heuristics
WEIGHT_PATTERNS = (
    re.compile(r"\bweight\b", re.IGNORECASE),
    re.compile(r"\bwgt\b", re.IGNORECASE),
    re.compile(r"\bsample_weight\b", re.IGNORECASE),
    re.compile(r"\bsampling_weight\b", re.IGNORECASE),
    re.compile(r"_wgt$", re.IGNORECASE),
    re.compile(r"_weight$", re.IGNORECASE),
    re.compile(r"^wt$", re.IGNORECASE),
    re.compile(r"_wt$", re.IGNORECASE),
)

DENOMINATOR_PATTERNS = (
    re.compile(r"\bdenominator\b", re.IGNORECASE),
    re.compile(r"\bdenom\b", re.IGNORECASE),
    re.compile(r"\btotal\b", re.IGNORECASE),
    re.compile(r"\bbase\b", re.IGNORECASE),
    re.compile(r"_denom$", re.IGNORECASE),
    re.compile(r"_denominator$", re.IGNORECASE),
    re.compile(r"^n$", re.IGNORECASE),
    re.compile(r"_n$", re.IGNORECASE),
    re.compile(r"\bpopulation\b", re.IGNORECASE),
    re.compile(r"\bcount\b", re.IGNORECASE),
)

SUPPRESSION_PATTERNS = (
    re.compile(r"\bsuppress\b", re.IGNORECASE),
    re.compile(r"\bredact\b", re.IGNORECASE),
    re.compile(r"\bsuppressed\b", re.IGNORECASE),
    re.compile(r"\bredacted\b", re.IGNORECASE),
    re.compile(r"\bsuppression\b", re.IGNORECASE),
    re.compile(r"_suppress$", re.IGNORECASE),
    re.compile(r"_redact$", re.IGNORECASE),
    re.compile(r"\bmasked\b", re.IGNORECASE),
    re.compile(r"\bcensored\b", re.IGNORECASE),
)

QUALITY_PATTERNS = (
    re.compile(r"\bquality\b", re.IGNORECASE),
    re.compile(r"\breliability\b", re.IGNORECASE),
    re.compile(r"\bconfidence\b", re.IGNORECASE),
    re.compile(r"\bqual\b", re.IGNORECASE),
    re.compile(r"_quality$", re.IGNORECASE),
    re.compile(r"_reliability$", re.IGNORECASE),
    re.compile(r"_conf$", re.IGNORECASE),
    re.compile(r"\bgrade\b", re.IGNORECASE),
    re.compile(r"\brating\b", re.IGNORECASE),
    re.compile(r"\bdata_quality\b", re.IGNORECASE),
)


def _matches_patterns(name: str, patterns: tuple[re.Pattern, ...]) -> list[str]:
    """Check if name matches patterns and return matched pattern strings."""
    matches = []
    for pattern in patterns:
        if pattern.search(name):
            matches.append(pattern.pattern)
    return matches


def detect_weight_column(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a weight column based on naming patterns.

    Weight columns are typically used in survey data to adjust for
    sampling methodology.

    Args:
        evidence: Column evidence from profiling.

    Returns:
        SpecialColumnFlag if detected, None otherwise.
    """
    matched = _matches_patterns(evidence.name, WEIGHT_PATTERNS)

    if not matched:
        return None

    evidence_list = [f"Name matches pattern: {p}" for p in matched]

    # Boost confidence if column is numeric
    from datasculpt.core.types import PrimitiveType

    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        evidence_list.append("Column is numeric (expected for weights)")
        confidence = 0.9 if len(matched) >= 2 else 0.8
    else:
        confidence = 0.6 if len(matched) >= 2 else 0.5

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=SpecialColumnType.WEIGHT,
        confidence=confidence,
        evidence=evidence_list,
    )


def detect_denominator_column(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a denominator column based on naming patterns.

    Denominator columns represent the base population or count for
    calculating rates and percentages.

    Args:
        evidence: Column evidence from profiling.

    Returns:
        SpecialColumnFlag if detected, None otherwise.
    """
    matched = _matches_patterns(evidence.name, DENOMINATOR_PATTERNS)

    if not matched:
        return None

    evidence_list = [f"Name matches pattern: {p}" for p in matched]

    # Boost confidence if column is numeric
    from datasculpt.core.types import PrimitiveType

    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        evidence_list.append("Column is numeric (expected for denominators)")
        confidence = 0.85 if len(matched) >= 2 else 0.75
    else:
        confidence = 0.55 if len(matched) >= 2 else 0.45

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=SpecialColumnType.DENOMINATOR,
        confidence=confidence,
        evidence=evidence_list,
    )


def detect_suppression_flag(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a suppression flag based on naming patterns.

    Suppression flags indicate data that should be redacted or masked,
    often for privacy or statistical reliability reasons.

    Args:
        evidence: Column evidence from profiling.

    Returns:
        SpecialColumnFlag if detected, None otherwise.
    """
    matched = _matches_patterns(evidence.name, SUPPRESSION_PATTERNS)

    if not matched:
        return None

    evidence_list = [f"Name matches pattern: {p}" for p in matched]

    # Suppression flags are often boolean or string
    from datasculpt.core.types import PrimitiveType

    if evidence.primitive_type in (PrimitiveType.BOOLEAN, PrimitiveType.STRING):
        evidence_list.append("Column is boolean/string (expected for flags)")
        confidence = 0.9 if len(matched) >= 2 else 0.8
    else:
        confidence = 0.6 if len(matched) >= 2 else 0.5

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=SpecialColumnType.SUPPRESSION_FLAG,
        confidence=confidence,
        evidence=evidence_list,
    )


def detect_quality_flag(evidence: ColumnEvidence) -> SpecialColumnFlag | None:
    """Detect if a column is a quality flag based on naming patterns.

    Quality flags indicate data reliability, confidence levels, or
    data quality grades.

    Args:
        evidence: Column evidence from profiling.

    Returns:
        SpecialColumnFlag if detected, None otherwise.
    """
    matched = _matches_patterns(evidence.name, QUALITY_PATTERNS)

    if not matched:
        return None

    evidence_list = [f"Name matches pattern: {p}" for p in matched]

    # Quality flags can be various types
    confidence = 0.85 if len(matched) >= 2 else 0.7

    return SpecialColumnFlag(
        column_name=evidence.name,
        flag_type=SpecialColumnType.QUALITY_FLAG,
        confidence=confidence,
        evidence=evidence_list,
    )


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


def get_special_columns(
    evidence: dict[str, ColumnEvidence],
    flags: list[SpecialColumnFlag] | None = None,
) -> list[SpecialColumnFlag]:
    """Get all special columns from evidence and user flags.

    Combines automatic detection with user-provided flags.
    User flags take precedence over automatic detection for the same column.

    Args:
        evidence: Dictionary mapping column names to ColumnEvidence.
        flags: Optional list of user-provided SpecialColumnFlags.

    Returns:
        List of all detected and flagged special columns.
    """
    if flags is None:
        flags = []

    # Track user-flagged columns to avoid duplicate detection
    user_flagged_columns = {f.column_name for f in flags}

    detected: list[SpecialColumnFlag] = []

    # Detect special columns from evidence
    for col_name, col_evidence in evidence.items():
        if col_name in user_flagged_columns:
            continue

        # Try each detector
        detectors = [
            detect_weight_column,
            detect_denominator_column,
            detect_suppression_flag,
            detect_quality_flag,
        ]

        for detector in detectors:
            result = detector(col_evidence)
            if result is not None:
                detected.append(result)
                break  # Only one special type per column

    # Combine user flags with detected flags (user flags first for precedence)
    return list(flags) + detected
