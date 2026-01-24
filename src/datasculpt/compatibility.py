"""Cross-dataset compatibility checking for Datasculpt.

This module provides functions to check compatibility between datasets
based on their grain, time axis, and potential join explosion risks.
It also provides mapping suggestions for reference systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

from datasculpt.core.time import TimeGranularity, TimeRangeResult
from datasculpt.core.types import (
    ColumnSpec,
    DecisionRecord,
    InvariantProposal,
    PrimitiveType,
    Role,
)

if TYPE_CHECKING:
    pass


# Known reference systems for mapping suggestions
REFERENCE_SYSTEMS: dict[str, dict[str, Any]] = {
    "iso_3166_1_alpha2": {
        "name": "ISO 3166-1 Alpha-2",
        "description": "Two-letter country codes (e.g., US, GB, DE)",
        "patterns": [r"^[A-Z]{2}$"],
        "example_values": ["US", "GB", "DE", "FR", "JP"],
        "category": "geography",
    },
    "iso_3166_1_alpha3": {
        "name": "ISO 3166-1 Alpha-3",
        "description": "Three-letter country codes (e.g., USA, GBR, DEU)",
        "patterns": [r"^[A-Z]{3}$"],
        "example_values": ["USA", "GBR", "DEU", "FRA", "JPN"],
        "category": "geography",
    },
    "iso_3166_1_numeric": {
        "name": "ISO 3166-1 Numeric",
        "description": "Three-digit country codes (e.g., 840, 826, 276)",
        "patterns": [r"^\d{3}$"],
        "example_values": ["840", "826", "276", "250", "392"],
        "category": "geography",
    },
    "fips_10_4": {
        "name": "FIPS 10-4",
        "description": "FIPS country codes (deprecated but still used)",
        "patterns": [r"^[A-Z]{2}$"],
        "example_values": ["US", "UK", "GM", "FR", "JA"],
        "category": "geography",
    },
    "fips_state": {
        "name": "FIPS State Codes",
        "description": "US state FIPS codes (e.g., 06 for California)",
        "patterns": [r"^\d{2}$"],
        "example_values": ["06", "36", "48", "12", "17"],
        "category": "geography",
    },
    "fips_county": {
        "name": "FIPS County Codes",
        "description": "US county FIPS codes (state + county, e.g., 06037)",
        "patterns": [r"^\d{5}$"],
        "example_values": ["06037", "36061", "17031", "48201", "04013"],
        "category": "geography",
    },
    "iso_4217": {
        "name": "ISO 4217",
        "description": "Three-letter currency codes (e.g., USD, EUR, GBP)",
        "patterns": [r"^[A-Z]{3}$"],
        "example_values": ["USD", "EUR", "GBP", "JPY", "CNY"],
        "category": "currency",
    },
    "naics": {
        "name": "NAICS",
        "description": "North American Industry Classification System",
        "patterns": [r"^\d{2,6}$"],
        "example_values": ["31", "311", "3111", "31111", "311111"],
        "category": "industry",
    },
    "sic": {
        "name": "SIC",
        "description": "Standard Industrial Classification",
        "patterns": [r"^\d{4}$"],
        "example_values": ["0100", "2000", "3500", "5200", "7000"],
        "category": "industry",
    },
    "iso_8601_date": {
        "name": "ISO 8601 Date",
        "description": "ISO date format (YYYY-MM-DD)",
        "patterns": [r"^\d{4}-\d{2}-\d{2}$"],
        "example_values": ["2024-01-15", "2023-12-31", "2025-06-01"],
        "category": "temporal",
    },
    "isin": {
        "name": "ISIN",
        "description": "International Securities Identification Number",
        "patterns": [r"^[A-Z]{2}[A-Z0-9]{9}\d$"],
        "example_values": ["US0378331005", "GB0002634946", "DE0007164600"],
        "category": "finance",
    },
    "cusip": {
        "name": "CUSIP",
        "description": "Committee on Uniform Securities Identification Procedures",
        "patterns": [r"^[A-Z0-9]{9}$"],
        "example_values": ["037833100", "594918104", "17275R102"],
        "category": "finance",
    },
}


class Severity(str, Enum):
    """Severity levels for compatibility issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class CompatibilityIssue:
    """A single compatibility issue or observation."""

    severity: Severity
    category: str
    message: str
    details: dict[str, str | int | float | None] = field(default_factory=dict)


@dataclass
class GrainCompatibilityResult:
    """Result of grain compatibility check."""

    is_compatible: bool
    score: float  # 0.0 to 1.0
    matching_columns: list[tuple[str, str]]  # (left_col, right_col)
    issues: list[CompatibilityIssue] = field(default_factory=list)


@dataclass
class TimeCompatibilityResult:
    """Result of time axis compatibility check."""

    is_compatible: bool
    left_granularity: TimeGranularity | None
    right_granularity: TimeGranularity | None
    granularity_compatible: bool
    overlapping_range: tuple[date, date] | None
    issues: list[CompatibilityIssue] = field(default_factory=list)


@dataclass
class JoinExplosionResult:
    """Result of join explosion analysis."""

    expected_multiplier: float
    is_safe: bool
    issues: list[CompatibilityIssue] = field(default_factory=list)


@dataclass
class CompatibilityResult:
    """Complete result of cross-dataset compatibility check."""

    is_compatible: bool
    grain_result: GrainCompatibilityResult
    time_result: TimeCompatibilityResult
    join_result: JoinExplosionResult
    issues: list[CompatibilityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[CompatibilityIssue]:
        """Return all error-level issues."""
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[CompatibilityIssue]:
        """Return all warning-level issues."""
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def infos(self) -> list[CompatibilityIssue]:
        """Return all info-level issues."""
        return [i for i in self.issues if i.severity == Severity.INFO]


# Type alias for inputs that can be either InvariantProposal or DecisionRecord
DatasetSpec = Union[InvariantProposal, DecisionRecord]


def _extract_grain_columns(spec: DatasetSpec) -> list[str]:
    """Extract grain column names from a dataset specification.

    Args:
        spec: Either an InvariantProposal or DecisionRecord.

    Returns:
        List of grain column names.
    """
    if isinstance(spec, InvariantProposal):
        return list(spec.grain)
    # DecisionRecord
    return list(spec.grain.key_columns)


def _extract_column_specs(spec: DatasetSpec) -> dict[str, ColumnSpec | None]:
    """Extract column specifications from a dataset specification.

    Args:
        spec: Either an InvariantProposal or DecisionRecord.

    Returns:
        Dictionary mapping column names to ColumnSpec objects.
        For DecisionRecord, returns a synthesized ColumnSpec from ColumnEvidence.
    """
    if isinstance(spec, InvariantProposal):
        return {col.name: col for col in spec.columns}

    # DecisionRecord - synthesize ColumnSpec from ColumnEvidence
    result: dict[str, ColumnSpec | None] = {}
    for name, evidence in spec.column_evidence.items():
        # Get primary role from evidence
        primary_role = Role.METADATA
        best_score = 0.0
        for role, score in evidence.role_scores.items():
            if score > best_score:
                primary_role = role
                best_score = score

        result[name] = ColumnSpec(
            name=name,
            role=primary_role,
            primitive_type=evidence.primitive_type,
            structural_type=evidence.structural_type,
        )
    return result


def _extract_column_types(spec: DatasetSpec) -> dict[str, PrimitiveType]:
    """Extract column primitive types from a dataset specification.

    Args:
        spec: Either an InvariantProposal or DecisionRecord.

    Returns:
        Dictionary mapping column names to primitive types.
    """
    if isinstance(spec, InvariantProposal):
        return {col.name: col.primitive_type for col in spec.columns}

    # DecisionRecord
    return {
        name: evidence.primitive_type
        for name, evidence in spec.column_evidence.items()
    }


def _extract_time_columns(spec: DatasetSpec) -> list[str]:
    """Extract time column names from a dataset specification.

    Args:
        spec: Either an InvariantProposal or DecisionRecord.

    Returns:
        List of column names identified as time columns.
    """
    time_columns: list[str] = []

    if isinstance(spec, InvariantProposal):
        for col in spec.columns:
            if col.role == Role.TIME:
                time_columns.append(col.name)
            elif col.primitive_type in (PrimitiveType.DATE, PrimitiveType.DATETIME):
                time_columns.append(col.name)
    else:
        # DecisionRecord
        for name, evidence in spec.column_evidence.items():
            time_score = evidence.role_scores.get(Role.TIME, 0.0)
            if time_score >= 0.3:
                time_columns.append(name)
            elif evidence.primitive_type in (PrimitiveType.DATE, PrimitiveType.DATETIME):
                time_columns.append(name)

    return time_columns


def _extract_time_granularity(spec: DatasetSpec, column_name: str) -> TimeGranularity | None:
    """Extract time granularity for a specific column.

    Args:
        spec: Either an InvariantProposal or DecisionRecord.
        column_name: Name of the column.

    Returns:
        TimeGranularity if available, None otherwise.
    """
    if isinstance(spec, InvariantProposal):
        for col in spec.columns:
            if col.name == column_name and col.time_granularity:
                # Map string granularity to enum
                granularity_map = {
                    "year": TimeGranularity.ANNUAL,
                    "annual": TimeGranularity.ANNUAL,
                    "quarter": TimeGranularity.QUARTERLY,
                    "month": TimeGranularity.MONTHLY,
                    "week": TimeGranularity.WEEKLY,
                    "day": TimeGranularity.DAILY,
                    "hour": TimeGranularity.DAILY,  # Treat sub-daily as daily
                    "minute": TimeGranularity.DAILY,
                    "second": TimeGranularity.DAILY,
                }
                return granularity_map.get(col.time_granularity.lower())
    return None


def _normalize_column_name(name: str) -> str:
    """Normalize a column name for comparison.

    Args:
        name: Column name to normalize.

    Returns:
        Normalized column name (lowercase, stripped, underscores for spaces).
    """
    return name.lower().strip().replace(" ", "_").replace("-", "_")


def _types_compatible(type1: PrimitiveType, type2: PrimitiveType) -> bool:
    """Check if two primitive types are compatible for joining.

    Args:
        type1: First primitive type.
        type2: Second primitive type.

    Returns:
        True if types are compatible.
    """
    # Exact match
    if type1 == type2:
        return True

    # Number and integer are compatible
    numeric_types = {PrimitiveType.NUMBER, PrimitiveType.INTEGER}
    if type1 in numeric_types and type2 in numeric_types:
        return True

    # Date and datetime are compatible
    temporal_types = {PrimitiveType.DATE, PrimitiveType.DATETIME}
    if type1 in temporal_types and type2 in temporal_types:
        return True

    # Unknown is compatible with anything (assume user knows what they're doing)
    if type1 == PrimitiveType.UNKNOWN or type2 == PrimitiveType.UNKNOWN:
        return True

    return False


def _granularity_compatible(
    gran1: TimeGranularity | None,
    gran2: TimeGranularity | None,
) -> bool:
    """Check if two time granularities are compatible for joining.

    Args:
        gran1: First granularity.
        gran2: Second granularity.

    Returns:
        True if granularities are compatible (same or one unknown).
    """
    if gran1 is None or gran2 is None:
        return True  # Unknown granularity - assume compatible
    if gran1 == TimeGranularity.UNKNOWN or gran2 == TimeGranularity.UNKNOWN:
        return True
    return gran1 == gran2


def _granularity_order(granularity: TimeGranularity) -> int:
    """Get numeric order for granularity comparison (finer = lower).

    Args:
        granularity: Time granularity.

    Returns:
        Integer order value.
    """
    order = {
        TimeGranularity.DAILY: 1,
        TimeGranularity.WEEKLY: 2,
        TimeGranularity.MONTHLY: 3,
        TimeGranularity.QUARTERLY: 4,
        TimeGranularity.ANNUAL: 5,
        TimeGranularity.UNKNOWN: 0,
    }
    return order.get(granularity, 0)


def check_grain_compatibility(
    left: DatasetSpec,
    right: DatasetSpec,
) -> GrainCompatibilityResult:
    """Check if two datasets have compatible grain keys.

    Compares grain column names and types to determine if datasets
    can be joined on their grain columns.

    Args:
        left: First dataset specification.
        right: Second dataset specification.

    Returns:
        GrainCompatibilityResult with compatibility score and issues.
    """
    issues: list[CompatibilityIssue] = []
    matching_columns: list[tuple[str, str]] = []

    left_grain = _extract_grain_columns(left)
    right_grain = _extract_grain_columns(right)

    left_types = _extract_column_types(left)
    right_types = _extract_column_types(right)

    # Handle empty grains
    if not left_grain:
        issues.append(CompatibilityIssue(
            severity=Severity.WARNING,
            category="grain",
            message="Left dataset has no grain columns defined",
        ))

    if not right_grain:
        issues.append(CompatibilityIssue(
            severity=Severity.WARNING,
            category="grain",
            message="Right dataset has no grain columns defined",
        ))

    if not left_grain or not right_grain:
        return GrainCompatibilityResult(
            is_compatible=False,
            score=0.0,
            matching_columns=[],
            issues=issues,
        )

    # Normalize grain column names
    left_normalized = {_normalize_column_name(col): col for col in left_grain}
    right_normalized = {_normalize_column_name(col): col for col in right_grain}

    # Find matching columns by normalized name
    for left_norm, left_orig in left_normalized.items():
        if left_norm in right_normalized:
            right_orig = right_normalized[left_norm]

            # Check type compatibility
            left_type = left_types.get(left_orig, PrimitiveType.UNKNOWN)
            right_type = right_types.get(right_orig, PrimitiveType.UNKNOWN)

            if _types_compatible(left_type, right_type):
                matching_columns.append((left_orig, right_orig))
                issues.append(CompatibilityIssue(
                    severity=Severity.INFO,
                    category="grain",
                    message=f"Grain column match: '{left_orig}' <-> '{right_orig}'",
                    details={
                        "left_type": left_type.value,
                        "right_type": right_type.value,
                    },
                ))
            else:
                issues.append(CompatibilityIssue(
                    severity=Severity.ERROR,
                    category="grain",
                    message=f"Type mismatch for grain column '{left_orig}': "
                            f"{left_type.value} vs {right_type.value}",
                    details={
                        "left_column": left_orig,
                        "right_column": right_orig,
                        "left_type": left_type.value,
                        "right_type": right_type.value,
                    },
                ))

    # Report unmatched grain columns
    matched_left = {m[0] for m in matching_columns}
    matched_right = {m[1] for m in matching_columns}

    for col in left_grain:
        if col not in matched_left:
            issues.append(CompatibilityIssue(
                severity=Severity.WARNING,
                category="grain",
                message=f"Left grain column '{col}' has no match in right dataset",
                details={"column": col, "side": "left"},
            ))

    for col in right_grain:
        if col not in matched_right:
            issues.append(CompatibilityIssue(
                severity=Severity.WARNING,
                category="grain",
                message=f"Right grain column '{col}' has no match in left dataset",
                details={"column": col, "side": "right"},
            ))

    # Calculate compatibility score
    # Score is based on proportion of grain columns that match
    total_grain_cols = len(set(left_grain) | set(right_grain))
    matched_grain_cols = len(matching_columns)

    if total_grain_cols == 0:
        score = 0.0
    else:
        score = matched_grain_cols / total_grain_cols

    # Determine overall compatibility
    has_errors = any(i.severity == Severity.ERROR for i in issues)
    is_compatible = len(matching_columns) > 0 and not has_errors

    return GrainCompatibilityResult(
        is_compatible=is_compatible,
        score=score,
        matching_columns=matching_columns,
        issues=issues,
    )


def check_time_compatibility(
    left: DatasetSpec,
    right: DatasetSpec,
    left_time_range: TimeRangeResult | None = None,
    right_time_range: TimeRangeResult | None = None,
) -> TimeCompatibilityResult:
    """Check if two datasets have compatible time representation.

    Compares time granularity and checks for overlapping time ranges.

    Args:
        left: First dataset specification.
        right: Second dataset specification.
        left_time_range: Optional pre-computed time range for left dataset.
        right_time_range: Optional pre-computed time range for right dataset.

    Returns:
        TimeCompatibilityResult with granularity comparison and overlap info.
    """
    issues: list[CompatibilityIssue] = []

    # Extract time columns
    left_time_cols = _extract_time_columns(left)
    right_time_cols = _extract_time_columns(right)

    if not left_time_cols:
        issues.append(CompatibilityIssue(
            severity=Severity.INFO,
            category="time",
            message="Left dataset has no time columns identified",
        ))

    if not right_time_cols:
        issues.append(CompatibilityIssue(
            severity=Severity.INFO,
            category="time",
            message="Right dataset has no time columns identified",
        ))

    # Determine granularities
    left_granularity: TimeGranularity | None = None
    right_granularity: TimeGranularity | None = None

    # Use provided time ranges if available
    if left_time_range:
        left_granularity = left_time_range.granularity
    elif left_time_cols:
        # Try to get granularity from first time column
        left_granularity = _extract_time_granularity(left, left_time_cols[0])

    if right_time_range:
        right_granularity = right_time_range.granularity
    elif right_time_cols:
        right_granularity = _extract_time_granularity(right, right_time_cols[0])

    # Check granularity compatibility
    granularity_compatible = _granularity_compatible(left_granularity, right_granularity)

    if not granularity_compatible:
        left_gran_str = left_granularity.value if left_granularity else "unknown"
        right_gran_str = right_granularity.value if right_granularity else "unknown"

        # Determine which is finer
        left_order = _granularity_order(left_granularity) if left_granularity else 0
        right_order = _granularity_order(right_granularity) if right_granularity else 0

        if left_order < right_order:
            issues.append(CompatibilityIssue(
                severity=Severity.WARNING,
                category="time",
                message=f"Time granularity mismatch: left ({left_gran_str}) is finer "
                        f"than right ({right_gran_str})",
                details={
                    "left_granularity": left_gran_str,
                    "right_granularity": right_gran_str,
                    "aggregation_needed": "left to right",
                },
            ))
        else:
            issues.append(CompatibilityIssue(
                severity=Severity.WARNING,
                category="time",
                message=f"Time granularity mismatch: right ({right_gran_str}) is finer "
                        f"than left ({left_gran_str})",
                details={
                    "left_granularity": left_gran_str,
                    "right_granularity": right_gran_str,
                    "aggregation_needed": "right to left",
                },
            ))
    elif left_granularity and right_granularity:
        issues.append(CompatibilityIssue(
            severity=Severity.INFO,
            category="time",
            message=f"Time granularities match: {left_granularity.value}",
        ))

    # Check time range overlap
    overlapping_range: tuple[date, date] | None = None

    if left_time_range and right_time_range:
        left_min = left_time_range.min_date
        left_max = left_time_range.max_date
        right_min = right_time_range.min_date
        right_max = right_time_range.max_date

        if all([left_min, left_max, right_min, right_max]):
            # Calculate overlap
            overlap_start = max(left_min, right_min)  # type: ignore
            overlap_end = min(left_max, right_max)  # type: ignore

            if overlap_start <= overlap_end:
                overlapping_range = (overlap_start, overlap_end)
                issues.append(CompatibilityIssue(
                    severity=Severity.INFO,
                    category="time",
                    message=f"Time ranges overlap: {overlap_start} to {overlap_end}",
                    details={
                        "overlap_start": str(overlap_start),
                        "overlap_end": str(overlap_end),
                    },
                ))
            else:
                issues.append(CompatibilityIssue(
                    severity=Severity.ERROR,
                    category="time",
                    message="Time ranges do not overlap",
                    details={
                        "left_range": f"{left_min} to {left_max}",
                        "right_range": f"{right_min} to {right_max}",
                    },
                ))

    # Determine overall compatibility
    has_errors = any(i.severity == Severity.ERROR for i in issues)
    is_compatible = granularity_compatible and not has_errors

    return TimeCompatibilityResult(
        is_compatible=is_compatible,
        left_granularity=left_granularity,
        right_granularity=right_granularity,
        granularity_compatible=granularity_compatible,
        overlapping_range=overlapping_range,
        issues=issues,
    )


def check_join_explosion(
    left: DatasetSpec,
    right: DatasetSpec,
    left_row_count: int | None = None,
    right_row_count: int | None = None,
    left_grain_cardinality: int | None = None,
    right_grain_cardinality: int | None = None,
    threshold: float = 10.0,
) -> JoinExplosionResult:
    """Detect if grain mismatch would cause row explosion on join.

    Calculates expected row count after join and warns if the multiplier
    exceeds the threshold.

    Args:
        left: First dataset specification.
        right: Second dataset specification.
        left_row_count: Number of rows in left dataset (optional).
        right_row_count: Number of rows in right dataset (optional).
        left_grain_cardinality: Number of unique grain values in left (optional).
        right_grain_cardinality: Number of unique grain values in right (optional).
        threshold: Maximum acceptable row multiplier (default 10x).

    Returns:
        JoinExplosionResult with expected multiplier and warnings.
    """
    issues: list[CompatibilityIssue] = []

    # Extract grain columns
    left_grain = _extract_grain_columns(left)
    right_grain = _extract_grain_columns(right)

    # Without row counts or cardinalities, we can only do heuristic analysis
    if left_row_count is None or right_row_count is None:
        issues.append(CompatibilityIssue(
            severity=Severity.INFO,
            category="join",
            message="Row counts not provided; cannot calculate exact explosion risk",
        ))

        # Check grain structure for potential issues
        if len(left_grain) != len(right_grain):
            issues.append(CompatibilityIssue(
                severity=Severity.WARNING,
                category="join",
                message=f"Grain column count mismatch: left has {len(left_grain)}, "
                        f"right has {len(right_grain)}",
                details={
                    "left_grain_count": len(left_grain),
                    "right_grain_count": len(right_grain),
                },
            ))

        return JoinExplosionResult(
            expected_multiplier=1.0,  # Unknown, assume safe
            is_safe=True,
            issues=issues,
        )

    # Calculate expected explosion
    # Worst case: Cartesian product of non-matching grain
    # Best case: 1:1 join on matching grain

    # Check grain compatibility first
    grain_result = check_grain_compatibility(left, right)
    matching_count = len(grain_result.matching_columns)

    if matching_count == 0:
        # No matching grain - Cartesian product risk
        expected_rows = left_row_count * right_row_count
        multiplier = expected_rows / max(left_row_count, right_row_count)

        issues.append(CompatibilityIssue(
            severity=Severity.ERROR,
            category="join",
            message=f"No matching grain columns - Cartesian product would produce "
                    f"{expected_rows:,} rows ({multiplier:.1f}x multiplier)",
            details={
                "left_rows": left_row_count,
                "right_rows": right_row_count,
                "expected_rows": expected_rows,
                "multiplier": multiplier,
            },
        ))

        return JoinExplosionResult(
            expected_multiplier=multiplier,
            is_safe=False,
            issues=issues,
        )

    # With matching grain, estimate based on cardinality
    if left_grain_cardinality is not None and right_grain_cardinality is not None:
        # Estimate average rows per grain key
        left_avg = left_row_count / left_grain_cardinality if left_grain_cardinality > 0 else 1
        right_avg = right_row_count / right_grain_cardinality if right_grain_cardinality > 0 else 1

        # If both have multiple rows per key, explosion can occur
        if left_avg > 1 and right_avg > 1:
            # Each left row joins with (right_avg) right rows on average
            expected_multiplier = left_avg * right_avg
            expected_rows = int(left_row_count * right_avg)

            if expected_multiplier > threshold:
                issues.append(CompatibilityIssue(
                    severity=Severity.WARNING,
                    category="join",
                    message=f"Join may cause row explosion: ~{expected_multiplier:.1f}x multiplier "
                            f"(~{expected_rows:,} rows)",
                    details={
                        "left_rows_per_key": left_avg,
                        "right_rows_per_key": right_avg,
                        "expected_multiplier": expected_multiplier,
                        "expected_rows": expected_rows,
                    },
                ))

                return JoinExplosionResult(
                    expected_multiplier=expected_multiplier,
                    is_safe=False,
                    issues=issues,
                )
        else:
            issues.append(CompatibilityIssue(
                severity=Severity.INFO,
                category="join",
                message="Grain cardinality suggests safe join (one-to-many or one-to-one)",
                details={
                    "left_rows_per_key": left_avg,
                    "right_rows_per_key": right_avg,
                },
            ))

    else:
        # Without cardinality, use heuristics
        # If full grain matches, likely safe
        left_grain_set = {_normalize_column_name(c) for c in left_grain}
        right_grain_set = {_normalize_column_name(c) for c in right_grain}

        if left_grain_set == right_grain_set:
            issues.append(CompatibilityIssue(
                severity=Severity.INFO,
                category="join",
                message="Full grain match - likely safe join",
            ))
        elif left_grain_set.issubset(right_grain_set):
            # Right is more granular - many-to-one
            issues.append(CompatibilityIssue(
                severity=Severity.INFO,
                category="join",
                message="Right grain is superset of left - many-to-one join expected",
            ))
        elif right_grain_set.issubset(left_grain_set):
            # Left is more granular - one-to-many
            issues.append(CompatibilityIssue(
                severity=Severity.INFO,
                category="join",
                message="Left grain is superset of right - one-to-many join expected",
            ))
        else:
            # Partial overlap - potential for explosion
            issues.append(CompatibilityIssue(
                severity=Severity.WARNING,
                category="join",
                message="Partial grain overlap - potential for join explosion",
                details={
                    "matching_columns": matching_count,
                    "left_grain_columns": len(left_grain),
                    "right_grain_columns": len(right_grain),
                },
            ))

    return JoinExplosionResult(
        expected_multiplier=1.0,
        is_safe=True,
        issues=issues,
    )


def check_compatibility(
    left: DatasetSpec,
    right: DatasetSpec,
    left_time_range: TimeRangeResult | None = None,
    right_time_range: TimeRangeResult | None = None,
    left_row_count: int | None = None,
    right_row_count: int | None = None,
    left_grain_cardinality: int | None = None,
    right_grain_cardinality: int | None = None,
    join_explosion_threshold: float = 10.0,
) -> CompatibilityResult:
    """Check full cross-dataset compatibility.

    This is the main entry point for compatibility checking. It runs all
    compatibility checks and aggregates the results.

    Args:
        left: First dataset specification (InvariantProposal or DecisionRecord).
        right: Second dataset specification (InvariantProposal or DecisionRecord).
        left_time_range: Optional pre-computed time range for left dataset.
        right_time_range: Optional pre-computed time range for right dataset.
        left_row_count: Number of rows in left dataset (optional).
        right_row_count: Number of rows in right dataset (optional).
        left_grain_cardinality: Number of unique grain values in left (optional).
        right_grain_cardinality: Number of unique grain values in right (optional).
        join_explosion_threshold: Maximum acceptable row multiplier (default 10x).

    Returns:
        CompatibilityResult with all check results and aggregated issues.
    """
    # Run individual checks
    grain_result = check_grain_compatibility(left, right)
    time_result = check_time_compatibility(left, right, left_time_range, right_time_range)
    join_result = check_join_explosion(
        left,
        right,
        left_row_count,
        right_row_count,
        left_grain_cardinality,
        right_grain_cardinality,
        join_explosion_threshold,
    )

    # Aggregate all issues
    all_issues = grain_result.issues + time_result.issues + join_result.issues

    # Determine overall compatibility
    # Compatible if grain is compatible, no errors, and join is safe
    has_errors = any(i.severity == Severity.ERROR for i in all_issues)
    is_compatible = grain_result.is_compatible and time_result.is_compatible and join_result.is_safe and not has_errors

    return CompatibilityResult(
        is_compatible=is_compatible,
        grain_result=grain_result,
        time_result=time_result,
        join_result=join_result,
        issues=all_issues,
    )


@dataclass
class MappingSuggestion:
    """A suggested reference system mapping for a column."""

    column_name: str
    reference_system: str
    reference_system_name: str
    confidence: float  # 0.0 to 1.0
    reason: str
    category: str
    example_matches: list[str] = field(default_factory=list)


@dataclass
class MappingTask:
    """A task for creating a mapping in Invariant."""

    task_id: str
    column_name: str
    source_reference_system: str | None
    target_reference_system: str
    status: str = "pending"
    notes: str = ""


def _check_pattern_match(values: list[str], patterns: list[str]) -> float:
    """Check what fraction of values match any of the given patterns.

    Args:
        values: List of string values to check.
        patterns: List of regex patterns to match against.

    Returns:
        Fraction of values matching at least one pattern (0.0 to 1.0).
    """
    import re

    if not values:
        return 0.0

    matches = 0
    for value in values:
        for pattern in patterns:
            if re.match(pattern, str(value)):
                matches += 1
                break

    return matches / len(values)


def _check_example_overlap(values: list[str], examples: list[str]) -> float:
    """Check what fraction of example values appear in the data.

    Args:
        values: List of string values from the data.
        examples: List of example values for a reference system.

    Returns:
        Fraction of examples found in values (0.0 to 1.0).
    """
    if not examples:
        return 0.0

    value_set = set(str(v).upper() for v in values)
    example_set = set(str(e).upper() for e in examples)

    overlap = value_set & example_set
    return len(overlap) / len(example_set)


def suggest_mappings(proposal: InvariantProposal) -> list[MappingSuggestion]:
    """Suggest reference system mappings for columns in a proposal.

    Analyzes column metadata and hints to suggest which reference systems
    (ISO country codes, FIPS codes, currency codes, etc.) might apply.

    Args:
        proposal: The InvariantProposal to analyze.

    Returns:
        List of MappingSuggestion objects with confidence scores.
    """
    suggestions: list[MappingSuggestion] = []

    for col in proposal.columns:
        # Skip columns that already have reference system hints
        if col.reference_system_hint:
            # Validate the existing hint
            if col.reference_system_hint in REFERENCE_SYSTEMS:
                ref_sys = REFERENCE_SYSTEMS[col.reference_system_hint]
                suggestions.append(MappingSuggestion(
                    column_name=col.name,
                    reference_system=col.reference_system_hint,
                    reference_system_name=ref_sys["name"],
                    confidence=1.0,
                    reason="Explicitly specified in column metadata",
                    category=ref_sys["category"],
                ))
            continue

        # Only analyze KEY and DIMENSION columns
        if col.role not in (Role.KEY, Role.DIMENSION):
            continue

        # Analyze column name for hints
        col_lower = col.name.lower()
        col_suggestions: list[MappingSuggestion] = []

        # Geography-related columns
        if any(kw in col_lower for kw in ["country", "nation", "cntry"]):
            if col.primitive_type == PrimitiveType.STRING:
                # Could be alpha-2 or alpha-3
                col_suggestions.append(MappingSuggestion(
                    column_name=col.name,
                    reference_system="iso_3166_1_alpha2",
                    reference_system_name="ISO 3166-1 Alpha-2",
                    confidence=0.6,
                    reason="Column name suggests country codes; common format is ISO 3166-1 Alpha-2",
                    category="geography",
                ))
                col_suggestions.append(MappingSuggestion(
                    column_name=col.name,
                    reference_system="iso_3166_1_alpha3",
                    reference_system_name="ISO 3166-1 Alpha-3",
                    confidence=0.5,
                    reason="Column name suggests country codes; could be ISO 3166-1 Alpha-3",
                    category="geography",
                ))
            elif col.primitive_type == PrimitiveType.INTEGER:
                col_suggestions.append(MappingSuggestion(
                    column_name=col.name,
                    reference_system="iso_3166_1_numeric",
                    reference_system_name="ISO 3166-1 Numeric",
                    confidence=0.6,
                    reason="Numeric country column suggests ISO 3166-1 numeric codes",
                    category="geography",
                ))

        elif any(kw in col_lower for kw in ["state", "province", "region"]):
            col_suggestions.append(MappingSuggestion(
                column_name=col.name,
                reference_system="fips_state",
                reference_system_name="FIPS State Codes",
                confidence=0.4,
                reason="Column name suggests US state codes; could be FIPS state codes",
                category="geography",
            ))

        elif any(kw in col_lower for kw in ["county", "fips"]):
            col_suggestions.append(MappingSuggestion(
                column_name=col.name,
                reference_system="fips_county",
                reference_system_name="FIPS County Codes",
                confidence=0.5,
                reason="Column name suggests county identifiers; could be FIPS county codes",
                category="geography",
            ))

        # Currency-related columns
        elif any(kw in col_lower for kw in ["currency", "curr", "ccy"]):
            col_suggestions.append(MappingSuggestion(
                column_name=col.name,
                reference_system="iso_4217",
                reference_system_name="ISO 4217",
                confidence=0.7,
                reason="Column name suggests currency codes; ISO 4217 is the standard",
                category="currency",
            ))

        # Industry-related columns
        elif any(kw in col_lower for kw in ["naics", "industry"]):
            col_suggestions.append(MappingSuggestion(
                column_name=col.name,
                reference_system="naics",
                reference_system_name="NAICS",
                confidence=0.6 if "naics" in col_lower else 0.4,
                reason="Column name suggests industry classification",
                category="industry",
            ))

        elif "sic" in col_lower:
            col_suggestions.append(MappingSuggestion(
                column_name=col.name,
                reference_system="sic",
                reference_system_name="SIC",
                confidence=0.7,
                reason="Column name contains 'sic', suggesting Standard Industrial Classification",
                category="industry",
            ))

        # Financial identifiers
        elif any(kw in col_lower for kw in ["isin", "security_id", "sec_id"]):
            col_suggestions.append(MappingSuggestion(
                column_name=col.name,
                reference_system="isin",
                reference_system_name="ISIN",
                confidence=0.6 if "isin" in col_lower else 0.4,
                reason="Column name suggests security identifier",
                category="finance",
            ))

        elif "cusip" in col_lower:
            col_suggestions.append(MappingSuggestion(
                column_name=col.name,
                reference_system="cusip",
                reference_system_name="CUSIP",
                confidence=0.7,
                reason="Column name contains 'cusip'",
                category="finance",
            ))

        # Add all suggestions for this column
        suggestions.extend(col_suggestions)

    # Sort by confidence descending
    suggestions.sort(key=lambda s: s.confidence, reverse=True)

    return suggestions


def create_mapping_tasks(suggestions: list[MappingSuggestion]) -> list[MappingTask]:
    """Create mapping tasks from suggestions for Invariant.

    This is a stub that would create tasks in Invariant for mapping
    columns to reference systems.

    Args:
        suggestions: List of mapping suggestions to create tasks for.

    Returns:
        List of MappingTask objects (placeholder implementation).

    Note:
        This is a stub implementation. When implemented, this function will:
        - Connect to the Invariant API
        - Create mapping tasks for each suggestion
        - Track task status and progress
        - Return task IDs for reference
    """
    tasks: list[MappingTask] = []

    for i, suggestion in enumerate(suggestions):
        # Only create tasks for high-confidence suggestions
        if suggestion.confidence < 0.5:
            continue

        task = MappingTask(
            task_id=f"mapping-task-{i + 1:04d}",
            column_name=suggestion.column_name,
            source_reference_system=None,  # To be determined
            target_reference_system=suggestion.reference_system,
            status="pending",
            notes=f"Auto-generated from suggestion: {suggestion.reason}",
        )
        tasks.append(task)

    return tasks
