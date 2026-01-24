"""Role scoring and assignment for dataset columns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from datasculpt.core.types import (
    ColumnEvidence,
    InferenceConfig,
    PrimitiveType,
    Role,
    StructuralType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# Naming patterns for role detection
KEY_NAME_PATTERNS = (
    re.compile(r"id$", re.IGNORECASE),  # ends with "id" (hhid, personid, geo_id)
    re.compile(r"^id_", re.IGNORECASE),  # starts with "id_" (id_hh, id_person)
    re.compile(r"_code$", re.IGNORECASE),
    re.compile(r"^code$", re.IGNORECASE),
    re.compile(r"_key$", re.IGNORECASE),
    re.compile(r"^key$", re.IGNORECASE),
    re.compile(r"_pk$", re.IGNORECASE),
    re.compile(r"^pk$", re.IGNORECASE),
    re.compile(r"_num$", re.IGNORECASE),  # record numbers
    re.compile(r"^uid$", re.IGNORECASE),  # unique id
    re.compile(r"uuid", re.IGNORECASE),  # uuid
)

TIME_NAME_PATTERNS = (
    re.compile(r"date", re.IGNORECASE),
    re.compile(r"time", re.IGNORECASE),
    re.compile(r"period", re.IGNORECASE),
    re.compile(r"year", re.IGNORECASE),
    re.compile(r"month", re.IGNORECASE),
    re.compile(r"day", re.IGNORECASE),
    re.compile(r"quarter", re.IGNORECASE),
    re.compile(r"week", re.IGNORECASE),
    re.compile(r"^dt$", re.IGNORECASE),
    re.compile(r"_dt$", re.IGNORECASE),
    re.compile(r"timestamp", re.IGNORECASE),
)

INDICATOR_NAME_PATTERNS = (
    re.compile(r"indicator", re.IGNORECASE),
    re.compile(r"metric", re.IGNORECASE),
    re.compile(r"^measure$", re.IGNORECASE),  # exactly "measure"
    re.compile(r"measure_?name", re.IGNORECASE),  # measure_name, measurename
    re.compile(r"variable", re.IGNORECASE),
    re.compile(r"series_?name", re.IGNORECASE),  # series_name, seriesname
    re.compile(r"statistic", re.IGNORECASE),  # statistic, not "state"
)

VALUE_NAME_PATTERNS = (
    re.compile(r"^value$", re.IGNORECASE),
    re.compile(r"^val$", re.IGNORECASE),
    re.compile(r"^amount$", re.IGNORECASE),
    re.compile(r"^obs$", re.IGNORECASE),
    re.compile(r"^observation$", re.IGNORECASE),
)

# Patterns for survey question codes - these are dimensions, not indicator names
# Examples: s1q2, q1, question_1, var_01, v101, hv001
SURVEY_QUESTION_PATTERNS = (
    re.compile(r"^s\d+q\d+", re.IGNORECASE),  # s1q2, s01q01
    re.compile(r"^q\d+", re.IGNORECASE),  # q1, q01, q1a
    re.compile(r"^v\d+", re.IGNORECASE),  # v101, v001
    re.compile(r"^hv\d+", re.IGNORECASE),  # hv001 (DHS household)
    re.compile(r"^mv\d+", re.IGNORECASE),  # mv001 (DHS men's)
    re.compile(r"^question", re.IGNORECASE),  # question_1, question1
    re.compile(r"^var_?\d+", re.IGNORECASE),  # var_01, var01
)


def _matches_any_pattern(name: str, patterns: tuple[re.Pattern, ...]) -> bool:
    """Check if name matches any pattern in the tuple."""
    return any(p.search(name) for p in patterns)


def _clamp(value: float) -> float:
    """Clamp value to 0.0-1.0 range."""
    return max(0.0, min(1.0, value))


def score_key_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
) -> float:
    """Score likelihood that column is a key (primary/foreign key).

    High scores for:
    - High cardinality (many distinct values relative to row count)
    - Low null rate
    - Naming patterns like _id, _code, _key

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.

    Returns:
        Score between 0.0 and 1.0.
    """
    if config is None:
        config = InferenceConfig()

    score = 0.0

    # High cardinality is strong signal for keys
    if evidence.distinct_ratio >= config.key_cardinality_threshold:
        score += 0.4
    elif evidence.distinct_ratio >= 0.7:
        score += 0.2

    # Low nulls expected for keys
    if evidence.null_rate <= config.null_rate_threshold:
        score += 0.2
    elif evidence.null_rate <= 0.05:
        score += 0.1

    # Naming patterns
    if _matches_any_pattern(evidence.name, KEY_NAME_PATTERNS):
        score += 0.3

    # Integer or string types are common for keys
    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.STRING):
        score += 0.1

    return _clamp(score)


def score_dimension_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
) -> float:
    """Score likelihood that column is a dimension (categorical grouping).

    High scores for:
    - Low to medium cardinality
    - String type
    - Scalar structural type

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.

    Returns:
        Score between 0.0 and 1.0.
    """
    if config is None:
        config = InferenceConfig()

    score = 0.0

    # Low-medium cardinality is ideal for dimensions
    if evidence.distinct_ratio <= config.dimension_cardinality_max:
        score += 0.4
    elif evidence.distinct_ratio <= 0.3:
        score += 0.2

    # String type is common for dimensions
    if evidence.primitive_type == PrimitiveType.STRING:
        score += 0.3

    # Scalar values (not arrays or objects)
    if evidence.structural_type == StructuralType.SCALAR:
        score += 0.1

    # Moderate null rate is acceptable
    if evidence.null_rate <= 0.1:
        score += 0.1

    # Penalize very high cardinality (likely not a dimension)
    if evidence.distinct_ratio > 0.5:
        score -= 0.2

    return _clamp(score)


def score_measure_role(evidence: ColumnEvidence) -> float:
    """Score likelihood that column is a measure (numeric fact/metric).

    High scores for:
    - Numeric type (integer or number)
    - Varying values (not constant)
    - Scalar structural type
    - High cardinality (continuous values)

    Low scores for:
    - Columns that look like codes/IDs (low cardinality integers)
    - Columns with time-related names (year, month)
    - Columns with ID-related names

    Args:
        evidence: Column evidence from profiling.

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # Numeric type is required for measures
    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        score += 0.4
    else:
        # Non-numeric types are very unlikely to be measures
        return 0.0

    # Varying values (not just one constant)
    if evidence.distinct_ratio > 0.01:
        score += 0.1

    # Scalar values
    if evidence.structural_type == StructuralType.SCALAR:
        score += 0.1

    # High cardinality is common for continuous measures
    if evidence.distinct_ratio > 0.3:
        score += 0.2
    elif evidence.distinct_ratio < 0.01:
        # Very low cardinality integers are likely codes, not measures
        score -= 0.2

    # Low null rate
    if evidence.null_rate <= 0.1:
        score += 0.1

    # Penalty for columns that look like IDs or time components
    if _matches_any_pattern(evidence.name, KEY_NAME_PATTERNS):
        score -= 0.3
    if _matches_any_pattern(evidence.name, TIME_NAME_PATTERNS):
        score -= 0.3

    return _clamp(score)


def score_time_role(evidence: ColumnEvidence) -> float:
    """Score likelihood that column is a time/date column.

    High scores for:
    - Date or datetime primitive type
    - Successful date parsing (from parse_results)
    - Naming patterns like date, time, period

    Args:
        evidence: Column evidence from profiling.

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # Already typed as date/datetime
    if evidence.primitive_type in (PrimitiveType.DATE, PrimitiveType.DATETIME):
        score += 0.5

    # Successful date parsing
    date_parse_success = evidence.parse_results.get("date", 0.0)
    if date_parse_success >= 0.9:
        score += 0.3
    elif date_parse_success >= 0.5:
        score += 0.15

    datetime_parse_success = evidence.parse_results.get("datetime", 0.0)
    if datetime_parse_success >= 0.9:
        score += 0.3
    elif datetime_parse_success >= 0.5:
        score += 0.15

    # Naming patterns
    if _matches_any_pattern(evidence.name, TIME_NAME_PATTERNS):
        score += 0.2

    return _clamp(score)


def score_indicator_name_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
) -> float:
    """Score likelihood that column is an indicator name (long format).

    High scores for:
    - Low cardinality (few distinct indicator names)
    - String type
    - Naming patterns suggesting indicator/metric names

    Low scores for:
    - Survey question codes (s1q2, q1, v101, etc.)
    - Generic dimension columns without indicator naming patterns

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.

    Returns:
        Score between 0.0 and 1.0.
    """
    if config is None:
        config = InferenceConfig()

    score = 0.0

    # String type required
    if evidence.primitive_type != PrimitiveType.STRING:
        return 0.0

    # Check for indicator naming patterns first - this is the strongest signal
    has_indicator_pattern = _matches_any_pattern(
        evidence.name, INDICATOR_NAME_PATTERNS
    )

    # Penalize survey question codes - these are dimensions, not indicator names
    is_survey_question = _matches_any_pattern(
        evidence.name, SURVEY_QUESTION_PATTERNS
    )

    if is_survey_question:
        # Survey questions are almost never indicator name columns
        return 0.1

    # Base score for string type
    score += 0.15

    # Low cardinality (typically few distinct indicator names)
    # But only moderate boost without naming pattern
    if evidence.distinct_ratio <= 0.05:
        score += 0.2 if has_indicator_pattern else 0.1
    elif evidence.distinct_ratio <= config.dimension_cardinality_max:
        score += 0.15 if has_indicator_pattern else 0.05

    # Naming patterns - strong signal
    if has_indicator_pattern:
        score += 0.4

    # Very low null rate expected
    if evidence.null_rate <= 0.01:
        score += 0.1

    # Scalar type
    if evidence.structural_type == StructuralType.SCALAR:
        score += 0.05

    return _clamp(score)


def score_value_role(
    evidence: ColumnEvidence,
    has_indicator_column: bool = False,
) -> float:
    """Score likelihood that column is a value column (paired with indicator).

    High scores for:
    - Numeric type WITH naming patterns like 'value', 'amount'
    - Numeric type WITH presence of a clear indicator column

    Low scores for:
    - Columns that look like IDs or time components
    - Low cardinality integers (likely codes)
    - Generic numeric columns without indicator context

    The VALUE role is specifically for indicator/value pair patterns in long
    format data. Without evidence of this pattern, numeric columns should
    score as MEASURE instead.

    Args:
        evidence: Column evidence from profiling.
        has_indicator_column: Whether dataset has a likely indicator name column.

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # Numeric type required
    if evidence.primitive_type not in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        return 0.0

    # Check for value naming patterns
    has_value_pattern = _matches_any_pattern(evidence.name, VALUE_NAME_PATTERNS)

    # VALUE role requires strong evidence: either naming pattern OR indicator context
    # Without these, the column should score as MEASURE instead
    if not has_value_pattern and not has_indicator_column:
        # No evidence of indicator/value pattern - return low score
        return 0.1

    # Base score for numeric type in indicator context
    score += 0.2

    # Indicator column presence is a strong signal
    if has_indicator_column:
        score += 0.3

    # Naming patterns
    if has_value_pattern:
        score += 0.35

    # Scalar type
    if evidence.structural_type == StructuralType.SCALAR:
        score += 0.05

    # Penalty for columns that look like IDs or time components
    if _matches_any_pattern(evidence.name, KEY_NAME_PATTERNS):
        score -= 0.4
    if _matches_any_pattern(evidence.name, TIME_NAME_PATTERNS):
        score -= 0.4

    # Low cardinality integers are likely codes, not values
    if evidence.distinct_ratio < 0.01:
        score -= 0.2

    return _clamp(score)


def score_series_role(evidence: ColumnEvidence) -> float:
    """Score likelihood that column contains series data (JSON arrays).

    High scores for:
    - Array structural type
    - Successful JSON array parsing

    Args:
        evidence: Column evidence from profiling.

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # Already typed as array
    if evidence.structural_type == StructuralType.ARRAY:
        score += 0.5

    # Successful JSON array parsing
    json_array_success = evidence.parse_results.get("json_array", 0.0)
    if json_array_success >= 0.9:
        score += 0.4
    elif json_array_success >= 0.5:
        score += 0.2

    # String type with high array parse rate
    if (
        evidence.primitive_type == PrimitiveType.STRING
        and json_array_success >= 0.8
    ):
        score += 0.1

    return _clamp(score)


@dataclass
class RoleAssignment:
    """Result of role assignment for a column."""

    role: Role
    score: float
    confidence: float
    all_scores: dict[Role, float]


def compute_role_scores(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
    has_indicator_column: bool = False,
) -> dict[Role, float]:
    """Compute all role scores for a column.

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.
        has_indicator_column: Whether dataset has a likely indicator column.

    Returns:
        Dictionary mapping roles to scores.
    """
    return {
        Role.KEY: score_key_role(evidence, config),
        Role.DIMENSION: score_dimension_role(evidence, config),
        Role.MEASURE: score_measure_role(evidence),
        Role.TIME: score_time_role(evidence),
        Role.INDICATOR_NAME: score_indicator_name_role(evidence, config),
        Role.VALUE: score_value_role(evidence, has_indicator_column),
        Role.SERIES: score_series_role(evidence),
        Role.METADATA: 0.1,  # Default low score; metadata is fallback
    }


def calculate_confidence(scores: dict[Role, float]) -> float:
    """Calculate confidence from gap between top two role scores.

    Confidence is the difference between the highest and second-highest
    score, normalized. A large gap means high confidence in the assignment.

    Args:
        scores: Dictionary mapping roles to scores.

    Returns:
        Confidence value between 0.0 and 1.0.
    """
    if not scores:
        return 0.0

    sorted_scores = sorted(scores.values(), reverse=True)

    if len(sorted_scores) < 2:
        return 1.0 if sorted_scores[0] > 0 else 0.0

    top_score = sorted_scores[0]
    second_score = sorted_scores[1]

    if top_score == 0:
        return 0.0

    # Confidence is the gap, normalized by top score
    gap = top_score - second_score
    confidence = gap / top_score if top_score > 0 else 0.0

    return _clamp(confidence)


def resolve_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
    has_indicator_column: bool = False,
) -> RoleAssignment:
    """Assign primary role based on highest scores.

    Handles ties by preferring more specific roles (e.g., KEY over DIMENSION).

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.
        has_indicator_column: Whether dataset has a likely indicator column.

    Returns:
        RoleAssignment with assigned role, score, and confidence.
    """
    scores = compute_role_scores(evidence, config, has_indicator_column)

    # Role priority for tie-breaking (more specific roles first)
    role_priority = [
        Role.KEY,
        Role.TIME,
        Role.INDICATOR_NAME,
        Role.VALUE,
        Role.SERIES,
        Role.MEASURE,
        Role.DIMENSION,
        Role.METADATA,
    ]

    best_role = Role.METADATA
    best_score = 0.0

    for role in role_priority:
        score = scores.get(role, 0.0)
        if score > best_score:
            best_role = role
            best_score = score

    confidence = calculate_confidence(scores)

    return RoleAssignment(
        role=best_role,
        score=best_score,
        confidence=confidence,
        all_scores=scores,
    )


def assign_roles(
    evidences: Sequence[ColumnEvidence],
    config: InferenceConfig | None = None,
) -> dict[str, RoleAssignment]:
    """Assign roles to all columns in a dataset.

    Performs two passes:
    1. First pass identifies potential indicator columns
    2. Second pass assigns all roles with indicator context

    Args:
        evidences: Sequence of column evidences.
        config: Optional inference configuration.

    Returns:
        Dictionary mapping column names to role assignments.
    """
    if config is None:
        config = InferenceConfig()

    # First pass: detect if there's likely an indicator column
    has_indicator = False
    for evidence in evidences:
        indicator_score = score_indicator_name_role(evidence, config)
        if indicator_score >= 0.5:
            has_indicator = True
            break

    # Second pass: assign roles with indicator context
    assignments: dict[str, RoleAssignment] = {}
    for evidence in evidences:
        assignments[evidence.name] = resolve_role(
            evidence,
            config,
            has_indicator_column=has_indicator,
        )

    return assignments


def update_evidence_with_roles(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
    has_indicator_column: bool = False,
) -> ColumnEvidence:
    """Update column evidence with computed role scores.

    Args:
        evidence: Column evidence to update.
        config: Optional inference configuration.
        has_indicator_column: Whether dataset has a likely indicator column.

    Returns:
        Updated evidence with role_scores populated.
    """
    scores = compute_role_scores(evidence, config, has_indicator_column)
    evidence.role_scores = scores
    return evidence
