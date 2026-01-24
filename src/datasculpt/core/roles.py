"""Role scoring and assignment for dataset columns.

This module provides scoring functions for semantic column roles and
supports a three-phase role assignment process:

1. Local scoring: Score each column independently for each role
2. Shape-conditional adjustment: Modify scores based on detected dataset shape
3. Global reconciliation: Enforce dataset-level constraints
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from datasculpt.core.types import (
    ColumnEvidence,
    InferenceConfig,
    PrimitiveType,
    Role,
    ShapeHypothesis,
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

# Patterns for pseudo-keys (columns that look unique but aren't meaningful keys)
# These need penalties in role scoring, not just grain inference
PSEUDO_KEY_PATTERNS = (
    (re.compile(r"^row_?id$", re.IGNORECASE), 0.6),  # row_id, rowid
    (re.compile(r"^index$", re.IGNORECASE), 0.6),  # index (common DataFrame index)
    (re.compile(r"^row_?num(ber)?$", re.IGNORECASE), 0.6),  # row_num, row_number
    (re.compile(r"^seq(uence)?$", re.IGNORECASE), 0.5),  # seq, sequence
    (re.compile(r"^auto_?inc(rement)?$", re.IGNORECASE), 0.6),  # auto_inc
    (re.compile(r"^record_?id$", re.IGNORECASE), 0.4),  # record_id
    (re.compile(r"^uuid$", re.IGNORECASE), 0.5),  # uuid column
    (re.compile(r"^guid$", re.IGNORECASE), 0.5),  # guid column
    (re.compile(r"^hash$", re.IGNORECASE), 0.4),  # hash column
    (re.compile(r"^created_at$", re.IGNORECASE), 0.4),  # ingestion timestamp
    (re.compile(r"^updated_at$", re.IGNORECASE), 0.4),  # update timestamp
    (re.compile(r"^inserted_at$", re.IGNORECASE), 0.4),  # insertion timestamp
    (re.compile(r"^_?id$", re.IGNORECASE), 0.2),  # just "id" - lower penalty, might be real
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


def _get_pseudo_key_penalty(name: str) -> float:
    """Get penalty for pseudo-key name patterns.

    Returns the highest matching penalty, or 0.0 if no pattern matches.
    Pseudo-keys are columns that appear unique but don't represent
    meaningful business keys (e.g., row indices, UUIDs, auto-increments).
    """
    max_penalty = 0.0
    for pattern, penalty in PSEUDO_KEY_PATTERNS:
        if pattern.search(name):
            max_penalty = max(max_penalty, penalty)
    return max_penalty


def _looks_like_numeric_code(name: str, distinct_ratio: float) -> bool:
    """Check if a column looks like a numeric code rather than a measure.

    Numeric codes (category IDs, status codes) have:
    - Low cardinality (few distinct values)
    - Names suggesting codes/IDs/status
    """
    if distinct_ratio >= 0.1:
        return False

    code_patterns = (
        re.compile(r"_code$", re.IGNORECASE),
        re.compile(r"_type$", re.IGNORECASE),
        re.compile(r"_status$", re.IGNORECASE),
        re.compile(r"_flag$", re.IGNORECASE),
        re.compile(r"_cat(egory)?$", re.IGNORECASE),
        re.compile(r"_class$", re.IGNORECASE),
        re.compile(r"_level$", re.IGNORECASE),
        re.compile(r"^code", re.IGNORECASE),
        re.compile(r"^type", re.IGNORECASE),
        re.compile(r"^status", re.IGNORECASE),
    )
    return any(p.search(name) for p in code_patterns)


def score_key_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
) -> float:
    """Score likelihood that column is a key (primary/foreign key).

    High scores for:
    - High cardinality (many distinct values relative to row count)
    - Low null rate
    - Naming patterns like _id, _code, _key

    Low scores (penalties) for:
    - Pseudo-key patterns (row_id, index, uuid, created_at)
    - Monotonic integer sequences (detected via parse_results)

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

    # === PSEUDO-KEY PENALTIES ===
    # Penalize columns that look like auto-generated/synthetic keys

    # Penalty for pseudo-key naming patterns (row_id, index, uuid, etc.)
    pseudo_penalty = _get_pseudo_key_penalty(evidence.name)
    if pseudo_penalty > 0:
        score -= pseudo_penalty

    # Check for monotonic sequence signal (if available from evidence profiler)
    # This catches 1, 2, 3, ... N row indices
    if evidence.parse_results.get("monotonic_sequence", 0.0) > 0.9:
        score -= 0.5

    # Check for UUID/hash-like signal (high entropy hex patterns)
    if evidence.parse_results.get("uuid_like", 0.0) > 0.9:
        score -= 0.4

    return _clamp(score)


def score_dimension_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
) -> float:
    """Score likelihood that column is a dimension (categorical grouping).

    Dimensions come in two flavors:
    1. Categorical dimensions: Low cardinality (country, status, category)
    2. Identifier dimensions: High cardinality but not the primary key
       (geo_id in fact tables, person_id as FK, product_code)

    High scores for:
    - String type with low-medium cardinality (classic categorical)
    - High cardinality with _id/_code suffix BUT not pseudo-key patterns
    - Scalar structural type

    Low scores for:
    - Numeric types without dimension-like naming
    - Date/time columns
    - Pseudo-key patterns (row_id, index)

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.

    Returns:
        Score between 0.0 and 1.0.
    """
    if config is None:
        config = InferenceConfig()

    score = 0.0

    # === CATEGORICAL DIMENSION SCORING ===
    # Low-medium cardinality strings are classic dimensions
    if evidence.distinct_ratio <= config.dimension_cardinality_max:
        score += 0.4
    elif evidence.distinct_ratio <= 0.3:
        score += 0.25
    elif evidence.distinct_ratio <= 0.5:
        score += 0.1

    # String type is common for categorical dimensions
    if evidence.primitive_type == PrimitiveType.STRING:
        score += 0.3

    # === IDENTIFIER DIMENSION SCORING ===
    # High-cardinality columns can still be dimensions if they look like
    # foreign keys or identifier references (geo_id, person_id, product_code)
    is_identifier_like = _matches_any_pattern(evidence.name, KEY_NAME_PATTERNS)
    is_pseudo_key = _get_pseudo_key_penalty(evidence.name) > 0.3

    if evidence.distinct_ratio > 0.5 and is_identifier_like and not is_pseudo_key:
        # This looks like an identifier dimension (FK reference)
        # Don't penalize high cardinality for these
        score += 0.15
    elif evidence.distinct_ratio > 0.98:
        # Extremely high cardinality without identifier naming - penalize
        # This is likely a key column, not a dimension
        score -= 0.2

    # Scalar values (not arrays or objects)
    if evidence.structural_type == StructuralType.SCALAR:
        score += 0.1

    # Moderate null rate is acceptable
    if evidence.null_rate <= 0.1:
        score += 0.1

    # Penalize numeric types without dimension-like naming
    # Numeric columns are usually measures, not dimensions
    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        if not is_identifier_like and not _looks_like_numeric_code(
            evidence.name, evidence.distinct_ratio
        ):
            score -= 0.15

    # Penalize date/time types - these should be TIME role, not DIMENSION
    if evidence.primitive_type in (PrimitiveType.DATE, PrimitiveType.DATETIME):
        score -= 0.3

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
    - Numeric code columns (type_code, status_flag)

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

    # Penalty for numeric code columns (status_code, type_flag, category_id)
    # These are dimensions encoded as numbers, not measures
    if _looks_like_numeric_code(evidence.name, evidence.distinct_ratio):
        score -= 0.4

    # Additional penalty for very low cardinality integers with ID-like names
    # These are almost certainly codes, not measures
    if evidence.distinct_ratio < 0.05 and evidence.primitive_type == PrimitiveType.INTEGER:
        if _matches_any_pattern(evidence.name, KEY_NAME_PATTERNS):
            score -= 0.2

    return _clamp(score)


def score_time_role(evidence: ColumnEvidence) -> float:
    """Score likelihood that column is a time/date column.

    High scores for:
    - Date or datetime primitive type
    - Successful date parsing (from parse_results)
    - Naming patterns like date, time, period

    Note: Date and datetime parse results are treated as mutually exclusive
    signals (using max, not sum) to avoid double-counting.

    Args:
        evidence: Column evidence from profiling.

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # Already typed as date/datetime - dominant signal
    if evidence.primitive_type in (PrimitiveType.DATE, PrimitiveType.DATETIME):
        score += 0.6

    # Successful date/datetime parsing - use max, not sum
    # These are overlapping signals, so we take the stronger one
    date_parse_success = evidence.parse_results.get("date", 0.0)
    datetime_parse_success = evidence.parse_results.get("datetime", 0.0)
    best_parse = max(date_parse_success, datetime_parse_success)

    if best_parse >= 0.9:
        score += 0.25
    elif best_parse >= 0.7:
        score += 0.15
    elif best_parse >= 0.5:
        score += 0.1

    # Naming patterns
    if _matches_any_pattern(evidence.name, TIME_NAME_PATTERNS):
        score += 0.2

    return _clamp(score)


def score_indicator_name_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
    detected_shape: ShapeHypothesis | None = None,
) -> float:
    """Score likelihood that column is an indicator name (long format).

    The INDICATOR_NAME role is shape-conditional:
    - High scores in LONG_INDICATORS shape
    - Suppressed in wide shapes (WIDE_OBSERVATIONS, WIDE_TIME_COLUMNS)

    Survey question codes (s1q2, q1, v101) are treated differently based on shape:
    - In LONG_INDICATORS: question codes can BE the indicator name
    - In other shapes: question codes are dimensions

    High scores for:
    - Low cardinality (few distinct indicator names)
    - String type
    - Naming patterns suggesting indicator/metric names
    - LONG_INDICATORS shape with appropriate cardinality

    Low scores for:
    - Wide shapes (WIDE_OBSERVATIONS, WIDE_TIME_COLUMNS)
    - Generic dimension columns without indicator naming patterns

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.
        detected_shape: The detected dataset shape (affects scoring).

    Returns:
        Score between 0.0 and 1.0.
    """
    if config is None:
        config = InferenceConfig()

    score = 0.0

    # String type required
    if evidence.primitive_type != PrimitiveType.STRING:
        return 0.0

    # === SHAPE-CONDITIONAL SCORING ===
    # INDICATOR_NAME role should be suppressed in wide shapes
    if detected_shape in (
        ShapeHypothesis.WIDE_OBSERVATIONS,
        ShapeHypothesis.WIDE_TIME_COLUMNS,
    ):
        # In wide shapes, there's typically no indicator name column
        # unless explicitly named (rare)
        has_indicator_pattern = _matches_any_pattern(
            evidence.name, INDICATOR_NAME_PATTERNS
        )
        if not has_indicator_pattern:
            return 0.0
        # Even with indicator pattern in wide shape, reduce score
        return 0.2 if has_indicator_pattern else 0.0

    # Check for indicator naming patterns - this is the strongest signal
    has_indicator_pattern = _matches_any_pattern(
        evidence.name, INDICATOR_NAME_PATTERNS
    )

    # Survey question codes treatment depends on shape
    is_survey_question = _matches_any_pattern(
        evidence.name, SURVEY_QUESTION_PATTERNS
    )

    # In LONG_INDICATORS shape, survey question codes can BE the indicator
    if is_survey_question:
        if detected_shape == ShapeHypothesis.LONG_INDICATORS:
            # In long-indicator format, question codes might be the indicator column
            # Check for moderate cardinality (suggests it's labeling different measures)
            if 0.01 < evidence.distinct_ratio < 0.3:
                score += 0.4
            else:
                score += 0.15
        else:
            # Survey questions are dimensions in non-indicator shapes
            return 0.1

    # Base score for string type
    score += 0.15

    # Low cardinality (typically few distinct indicator names)
    # But only moderate boost without naming pattern
    if evidence.distinct_ratio <= 0.05:
        score += 0.2 if has_indicator_pattern else 0.1
    elif evidence.distinct_ratio <= config.dimension_cardinality_max:
        score += 0.15 if has_indicator_pattern else 0.05
    elif evidence.distinct_ratio <= 0.3:
        # Moderate cardinality is acceptable for indicators
        score += 0.1 if has_indicator_pattern else 0.0

    # Naming patterns - strong signal
    if has_indicator_pattern:
        score += 0.4

    # Boost for LONG_INDICATORS shape
    if detected_shape == ShapeHypothesis.LONG_INDICATORS:
        score += 0.1

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
    detected_shape: ShapeHypothesis | None = None,
) -> float:
    """Score likelihood that column is a value column (paired with indicator).

    The VALUE role is specifically for indicator/value pair patterns in long
    format data. It is shape-conditional:
    - High scores in LONG_INDICATORS shape with indicator column present
    - Very low scores in WIDE_OBSERVATIONS or other wide shapes

    High scores for:
    - Numeric type WITH naming patterns like 'value', 'amount'
    - Numeric type WITH presence of a clear indicator column
    - LONG_INDICATORS shape detected

    Low scores for:
    - Columns that look like IDs or time components
    - Low cardinality integers (likely codes)
    - Generic numeric columns without indicator context
    - Wide shapes (WIDE_OBSERVATIONS, WIDE_TIME_COLUMNS)

    Args:
        evidence: Column evidence from profiling.
        has_indicator_column: Whether dataset has a likely indicator name column.
        detected_shape: The detected dataset shape (affects VALUE scoring).

    Returns:
        Score between 0.0 and 1.0.
    """
    score = 0.0

    # Numeric type required
    if evidence.primitive_type not in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        return 0.0

    # Check for value naming patterns
    has_value_pattern = _matches_any_pattern(evidence.name, VALUE_NAME_PATTERNS)

    # === SHAPE-CONDITIONAL SCORING ===
    # VALUE role should be suppressed in wide shapes
    if detected_shape in (
        ShapeHypothesis.WIDE_OBSERVATIONS,
        ShapeHypothesis.WIDE_TIME_COLUMNS,
    ):
        # In wide shapes, numeric columns should be MEASURE, not VALUE
        # Only allow VALUE if explicitly named
        if not has_value_pattern:
            return 0.0
        # Even with value pattern in wide shape, reduce score
        return 0.3 if has_value_pattern else 0.0

    # VALUE role requires strong evidence: either naming pattern OR indicator context
    # Without these, the column should score as MEASURE instead
    if not has_value_pattern and not has_indicator_column:
        # No evidence of indicator/value pattern - return zero
        # This is stricter than before to prevent VALUE displacing MEASURE
        return 0.0

    # Base score for numeric type in indicator context
    score += 0.2

    # Indicator column presence is a strong signal
    if has_indicator_column:
        score += 0.3

    # Boost for LONG_INDICATORS shape
    if detected_shape == ShapeHypothesis.LONG_INDICATORS:
        score += 0.15

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
    secondary_role: Role | None = None
    secondary_score: float = 0.0
    reasons: list[str] = field(default_factory=list)


def compute_role_scores(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
    has_indicator_column: bool = False,
    detected_shape: ShapeHypothesis | None = None,
) -> dict[Role, float]:
    """Compute all role scores for a column.

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.
        has_indicator_column: Whether dataset has a likely indicator column.
        detected_shape: Detected dataset shape for shape-conditional scoring.

    Returns:
        Dictionary mapping roles to scores.
    """
    return {
        Role.KEY: score_key_role(evidence, config),
        Role.DIMENSION: score_dimension_role(evidence, config),
        Role.MEASURE: score_measure_role(evidence),
        Role.TIME: score_time_role(evidence),
        Role.INDICATOR_NAME: score_indicator_name_role(evidence, config, detected_shape),
        Role.VALUE: score_value_role(evidence, has_indicator_column, detected_shape),
        Role.SERIES: score_series_role(evidence),
        Role.METADATA: 0.1,  # Default low score; metadata is fallback
    }


def calculate_confidence(
    scores: dict[Role, float],
    pseudo_key_penalty: float = 0.0,
) -> float:
    """Calculate confidence from absolute score strength and separation.

    Improved confidence formula that considers:
    - Absolute strength: The top score value (high score = more confident)
    - Relative separation: Gap between top and second scores
    - Pseudo-key penalty: Reduces confidence if column looks auto-generated

    Formula: confidence = 0.5 * best_score + 0.5 * ((best - second) / max(best, eps))
    Then adjusted for pseudo-key penalty.

    Args:
        scores: Dictionary mapping roles to scores.
        pseudo_key_penalty: Penalty from pseudo-key detection (0.0-0.5).

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

    # Improved confidence formula:
    # - 50% weight on absolute strength (high score = confident)
    # - 50% weight on relative separation (clear winner = confident)
    eps = 0.001  # Avoid division by zero
    absolute_strength = top_score
    relative_separation = (top_score - second_score) / max(top_score, eps)

    confidence = 0.5 * absolute_strength + 0.5 * relative_separation

    # Apply pseudo-key penalty (reduces confidence in the assignment)
    # If this looks like a pseudo-key, we're less confident it's a real key
    if pseudo_key_penalty > 0:
        confidence -= pseudo_key_penalty * 0.3

    return _clamp(confidence)


def resolve_role(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
    has_indicator_column: bool = False,
    detected_shape: ShapeHypothesis | None = None,
) -> RoleAssignment:
    """Assign primary role based on highest scores.

    Returns both primary and secondary roles with explanatory reasons
    to support downstream decisions (grain inference, user confirmation).

    Handles ties by preferring more specific roles (e.g., KEY over DIMENSION),
    but only as a soft nudge when scores are very close.

    Args:
        evidence: Column evidence from profiling.
        config: Optional inference configuration.
        has_indicator_column: Whether dataset has a likely indicator column.
        detected_shape: Detected dataset shape for shape-conditional scoring.

    Returns:
        RoleAssignment with assigned role, score, confidence, secondary role, and reasons.
    """
    scores = compute_role_scores(evidence, config, has_indicator_column, detected_shape)
    reasons: list[str] = []

    # Role priority for tie-breaking (more specific roles first)
    # Only used when scores are within 0.05 of each other
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

    # Sort roles by score descending
    sorted_roles = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_role = sorted_roles[0][0]
    best_score = sorted_roles[0][1]

    # Get second-best for reporting
    secondary_role = sorted_roles[1][0] if len(sorted_roles) > 1 else None
    secondary_score = sorted_roles[1][1] if len(sorted_roles) > 1 else 0.0

    # Handle near-ties with priority (only if gap < 0.05)
    if len(sorted_roles) >= 2 and best_score - secondary_score < 0.05:
        # Use priority to break tie
        best_priority = role_priority.index(best_role) if best_role in role_priority else 99
        second_priority = role_priority.index(sorted_roles[1][0]) if sorted_roles[1][0] in role_priority else 99
        if second_priority < best_priority:
            # Swap: the higher-priority role wins the tie
            reasons.append(f"Tie-break: preferred {sorted_roles[1][0].value} over {best_role.value}")
            best_role, secondary_role = sorted_roles[1][0], best_role
            best_score, secondary_score = sorted_roles[1][1], sorted_roles[0][1]

    # Build explanatory reasons
    if _matches_any_pattern(evidence.name, KEY_NAME_PATTERNS):
        reasons.append(f"name matches key pattern (*_id, *_code)")
    if _matches_any_pattern(evidence.name, TIME_NAME_PATTERNS):
        reasons.append(f"name matches time pattern (date, time, year)")
    if _matches_any_pattern(evidence.name, INDICATOR_NAME_PATTERNS):
        reasons.append(f"name matches indicator pattern")
    if _matches_any_pattern(evidence.name, VALUE_NAME_PATTERNS):
        reasons.append(f"name matches value pattern")

    pseudo_penalty = _get_pseudo_key_penalty(evidence.name)
    if pseudo_penalty > 0:
        reasons.append(f"pseudo-key penalty: {pseudo_penalty:.2f}")

    if evidence.distinct_ratio >= 0.95:
        reasons.append(f"high cardinality ({evidence.distinct_ratio:.2%})")
    elif evidence.distinct_ratio <= 0.05:
        reasons.append(f"low cardinality ({evidence.distinct_ratio:.2%})")

    date_parse = max(
        evidence.parse_results.get("date", 0.0),
        evidence.parse_results.get("datetime", 0.0),
    )
    if date_parse >= 0.5:
        reasons.append(f"date parse: {date_parse:.2%}")

    confidence = calculate_confidence(scores, pseudo_penalty)

    return RoleAssignment(
        role=best_role,
        score=best_score,
        confidence=confidence,
        all_scores=scores,
        secondary_role=secondary_role,
        secondary_score=secondary_score,
        reasons=reasons,
    )


def assign_roles(
    evidences: Sequence[ColumnEvidence],
    config: InferenceConfig | None = None,
    detected_shape: ShapeHypothesis | None = None,
) -> dict[str, RoleAssignment]:
    """Assign roles to all columns in a dataset.

    Performs a three-phase process:
    1. First pass: Detect if there's likely an indicator column
    2. Second pass: Assign roles with shape and indicator context
    3. Third pass: Global reconciliation to enforce shape constraints

    Args:
        evidences: Sequence of column evidences.
        config: Optional inference configuration.
        detected_shape: Detected dataset shape (for shape-conditional scoring).

    Returns:
        Dictionary mapping column names to role assignments.
    """
    if config is None:
        config = InferenceConfig()

    # === PHASE 1: Detect indicator column presence ===
    has_indicator = False
    for evidence in evidences:
        indicator_score = score_indicator_name_role(evidence, config, detected_shape)
        if indicator_score >= 0.5:
            has_indicator = True
            break

    # === PHASE 2: Assign roles with context ===
    assignments: dict[str, RoleAssignment] = {}
    for evidence in evidences:
        assignments[evidence.name] = resolve_role(
            evidence,
            config,
            has_indicator_column=has_indicator,
            detected_shape=detected_shape,
        )

    # === PHASE 3: Global reconciliation ===
    assignments = _reconcile_roles(assignments, detected_shape)

    return assignments


def _reconcile_roles(
    assignments: dict[str, RoleAssignment],
    detected_shape: ShapeHypothesis | None,
) -> dict[str, RoleAssignment]:
    """Apply global reconciliation to enforce shape-specific constraints.

    Constraints by shape:
    - LONG_INDICATORS: Expect exactly one VALUE column, one INDICATOR_NAME
    - WIDE_OBSERVATIONS: VALUE/INDICATOR_NAME roles suppressed
    - WIDE_TIME_COLUMNS: VALUE/INDICATOR_NAME roles suppressed

    Args:
        assignments: Initial role assignments.
        detected_shape: Detected dataset shape.

    Returns:
        Reconciled role assignments.
    """
    if detected_shape is None:
        return assignments

    # Count current role assignments
    role_counts: dict[Role, list[str]] = {role: [] for role in Role}
    for name, assignment in assignments.items():
        role_counts[assignment.role].append(name)

    # === LONG_INDICATORS constraints ===
    if detected_shape == ShapeHypothesis.LONG_INDICATORS:
        # Expect exactly one VALUE column
        value_cols = role_counts[Role.VALUE]
        if len(value_cols) > 1:
            # Multiple VALUE columns - keep the one with highest score
            best_value = max(value_cols, key=lambda n: assignments[n].score)
            for col in value_cols:
                if col != best_value:
                    # Demote to MEASURE
                    old_assignment = assignments[col]
                    old_assignment.reasons.append("demoted: only one VALUE allowed in LONG_INDICATORS")
                    assignments[col] = RoleAssignment(
                        role=Role.MEASURE,
                        score=old_assignment.all_scores.get(Role.MEASURE, 0.0),
                        confidence=old_assignment.confidence * 0.8,
                        all_scores=old_assignment.all_scores,
                        secondary_role=Role.VALUE,
                        secondary_score=old_assignment.score,
                        reasons=old_assignment.reasons,
                    )

        # Expect exactly one INDICATOR_NAME column
        indicator_cols = role_counts[Role.INDICATOR_NAME]
        if len(indicator_cols) > 1:
            # Multiple INDICATOR_NAME - keep highest, demote others to DIMENSION
            best_ind = max(indicator_cols, key=lambda n: assignments[n].score)
            for col in indicator_cols:
                if col != best_ind:
                    old_assignment = assignments[col]
                    old_assignment.reasons.append("demoted: only one INDICATOR_NAME allowed in LONG_INDICATORS")
                    assignments[col] = RoleAssignment(
                        role=Role.DIMENSION,
                        score=old_assignment.all_scores.get(Role.DIMENSION, 0.0),
                        confidence=old_assignment.confidence * 0.8,
                        all_scores=old_assignment.all_scores,
                        secondary_role=Role.INDICATOR_NAME,
                        secondary_score=old_assignment.score,
                        reasons=old_assignment.reasons,
                    )

    # === WIDE_OBSERVATIONS / WIDE_TIME_COLUMNS constraints ===
    elif detected_shape in (ShapeHypothesis.WIDE_OBSERVATIONS, ShapeHypothesis.WIDE_TIME_COLUMNS):
        # VALUE and INDICATOR_NAME roles should not exist in wide shapes
        # Demote VALUE to MEASURE, INDICATOR_NAME to DIMENSION
        for col in role_counts[Role.VALUE]:
            old_assignment = assignments[col]
            old_assignment.reasons.append(f"demoted: VALUE not expected in {detected_shape.value}")
            assignments[col] = RoleAssignment(
                role=Role.MEASURE,
                score=old_assignment.all_scores.get(Role.MEASURE, 0.0),
                confidence=old_assignment.confidence * 0.8,
                all_scores=old_assignment.all_scores,
                secondary_role=Role.VALUE,
                secondary_score=old_assignment.score,
                reasons=old_assignment.reasons,
            )

        for col in role_counts[Role.INDICATOR_NAME]:
            old_assignment = assignments[col]
            old_assignment.reasons.append(f"demoted: INDICATOR_NAME not expected in {detected_shape.value}")
            assignments[col] = RoleAssignment(
                role=Role.DIMENSION,
                score=old_assignment.all_scores.get(Role.DIMENSION, 0.0),
                confidence=old_assignment.confidence * 0.8,
                all_scores=old_assignment.all_scores,
                secondary_role=Role.INDICATOR_NAME,
                secondary_score=old_assignment.score,
                reasons=old_assignment.reasons,
            )

    return assignments


def update_evidence_with_roles(
    evidence: ColumnEvidence,
    config: InferenceConfig | None = None,
    has_indicator_column: bool = False,
    detected_shape: ShapeHypothesis | None = None,
) -> ColumnEvidence:
    """Update column evidence with computed role scores.

    Args:
        evidence: Column evidence to update.
        config: Optional inference configuration.
        has_indicator_column: Whether dataset has a likely indicator column.
        detected_shape: Detected dataset shape for shape-conditional scoring.

    Returns:
        Updated evidence with role_scores populated.
    """
    scores = compute_role_scores(evidence, config, has_indicator_column, detected_shape)
    evidence.role_scores = scores
    return evidence
