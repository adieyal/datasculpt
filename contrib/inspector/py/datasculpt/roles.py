"""Role scoring and assignment for Datasculpt (browser bundle)."""


import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from datasculpt.types import (
    ColumnEvidence,
    InferenceConfig,
    PrimitiveType,
    Role,
    StructuralType,
)

if TYPE_CHECKING:
    pass


@dataclass
class RoleAssignment:
    """Role assignment result for a column."""

    role: Role
    confidence: float
    reasons: list[str]


# Name patterns for role detection
KEY_PATTERNS = [
    re.compile(r"_id$", re.IGNORECASE),
    re.compile(r"^id$", re.IGNORECASE),
    re.compile(r"_key$", re.IGNORECASE),
    re.compile(r"_code$", re.IGNORECASE),
    re.compile(r"^code$", re.IGNORECASE),
]

DIMENSION_PATTERNS = [
    re.compile(r"_name$", re.IGNORECASE),
    re.compile(r"_category$", re.IGNORECASE),
    re.compile(r"_type$", re.IGNORECASE),
    re.compile(r"^category$", re.IGNORECASE),
    re.compile(r"^type$", re.IGNORECASE),
    re.compile(r"^status$", re.IGNORECASE),
    re.compile(r"^country$", re.IGNORECASE),
    re.compile(r"^region$", re.IGNORECASE),
    re.compile(r"^state$", re.IGNORECASE),
]

TIME_PATTERNS = [
    re.compile(r"_date$", re.IGNORECASE),
    re.compile(r"^date$", re.IGNORECASE),
    re.compile(r"_time$", re.IGNORECASE),
    re.compile(r"^time$", re.IGNORECASE),
    re.compile(r"_year$", re.IGNORECASE),
    re.compile(r"^year$", re.IGNORECASE),
    re.compile(r"_month$", re.IGNORECASE),
    re.compile(r"^month$", re.IGNORECASE),
    re.compile(r"timestamp", re.IGNORECASE),
]

INDICATOR_PATTERNS = [
    re.compile(r"^indicator$", re.IGNORECASE),
    re.compile(r"_indicator$", re.IGNORECASE),
    re.compile(r"^variable$", re.IGNORECASE),
    re.compile(r"^metric$", re.IGNORECASE),
    re.compile(r"^measure$", re.IGNORECASE),
    re.compile(r"_name$", re.IGNORECASE),
]

VALUE_PATTERNS = [
    re.compile(r"^value$", re.IGNORECASE),
    re.compile(r"_value$", re.IGNORECASE),
    re.compile(r"^amount$", re.IGNORECASE),
    re.compile(r"^observation$", re.IGNORECASE),
]

SERIES_PATTERNS = [
    re.compile(r"^series$", re.IGNORECASE),
    re.compile(r"_series$", re.IGNORECASE),
    re.compile(r"^data$", re.IGNORECASE),
    re.compile(r"_data$", re.IGNORECASE),
    re.compile(r"^values$", re.IGNORECASE),
]


def _matches_patterns(name: str, patterns: list[re.Pattern]) -> bool:
    """Check if name matches any of the patterns."""
    return any(p.search(name) for p in patterns)


def score_key_role(evidence: ColumnEvidence, config: InferenceConfig) -> float:
    """Score a column for KEY role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0
    name = evidence.name

    # Name-based scoring
    if _matches_patterns(name, KEY_PATTERNS):
        score += 0.4

    # High cardinality is key-like
    if evidence.distinct_ratio >= config.key_cardinality_threshold:
        score += 0.3

    # String or integer types are common for keys
    if evidence.primitive_type in (PrimitiveType.STRING, PrimitiveType.INTEGER):
        score += 0.15

    # Low null rate
    if evidence.null_rate < config.null_rate_threshold:
        score += 0.15

    return min(1.0, score)


def score_dimension_role(evidence: ColumnEvidence, config: InferenceConfig) -> float:
    """Score a column for DIMENSION role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0
    name = evidence.name

    # Name-based scoring
    if _matches_patterns(name, DIMENSION_PATTERNS):
        score += 0.35

    # Low cardinality is dimension-like
    if evidence.distinct_ratio <= config.dimension_cardinality_max:
        score += 0.35

    # String type is common for dimensions
    if evidence.primitive_type == PrimitiveType.STRING:
        score += 0.15

    # Boolean can be dimension
    if evidence.primitive_type == PrimitiveType.BOOLEAN:
        score += 0.25

    # Low null rate
    if evidence.null_rate < config.null_rate_threshold:
        score += 0.15

    return min(1.0, score)


def score_measure_role(evidence: ColumnEvidence, config: InferenceConfig) -> float:
    """Score a column for MEASURE role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0

    # Numeric types are measure-like
    if evidence.primitive_type in (PrimitiveType.NUMBER, PrimitiveType.INTEGER):
        score += 0.4

    # Non-negative values are measure-like
    if evidence.value_profile.non_negative_ratio > 0.9:
        score += 0.2

    # High cardinality (continuous values)
    if evidence.distinct_ratio > config.dimension_cardinality_max:
        score += 0.2

    # Presence of decimal values
    if evidence.primitive_type == PrimitiveType.NUMBER:
        if evidence.value_profile.integer_ratio < 0.5:
            score += 0.1

    # Common measure name patterns
    measure_patterns = [
        r"amount", r"total", r"sum", r"count", r"rate", r"ratio",
        r"percent", r"price", r"cost", r"revenue", r"sales", r"quantity",
    ]
    name_lower = evidence.name.lower()
    if any(p in name_lower for p in measure_patterns):
        score += 0.3

    return min(1.0, score)


def score_time_role(evidence: ColumnEvidence, config: InferenceConfig) -> float:
    """Score a column for TIME role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0

    # Date/datetime types
    if evidence.primitive_type in (PrimitiveType.DATE, PrimitiveType.DATETIME):
        score += 0.5

    # Name patterns
    if _matches_patterns(evidence.name, TIME_PATTERNS):
        score += 0.3

    # Parse results show date-like
    if evidence.parse_results.date_parse_rate > 0.8:
        score += 0.3

    # Header looks like a date (for wide format detection)
    if evidence.header_date_like:
        score += 0.4

    return min(1.0, score)


def score_indicator_name_role(
    evidence: ColumnEvidence,
    config: InferenceConfig,
    has_value_column: bool = False,
) -> float:
    """Score a column for INDICATOR_NAME role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.
        has_value_column: Whether a VALUE column candidate exists.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0

    # Name patterns
    if _matches_patterns(evidence.name, INDICATOR_PATTERNS):
        score += 0.4

    # String type
    if evidence.primitive_type == PrimitiveType.STRING:
        score += 0.2

    # Low cardinality (repeating indicator names)
    if evidence.distinct_ratio < 0.1 and evidence.unique_count > 1:
        score += 0.2

    # Paired with value column
    if has_value_column:
        score += 0.2

    return min(1.0, score)


def score_value_role(
    evidence: ColumnEvidence,
    config: InferenceConfig,
    has_indicator_column: bool = False,
) -> float:
    """Score a column for VALUE role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.
        has_indicator_column: Whether an INDICATOR_NAME column candidate exists.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0

    # Name patterns
    if _matches_patterns(evidence.name, VALUE_PATTERNS):
        score += 0.4

    # Numeric types
    if evidence.primitive_type in (PrimitiveType.NUMBER, PrimitiveType.INTEGER):
        score += 0.2

    # Paired with indicator column
    if has_indicator_column:
        score += 0.3

    return min(1.0, score)


def score_series_role(evidence: ColumnEvidence, config: InferenceConfig) -> float:
    """Score a column for SERIES role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0

    # Array structural type
    if evidence.structural_type == StructuralType.ARRAY:
        score += 0.5

    # Name patterns
    if _matches_patterns(evidence.name, SERIES_PATTERNS):
        score += 0.3

    # JSON array parse results
    if evidence.parse_results.json_array_rate > 0.5:
        score += 0.3

    # Has array profile
    if evidence.array_profile is not None:
        score += 0.2

    return min(1.0, score)


def score_metadata_role(evidence: ColumnEvidence, config: InferenceConfig) -> float:
    """Score a column for METADATA role.

    Args:
        evidence: Column evidence.
        config: Inference configuration.

    Returns:
        Score from 0.0 to 1.0.
    """
    score = 0.0

    # High null rate suggests metadata
    if evidence.null_rate > 0.5:
        score += 0.3

    # Common metadata patterns
    metadata_patterns = [
        r"note", r"comment", r"description", r"remark", r"source",
        r"footnote", r"created", r"updated", r"modified", r"_at$",
    ]
    name_lower = evidence.name.lower()
    if any(re.search(p, name_lower) for p in metadata_patterns):
        score += 0.4

    # String type with high cardinality (free text)
    if evidence.primitive_type == PrimitiveType.STRING and evidence.distinct_ratio > 0.8:
        score += 0.2

    return min(1.0, score)


def resolve_role(role_scores: dict[Role, float]) -> tuple[Role, float]:
    """Resolve the best role from scores.

    Args:
        role_scores: Dictionary of role to score.

    Returns:
        Tuple of (best role, confidence).
    """
    if not role_scores:
        return Role.METADATA, 0.0

    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    best_role, best_score = sorted_roles[0]

    # Calculate confidence based on margin over second best
    if len(sorted_roles) > 1:
        second_score = sorted_roles[1][1]
        margin = best_score - second_score
        confidence = min(1.0, best_score * 0.7 + margin * 0.3)
    else:
        confidence = best_score

    return best_role, confidence


def update_evidence_with_roles(
    evidence: ColumnEvidence,
    config: InferenceConfig,
    has_indicator: bool = False,
) -> None:
    """Update column evidence with role scores.

    Args:
        evidence: Column evidence to update.
        config: Inference configuration.
        has_indicator: Whether an indicator column was detected.
    """
    evidence.role_scores = {
        Role.KEY: score_key_role(evidence, config),
        Role.DIMENSION: score_dimension_role(evidence, config),
        Role.MEASURE: score_measure_role(evidence, config),
        Role.TIME: score_time_role(evidence, config),
        Role.INDICATOR_NAME: score_indicator_name_role(evidence, config, has_indicator),
        Role.VALUE: score_value_role(evidence, config, has_indicator),
        Role.SERIES: score_series_role(evidence, config),
        Role.METADATA: score_metadata_role(evidence, config),
    }


def assign_roles(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> dict[str, RoleAssignment]:
    """Assign roles to all columns.

    Args:
        evidences: List of column evidence.
        config: Inference configuration.

    Returns:
        Dictionary mapping column names to RoleAssignment.
    """
    # First pass: check for indicator/value pairs
    has_indicator = False
    has_value = False

    for evidence in evidences:
        if _matches_patterns(evidence.name, INDICATOR_PATTERNS):
            has_indicator = True
        if _matches_patterns(evidence.name, VALUE_PATTERNS):
            has_value = True

    # Score all columns
    assignments: dict[str, RoleAssignment] = {}
    for evidence in evidences:
        update_evidence_with_roles(evidence, config, has_indicator and has_value)
        role, confidence = resolve_role(evidence.role_scores)

        # Build reasons
        reasons = []
        if evidence.role_scores.get(role, 0) > 0:
            reasons.append(f"Highest scoring role: {role.value}")

        assignments[evidence.name] = RoleAssignment(
            role=role,
            confidence=confidence,
            reasons=reasons,
        )

    return assignments
