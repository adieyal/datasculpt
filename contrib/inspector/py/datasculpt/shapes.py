"""Shape detection for Datasculpt (browser bundle)."""


from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from datasculpt.types import (
    ColumnEvidence,
    HypothesisScore,
    InferenceConfig,
    PrimitiveType,
    Role,
    ShapeHypothesis,
    StructuralType,
)

if TYPE_CHECKING:
    pass


@dataclass
class ShapeResult:
    """Result of shape detection."""

    selected: ShapeHypothesis
    ranked_hypotheses: list[HypothesisScore]
    is_ambiguous: bool = False
    ambiguity_details: list[str] = field(default_factory=list)


def score_long_observations(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score the LONG_OBSERVATIONS hypothesis.

    Long observations have:
    - Multiple dimension columns (low cardinality)
    - Multiple measure columns (numeric, high cardinality)
    - No indicator/value pairs
    - No date-like column headers
    """
    score = 0.0
    reasons: list[str] = []

    dimension_count = 0
    measure_count = 0
    time_count = 0
    date_header_count = 0

    for evidence in evidences:
        # Count date-like headers
        if evidence.header_date_like:
            date_header_count += 1

        # Check role scores
        dim_score = evidence.role_scores.get(Role.DIMENSION, 0.0)
        meas_score = evidence.role_scores.get(Role.MEASURE, 0.0)
        time_score = evidence.role_scores.get(Role.TIME, 0.0)

        if dim_score > 0.4:
            dimension_count += 1
        if meas_score > 0.4:
            measure_count += 1
        if time_score > 0.4:
            time_count += 1

    # Expect multiple dimensions and measures
    if dimension_count >= 1:
        score += 0.25
        reasons.append(f"Found {dimension_count} dimension column(s)")

    if measure_count >= 1:
        score += 0.25
        reasons.append(f"Found {measure_count} measure column(s)")

    # Time column is common in long observations
    if time_count >= 1:
        score += 0.15
        reasons.append(f"Found {time_count} time column(s)")

    # No date-like headers (that would suggest wide format)
    if date_header_count == 0:
        score += 0.2
        reasons.append("No date-like column headers")
    else:
        score -= 0.3
        reasons.append(f"Found {date_header_count} date-like headers (suggests wide format)")

    # Bonus if we have the classic dimension + measure pattern
    if dimension_count >= 2 and measure_count >= 1:
        score += 0.15
        reasons.append("Classic dimension + measure pattern")

    return HypothesisScore(
        hypothesis=ShapeHypothesis.LONG_OBSERVATIONS,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
    )


def score_long_indicators(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score the LONG_INDICATORS hypothesis.

    Long indicators have:
    - One indicator_name column (low cardinality strings like metric names)
    - One value column (numeric)
    - Dimension columns for grouping
    """
    score = 0.0
    reasons: list[str] = []

    indicator_candidates = []
    value_candidates = []

    for evidence in evidences:
        ind_score = evidence.role_scores.get(Role.INDICATOR_NAME, 0.0)
        val_score = evidence.role_scores.get(Role.VALUE, 0.0)

        if ind_score > 0.3:
            indicator_candidates.append(evidence.name)
        if val_score > 0.3:
            value_candidates.append(evidence.name)

    # Must have indicator/value pair
    if indicator_candidates and value_candidates:
        score += 0.5
        reasons.append(f"Found indicator column(s): {', '.join(indicator_candidates)}")
        reasons.append(f"Found value column(s): {', '.join(value_candidates)}")

        # Single indicator + single value is ideal
        if len(indicator_candidates) == 1 and len(value_candidates) == 1:
            score += 0.25
            reasons.append("Single indicator-value pair (ideal)")

    else:
        if not indicator_candidates:
            reasons.append("No indicator column detected")
        if not value_candidates:
            reasons.append("No value column detected")

    # Check for low cardinality in indicator column
    for evidence in evidences:
        if evidence.name in indicator_candidates:
            if evidence.distinct_ratio < 0.1 and evidence.unique_count > 1:
                score += 0.15
                reasons.append(f"Indicator '{evidence.name}' has low cardinality")

    return HypothesisScore(
        hypothesis=ShapeHypothesis.LONG_INDICATORS,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
    )


def score_wide_observations(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score the WIDE_OBSERVATIONS hypothesis.

    Wide observations have:
    - Multiple measure columns (numeric)
    - Few dimension/key columns
    - No date-like column headers
    - No indicator/value pattern
    """
    score = 0.0
    reasons: list[str] = []

    measure_count = 0
    key_dim_count = 0
    date_header_count = 0

    for evidence in evidences:
        if evidence.header_date_like:
            date_header_count += 1

        meas_score = evidence.role_scores.get(Role.MEASURE, 0.0)
        key_score = evidence.role_scores.get(Role.KEY, 0.0)
        dim_score = evidence.role_scores.get(Role.DIMENSION, 0.0)

        if meas_score > 0.4:
            measure_count += 1
        if key_score > 0.4 or dim_score > 0.4:
            key_dim_count += 1

    # Multiple measures
    if measure_count >= 3:
        score += 0.35
        reasons.append(f"Found {measure_count} measure columns")
    elif measure_count >= 2:
        score += 0.2
        reasons.append(f"Found {measure_count} measure columns")

    # Few key/dimension columns
    if key_dim_count <= 3:
        score += 0.2
        reasons.append(f"Found {key_dim_count} key/dimension columns")

    # No date-like headers
    if date_header_count == 0:
        score += 0.25
        reasons.append("No date-like column headers")
    else:
        score -= 0.2
        reasons.append(f"Found {date_header_count} date-like headers")

    return HypothesisScore(
        hypothesis=ShapeHypothesis.WIDE_OBSERVATIONS,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
    )


def score_wide_time_columns(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score the WIDE_TIME_COLUMNS hypothesis.

    Wide time columns have:
    - Multiple columns with date-like headers (years, months, etc.)
    - Key/dimension columns for identification
    """
    score = 0.0
    reasons: list[str] = []

    date_header_count = 0
    date_headers: list[str] = []

    for evidence in evidences:
        if evidence.header_date_like:
            date_header_count += 1
            date_headers.append(evidence.name)

    # Need multiple date-like headers
    if date_header_count >= config.min_time_columns_for_wide:
        score += 0.6
        reasons.append(f"Found {date_header_count} date-like column headers: {', '.join(date_headers[:5])}")
    elif date_header_count >= 2:
        score += 0.3
        reasons.append(f"Found {date_header_count} date-like column headers")
    elif date_header_count == 1:
        reasons.append(f"Only 1 date-like header found")
    else:
        reasons.append("No date-like column headers found")

    # Should have some non-date columns for identification
    non_date_count = len(evidences) - date_header_count
    if non_date_count >= 1 and date_header_count > non_date_count:
        score += 0.2
        reasons.append(f"Good ratio of date columns ({date_header_count}) to identifier columns ({non_date_count})")

    return HypothesisScore(
        hypothesis=ShapeHypothesis.WIDE_TIME_COLUMNS,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
    )


def score_series_column(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score the SERIES_COLUMN hypothesis.

    Series column format has:
    - One or more columns containing JSON arrays (time series data)
    - Key/dimension columns for identification
    """
    score = 0.0
    reasons: list[str] = []

    series_candidates: list[str] = []

    for evidence in evidences:
        # Check for array structural type
        if evidence.structural_type == StructuralType.ARRAY:
            series_candidates.append(evidence.name)
            continue

        # Check for JSON array in parse results
        if evidence.parse_results.json_array_rate > 0.5:
            series_candidates.append(evidence.name)
            continue

        # Check series role score
        series_score = evidence.role_scores.get(Role.SERIES, 0.0)
        if series_score > 0.4:
            series_candidates.append(evidence.name)

    if series_candidates:
        score += 0.6
        reasons.append(f"Found series column(s): {', '.join(series_candidates)}")

        # Check array profile for consistency
        for evidence in evidences:
            if evidence.name in series_candidates and evidence.array_profile:
                if evidence.array_profile.consistent_length:
                    score += 0.15
                    reasons.append(f"Column '{evidence.name}' has consistent array lengths")
                if evidence.array_profile.avg_length > 5:
                    score += 0.1
                    reasons.append(f"Column '{evidence.name}' has substantial arrays (avg length: {evidence.array_profile.avg_length:.1f})")
    else:
        reasons.append("No series/array columns detected")

    return HypothesisScore(
        hypothesis=ShapeHypothesis.SERIES_COLUMN,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
    )


def detect_shape(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> ShapeResult:
    """Detect the dataset shape from column evidence.

    Args:
        evidences: List of column evidence.
        config: Inference configuration.

    Returns:
        ShapeResult with selected shape and all hypothesis scores.
    """
    # Score all hypotheses
    hypotheses = [
        score_long_observations(evidences, config),
        score_long_indicators(evidences, config),
        score_wide_observations(evidences, config),
        score_wide_time_columns(evidences, config),
        score_series_column(evidences, config),
    ]

    # Sort by score descending
    hypotheses.sort(key=lambda h: h.score, reverse=True)

    # Select winner
    selected = hypotheses[0].hypothesis

    # Check for ambiguity
    is_ambiguous = False
    ambiguity_details: list[str] = []

    if len(hypotheses) >= 2:
        gap = hypotheses[0].score - hypotheses[1].score
        if gap < config.hypothesis_confidence_gap:
            is_ambiguous = True
            ambiguity_details.append(
                f"Close scores: {hypotheses[0].hypothesis.value} ({hypotheses[0].score:.2f}) "
                f"vs {hypotheses[1].hypothesis.value} ({hypotheses[1].score:.2f})"
            )

    if hypotheses[0].score < 0.5:
        is_ambiguous = True
        ambiguity_details.append(f"Low confidence: {hypotheses[0].score:.2f}")

    return ShapeResult(
        selected=selected,
        ranked_hypotheses=hypotheses,
        is_ambiguous=is_ambiguous,
        ambiguity_details=ambiguity_details,
    )
