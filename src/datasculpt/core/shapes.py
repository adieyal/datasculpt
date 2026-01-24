"""Shape hypothesis detection for Datasculpt.

This module scores datasets against different shape hypotheses and selects
the best matching shape with confidence assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from datasculpt.core.types import (
    ColumnEvidence,
    HypothesisScore,
    InferenceConfig,
    Role,
    ShapeHypothesis,
    StructuralType,
)


@dataclass
class ShapeResult:
    """Result of shape hypothesis detection."""

    selected: ShapeHypothesis
    confidence: float
    ranked_hypotheses: list[HypothesisScore]
    is_ambiguous: bool
    explanation: str
    ambiguity_details: list[str] = field(default_factory=list)


def score_long_observations(
    columns: Sequence[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score for standard tidy data pattern (dimensions + measures as rows).

    Long observations is the classic tidy data format where:
    - Each row is one observation
    - Dimensions identify the observation (who, where, when)
    - Measures are numeric values for that observation

    Args:
        columns: Column evidence from profiling.
        config: Inference configuration.

    Returns:
        HypothesisScore with score and reasons.
    """
    score = 0.0
    reasons: list[str] = []

    dimension_cols = _columns_with_role(columns, Role.DIMENSION)
    measure_cols = _columns_with_role(columns, Role.MEASURE)
    time_cols = _columns_with_role(columns, Role.TIME)
    key_cols = _columns_with_role(columns, Role.KEY)

    # Positive: has dimensions
    if dimension_cols:
        score += 0.2
        reasons.append(f"Found {len(dimension_cols)} dimension column(s): {_names(dimension_cols)}")

    # Positive: has measures
    if measure_cols:
        score += 0.25
        reasons.append(f"Found {len(measure_cols)} measure column(s): {_names(measure_cols)}")

    # Positive: has time column (not in headers)
    if time_cols:
        score += 0.15
        reasons.append(f"Found time column(s) as rows: {_names(time_cols)}")

    # Positive: has key columns
    if key_cols:
        score += 0.1
        reasons.append(f"Found key column(s): {_names(key_cols)}")

    # Negative: indicator_name/value pattern suggests long_indicators
    indicator_cols = _columns_with_role(columns, Role.INDICATOR_NAME)
    value_cols = _columns_with_role(columns, Role.VALUE)
    if indicator_cols and value_cols:
        score -= 0.3
        reasons.append("Has indicator_name/value pattern (suggests long_indicators)")

    # Negative: series columns suggest series_column shape
    series_cols = _columns_with_role(columns, Role.SERIES)
    if series_cols:
        score -= 0.25
        reasons.append(f"Has series column(s): {_names(series_cols)} (suggests series_column)")

    # Ideal case: dimensions + measures without indicator pattern
    if dimension_cols and measure_cols and not indicator_cols:
        score += 0.15
        reasons.append("Classic tidy data pattern: dimensions + measures")

    # Clamp score
    score = max(0.0, min(1.0, score))

    return HypothesisScore(
        hypothesis=ShapeHypothesis.LONG_OBSERVATIONS,
        score=score,
        reasons=reasons,
    )


def score_long_indicators(
    columns: Sequence[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score for indicator/value pair pattern (unpivoted data).

    Long indicators is a common pattern where:
    - One column contains indicator names (e.g., "GDP", "Population")
    - Another column contains the values
    - Additional columns provide context (country, year)

    Args:
        columns: Column evidence from profiling.
        config: Inference configuration.

    Returns:
        HypothesisScore with score and reasons.
    """
    score = 0.0
    reasons: list[str] = []

    indicator_cols = _columns_with_role(columns, Role.INDICATOR_NAME)
    value_cols = _columns_with_role(columns, Role.VALUE)
    dimension_cols = _columns_with_role(columns, Role.DIMENSION)
    time_cols = _columns_with_role(columns, Role.TIME)

    # Core pattern: indicator_name + value columns
    if indicator_cols and value_cols:
        score += 0.5
        reasons.append(
            f"Found indicator/value pattern: {_names(indicator_cols)} -> {_names(value_cols)}"
        )

    # Bonus: has dimensions for context
    if dimension_cols:
        score += 0.15
        reasons.append(f"Has dimension columns for context: {_names(dimension_cols)}")

    # Bonus: has time column
    if time_cols:
        score += 0.1
        reasons.append(f"Has time column: {_names(time_cols)}")

    # Penalty: multiple measure columns (suggests observations, not indicators)
    measure_cols = _columns_with_role(columns, Role.MEASURE)
    if len(measure_cols) > 1 and not value_cols:
        score -= 0.2
        reasons.append(
            f"Multiple measure columns without value column suggests observations"
        )

    # Penalty: no indicator column found
    if not indicator_cols:
        score -= 0.3
        reasons.append("No indicator_name column detected")

    # Penalty: no value column found
    if not value_cols:
        score -= 0.3
        reasons.append("No value column detected")

    # Clamp score
    score = max(0.0, min(1.0, score))

    return HypothesisScore(
        hypothesis=ShapeHypothesis.LONG_INDICATORS,
        score=score,
        reasons=reasons,
    )


def score_wide_observations(
    columns: Sequence[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score for wide data with multiple measures as columns.

    Wide observations is a denormalized format where:
    - Each row represents an entity
    - Multiple measure columns exist (not time-based)
    - Example: person_id, height, weight, age

    Args:
        columns: Column evidence from profiling.
        config: Inference configuration.

    Returns:
        HypothesisScore with score and reasons.
    """
    score = 0.0
    reasons: list[str] = []

    measure_cols = _columns_with_role(columns, Role.MEASURE)
    dimension_cols = _columns_with_role(columns, Role.DIMENSION)
    key_cols = _columns_with_role(columns, Role.KEY)
    time_cols = _columns_with_role(columns, Role.TIME)

    # Core pattern: multiple measure columns
    if len(measure_cols) >= 2:
        score += 0.35
        reasons.append(f"Found {len(measure_cols)} measure columns: {_names(measure_cols)}")

    # Bonus: has key or dimension columns
    if key_cols or dimension_cols:
        score += 0.15
        id_cols = key_cols or dimension_cols
        reasons.append(f"Has identifier columns: {_names(id_cols)}")

    # Check if measures look like time periods (suggests wide_time_columns instead)
    time_like_measures = [
        col for col in measure_cols
        if _looks_like_time_header(col.name)
    ]

    if time_like_measures and len(time_like_measures) >= config.min_time_columns_for_wide:
        score -= 0.3
        reasons.append(
            f"Measure columns look like time periods: {_names(time_like_measures)} "
            "(suggests wide_time_columns)"
        )

    # Penalty: indicator/value pattern present
    indicator_cols = _columns_with_role(columns, Role.INDICATOR_NAME)
    value_cols = _columns_with_role(columns, Role.VALUE)
    if indicator_cols and value_cols:
        score -= 0.25
        reasons.append("Has indicator/value pattern (suggests long_indicators)")

    # Penalty: series columns present
    series_cols = _columns_with_role(columns, Role.SERIES)
    if series_cols:
        score -= 0.2
        reasons.append(f"Has series column(s): {_names(series_cols)} (suggests series_column)")

    # Ideal case: key + multiple non-time measures
    non_time_measures = [col for col in measure_cols if not _looks_like_time_header(col.name)]
    if (key_cols or dimension_cols) and len(non_time_measures) >= 2:
        score += 0.2
        reasons.append("Classic wide format: identifiers + multiple measures")

    # Clamp score
    score = max(0.0, min(1.0, score))

    return HypothesisScore(
        hypothesis=ShapeHypothesis.WIDE_OBSERVATIONS,
        score=score,
        reasons=reasons,
    )


def score_wide_time_columns(
    columns: Sequence[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score for time periods as column headers.

    Wide time columns is common in economic/demographic data where:
    - Column headers are years, months, or dates
    - Each row is an entity (country, indicator)
    - Values are measures for that time period

    Args:
        columns: Column evidence from profiling.
        config: Inference configuration.

    Returns:
        HypothesisScore with score and reasons.
    """
    score = 0.0
    reasons: list[str] = []

    measure_cols = _columns_with_role(columns, Role.MEASURE)
    dimension_cols = _columns_with_role(columns, Role.DIMENSION)
    indicator_cols = _columns_with_role(columns, Role.INDICATOR_NAME)

    # Count columns that look like time periods
    time_like_columns = [col for col in columns if _looks_like_time_header(col.name)]

    # Core pattern: multiple time-like column headers
    if len(time_like_columns) >= config.min_time_columns_for_wide:
        score += 0.5
        reasons.append(
            f"Found {len(time_like_columns)} time-like column headers: "
            f"{_names(time_like_columns[:5])}{'...' if len(time_like_columns) > 5 else ''}"
        )

    # Bonus: has entity identifiers
    if dimension_cols or indicator_cols:
        id_cols = dimension_cols or indicator_cols
        score += 0.15
        reasons.append(f"Has entity identifier columns: {_names(id_cols)}")

    # Bonus: time columns are contiguous in headers (suggests intentional layout)
    if len(time_like_columns) >= 3:
        all_names = [col.name for col in columns]
        time_indices = [all_names.index(col.name) for col in time_like_columns]
        if _is_mostly_contiguous(time_indices):
            score += 0.1
            reasons.append("Time columns are contiguous (typical for time-wide format)")

    # Penalty: too few time-like columns
    if len(time_like_columns) < config.min_time_columns_for_wide:
        score -= 0.3
        reasons.append(
            f"Only {len(time_like_columns)} time-like columns "
            f"(need at least {config.min_time_columns_for_wide})"
        )

    # Penalty: series columns present
    series_cols = _columns_with_role(columns, Role.SERIES)
    if series_cols:
        score -= 0.2
        reasons.append(f"Has series column(s): {_names(series_cols)} (suggests series_column)")

    # Clamp score
    score = max(0.0, min(1.0, score))

    return HypothesisScore(
        hypothesis=ShapeHypothesis.WIDE_TIME_COLUMNS,
        score=score,
        reasons=reasons,
    )


def score_series_column(
    columns: Sequence[ColumnEvidence],
    config: InferenceConfig,
) -> HypothesisScore:
    """Score for JSON arrays representing time series in a column.

    Series column format stores entire time series as JSON arrays:
    - One column contains JSON arrays of values
    - Entity columns identify what the series represents
    - May have metadata columns for time range, frequency, etc.

    Args:
        columns: Column evidence from profiling.
        config: Inference configuration.

    Returns:
        HypothesisScore with score and reasons.
    """
    score = 0.0
    reasons: list[str] = []

    series_cols = _columns_with_role(columns, Role.SERIES)
    dimension_cols = _columns_with_role(columns, Role.DIMENSION)
    indicator_cols = _columns_with_role(columns, Role.INDICATOR_NAME)

    # Core pattern: series column with array structural type
    array_cols = [col for col in columns if col.structural_type == StructuralType.ARRAY]

    if series_cols:
        score += 0.5
        reasons.append(f"Found series column(s): {_names(series_cols)}")

    # Also check for array-typed columns even without explicit role
    if array_cols and not series_cols:
        score += 0.3
        reasons.append(f"Found array-typed column(s): {_names(array_cols)}")

    # Bonus: has entity identifiers
    if dimension_cols or indicator_cols:
        id_cols = dimension_cols or indicator_cols
        score += 0.15
        reasons.append(f"Has entity identifier columns: {_names(id_cols)}")

    # Bonus: has metadata columns (often present with series data)
    metadata_cols = _columns_with_role(columns, Role.METADATA)
    if metadata_cols:
        score += 0.1
        reasons.append(f"Has metadata columns: {_names(metadata_cols)}")

    # Penalty: no array or series columns
    if not series_cols and not array_cols:
        score -= 0.4
        reasons.append("No series or array columns detected")

    # Penalty: many time-like column headers (suggests wide_time_columns)
    time_like_columns = [col for col in columns if _looks_like_time_header(col.name)]
    if len(time_like_columns) >= config.min_time_columns_for_wide:
        score -= 0.2
        reasons.append(
            f"Has {len(time_like_columns)} time-like columns (suggests wide_time_columns)"
        )

    # Clamp score
    score = max(0.0, min(1.0, score))

    return HypothesisScore(
        hypothesis=ShapeHypothesis.SERIES_COLUMN,
        score=score,
        reasons=reasons,
    )


def compare_hypotheses(
    columns: Sequence[ColumnEvidence],
    config: InferenceConfig | None = None,
) -> list[HypothesisScore]:
    """Score all hypotheses and return ranked list.

    Args:
        columns: Column evidence from profiling.
        config: Inference configuration (uses defaults if None).

    Returns:
        List of HypothesisScore sorted by score descending.
    """
    if config is None:
        config = InferenceConfig()

    scores = [
        score_long_observations(columns, config),
        score_long_indicators(columns, config),
        score_wide_observations(columns, config),
        score_wide_time_columns(columns, config),
        score_series_column(columns, config),
    ]

    # Sort by score descending
    scores.sort(key=lambda s: s.score, reverse=True)

    return scores


def detect_shape(
    columns: Sequence[ColumnEvidence],
    config: InferenceConfig | None = None,
) -> ShapeResult:
    """Detect the most likely shape hypothesis for a dataset.

    This is the main entry point for shape detection. It scores all hypotheses,
    selects the best one, and provides confidence and ambiguity assessment.

    Args:
        columns: Column evidence from profiling.
        config: Inference configuration (uses defaults if None).

    Returns:
        ShapeResult with selected hypothesis, confidence, and explanation.
    """
    if config is None:
        config = InferenceConfig()

    ranked = compare_hypotheses(columns, config)

    if not ranked:
        # Fallback to long_observations if no hypotheses could be scored
        return ShapeResult(
            selected=ShapeHypothesis.LONG_OBSERVATIONS,
            confidence=0.0,
            ranked_hypotheses=[],
            is_ambiguous=True,
            explanation="No hypotheses could be evaluated. Defaulting to long_observations.",
            ambiguity_details=["No column evidence available for hypothesis scoring."],
        )

    best = ranked[0]
    confidence = best.score

    # Check for ambiguity: top 2 scores within threshold
    is_ambiguous = False
    ambiguity_details: list[str] = []

    if len(ranked) >= 2:
        second = ranked[1]
        gap = best.score - second.score
        if gap < config.hypothesis_confidence_gap:
            is_ambiguous = True
            ambiguity_details.append(
                f"Top hypotheses are close: {best.hypothesis.value} ({best.score:.2f}) "
                f"vs {second.hypothesis.value} ({second.score:.2f}), gap={gap:.2f}"
            )

    # Low confidence is also ambiguous
    if confidence < 0.3:
        is_ambiguous = True
        ambiguity_details.append(f"Low confidence score: {confidence:.2f}")

    explanation = generate_explanation(columns, best, ranked)

    return ShapeResult(
        selected=best.hypothesis,
        confidence=confidence,
        ranked_hypotheses=ranked,
        is_ambiguous=is_ambiguous,
        explanation=explanation,
        ambiguity_details=ambiguity_details,
    )


def generate_explanation(
    columns: Sequence[ColumnEvidence],
    selected: HypothesisScore,
    all_scores: list[HypothesisScore],
) -> str:
    """Generate human-readable explanation for shape selection.

    Args:
        columns: Column evidence from profiling.
        selected: The selected hypothesis score.
        all_scores: All hypothesis scores for comparison.

    Returns:
        Human-readable explanation text.
    """
    lines: list[str] = []

    # Opening statement
    lines.append(
        f"Selected shape: {_format_hypothesis_name(selected.hypothesis)} "
        f"(score: {selected.score:.2f})"
    )
    lines.append("")

    # Why this shape
    lines.append("Evidence for this shape:")
    for reason in selected.reasons:
        if not reason.startswith("-"):
            lines.append(f"  - {reason}")

    # Column summary
    lines.append("")
    lines.append("Column analysis:")
    for col in columns:
        top_roles = _top_roles(col, limit=2)
        if top_roles:
            role_str = ", ".join(f"{r.value}({s:.2f})" for r, s in top_roles)
            lines.append(f"  - {col.name}: {role_str}")
        else:
            lines.append(f"  - {col.name}: {col.primitive_type.value}")

    # Comparison with alternatives
    if len(all_scores) > 1:
        lines.append("")
        lines.append("Alternative hypotheses:")
        for score in all_scores[1:]:
            lines.append(f"  - {_format_hypothesis_name(score.hypothesis)}: {score.score:.2f}")

    return "\n".join(lines)


# Helper functions


def _columns_with_role(
    columns: Sequence[ColumnEvidence],
    role: Role,
    threshold: float = 0.3,
) -> list[ColumnEvidence]:
    """Filter columns that have a role score above threshold."""
    return [
        col for col in columns
        if col.role_scores.get(role, 0.0) >= threshold
    ]


def _names(columns: Sequence[ColumnEvidence]) -> str:
    """Format column names as comma-separated string."""
    if not columns:
        return "(none)"
    names = [col.name for col in columns]
    if len(names) <= 3:
        return ", ".join(names)
    return f"{', '.join(names[:3])}, ... ({len(names)} total)"


def _looks_like_time_header(name: str) -> bool:
    """Check if a column name looks like a time period.

    Detects patterns like:
    - Years: 2020, 2021, FY2020
    - Months: Jan, January, 2020-01
    - Quarters: Q1, Q2, 2020Q1
    - Dates: 2020-01-01
    """
    import re

    name = name.strip()

    # Year patterns
    if re.match(r"^(FY)?\d{4}$", name):
        return True

    # Quarter patterns
    if re.match(r"^(Q[1-4]|q[1-4]|\d{4}[Qq][1-4])$", name):
        return True

    # Month patterns
    month_names = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
        "january", "february", "march", "april", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    if name.lower() in month_names:
        return True

    # ISO date patterns (YYYY-MM, YYYY-MM-DD)
    if re.match(r"^\d{4}-\d{2}(-\d{2})?$", name):
        return True

    # Year-Month combo (2020-Jan, Jan-2020)
    if re.match(r"^\d{4}[-_]?[A-Za-z]{3,}$", name) or re.match(r"^[A-Za-z]{3,}[-_]?\d{4}$", name):
        return True

    return False


def _is_mostly_contiguous(indices: list[int]) -> bool:
    """Check if indices are mostly contiguous (allowing small gaps)."""
    if len(indices) < 2:
        return True

    sorted_indices = sorted(indices)
    gaps = sum(1 for i in range(1, len(sorted_indices)) if sorted_indices[i] - sorted_indices[i - 1] > 1)

    # Allow up to 20% gaps
    return gaps <= len(indices) * 0.2


def _top_roles(col: ColumnEvidence, limit: int = 3) -> list[tuple[Role, float]]:
    """Get top N roles by score for a column."""
    if not col.role_scores:
        return []

    sorted_roles = sorted(col.role_scores.items(), key=lambda x: x[1], reverse=True)
    return [(role, score) for role, score in sorted_roles[:limit] if score > 0]


def _format_hypothesis_name(hypothesis: ShapeHypothesis) -> str:
    """Format hypothesis enum as human-readable name."""
    return hypothesis.value.replace("_", " ").title()
