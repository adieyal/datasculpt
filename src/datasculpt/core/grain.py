"""Grain inference module for determining dataset unique keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

from datasculpt.core.types import (
    ColumnEvidence,
    GrainInference,
    InferenceConfig,
    Role,
    ShapeHypothesis,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# Naming patterns for common survey/roster identifiers that should be
# considered for composite keys even if they have low cardinality
SURVEY_ID_PATTERNS = (
    re.compile(r"^indiv", re.IGNORECASE),  # indiv, individual, individual_id
    re.compile(r"^person", re.IGNORECASE),  # person, person_id, person_num
    re.compile(r"^member", re.IGNORECASE),  # member, member_id, member_no
    re.compile(r"^line", re.IGNORECASE),  # line, line_no, line_number
    re.compile(r"^roster", re.IGNORECASE),  # roster_id, roster_line
    re.compile(r"^hhmem", re.IGNORECASE),  # hhmem, hhmember
    re.compile(r"^pid$", re.IGNORECASE),  # pid (person id)
    re.compile(r"^mid$", re.IGNORECASE),  # mid (member id)
    re.compile(r"_num$", re.IGNORECASE),  # person_num, member_num
    re.compile(r"_no$", re.IGNORECASE),  # person_no, member_no
    re.compile(r"_line$", re.IGNORECASE),  # roster_line
)


def _matches_survey_id_pattern(name: str) -> bool:
    """Check if name matches survey identifier patterns."""
    return any(p.search(name) for p in SURVEY_ID_PATTERNS)


@dataclass
class KeyCandidate:
    """A candidate column for grain inference with ranking metrics."""

    name: str
    cardinality: int
    cardinality_ratio: float
    null_rate: float
    score: float


def rank_key_candidates(
    df: pd.DataFrame,
    column_evidence: dict[str, ColumnEvidence],
    detected_shape: ShapeHypothesis | None = None,
) -> list[KeyCandidate]:
    """Rank columns by their likelihood of being grain columns.

    Grain columns are determined by semantic role, not just cardinality.
    For long_indicators shape: grain = dimensions + time + indicator_name
    For other shapes: grain = key + dimensions + time (excluding measures)

    Args:
        df: Input DataFrame.
        column_evidence: Pre-computed evidence about each column.
        detected_shape: The detected dataset shape (affects grain semantics).

    Returns:
        List of KeyCandidate objects sorted by grain likelihood (best first).
    """
    total_rows = len(df)
    if total_rows == 0:
        return []

    candidates: list[KeyCandidate] = []

    # Roles that should be part of grain (not measures/values)
    grain_roles = {Role.KEY, Role.DIMENSION, Role.TIME, Role.INDICATOR_NAME}

    # Roles that should NOT be part of grain
    non_grain_roles = {Role.MEASURE, Role.VALUE, Role.SERIES, Role.METADATA}

    for col in df.columns:
        col_str = str(col)
        series = df[col]

        # Calculate metrics
        non_null_count = series.notna().sum()
        null_rate = 1.0 - (non_null_count / total_rows) if total_rows > 0 else 1.0
        cardinality = series.nunique(dropna=True)
        cardinality_ratio = cardinality / total_rows if total_rows > 0 else 0.0

        # Base score from cardinality and null rate
        base_score = cardinality_ratio * (1.0 - null_rate)
        score = base_score

        # Apply role-based scoring from column evidence
        if col_str in column_evidence:
            evidence = column_evidence[col_str]

            # Find the primary role (highest scoring)
            primary_role = None
            primary_role_score = 0.0
            for role, role_score in evidence.role_scores.items():
                if role_score > primary_role_score:
                    primary_role = role
                    primary_role_score = role_score

            # Heavily penalize columns with non-grain roles
            if primary_role in non_grain_roles:
                # Measure/Value columns should NOT be in grain
                score *= 0.1  # 90% penalty
            elif primary_role in grain_roles:
                # Boost columns with grain-appropriate roles
                score = base_score * 0.5 + primary_role_score * 0.5

                # Extra boost for key/dimension/time roles
                if primary_role == Role.KEY:
                    score += 0.2
                elif primary_role == Role.DIMENSION:
                    score += 0.15
                elif primary_role == Role.TIME:
                    score += 0.15
                elif primary_role == Role.INDICATOR_NAME:
                    # For long_indicators, indicator_name is critical
                    if detected_shape == ShapeHypothesis.LONG_INDICATORS:
                        score += 0.2

        # Boost columns matching survey identifier patterns (person/member IDs)
        # These are often low-cardinality but critical for composite keys
        if _matches_survey_id_pattern(col_str):
            score += 0.25

        candidates.append(
            KeyCandidate(
                name=col_str,
                cardinality=cardinality,
                cardinality_ratio=cardinality_ratio,
                null_rate=null_rate,
                score=score,
            )
        )

    # Sort by score descending (best candidates first)
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def calculate_uniqueness_ratio(df: pd.DataFrame, columns: Sequence[str]) -> float:
    """Calculate the uniqueness ratio for a set of columns.

    Uniqueness ratio = unique_combinations / total_rows

    Args:
        df: Input DataFrame.
        columns: Column names to check for uniqueness.

    Returns:
        Float between 0 and 1 representing uniqueness.
    """
    total_rows = len(df)
    if total_rows == 0:
        return 0.0

    if not columns:
        return 0.0

    # Drop rows with nulls in key columns and count unique combinations
    subset = df[list(columns)].dropna()
    unique_count = len(subset.drop_duplicates())

    return unique_count / total_rows


def test_single_column_uniqueness(
    df: pd.DataFrame,
    candidates: list[KeyCandidate],
    min_score_threshold: float = 0.2,
) -> tuple[str, float] | None:
    """Test if any single column provides perfect or near-perfect uniqueness.

    Only considers candidates with score above threshold. This prevents
    measure/value columns (which are penalized in scoring) from being
    selected as grain even if they happen to be unique.

    Args:
        df: Input DataFrame.
        candidates: Ranked list of key candidates.
        min_score_threshold: Minimum score to consider (filters out low-score columns).

    Returns:
        Tuple of (column_name, uniqueness_ratio) if a suitable column found,
        None otherwise.
    """
    total_rows = len(df)
    if total_rows == 0:
        return None

    for candidate in candidates:
        # Skip candidates with low scores (likely measure/value columns)
        if candidate.score < min_score_threshold:
            continue

        series = df[candidate.name]
        non_null_count = series.notna().sum()
        unique_count = series.nunique(dropna=True)

        # Check if this column is unique (considering non-null values)
        if unique_count == non_null_count == total_rows:
            return (candidate.name, 1.0)

        # Check uniqueness ratio
        uniqueness = unique_count / total_rows
        if uniqueness >= 0.99:  # Near-perfect uniqueness
            return (candidate.name, uniqueness)

    return None


def search_composite_keys(
    df: pd.DataFrame,
    candidates: list[KeyCandidate],
    max_columns: int = 4,
    min_uniqueness: float = 0.95,
    min_score_threshold: float = 0.2,
) -> tuple[list[str], float] | None:
    """Search for composite keys by trying column combinations.

    Prefers smaller key sets and exits early when perfect uniqueness found.
    Only considers candidates with score above threshold.

    Args:
        df: Input DataFrame.
        candidates: Ranked list of key candidates to consider.
        max_columns: Maximum number of columns in composite key.
        min_uniqueness: Minimum uniqueness ratio to accept.
        min_score_threshold: Minimum score to consider (filters out low-score columns).

    Returns:
        Tuple of (column_names, uniqueness_ratio) if found, None otherwise.
    """
    total_rows = len(df)
    if total_rows == 0:
        return None

    # Only consider candidates above score threshold (exclude measure/value columns)
    valid_candidates = [c for c in candidates if c.score >= min_score_threshold]

    # Limit search space - use 20 to handle datasets with many dimension columns
    # (e.g., survey data where composite keys may involve lower-ranked columns)
    max_candidates = min(len(valid_candidates), 20)
    candidate_names = [c.name for c in valid_candidates[:max_candidates]]

    best_result: tuple[list[str], float] | None = None

    # Try combinations starting from size 2 (size 1 already tested)
    for size in range(2, max_columns + 1):
        for combo in combinations(candidate_names, size):
            columns = list(combo)
            uniqueness = calculate_uniqueness_ratio(df, columns)

            # Perfect uniqueness - early exit
            if uniqueness == 1.0:
                return (columns, 1.0)

            # Track best result above threshold
            if uniqueness >= min_uniqueness:
                if best_result is None or uniqueness > best_result[1]:
                    best_result = (columns, uniqueness)

        # If we found a perfect result at this size, no need to try larger
        if best_result is not None and best_result[1] == 1.0:
            break

    return best_result


def calculate_confidence(uniqueness_ratio: float, key_size: int) -> float:
    """Calculate confidence score based on uniqueness and key size.

    Confidence is higher for:
    - Higher uniqueness ratios
    - Smaller key sets (single column preferred)

    Args:
        uniqueness_ratio: The uniqueness ratio (0-1).
        key_size: Number of columns in the key.

    Returns:
        Confidence score between 0 and 1.
    """
    # Base confidence from uniqueness
    base_confidence = uniqueness_ratio

    # Penalty for larger key sizes (diminishing penalty)
    # Single column: no penalty
    # 2 columns: 5% penalty
    # 3 columns: 10% penalty
    # 4 columns: 15% penalty
    size_penalty = 0.05 * (key_size - 1) if key_size > 1 else 0.0

    confidence = base_confidence - size_penalty

    # Clamp to valid range
    return max(0.0, min(1.0, confidence))


def infer_grain(
    df: pd.DataFrame,
    column_evidence: dict[str, ColumnEvidence] | None = None,
    config: InferenceConfig | None = None,
    detected_shape: ShapeHypothesis | None = None,
) -> GrainInference:
    """Infer the grain (unique key) for a dataset.

    The grain represents the set of columns that uniquely identify each row.
    This function uses semantic roles to determine grain columns:
    - For long_indicators: grain = dimensions + time + indicator_name
    - For other shapes: grain = key + dimensions + time (excluding measures)

    Args:
        df: Input DataFrame to analyze.
        column_evidence: Optional pre-computed evidence about columns.
        config: Optional inference configuration.
        detected_shape: The detected dataset shape (affects grain semantics).

    Returns:
        GrainInference with key columns, confidence, and evidence.
    """
    if config is None:
        config = InferenceConfig()

    if column_evidence is None:
        column_evidence = {}

    evidence_notes: list[str] = []
    total_rows = len(df)

    # Handle empty dataframe
    if total_rows == 0:
        return GrainInference(
            key_columns=[],
            confidence=0.0,
            uniqueness_ratio=0.0,
            evidence=["Dataset is empty - no grain can be inferred"],
        )

    # Step 1: Rank candidates using role-based scoring
    candidates = rank_key_candidates(df, column_evidence, detected_shape)

    if not candidates:
        return GrainInference(
            key_columns=[],
            confidence=0.0,
            uniqueness_ratio=0.0,
            evidence=["No columns available for grain inference"],
        )

    evidence_notes.append(
        f"Analyzed {len(candidates)} columns as key candidates"
    )

    # Step 2: Try single-column uniqueness
    single_result = test_single_column_uniqueness(df, candidates)

    if single_result is not None:
        col_name, uniqueness = single_result
        confidence = calculate_confidence(uniqueness, key_size=1)

        if uniqueness == 1.0:
            evidence_notes.append(f"Column '{col_name}' is perfectly unique")
        else:
            evidence_notes.append(
                f"Column '{col_name}' has {uniqueness:.2%} uniqueness"
            )

        return GrainInference(
            key_columns=[col_name],
            confidence=confidence,
            uniqueness_ratio=uniqueness,
            evidence=evidence_notes,
        )

    evidence_notes.append("No single column provides sufficient uniqueness")

    # Step 3: Search for composite keys
    composite_result = search_composite_keys(
        df,
        candidates,
        max_columns=config.max_grain_columns,
        min_uniqueness=config.min_uniqueness_confidence,
    )

    if composite_result is not None:
        columns, uniqueness = composite_result
        confidence = calculate_confidence(uniqueness, key_size=len(columns))

        evidence_notes.append(
            f"Composite key [{', '.join(columns)}] has {uniqueness:.2%} uniqueness"
        )

        return GrainInference(
            key_columns=columns,
            confidence=confidence,
            uniqueness_ratio=uniqueness,
            evidence=evidence_notes,
        )

    # Step 4: No stable grain found - return warning
    evidence_notes.append(
        f"No stable grain found with up to {config.max_grain_columns} columns"
    )
    evidence_notes.append(
        "Dataset may have duplicate rows or require all columns as grain"
    )

    # Return best single-column candidate with its actual uniqueness
    best_candidate = candidates[0]
    best_uniqueness = calculate_uniqueness_ratio(df, [best_candidate.name])

    return GrainInference(
        key_columns=[best_candidate.name],
        confidence=0.0,  # Zero confidence indicates no stable grain
        uniqueness_ratio=best_uniqueness,
        evidence=evidence_notes,
    )


def has_stable_grain(grain: GrainInference, min_confidence: float = 0.9) -> bool:
    """Check if a grain inference result represents a stable grain.

    Args:
        grain: The grain inference result.
        min_confidence: Minimum confidence threshold.

    Returns:
        True if the grain is stable, False otherwise.
    """
    return grain.confidence >= min_confidence and grain.uniqueness_ratio >= 0.95
