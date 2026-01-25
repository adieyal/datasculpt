"""Grain inference for Datasculpt (browser bundle)."""


from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

from datasculpt.types import (
    ColumnEvidence,
    GrainDiagnostics,
    GrainInference,
    InferenceConfig,
    PseudoKeySignals,
    Role,
    ShapeHypothesis,
)

if TYPE_CHECKING:
    pass


def detect_pseudo_key_signals(
    df: pd.DataFrame,
    evidence: ColumnEvidence,
) -> PseudoKeySignals:
    """Detect signals that a column is a pseudo-key (surrogate/synthetic).

    Args:
        df: DataFrame containing the column.
        evidence: Column evidence.

    Returns:
        PseudoKeySignals with detected signals.
    """
    signals = PseudoKeySignals()
    col_name = evidence.name
    series = df[col_name]

    # Check for monotonic sequence (row numbers)
    non_null = series.dropna()
    if len(non_null) > 0:
        # Check if values are sequential integers
        if pd.api.types.is_integer_dtype(non_null):
            sorted_vals = non_null.sort_values()
            is_sequential = (sorted_vals.diff().dropna() == 1).all()
            if is_sequential:
                signals.is_monotonic_sequence = True
                signals.total_penalty += 0.4

    # Check for UUID-like patterns
    name_lower = col_name.lower()
    uuid_indicators = ["uuid", "guid", "uid", "_id"]
    if any(ind in name_lower for ind in uuid_indicators):
        # Check if values look like UUIDs
        sample = non_null.head(10).astype(str)
        uuid_pattern = r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$"
        uuid_matches = sample.str.match(uuid_pattern, case=False)
        if uuid_matches.mean() > 0.5:
            signals.is_uuid_like = True
            signals.total_penalty += 0.3

    # Check for ingestion timestamp patterns
    timestamp_indicators = ["created", "inserted", "loaded", "ingested", "_at"]
    if any(ind in name_lower for ind in timestamp_indicators):
        signals.is_ingestion_timestamp = True
        signals.total_penalty += 0.25

    # Name signal penalties
    synthetic_indicators = ["row", "index", "sequence", "auto", "synthetic"]
    if any(ind in name_lower for ind in synthetic_indicators):
        signals.name_signal_penalty = 0.3
        signals.total_penalty += 0.3

    return signals


def compute_uniqueness_ratio(
    df: pd.DataFrame,
    columns: list[str],
) -> float:
    """Compute the uniqueness ratio for a combination of columns.

    Args:
        df: DataFrame to check.
        columns: Column names to check.

    Returns:
        Ratio of unique combinations to total rows.
    """
    if not columns:
        return 0.0

    subset = df[columns].dropna()
    if len(subset) == 0:
        return 0.0

    unique_count = subset.drop_duplicates().shape[0]
    return unique_count / len(subset)


def compute_grain_diagnostics(
    df: pd.DataFrame,
    key_columns: list[str],
) -> GrainDiagnostics:
    """Compute diagnostics for grain quality assessment.

    Args:
        df: DataFrame to analyze.
        key_columns: Selected key columns.

    Returns:
        GrainDiagnostics with detailed information.
    """
    diagnostics = GrainDiagnostics()

    if not key_columns:
        return diagnostics

    # Check for nulls in key columns
    for col in key_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            diagnostics.rows_with_null_in_key += null_count
            diagnostics.null_columns.append(col)

    # Find duplicate groups
    subset = df[key_columns].dropna()
    if len(subset) > 0:
        duplicates = subset[subset.duplicated(keep=False)]
        if len(duplicates) > 0:
            grouped = duplicates.groupby(key_columns, observed=True)
            diagnostics.duplicate_groups = grouped.ngroups
            diagnostics.max_group_size = grouped.size().max()

            # Get example duplicate keys
            for name, _group in grouped:
                if len(diagnostics.example_duplicate_keys) < 5:
                    if isinstance(name, tuple):
                        diagnostics.example_duplicate_keys.append(name)
                    else:
                        diagnostics.example_duplicate_keys.append((name,))

    return diagnostics


def rank_column_for_grain(
    evidence: ColumnEvidence,
    pseudo_signals: PseudoKeySignals,
) -> float:
    """Rank a column's suitability for grain.

    Args:
        evidence: Column evidence.
        pseudo_signals: Pseudo-key detection signals.

    Returns:
        Suitability score (higher is better).
    """
    score = 0.0

    # Key role score is primary signal
    key_score = evidence.role_scores.get(Role.KEY, 0.0)
    score += key_score * 0.4

    # Dimension role is also grain-relevant
    dim_score = evidence.role_scores.get(Role.DIMENSION, 0.0)
    score += dim_score * 0.3

    # Time role can be part of grain
    time_score = evidence.role_scores.get(Role.TIME, 0.0)
    score += time_score * 0.2

    # Penalize measures and values (not grain columns)
    meas_score = evidence.role_scores.get(Role.MEASURE, 0.0)
    val_score = evidence.role_scores.get(Role.VALUE, 0.0)
    score -= (meas_score + val_score) * 0.3

    # Apply pseudo-key penalty
    score -= pseudo_signals.total_penalty

    # High distinct ratio is good for keys
    if evidence.distinct_ratio > 0.9:
        score += 0.2

    # Low null rate is good
    if evidence.null_rate < 0.01:
        score += 0.15

    return max(0.0, score)


def infer_grain(
    df: pd.DataFrame,
    column_evidence: dict[str, ColumnEvidence],
    config: InferenceConfig,
    shape: ShapeHypothesis | None = None,
) -> GrainInference:
    """Infer the grain (unique key) for a dataset.

    Args:
        df: DataFrame to analyze.
        column_evidence: Evidence for each column.
        config: Inference configuration.
        shape: Optional detected shape hypothesis.

    Returns:
        GrainInference with key columns and confidence.
    """
    evidence_list: list[str] = []

    # Detect pseudo-key signals for each column
    pseudo_signals = {
        name: detect_pseudo_key_signals(df, ev)
        for name, ev in column_evidence.items()
    }

    # Rank columns by grain suitability
    column_ranks = [
        (name, rank_column_for_grain(ev, pseudo_signals[name]))
        for name, ev in column_evidence.items()
    ]
    column_ranks.sort(key=lambda x: x[1], reverse=True)

    # Get candidate columns (positive rank)
    candidates = [name for name, rank in column_ranks if rank > 0.2]

    # Limit candidates
    candidates = candidates[:config.max_grain_columns * 2]

    if not candidates:
        # Fall back to all non-measure columns
        candidates = [
            name for name, ev in column_evidence.items()
            if ev.role_scores.get(Role.MEASURE, 0) < 0.5
            and ev.role_scores.get(Role.VALUE, 0) < 0.5
        ][:config.max_grain_columns * 2]

    # Try single columns first
    best_columns: list[str] = []
    best_uniqueness = 0.0

    for col in candidates:
        uniqueness = compute_uniqueness_ratio(df, [col])
        if uniqueness >= config.min_uniqueness_confidence:
            if uniqueness > best_uniqueness:
                best_columns = [col]
                best_uniqueness = uniqueness
                evidence_list.append(f"Column '{col}' is unique ({uniqueness:.1%})")

    # If no single column works, try combinations
    if best_uniqueness < config.min_uniqueness_confidence:
        for r in range(2, min(len(candidates), config.max_grain_columns) + 1):
            for combo in combinations(candidates, r):
                uniqueness = compute_uniqueness_ratio(df, list(combo))
                if uniqueness > best_uniqueness:
                    best_columns = list(combo)
                    best_uniqueness = uniqueness

                if uniqueness >= config.min_uniqueness_confidence:
                    evidence_list.append(
                        f"Combination {combo} achieves {uniqueness:.1%} uniqueness"
                    )
                    break

            if best_uniqueness >= config.min_uniqueness_confidence:
                break

    # Calculate confidence
    confidence = best_uniqueness
    if best_uniqueness >= config.min_uniqueness_confidence:
        confidence = min(1.0, best_uniqueness + 0.1)

    # Compute diagnostics
    diagnostics = compute_grain_diagnostics(df, best_columns)

    # Add diagnostics to evidence
    if diagnostics.duplicate_groups > 0:
        evidence_list.append(
            f"Found {diagnostics.duplicate_groups} duplicate key groups"
        )
    if diagnostics.rows_with_null_in_key > 0:
        evidence_list.append(
            f"{diagnostics.rows_with_null_in_key} rows have null values in key columns"
        )

    return GrainInference(
        key_columns=best_columns,
        confidence=confidence,
        uniqueness_ratio=best_uniqueness,
        evidence=evidence_list,
        diagnostics=diagnostics,
    )
