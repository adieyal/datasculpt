"""Grain inference module for determining dataset unique keys."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

logger = logging.getLogger(__name__)

from datasculpt.core.types import (
    ColumnEvidence,
    GrainDiagnostics,
    GrainInference,
    InferenceConfig,
    PseudoKeySignals,
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

# Naming patterns for pseudo-keys (columns that look unique but aren't meaningful keys)
PSEUDO_KEY_NAME_PATTERNS = (
    (re.compile(r"^row_?id$", re.IGNORECASE), 0.3),  # row_id, rowid
    (re.compile(r"^index$", re.IGNORECASE), 0.3),  # index
    (re.compile(r"^_?id$", re.IGNORECASE), 0.1),  # id, _id (lower penalty - might be real)
    (re.compile(r"^uuid$", re.IGNORECASE), 0.25),  # uuid
    (re.compile(r"^guid$", re.IGNORECASE), 0.25),  # guid
    (re.compile(r"^hash$", re.IGNORECASE), 0.2),  # hash
    (re.compile(r"^created_at$", re.IGNORECASE), 0.2),  # created_at
    (re.compile(r"^updated_at$", re.IGNORECASE), 0.2),  # updated_at
    (re.compile(r"^inserted_at$", re.IGNORECASE), 0.2),  # inserted_at
    (re.compile(r"^timestamp$", re.IGNORECASE), 0.15),  # timestamp
    (re.compile(r"^record_?id$", re.IGNORECASE), 0.15),  # record_id
    (re.compile(r"^row_?num(ber)?$", re.IGNORECASE), 0.3),  # row_num, row_number
    (re.compile(r"^seq(uence)?$", re.IGNORECASE), 0.25),  # seq, sequence
    (re.compile(r"^auto_?inc(rement)?$", re.IGNORECASE), 0.3),  # auto_inc, autoincrement
)


def _matches_survey_id_pattern(name: str) -> bool:
    """Check if name matches survey identifier patterns."""
    return any(p.search(name) for p in SURVEY_ID_PATTERNS)


def _get_pseudo_key_name_penalty(name: str) -> float:
    """Get penalty for pseudo-key name patterns.

    Returns the highest matching penalty, or 0.0 if no pattern matches.
    """
    max_penalty = 0.0
    for pattern, penalty in PSEUDO_KEY_NAME_PATTERNS:
        if pattern.search(name):
            max_penalty = max(max_penalty, penalty)
    return max_penalty


def _detect_monotonic_sequence(series: pd.Series) -> bool:
    """Check if a series is a monotonic integer sequence (1, 2, 3, ... N).

    This pattern indicates a row index or auto-increment ID.
    """
    if series.isna().any():
        return False

    # Must be numeric
    if not pd.api.types.is_numeric_dtype(series):
        return False

    values = series.values
    n = len(values)
    if n < 2:
        return False

    # Check if it's 0..N-1 or 1..N (common patterns)
    try:
        is_zero_based = all(v == i for i, v in enumerate(values))
        is_one_based = all(v == i + 1 for i, v in enumerate(values))
        return is_zero_based or is_one_based
    except (TypeError, ValueError) as e:
        logger.debug(f"Monotonic sequence check failed due to type/value error: {e}")
        return False


def _detect_uuid_like(series: pd.Series) -> bool:
    """Check if a series contains UUID-like values.

    Characteristics:
    - String type with fixed length (32-36 chars for UUIDs)
    - High entropy (many unique characters)
    - All unique values
    - Contains hex-like patterns
    """
    if series.isna().any():
        return False

    # Must be string type
    if not pd.api.types.is_string_dtype(series):
        # Try to detect object dtype with string values
        if series.dtype == object:
            try:
                sample = series.dropna().head(100).astype(str)
            except (ValueError, TypeError):
                return False
        else:
            return False
    else:
        sample = series.head(100)

    if len(sample) < 2:
        return False

    # Check for fixed length (UUID pattern: 32 hex + 4 dashes = 36, or 32 without dashes)
    lengths = sample.str.len()
    unique_lengths = lengths.nunique()
    if unique_lengths > 2:  # Allow some variation but not too much
        return False

    mode_length = lengths.mode().iloc[0] if len(lengths) > 0 else 0
    if mode_length not in (32, 36):
        return False

    # Check for hex-like content (allow dashes)
    hex_pattern = re.compile(r"^[0-9a-fA-F-]+$")
    hex_matches = sample.apply(lambda x: bool(hex_pattern.match(str(x))) if pd.notna(x) else False)
    if hex_matches.sum() < len(sample) * 0.9:  # At least 90% match
        return False

    # Check uniqueness - should be 100% unique
    return series.nunique() == len(series)


def _detect_ingestion_timestamp(series: pd.Series, name: str) -> bool:
    """Check if a series is likely an ingestion timestamp.

    Characteristics:
    - Name suggests creation/insertion time
    - Datetime values
    - Very small time deltas between consecutive rows (suggests bulk insert)
    """
    # Check name signals first
    timestamp_names = ("created_at", "updated_at", "inserted_at", "ingested_at", "load_time")
    if name.lower() not in timestamp_names:
        return False

    # Try to convert to datetime
    try:
        if pd.api.types.is_datetime64_any_dtype(series):
            dt_series = series
        else:
            dt_series = pd.to_datetime(series, errors="coerce")

        if dt_series.isna().sum() > len(series) * 0.5:
            return False

        # Check for tiny deltas (suggests automated ingestion)
        sorted_times = dt_series.dropna().sort_values()
        if len(sorted_times) < 2:
            return False

        deltas = sorted_times.diff().dropna()
        median_delta = deltas.median()

        # If median delta is less than 1 second, likely auto-generated
        if isinstance(median_delta, pd.Timedelta):
            one_second = pd.Timedelta(seconds=1)
            if median_delta < one_second:
                return True

    except (ValueError, TypeError):
        pass

    return False


def detect_pseudo_key_signals(series: pd.Series, name: str) -> PseudoKeySignals:
    """Detect signals that a column may be a pseudo-key.

    Pseudo-keys are columns that uniquely identify rows but don't represent
    meaningful business keys (e.g., row indices, UUIDs, ingestion timestamps).

    Args:
        series: Column data to analyze.
        name: Column name.

    Returns:
        PseudoKeySignals with detection results and computed penalty.
    """
    is_monotonic = _detect_monotonic_sequence(series)
    is_uuid = _detect_uuid_like(series)
    is_timestamp = _detect_ingestion_timestamp(series, name)
    name_penalty = _get_pseudo_key_name_penalty(name)

    # Calculate total penalty (capped at 0.5)
    total_penalty = name_penalty
    if is_monotonic:
        total_penalty += 0.3
    if is_uuid:
        total_penalty += 0.25
    if is_timestamp:
        total_penalty += 0.2

    total_penalty = min(0.5, total_penalty)

    return PseudoKeySignals(
        is_monotonic_sequence=is_monotonic,
        is_uuid_like=is_uuid,
        is_ingestion_timestamp=is_timestamp,
        name_signal_penalty=name_penalty,
        total_penalty=total_penalty,
    )


@dataclass
class KeyCandidate:
    """A candidate column for grain inference with ranking metrics."""

    name: str
    cardinality: int
    cardinality_ratio: float
    null_rate: float
    score: float
    pseudo_key_signals: PseudoKeySignals | None = None


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
                elif primary_role == Role.DIMENSION or primary_role == Role.TIME:
                    score += 0.15
                elif primary_role == Role.INDICATOR_NAME:
                    # For long_indicators, indicator_name is critical
                    if detected_shape == ShapeHypothesis.LONG_INDICATORS:
                        score += 0.2

        # Boost columns matching survey identifier patterns (person/member IDs)
        # These are often low-cardinality but critical for composite keys
        if _matches_survey_id_pattern(col_str):
            score += 0.25

        # Detect and apply pseudo-key penalties
        pseudo_signals = detect_pseudo_key_signals(series, col_str)
        if pseudo_signals.total_penalty > 0:
            score *= 1.0 - pseudo_signals.total_penalty

        candidates.append(
            KeyCandidate(
                name=col_str,
                cardinality=cardinality,
                cardinality_ratio=cardinality_ratio,
                null_rate=null_rate,
                score=score,
                pseudo_key_signals=pseudo_signals,
            )
        )

    # Sort by score descending (best candidates first)
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Hybrid filtering: keep top-K candidates OR those above threshold
    # This ensures we don't miss good candidates just because of a threshold
    top_k = 8
    score_threshold = 0.15
    filtered = []
    for i, c in enumerate(candidates):
        if i < top_k or c.score >= score_threshold:
            filtered.append(c)

    return filtered


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


def score_key_combo(
    df: pd.DataFrame,
    columns: list[str],
    candidates: list[KeyCandidate],
    uniqueness: float,
) -> float:
    """Score a key combination based on multiple factors.

    Considers:
    - Uniqueness ratio (primary factor)
    - Key size penalty (prefer smaller keys)
    - Pseudo-key penalty (avoid auto-generated columns)
    - Null rate penalty (prefer columns with few nulls)
    - Candidate score bonus (prefer columns with good role alignment)

    Args:
        df: Input DataFrame.
        columns: Columns in the combo.
        candidates: Full list of key candidates (for lookup).
        uniqueness: Pre-computed uniqueness ratio.

    Returns:
        Combined score (higher is better).
    """
    score = uniqueness

    # Size penalty: 3% per additional column
    score -= 0.03 * (len(columns) - 1)

    # Build lookup for candidates
    candidate_map = {c.name: c for c in candidates}

    # Check for pseudo-keys and calculate average metrics
    has_pseudo_key = False
    total_null_rate = 0.0
    total_candidate_score = 0.0
    valid_count = 0

    for col in columns:
        if col in candidate_map:
            c = candidate_map[col]
            if c.pseudo_key_signals and c.pseudo_key_signals.total_penalty > 0.2:
                has_pseudo_key = True
            total_null_rate += c.null_rate
            total_candidate_score += c.score
            valid_count += 1

    # Pseudo-key penalty
    if has_pseudo_key:
        score -= 0.2

    # Null rate penalty
    if valid_count > 0:
        avg_null_rate = total_null_rate / valid_count
        score -= 0.1 * avg_null_rate

        # Role alignment bonus
        avg_candidate_score = total_candidate_score / valid_count
        score += 0.05 * avg_candidate_score

    return max(0.0, score)


def search_composite_keys(
    df: pd.DataFrame,
    candidates: list[KeyCandidate],
    max_columns: int = 4,
    min_uniqueness: float = 0.95,
    min_score_threshold: float = 0.2,
) -> tuple[list[str], float, float] | None:
    """Search for composite keys by trying column combinations.

    Uses best-combo selection: evaluates all combinations at each size level
    and picks the one with the best overall score (not just first-hit).

    Args:
        df: Input DataFrame.
        candidates: Ranked list of key candidates to consider.
        max_columns: Maximum number of columns in composite key.
        min_uniqueness: Minimum uniqueness ratio to accept.
        min_score_threshold: Minimum score to consider (filters out low-score columns).

    Returns:
        Tuple of (column_names, uniqueness_ratio, combo_score) if found, None otherwise.
    """
    total_rows = len(df)
    if total_rows == 0:
        return None

    # Only consider candidates above score threshold (exclude measure/value columns)
    valid_candidates = [c for c in candidates if c.score >= min_score_threshold]

    # Limit search space - use 20 to handle datasets with many dimension columns
    max_candidates = min(len(valid_candidates), 20)
    top_candidates = valid_candidates[:max_candidates]
    candidate_names = [c.name for c in top_candidates]

    # Track best result across all sizes: (columns, uniqueness, score)
    best_result: tuple[list[str], float, float] | None = None

    # Try combinations starting from size 2 (size 1 already tested)
    for size in range(2, max_columns + 1):
        # Track best at this size level
        best_at_size: tuple[list[str], float, float] | None = None

        for combo in combinations(candidate_names, size):
            columns = list(combo)
            uniqueness = calculate_uniqueness_ratio(df, columns)

            # Skip if below minimum threshold
            if uniqueness < min_uniqueness:
                continue

            # Score this combo
            combo_score = score_key_combo(df, columns, candidates, uniqueness)

            # Track best at this size
            if best_at_size is None or combo_score > best_at_size[2]:
                best_at_size = (columns, uniqueness, combo_score)

        # Update overall best if this size found something better
        if best_at_size is not None:
            if best_result is None or best_at_size[2] > best_result[2]:
                best_result = best_at_size

            # Early exit: if we have 100% uniqueness AND high score, stop searching
            if best_at_size[1] == 1.0 and best_at_size[2] > 0.8:
                break

    return best_result


def minimize_key(
    df: pd.DataFrame,
    columns: list[str],
    min_uniqueness: float = 0.95,
) -> list[str]:
    """Minimize a key by removing redundant columns.

    Attempts to remove columns from the end while maintaining uniqueness.
    This helps produce cleaner keys when extra columns were included.

    Args:
        df: Input DataFrame.
        columns: Initial key columns (order matters - removes from end first).
        min_uniqueness: Minimum uniqueness to maintain.

    Returns:
        Minimized list of key columns.
    """
    if len(columns) <= 1:
        return columns

    current = list(columns)

    # Try removing columns from the end (typically lower priority)
    for col in reversed(columns):
        if len(current) <= 1:
            break

        test = [c for c in current if c != col]
        if calculate_uniqueness_ratio(df, test) >= min_uniqueness:
            current = test

    return current


def calculate_grain_diagnostics(
    df: pd.DataFrame,
    columns: list[str],
) -> GrainDiagnostics:
    """Calculate diagnostics about grain quality.

    Provides information useful for user feedback:
    - How many rows have nulls in key columns
    - How many duplicate groups exist
    - Example duplicate keys for investigation

    Args:
        df: Input DataFrame.
        columns: Key columns to analyze.

    Returns:
        GrainDiagnostics with quality metrics.
    """
    if not columns or len(df) == 0:
        return GrainDiagnostics()

    key_df = df[columns]

    # Count nulls in key columns
    null_mask = key_df.isna().any(axis=1)
    rows_with_null = int(null_mask.sum())
    null_columns = [col for col in columns if df[col].isna().any()]

    # Find duplicates
    non_null_df = key_df.dropna()
    if len(non_null_df) == 0:
        return GrainDiagnostics(
            rows_with_null_in_key=rows_with_null,
            null_columns=null_columns,
        )

    # Group by key and find duplicates
    value_counts = non_null_df.groupby(list(columns), dropna=False).size()
    duplicates = value_counts[value_counts > 1]

    duplicate_groups = len(duplicates)
    max_group_size = int(duplicates.max()) if len(duplicates) > 0 else 0

    # Extract example duplicate keys (max 3)
    example_keys: list[tuple] = []
    if duplicate_groups > 0:
        for key_tuple in duplicates.head(3).index:
            if isinstance(key_tuple, tuple):
                example_keys.append(key_tuple)
            else:
                # Single column case
                example_keys.append((key_tuple,))

    return GrainDiagnostics(
        duplicate_groups=duplicate_groups,
        max_group_size=max_group_size,
        rows_with_null_in_key=rows_with_null,
        null_columns=null_columns,
        example_duplicate_keys=example_keys,
    )


def calculate_confidence(
    uniqueness_ratio: float,
    key_size: int,
    pseudo_key_penalty: float = 0.0,
    null_contamination: float = 0.0,
    margin_vs_runner_up: float = 0.0,
) -> float:
    """Calculate confidence score based on multiple factors.

    Confidence is higher for:
    - Higher uniqueness ratios
    - Smaller key sets (single column preferred)
    - No pseudo-key columns
    - Low null rates in key columns
    - Clear margin vs alternative candidates

    Args:
        uniqueness_ratio: The uniqueness ratio (0-1).
        key_size: Number of columns in the key.
        pseudo_key_penalty: Penalty from pseudo-key columns (0-0.5).
        null_contamination: Rate of rows with nulls in key (0-1).
        margin_vs_runner_up: Score margin vs next best candidate (higher = more confident).

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

    # Apply penalties
    confidence = base_confidence - size_penalty
    confidence -= pseudo_key_penalty * 0.3  # Pseudo-key penalty (scaled)
    confidence -= null_contamination * 0.2  # Null penalty

    # Margin bonus (if clear winner)
    if margin_vs_runner_up > 0.1:
        confidence += 0.05

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

    # Helper to get pseudo-key penalty for columns
    def get_pseudo_penalty(cols: list[str]) -> float:
        candidate_map = {c.name: c for c in candidates}
        max_penalty = 0.0
        for col in cols:
            if col in candidate_map:
                c = candidate_map[col]
                if c.pseudo_key_signals:
                    max_penalty = max(max_penalty, c.pseudo_key_signals.total_penalty)
        return max_penalty

    # Step 2: Try single-column uniqueness
    single_result = test_single_column_uniqueness(df, candidates)

    if single_result is not None:
        col_name, uniqueness = single_result

        # Calculate diagnostics
        diagnostics = calculate_grain_diagnostics(df, [col_name])
        null_contamination = diagnostics.rows_with_null_in_key / total_rows if total_rows > 0 else 0.0
        pseudo_penalty = get_pseudo_penalty([col_name])

        confidence = calculate_confidence(
            uniqueness,
            key_size=1,
            pseudo_key_penalty=pseudo_penalty,
            null_contamination=null_contamination,
        )

        if uniqueness == 1.0:
            evidence_notes.append(f"Column '{col_name}' is perfectly unique")
        else:
            evidence_notes.append(
                f"Column '{col_name}' has {uniqueness:.2%} uniqueness"
            )

        # Add diagnostic warnings
        if pseudo_penalty > 0.1:
            evidence_notes.append(f"Warning: '{col_name}' may be a pseudo-key (auto-generated)")
        if diagnostics.rows_with_null_in_key > 0:
            evidence_notes.append(f"Warning: {diagnostics.rows_with_null_in_key} rows have null in key column")

        return GrainInference(
            key_columns=[col_name],
            confidence=confidence,
            uniqueness_ratio=uniqueness,
            evidence=evidence_notes,
            diagnostics=diagnostics,
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
        columns, uniqueness, combo_score = composite_result

        # Minimize the key (remove redundant columns)
        minimized_columns = minimize_key(df, columns, config.min_uniqueness_confidence)
        if len(minimized_columns) < len(columns):
            evidence_notes.append(
                f"Minimized key from {len(columns)} to {len(minimized_columns)} columns"
            )
            columns = minimized_columns
            # Recalculate uniqueness for minimized key
            uniqueness = calculate_uniqueness_ratio(df, columns)

        # Calculate diagnostics
        diagnostics = calculate_grain_diagnostics(df, columns)
        null_contamination = diagnostics.rows_with_null_in_key / total_rows if total_rows > 0 else 0.0
        pseudo_penalty = get_pseudo_penalty(columns)

        confidence = calculate_confidence(
            uniqueness,
            key_size=len(columns),
            pseudo_key_penalty=pseudo_penalty,
            null_contamination=null_contamination,
        )

        evidence_notes.append(
            f"Composite key [{', '.join(columns)}] has {uniqueness:.2%} uniqueness"
        )

        # Add diagnostic warnings
        if pseudo_penalty > 0.1:
            evidence_notes.append("Warning: Key contains potential pseudo-key column(s)")
        if diagnostics.rows_with_null_in_key > 0:
            evidence_notes.append(f"Warning: {diagnostics.rows_with_null_in_key} rows have nulls in key columns")
        if diagnostics.duplicate_groups > 0:
            evidence_notes.append(
                f"Warning: {diagnostics.duplicate_groups} duplicate key combinations found "
                f"(max group size: {diagnostics.max_group_size})"
            )

        return GrainInference(
            key_columns=columns,
            confidence=confidence,
            uniqueness_ratio=uniqueness,
            evidence=evidence_notes,
            diagnostics=diagnostics,
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
    diagnostics = calculate_grain_diagnostics(df, [best_candidate.name])

    return GrainInference(
        key_columns=[best_candidate.name],
        confidence=0.0,  # Zero confidence indicates no stable grain
        uniqueness_ratio=best_uniqueness,
        evidence=evidence_notes,
        diagnostics=diagnostics,
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
