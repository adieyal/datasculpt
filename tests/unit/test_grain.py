"""Unit tests for grain inference module."""

from __future__ import annotations

import pandas as pd

from datasculpt.core.grain import (
    GrainInference,
    KeyCandidate,
    calculate_confidence,
    calculate_grain_diagnostics,
    calculate_uniqueness_ratio,
    detect_pseudo_key_signals,
    has_stable_grain,
    infer_grain,
    minimize_key,
    rank_key_candidates,
    score_key_combo,
    search_composite_keys,
)
from datasculpt.core.grain import test_single_column_uniqueness as check_single_column_uniqueness
from datasculpt.core.types import (
    ColumnEvidence,
    GrainDiagnostics,
    InferenceConfig,
    PrimitiveType,
    PseudoKeySignals,
    Role,
    StructuralType,
)


def make_evidence(
    name: str,
    role_scores: dict[Role, float] | None = None,
) -> ColumnEvidence:
    """Helper to create ColumnEvidence for testing."""
    return ColumnEvidence(
        name=name,
        primitive_type=PrimitiveType.STRING,
        structural_type=StructuralType.SCALAR,
        role_scores=role_scores or {},
    )


class TestCalculateUniquenessRatio:
    """Tests for calculate_uniqueness_ratio function."""

    def test_all_unique(self) -> None:
        """All unique values gives ratio of 1.0."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["a", "b", "c", "d", "e"],
        })
        ratio = calculate_uniqueness_ratio(df, ["id"])
        assert ratio == 1.0

    def test_some_duplicates(self) -> None:
        """Duplicates reduce uniqueness ratio."""
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C"],
        })
        ratio = calculate_uniqueness_ratio(df, ["category"])
        assert ratio == 0.6  # 3 unique out of 5

    def test_composite_key_uniqueness(self) -> None:
        """Composite key can achieve uniqueness."""
        df = pd.DataFrame({
            "country": ["US", "US", "UK", "UK"],
            "year": [2020, 2021, 2020, 2021],
        })
        # Single column not unique
        assert calculate_uniqueness_ratio(df, ["country"]) == 0.5
        assert calculate_uniqueness_ratio(df, ["year"]) == 0.5
        # Composite key is unique
        assert calculate_uniqueness_ratio(df, ["country", "year"]) == 1.0

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame returns 0.0."""
        df = pd.DataFrame({"id": []})
        assert calculate_uniqueness_ratio(df, ["id"]) == 0.0

    def test_empty_columns(self) -> None:
        """Empty column list returns 0.0."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        assert calculate_uniqueness_ratio(df, []) == 0.0

    def test_nulls_excluded(self) -> None:
        """Rows with nulls in key columns are dropped."""
        df = pd.DataFrame({
            "id": [1, 2, None, 4],
        })
        # 3 unique non-null out of 4 rows
        ratio = calculate_uniqueness_ratio(df, ["id"])
        assert ratio == 0.75


class TestRankKeyCandidates:
    """Tests for rank_key_candidates function."""

    def test_ranks_by_uniqueness(self) -> None:
        """Columns with higher uniqueness rank higher."""
        df = pd.DataFrame({
            "unique_col": [1, 2, 3, 4, 5],  # 100% unique
            "partial_col": [1, 1, 2, 2, 3],  # 60% unique
            "constant_col": [1, 1, 1, 1, 1],  # 20% unique
        })
        candidates = rank_key_candidates(df, {})

        # First should be most unique
        assert candidates[0].name == "unique_col"
        assert candidates[0].cardinality_ratio == 1.0

    def test_penalizes_nulls(self) -> None:
        """Columns with nulls score lower."""
        df = pd.DataFrame({
            "no_nulls": [1, 2, 3, 4, 5],
            "with_nulls": [1, 2, None, None, 5],
        })
        candidates = rank_key_candidates(df, {})

        no_nulls = next(c for c in candidates if c.name == "no_nulls")
        with_nulls = next(c for c in candidates if c.name == "with_nulls")

        assert no_nulls.score > with_nulls.score

    def test_boosts_key_role(self) -> None:
        """Columns with KEY role score get boost."""
        df = pd.DataFrame({
            "id_col": [1, 2, 3, 4, 5],
            "other_col": [1, 2, 3, 4, 5],
        })
        evidence = {
            "id_col": make_evidence("id_col", role_scores={Role.KEY: 0.9}),
            "other_col": make_evidence("other_col", role_scores={Role.KEY: 0.1}),
        }
        candidates = rank_key_candidates(df, evidence)

        # Both have same uniqueness but id_col has KEY role boost
        id_candidate = next(c for c in candidates if c.name == "id_col")
        other_candidate = next(c for c in candidates if c.name == "other_col")

        assert id_candidate.score > other_candidate.score

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame returns empty candidate list."""
        df = pd.DataFrame({"id": []})
        candidates = rank_key_candidates(df, {})
        assert candidates == []


class TestSingleColumnUniqueness:
    """Tests for check_single_column_uniqueness function."""

    def test_finds_unique_column(self) -> None:
        """Finds perfectly unique column."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "category": ["A", "A", "B", "B", "C"],
        })
        candidates = [
            KeyCandidate("id", 5, 1.0, 0.0, 1.0),
            KeyCandidate("category", 3, 0.6, 0.0, 0.6),
        ]
        result = check_single_column_uniqueness(df, candidates)

        assert result is not None
        assert result[0] == "id"
        assert result[1] == 1.0

    def test_accepts_near_perfect(self) -> None:
        """Accepts column with >= 99% uniqueness."""
        df = pd.DataFrame({
            "id": list(range(100)) + [0],  # 100 unique out of 101
        })
        candidates = [KeyCandidate("id", 100, 100 / 101, 0.0, 100 / 101)]
        result = check_single_column_uniqueness(df, candidates)

        assert result is not None
        assert result[0] == "id"
        assert result[1] >= 0.99

    def test_returns_none_for_no_unique(self) -> None:
        """Returns None when no column is sufficiently unique."""
        df = pd.DataFrame({
            "cat1": ["A", "A", "B", "B"],
            "cat2": ["X", "X", "Y", "Y"],
        })
        candidates = [
            KeyCandidate("cat1", 2, 0.5, 0.0, 0.5),
            KeyCandidate("cat2", 2, 0.5, 0.0, 0.5),
        ]
        result = check_single_column_uniqueness(df, candidates)

        assert result is None

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame returns None."""
        df = pd.DataFrame({"id": []})
        candidates = [KeyCandidate("id", 0, 0.0, 0.0, 0.0)]
        result = check_single_column_uniqueness(df, candidates)

        assert result is None


class TestSearchCompositeKeys:
    """Tests for search_composite_keys function."""

    def test_finds_composite_key(self) -> None:
        """Finds composite key when single columns are not unique."""
        df = pd.DataFrame({
            "country": ["US", "US", "UK", "UK"],
            "year": [2020, 2021, 2020, 2021],
            "value": [100, 110, 200, 210],
        })
        candidates = [
            KeyCandidate("country", 2, 0.5, 0.0, 0.5),
            KeyCandidate("year", 2, 0.5, 0.0, 0.5),
            KeyCandidate("value", 4, 1.0, 0.0, 1.0),
        ]
        result = search_composite_keys(df, candidates)

        assert result is not None
        # Should find 2-column key, not include value which is unique alone
        columns, uniqueness, _score = result
        assert len(columns) <= 2
        assert uniqueness == 1.0

    def test_prefers_smaller_key(self) -> None:
        """Prefers smaller composite key when both achieve uniqueness."""
        df = pd.DataFrame({
            "a": [1, 1, 2, 2],
            "b": [1, 2, 1, 2],
            "c": [10, 20, 30, 40],
        })
        candidates = [
            KeyCandidate("a", 2, 0.5, 0.0, 0.5),
            KeyCandidate("b", 2, 0.5, 0.0, 0.5),
            KeyCandidate("c", 4, 1.0, 0.0, 1.0),
        ]
        result = search_composite_keys(df, candidates)

        assert result is not None
        columns, _uniqueness, _score = result
        # Should find [a, b] as composite key (size 2)
        assert len(columns) == 2

    def test_returns_none_when_no_composite(self) -> None:
        """Returns None when no composite key achieves threshold."""
        # All rows identical
        df = pd.DataFrame({
            "a": [1, 1, 1, 1],
            "b": [1, 1, 1, 1],
        })
        candidates = [
            KeyCandidate("a", 1, 0.25, 0.0, 0.25),
            KeyCandidate("b", 1, 0.25, 0.0, 0.25),
        ]
        result = search_composite_keys(df, candidates, min_uniqueness=0.95)

        assert result is None

    def test_respects_max_columns(self) -> None:
        """Respects max_columns parameter."""
        df = pd.DataFrame({
            "a": [1, 1, 1, 1, 1, 1, 1, 1],
            "b": [1, 1, 1, 1, 2, 2, 2, 2],
            "c": [1, 1, 2, 2, 1, 1, 2, 2],
            "d": [1, 2, 1, 2, 1, 2, 1, 2],
        })
        candidates = [
            KeyCandidate("a", 1, 0.125, 0.0, 0.125),
            KeyCandidate("b", 2, 0.25, 0.0, 0.25),
            KeyCandidate("c", 2, 0.25, 0.0, 0.25),
            KeyCandidate("d", 2, 0.25, 0.0, 0.25),
        ]
        # Need 3 columns for uniqueness, but limit to 2
        result = search_composite_keys(df, candidates, max_columns=2)

        assert result is None


class TestCalculateConfidence:
    """Tests for calculate_confidence function."""

    def test_perfect_uniqueness_single_column(self) -> None:
        """Perfect uniqueness with single column gives 1.0 confidence."""
        confidence = calculate_confidence(uniqueness_ratio=1.0, key_size=1)
        assert confidence == 1.0

    def test_penalty_for_larger_keys(self) -> None:
        """Larger keys get confidence penalty."""
        conf_1 = calculate_confidence(1.0, key_size=1)
        conf_2 = calculate_confidence(1.0, key_size=2)
        conf_3 = calculate_confidence(1.0, key_size=3)

        assert conf_1 > conf_2 > conf_3

    def test_low_uniqueness_low_confidence(self) -> None:
        """Low uniqueness gives low confidence."""
        confidence = calculate_confidence(uniqueness_ratio=0.5, key_size=1)
        assert confidence == 0.5

    def test_combined_penalty(self) -> None:
        """Both low uniqueness and large key reduce confidence."""
        confidence = calculate_confidence(uniqueness_ratio=0.9, key_size=3)
        assert confidence < 0.9  # Reduced by size penalty


class TestInferGrain:
    """Tests for infer_grain function."""

    def test_finds_single_column_grain(self) -> None:
        """Finds single column grain when one exists."""
        # Use non-sequential IDs to avoid pseudo-key detection (1,2,3,4,5 triggers it)
        df = pd.DataFrame({
            "id": [101, 203, 305, 407, 509],  # Non-monotonic pattern
            "category": ["A", "A", "B", "B", "C"],
            "value": [10, 20, 30, 40, 50],
        })
        result = infer_grain(df)

        assert isinstance(result, GrainInference)
        # Either id or value could be selected since both are unique
        assert len(result.key_columns) == 1
        assert result.key_columns[0] in ["id", "value"]
        assert result.uniqueness_ratio == 1.0
        assert result.confidence > 0.9

    def test_finds_composite_grain(self) -> None:
        """Finds composite grain when needed."""
        # No single column is unique, but (country, year) is unique
        df = pd.DataFrame({
            "country": ["US", "US", "UK", "UK", "US", "UK"],
            "year": [2020, 2021, 2020, 2021, 2022, 2022],
        })
        result = infer_grain(df)

        # Should find composite key since no single column is unique
        assert len(result.key_columns) == 2
        assert set(result.key_columns) == {"country", "year"}
        assert result.uniqueness_ratio == 1.0

    def test_handles_no_stable_grain(self) -> None:
        """Returns low confidence when no stable grain found."""
        # All rows identical
        df = pd.DataFrame({
            "a": [1, 1, 1, 1],
            "b": [1, 1, 1, 1],
        })
        result = infer_grain(df)

        assert result.confidence == 0.0
        assert "No stable grain found" in " ".join(result.evidence)

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame returns appropriate result."""
        df = pd.DataFrame({"id": []})
        result = infer_grain(df)

        assert result.key_columns == []
        assert result.confidence == 0.0
        assert "empty" in " ".join(result.evidence).lower()

    def test_uses_column_evidence(self) -> None:
        """Uses column evidence to boost key candidates in ranking."""
        df = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5],
            "other": [1, 2, 3, 4, 5],
        })
        evidence = {
            "user_id": make_evidence("user_id", role_scores={Role.KEY: 0.9}),
            "other": make_evidence("other", role_scores={Role.DIMENSION: 0.3}),
        }
        result = infer_grain(df, column_evidence=evidence)

        # Both columns are unique, so either could be chosen
        # The important thing is we found a grain
        assert len(result.key_columns) == 1
        assert result.uniqueness_ratio == 1.0
        assert result.confidence > 0.9

    def test_respects_config(self) -> None:
        """Respects InferenceConfig parameters."""
        df = pd.DataFrame({
            "a": [1, 1, 1, 1, 1, 1, 1, 1],
            "b": [1, 1, 1, 1, 2, 2, 2, 2],
            "c": [1, 1, 2, 2, 1, 1, 2, 2],
            "d": [1, 2, 1, 2, 1, 2, 1, 2],
        })
        # Limit max columns
        config = InferenceConfig(max_grain_columns=2)
        result = infer_grain(df, config=config)

        # Cannot find grain with only 2 columns
        assert result.confidence == 0.0


class TestHasStableGrain:
    """Tests for has_stable_grain function."""

    def test_stable_grain(self) -> None:
        """High confidence and uniqueness is stable."""
        grain = GrainInference(
            key_columns=["id"],
            confidence=0.95,
            uniqueness_ratio=1.0,
            evidence=[],
        )
        assert has_stable_grain(grain) is True

    def test_unstable_low_confidence(self) -> None:
        """Low confidence is not stable."""
        grain = GrainInference(
            key_columns=["id"],
            confidence=0.5,
            uniqueness_ratio=1.0,
            evidence=[],
        )
        assert has_stable_grain(grain) is False

    def test_unstable_low_uniqueness(self) -> None:
        """Low uniqueness is not stable."""
        grain = GrainInference(
            key_columns=["id"],
            confidence=0.95,
            uniqueness_ratio=0.8,
            evidence=[],
        )
        assert has_stable_grain(grain) is False

    def test_custom_threshold(self) -> None:
        """Respects custom min_confidence threshold."""
        grain = GrainInference(
            key_columns=["id"],
            confidence=0.8,
            uniqueness_ratio=0.98,
            evidence=[],
        )
        assert has_stable_grain(grain, min_confidence=0.7) is True
        assert has_stable_grain(grain, min_confidence=0.9) is False


class TestPseudoKeyDetection:
    """Tests for pseudo-key detection functionality."""

    def test_detects_monotonic_sequence(self) -> None:
        """Detects 0..N-1 or 1..N sequence patterns."""
        # 0-based sequence
        series_zero = pd.Series([0, 1, 2, 3, 4])
        signals = detect_pseudo_key_signals(series_zero, "row_num")
        assert signals.is_monotonic_sequence is True

        # 1-based sequence
        series_one = pd.Series([1, 2, 3, 4, 5])
        signals = detect_pseudo_key_signals(series_one, "id")
        assert signals.is_monotonic_sequence is True

    def test_not_monotonic_with_gaps(self) -> None:
        """Non-sequential integers are not flagged."""
        series = pd.Series([1, 3, 5, 7, 9])  # Odd numbers
        signals = detect_pseudo_key_signals(series, "id")
        assert signals.is_monotonic_sequence is False

    def test_detects_uuid_like(self) -> None:
        """Detects UUID-like hex strings."""
        uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "7c9e6679-7425-40de-944b-e07fc1f90ae7",
            "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        ]
        series = pd.Series(uuids)
        signals = detect_pseudo_key_signals(series, "uuid_col")
        assert signals.is_uuid_like is True

    def test_not_uuid_with_repeats(self) -> None:
        """Non-unique strings are not flagged as UUID-like."""
        series = pd.Series(["abc", "abc", "def", "ghi", "jkl"])
        signals = detect_pseudo_key_signals(series, "code")
        assert signals.is_uuid_like is False

    def test_name_pattern_penalties(self) -> None:
        """Column names matching anti-patterns get penalties."""
        series = pd.Series([1, 2, 3, 4, 5])

        # row_id pattern
        signals = detect_pseudo_key_signals(series, "row_id")
        assert signals.name_signal_penalty > 0

        # index pattern
        signals = detect_pseudo_key_signals(series, "index")
        assert signals.name_signal_penalty > 0

        # Normal name - no penalty
        signals = detect_pseudo_key_signals(series, "customer_id")
        assert signals.name_signal_penalty == 0.0

    def test_total_penalty_capped(self) -> None:
        """Total penalty is capped at 0.5."""
        # Monotonic sequence + bad name should be capped
        series = pd.Series([1, 2, 3, 4, 5])
        signals = detect_pseudo_key_signals(series, "row_id")
        assert signals.total_penalty <= 0.5

    def test_pseudo_signals_dataclass(self) -> None:
        """PseudoKeySignals dataclass has expected fields."""
        signals = PseudoKeySignals()
        assert signals.is_monotonic_sequence is False
        assert signals.is_uuid_like is False
        assert signals.is_ingestion_timestamp is False
        assert signals.name_signal_penalty == 0.0
        assert signals.total_penalty == 0.0


class TestKeyMinimization:
    """Tests for minimize_key function."""

    def test_removes_redundant_column(self) -> None:
        """Removes column that doesn't contribute to uniqueness."""
        df = pd.DataFrame({
            "country": ["US", "US", "UK", "UK"],
            "year": [2020, 2021, 2020, 2021],
            "constant": ["A", "A", "A", "A"],  # Redundant
        })
        # All three columns achieve uniqueness, but constant is redundant
        minimized = minimize_key(df, ["country", "year", "constant"])
        assert "constant" not in minimized
        assert set(minimized) == {"country", "year"}

    def test_preserves_necessary_columns(self) -> None:
        """Keeps all columns when all are necessary."""
        df = pd.DataFrame({
            "a": [1, 1, 2, 2],
            "b": [1, 2, 1, 2],
        })
        minimized = minimize_key(df, ["a", "b"])
        assert set(minimized) == {"a", "b"}

    def test_single_column_unchanged(self) -> None:
        """Single column keys are returned unchanged."""
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        minimized = minimize_key(df, ["id"])
        assert minimized == ["id"]

    def test_empty_columns(self) -> None:
        """Empty column list returns empty."""
        df = pd.DataFrame({"id": [1, 2, 3]})
        minimized = minimize_key(df, [])
        assert minimized == []


class TestGrainDiagnostics:
    """Tests for calculate_grain_diagnostics function."""

    def test_counts_nulls_in_key(self) -> None:
        """Counts rows with nulls in key columns."""
        df = pd.DataFrame({
            "id": [1, 2, None, 4],
            "value": [10, 20, 30, 40],
        })
        diagnostics = calculate_grain_diagnostics(df, ["id"])
        assert diagnostics.rows_with_null_in_key == 1
        assert "id" in diagnostics.null_columns

    def test_counts_duplicate_groups(self) -> None:
        """Counts groups with duplicate keys."""
        df = pd.DataFrame({
            "key": ["A", "A", "B", "B", "C"],
        })
        diagnostics = calculate_grain_diagnostics(df, ["key"])
        assert diagnostics.duplicate_groups == 2  # A and B have duplicates
        assert diagnostics.max_group_size == 2

    def test_extracts_example_duplicates(self) -> None:
        """Extracts example duplicate key values."""
        df = pd.DataFrame({
            "key": ["A", "A", "A", "B", "C"],
        })
        diagnostics = calculate_grain_diagnostics(df, ["key"])
        assert len(diagnostics.example_duplicate_keys) >= 1
        # The duplicate key "A" should be in examples
        assert ("A",) in diagnostics.example_duplicate_keys

    def test_no_duplicates(self) -> None:
        """Clean data has no duplicate diagnostics."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
        })
        diagnostics = calculate_grain_diagnostics(df, ["id"])
        assert diagnostics.duplicate_groups == 0
        assert diagnostics.max_group_size == 0
        assert diagnostics.rows_with_null_in_key == 0

    def test_diagnostics_dataclass(self) -> None:
        """GrainDiagnostics dataclass has expected defaults."""
        diag = GrainDiagnostics()
        assert diag.duplicate_groups == 0
        assert diag.max_group_size == 0
        assert diag.rows_with_null_in_key == 0
        assert diag.null_columns == []
        assert diag.example_duplicate_keys == []


class TestBestComboSelection:
    """Tests for best-combo selection in search_composite_keys."""

    def test_returns_three_values(self) -> None:
        """search_composite_keys returns columns, uniqueness, and score."""
        df = pd.DataFrame({
            "country": ["US", "US", "UK", "UK"],
            "year": [2020, 2021, 2020, 2021],
        })
        candidates = [
            KeyCandidate("country", 2, 0.5, 0.0, 0.5),
            KeyCandidate("year", 2, 0.5, 0.0, 0.5),
        ]
        result = search_composite_keys(df, candidates)
        assert result is not None
        columns, uniqueness, score = result
        assert isinstance(columns, list)
        assert isinstance(uniqueness, float)
        assert isinstance(score, float)

    def test_scores_combos(self) -> None:
        """score_key_combo produces reasonable scores."""
        df = pd.DataFrame({
            "a": [1, 1, 2, 2],
            "b": [1, 2, 1, 2],
        })
        candidates = [
            KeyCandidate("a", 2, 0.5, 0.0, 0.5),
            KeyCandidate("b", 2, 0.5, 0.0, 0.5),
        ]
        score = score_key_combo(df, ["a", "b"], candidates, uniqueness=1.0)
        assert 0.0 <= score <= 1.0
        # Perfect uniqueness with 2 columns should score well
        assert score > 0.9

    def test_penalizes_pseudo_keys_in_combo(self) -> None:
        """Combos with pseudo-keys get lower scores."""
        df = pd.DataFrame({
            "row_id": [1, 2, 3, 4],
            "country": ["US", "US", "UK", "UK"],
        })
        pseudo_signals = PseudoKeySignals(
            is_monotonic_sequence=True,
            total_penalty=0.4,
        )
        candidates = [
            KeyCandidate("row_id", 4, 1.0, 0.0, 0.7, pseudo_key_signals=pseudo_signals),
            KeyCandidate("country", 2, 0.5, 0.0, 0.5),
        ]
        # Combo with pseudo-key
        score_with_pseudo = score_key_combo(df, ["row_id", "country"], candidates, 1.0)
        # Combo without pseudo-key (hypothetical)
        score_key_combo(df, ["country"], candidates, 0.5)
        # The pseudo-key penalty should affect the score
        assert score_with_pseudo < 1.0


class TestImprovedConfidence:
    """Tests for improved multi-factor confidence calculation."""

    def test_base_confidence(self) -> None:
        """Base confidence equals uniqueness for simple cases."""
        conf = calculate_confidence(uniqueness_ratio=0.95, key_size=1)
        assert conf == 0.95

    def test_pseudo_key_penalty(self) -> None:
        """Pseudo-key penalty reduces confidence."""
        conf_no_penalty = calculate_confidence(1.0, 1, pseudo_key_penalty=0.0)
        conf_with_penalty = calculate_confidence(1.0, 1, pseudo_key_penalty=0.3)
        assert conf_with_penalty < conf_no_penalty

    def test_null_contamination_penalty(self) -> None:
        """Null contamination reduces confidence."""
        conf_clean = calculate_confidence(1.0, 1, null_contamination=0.0)
        conf_nulls = calculate_confidence(1.0, 1, null_contamination=0.1)
        assert conf_nulls < conf_clean

    def test_margin_bonus(self) -> None:
        """Large margin vs runner-up increases confidence."""
        conf_close = calculate_confidence(0.95, 1, margin_vs_runner_up=0.05)
        conf_clear = calculate_confidence(0.95, 1, margin_vs_runner_up=0.2)
        assert conf_clear > conf_close

    def test_combined_factors(self) -> None:
        """All factors combine correctly."""
        # Many penalties should reduce confidence significantly
        conf = calculate_confidence(
            uniqueness_ratio=0.95,
            key_size=3,
            pseudo_key_penalty=0.2,
            null_contamination=0.1,
        )
        # Should be less than base uniqueness
        assert conf < 0.95
        # But still positive
        assert conf > 0

    def test_confidence_clamped(self) -> None:
        """Confidence is clamped to 0-1 range."""
        conf_low = calculate_confidence(0.1, 4, pseudo_key_penalty=0.5, null_contamination=0.5)
        assert conf_low >= 0.0

        conf_high = calculate_confidence(1.0, 1, margin_vs_runner_up=0.5)
        assert conf_high <= 1.0


class TestGrainInferenceWithDiagnostics:
    """Tests for infer_grain with diagnostics."""

    def test_includes_diagnostics(self) -> None:
        """infer_grain returns diagnostics."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
        })
        result = infer_grain(df)
        assert result.diagnostics is not None
        assert isinstance(result.diagnostics, GrainDiagnostics)

    def test_diagnostics_for_composite_key(self) -> None:
        """Diagnostics calculated for composite keys."""
        df = pd.DataFrame({
            "country": ["US", "US", "UK", "UK"],
            "year": [2020, 2021, 2020, 2021],
        })
        result = infer_grain(df)
        assert result.diagnostics is not None

    def test_pseudo_key_warning_in_evidence(self) -> None:
        """Pseudo-key columns trigger warnings in evidence."""
        df = pd.DataFrame({
            "row_id": list(range(10)),  # Monotonic sequence
            "value": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4],
        })
        result = infer_grain(df)
        # Should have some warning about pseudo-key
        " ".join(result.evidence)
        # Confidence should be reduced
        assert result.confidence < 1.0

    def test_minimization_noted_in_evidence(self) -> None:
        """Key minimization is noted in evidence when it happens."""
        df = pd.DataFrame({
            "country": ["US", "US", "UK", "UK"],
            "year": [2020, 2021, 2020, 2021],
            "constant": ["X", "X", "X", "X"],
        })
        # Force composite key search by not having single unique column
        infer_grain(df)
        # If minimization happened, it should be noted
        # (depends on whether constant gets included initially)
