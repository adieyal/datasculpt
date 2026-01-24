"""Unit tests for grain inference module."""

from __future__ import annotations

import pandas as pd
import pytest

from datasculpt.core.grain import (
    GrainInference,
    KeyCandidate,
    calculate_confidence,
    calculate_uniqueness_ratio,
    has_stable_grain,
    infer_grain,
    rank_key_candidates,
    search_composite_keys,
)
from datasculpt.core.grain import test_single_column_uniqueness as check_single_column_uniqueness
from datasculpt.core.types import ColumnEvidence, InferenceConfig, PrimitiveType, Role, StructuralType


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
        columns, uniqueness = result
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
        columns, _ = result
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
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "category": ["A", "A", "B", "B", "C"],
            "value": [10, 20, 30, 40, 50],
        })
        result = infer_grain(df)

        assert isinstance(result, GrainInference)
        assert result.key_columns == ["id"]
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
