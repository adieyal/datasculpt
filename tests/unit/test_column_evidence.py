"""Unit tests for extended ColumnEvidence fields."""

from __future__ import annotations

import pandas as pd
import pytest

from datasculpt.core.evidence import extract_column_evidence


class TestNRows:
    """Tests for n_rows field - total row count."""

    def test_n_rows_basic(self) -> None:
        """n_rows should equal total rows in series."""
        series = pd.Series([1, 2, 3, 4, 5])
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_rows == 5

    def test_n_rows_with_nulls(self) -> None:
        """n_rows should include null rows."""
        series = pd.Series([1, None, 3, None, 5])
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_rows == 5

    def test_n_rows_empty_series(self) -> None:
        """n_rows should be 0 for empty series."""
        series = pd.Series([], dtype=object)
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_rows == 0

    def test_n_rows_all_nulls(self) -> None:
        """n_rows should count all null rows."""
        series = pd.Series([None, None, None])
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_rows == 3


class TestNNonNull:
    """Tests for n_non_null field - non-null count."""

    def test_n_non_null_no_nulls(self) -> None:
        """n_non_null should equal n_rows when no nulls."""
        series = pd.Series([1, 2, 3, 4, 5])
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_non_null == 5

    def test_n_non_null_with_nulls(self) -> None:
        """n_non_null should exclude null values."""
        series = pd.Series([1, None, 3, None, 5])
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_non_null == 3

    def test_n_non_null_empty_series(self) -> None:
        """n_non_null should be 0 for empty series."""
        series = pd.Series([], dtype=object)
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_non_null == 0

    def test_n_non_null_all_nulls(self) -> None:
        """n_non_null should be 0 when all values are null."""
        series = pd.Series([None, None, None])
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_non_null == 0

    def test_n_non_null_excludes_nan(self) -> None:
        """n_non_null should exclude NaN values."""
        series = pd.Series([1.0, float("nan"), 3.0, float("nan")])
        evidence = extract_column_evidence(series, "col")
        assert evidence.n_non_null == 2


class TestTopValues:
    """Tests for top_values field - list of (value, count) tuples."""

    def test_top_values_basic(self) -> None:
        """top_values should return value counts as tuples."""
        series = pd.Series(["a", "b", "a", "c", "a", "b"])
        evidence = extract_column_evidence(series, "col")

        assert isinstance(evidence.top_values, list)
        assert len(evidence.top_values) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in evidence.top_values)

        values_dict = dict(evidence.top_values)
        assert "a" in values_dict
        assert values_dict["a"] == 3

    def test_top_values_limited_to_10(self) -> None:
        """top_values should return at most 10 entries."""
        values = [f"val_{i}" for i in range(15)] * 2
        series = pd.Series(values)
        evidence = extract_column_evidence(series, "col")

        assert len(evidence.top_values) <= 10

    def test_top_values_sorted_by_count(self) -> None:
        """top_values should be sorted by count descending."""
        series = pd.Series(["a", "a", "a", "b", "b", "c"])
        evidence = extract_column_evidence(series, "col")

        assert evidence.top_values[0] == ("a", 3)

        counts = [count for _, count in evidence.top_values]
        assert counts == sorted(counts, reverse=True)

    def test_top_values_empty_series(self) -> None:
        """top_values should be empty list for empty series."""
        series = pd.Series([], dtype=object)
        evidence = extract_column_evidence(series, "col")
        assert evidence.top_values == []

    def test_top_values_all_nulls(self) -> None:
        """top_values should be empty list when all values are null."""
        series = pd.Series([None, None, None])
        evidence = extract_column_evidence(series, "col")
        assert evidence.top_values == []

    def test_top_values_excludes_nulls(self) -> None:
        """top_values should not include null values."""
        series = pd.Series(["a", None, "a", None, "b"])
        evidence = extract_column_evidence(series, "col")

        values_dict = dict(evidence.top_values)
        assert None not in values_dict
        assert "None" not in values_dict
        assert values_dict.get("a") == 2
        assert values_dict.get("b") == 1

    def test_top_values_numeric_stringified(self) -> None:
        """top_values should stringify numeric values."""
        series = pd.Series([1, 2, 1, 1, 2])
        evidence = extract_column_evidence(series, "col")

        values = [v for v, _ in evidence.top_values]
        assert all(isinstance(v, str) for v in values)
        assert "1" in values
        assert "2" in values


class TestValueLengthStats:
    """Tests for value_length_stats field - string length statistics."""

    def test_value_length_stats_string_column(self) -> None:
        """value_length_stats should compute min/max/mean for strings."""
        series = pd.Series(["a", "bb", "ccc", "dddd"])
        evidence = extract_column_evidence(series, "col")

        assert evidence.value_length_stats is not None
        assert evidence.value_length_stats["min"] == 1
        assert evidence.value_length_stats["max"] == 4
        assert evidence.value_length_stats["mean"] == pytest.approx(2.5)

    def test_value_length_stats_uniform_length(self) -> None:
        """value_length_stats should handle uniform length strings."""
        series = pd.Series(["aa", "bb", "cc"])
        evidence = extract_column_evidence(series, "col")

        assert evidence.value_length_stats is not None
        assert evidence.value_length_stats["min"] == 2
        assert evidence.value_length_stats["max"] == 2
        assert evidence.value_length_stats["mean"] == 2.0

    def test_value_length_stats_with_nulls(self) -> None:
        """value_length_stats should exclude nulls from calculation."""
        series = pd.Series(["a", None, "ccc", None])
        evidence = extract_column_evidence(series, "col")

        assert evidence.value_length_stats is not None
        assert evidence.value_length_stats["min"] == 1
        assert evidence.value_length_stats["max"] == 3
        assert evidence.value_length_stats["mean"] == pytest.approx(2.0)

    def test_value_length_stats_empty_series(self) -> None:
        """value_length_stats should be None for empty series."""
        series = pd.Series([], dtype=object)
        evidence = extract_column_evidence(series, "col")
        assert evidence.value_length_stats is None

    def test_value_length_stats_all_nulls(self) -> None:
        """value_length_stats should be None when all values are null."""
        series = pd.Series([None, None, None])
        evidence = extract_column_evidence(series, "col")
        assert evidence.value_length_stats is None

    def test_value_length_stats_numeric_column(self) -> None:
        """value_length_stats should compute for stringified numerics."""
        series = pd.Series([1, 22, 333, 4444])
        evidence = extract_column_evidence(series, "col")

        assert evidence.value_length_stats is not None
        assert evidence.value_length_stats["min"] == 1
        assert evidence.value_length_stats["max"] == 4


class TestRegexHits:
    """Tests for regex_hits field - pattern match counts."""

    def test_regex_hits_default_empty(self) -> None:
        """regex_hits should default to empty dict."""
        series = pd.Series(["test@email.com", "hello", "another@test.org"])
        evidence = extract_column_evidence(series, "col")

        assert evidence.regex_hits == {}
        assert isinstance(evidence.regex_hits, dict)
