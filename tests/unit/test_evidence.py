"""Unit tests for evidence extraction module."""

from __future__ import annotations

import pandas as pd
import pytest

from datasculpt.core.evidence import (
    attempt_date_parse,
    compute_distinct_ratio,
    compute_null_rate,
    detect_json_array,
    detect_primitive_type,
    detect_structural_type,
    extract_column_evidence,
)
from datasculpt.core.types import PrimitiveType, StructuralType


class TestDetectPrimitiveType:
    """Tests for detect_primitive_type function."""

    def test_integer_dtype(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5])
        assert detect_primitive_type(series) == PrimitiveType.INTEGER

    def test_float_dtype_whole_numbers(self) -> None:
        """Float dtype with whole numbers should be INTEGER."""
        series = pd.Series([1.0, 2.0, 3.0])
        assert detect_primitive_type(series) == PrimitiveType.INTEGER

    def test_float_dtype_decimals(self) -> None:
        """Float dtype with actual decimals should be NUMBER."""
        series = pd.Series([1.5, 2.7, 3.14])
        assert detect_primitive_type(series) == PrimitiveType.NUMBER

    def test_boolean_dtype(self) -> None:
        series = pd.Series([True, False, True])
        assert detect_primitive_type(series) == PrimitiveType.BOOLEAN

    def test_string_dtype(self) -> None:
        series = pd.Series(["apple", "banana", "cherry"])
        assert detect_primitive_type(series) == PrimitiveType.STRING

    def test_datetime_dtype_date_only(self) -> None:
        """Datetime without time component should be DATE."""
        series = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
        assert detect_primitive_type(series) == PrimitiveType.DATE

    def test_datetime_dtype_with_time(self) -> None:
        """Datetime with time component should be DATETIME."""
        series = pd.Series(pd.to_datetime(["2020-01-01 10:30:00", "2020-01-02 14:45:00"]))
        assert detect_primitive_type(series) == PrimitiveType.DATETIME

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        assert detect_primitive_type(series) == PrimitiveType.UNKNOWN

    def test_all_null_series(self) -> None:
        series = pd.Series([None, None, None])
        assert detect_primitive_type(series) == PrimitiveType.UNKNOWN

    def test_object_dtype_boolean_strings(self) -> None:
        """Object dtype with boolean strings should be BOOLEAN."""
        series = pd.Series(["true", "false", "true"])
        assert detect_primitive_type(series) == PrimitiveType.BOOLEAN

    def test_object_dtype_yes_no(self) -> None:
        """Object dtype with yes/no values should be BOOLEAN."""
        series = pd.Series(["yes", "no", "yes"])
        assert detect_primitive_type(series) == PrimitiveType.BOOLEAN

    def test_object_dtype_numeric_strings(self) -> None:
        """Object dtype with numeric strings should detect number type."""
        series = pd.Series(["1", "2", "3"])
        assert detect_primitive_type(series) == PrimitiveType.INTEGER

    def test_object_dtype_date_strings(self) -> None:
        """Object dtype with date strings should detect DATE."""
        series = pd.Series(["2020-01-01", "2020-02-15", "2020-03-20"])
        assert detect_primitive_type(series) == PrimitiveType.DATE


class TestComputeNullRate:
    """Tests for compute_null_rate function."""

    def test_no_nulls(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5])
        assert compute_null_rate(series) == 0.0

    def test_all_nulls(self) -> None:
        series = pd.Series([None, None, None])
        assert compute_null_rate(series) == 1.0

    def test_partial_nulls(self) -> None:
        series = pd.Series([1, None, 3, None, 5])
        assert compute_null_rate(series) == 0.4

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        assert compute_null_rate(series) == 0.0

    def test_na_values(self) -> None:
        """NaN values should be counted as nulls."""
        series = pd.Series([1.0, float("nan"), 3.0])
        assert compute_null_rate(series) == pytest.approx(1 / 3, rel=1e-3)


class TestComputeDistinctRatio:
    """Tests for compute_distinct_ratio (cardinality) function."""

    def test_all_unique(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5])
        assert compute_distinct_ratio(series) == 1.0

    def test_all_same(self) -> None:
        series = pd.Series([1, 1, 1, 1, 1])
        assert compute_distinct_ratio(series) == 0.2

    def test_mixed_cardinality(self) -> None:
        series = pd.Series([1, 1, 2, 2, 3])  # 3 unique out of 5
        assert compute_distinct_ratio(series) == 0.6

    def test_with_nulls(self) -> None:
        """Nulls should be excluded from ratio calculation."""
        series = pd.Series([1, 2, None, None, 3])  # 3 unique out of 3 non-null
        assert compute_distinct_ratio(series) == 1.0

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        assert compute_distinct_ratio(series) == 0.0


class TestAttemptDateParse:
    """Tests for attempt_date_parse function."""

    def test_iso_dates(self) -> None:
        series = pd.Series(["2020-01-15", "2020-02-20", "2020-03-25"])
        result = attempt_date_parse(series)
        assert result["success_rate"] > 0.9
        assert result["has_time"] == False  # noqa: E712

    def test_datetime_with_time(self) -> None:
        series = pd.Series(["2020-01-15 10:30:00", "2020-02-20 14:45:00"])
        result = attempt_date_parse(series)
        assert result["success_rate"] > 0.9
        assert result["has_time"] == True  # noqa: E712

    def test_iso_datetime(self) -> None:
        series = pd.Series(["2020-01-15T10:30:00", "2020-02-20T14:45:00"])
        result = attempt_date_parse(series)
        assert result["success_rate"] > 0.9
        assert result["has_time"] == True  # noqa: E712

    def test_non_date_strings(self) -> None:
        series = pd.Series(["apple", "banana", "cherry"])
        result = attempt_date_parse(series)
        assert result["success_rate"] < 0.1

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        result = attempt_date_parse(series)
        assert result["success_rate"] == 0.0
        assert result["best_format"] is None


class TestDetectJsonArray:
    """Tests for detect_json_array function."""

    def test_json_array_strings(self) -> None:
        series = pd.Series(['[1, 2, 3]', '[4, 5, 6]', '[7, 8, 9]'])
        result = detect_json_array(series)
        assert result["is_json_array"] is True
        assert result["success_rate"] > 0.9

    def test_python_lists(self) -> None:
        series = pd.Series([[1, 2], [3, 4], [5, 6]])
        result = detect_json_array(series)
        assert result["is_json_array"] is True
        assert result["success_rate"] > 0.9

    def test_non_array_values(self) -> None:
        series = pd.Series(["hello", "world", "test"])
        result = detect_json_array(series)
        assert result["is_json_array"] is False
        assert result["success_rate"] < 0.1

    def test_mixed_valid_invalid(self) -> None:
        series = pd.Series(['[1, 2]', 'not json', '[3, 4]'])
        result = detect_json_array(series)
        assert result["success_rate"] == pytest.approx(2 / 3, rel=1e-3)

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        result = detect_json_array(series)
        assert result["is_json_array"] is False
        assert result["success_rate"] == 0.0


class TestDetectStructuralType:
    """Tests for detect_structural_type function."""

    def test_scalar_values(self) -> None:
        series = pd.Series([1, 2, 3])
        assert detect_structural_type(series) == StructuralType.SCALAR

    def test_array_values_python_list(self) -> None:
        series = pd.Series([[1, 2], [3, 4], [5, 6]])
        assert detect_structural_type(series) == StructuralType.ARRAY

    def test_array_values_json_string(self) -> None:
        series = pd.Series(['[1, 2]', '[3, 4]', '[5, 6]'])
        assert detect_structural_type(series) == StructuralType.ARRAY

    def test_object_values_python_dict(self) -> None:
        series = pd.Series([{"a": 1}, {"b": 2}, {"c": 3}])
        assert detect_structural_type(series) == StructuralType.OBJECT

    def test_object_values_json_string(self) -> None:
        series = pd.Series(['{"a": 1}', '{"b": 2}', '{"c": 3}'])
        assert detect_structural_type(series) == StructuralType.OBJECT

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        assert detect_structural_type(series) == StructuralType.UNKNOWN


class TestExtractColumnEvidence:
    """Tests for extract_column_evidence function."""

    def test_integer_column(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5])
        evidence = extract_column_evidence(series, "my_column")

        assert evidence.name == "my_column"
        assert evidence.primitive_type == PrimitiveType.INTEGER
        assert evidence.structural_type == StructuralType.SCALAR
        assert evidence.null_rate == 0.0
        assert evidence.distinct_ratio == 1.0

    def test_string_column_with_nulls(self) -> None:
        series = pd.Series(["a", "b", None, "a", None])
        evidence = extract_column_evidence(series, "category")

        assert evidence.name == "category"
        assert evidence.primitive_type == PrimitiveType.STRING
        assert evidence.null_rate == 0.4
        assert evidence.distinct_ratio == pytest.approx(2 / 3, rel=1e-3)

    def test_json_array_column(self) -> None:
        series = pd.Series(['[1, 2]', '[3, 4]', '[5, 6]'])
        evidence = extract_column_evidence(series, "data")

        assert evidence.structural_type == StructuralType.ARRAY
        assert evidence.parse_results.get("json_array", 0) > 0.9
        assert "Contains JSON arrays" in evidence.notes
