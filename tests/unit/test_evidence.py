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
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.INTEGER

    def test_float_dtype_whole_numbers(self) -> None:
        """Float dtype with whole numbers should be INTEGER."""
        series = pd.Series([1.0, 2.0, 3.0])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.INTEGER
        # Should have note about float-to-integer coercion
        assert any("float dtype" in note for note in notes)

    def test_float_dtype_decimals(self) -> None:
        """Float dtype with actual decimals should be NUMBER."""
        series = pd.Series([1.5, 2.7, 3.14])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.NUMBER

    def test_boolean_dtype(self) -> None:
        series = pd.Series([True, False, True])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.BOOLEAN

    def test_string_dtype(self) -> None:
        series = pd.Series(["apple", "banana", "cherry"])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.STRING

    def test_datetime_dtype_date_only(self) -> None:
        """Datetime without time component should be DATE."""
        series = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02"]))
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.DATE

    def test_datetime_dtype_with_time(self) -> None:
        """Datetime with time component should be DATETIME."""
        series = pd.Series(pd.to_datetime(["2020-01-01 10:30:00", "2020-01-02 14:45:00"]))
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.DATETIME

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.UNKNOWN

    def test_all_null_series(self) -> None:
        series = pd.Series([None, None, None])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.UNKNOWN

    def test_object_dtype_boolean_strings(self) -> None:
        """Object dtype with boolean strings should be BOOLEAN."""
        series = pd.Series(["true", "false", "true"])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.BOOLEAN

    def test_object_dtype_yes_no(self) -> None:
        """Object dtype with yes/no values should be BOOLEAN."""
        series = pd.Series(["yes", "no", "yes"])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.BOOLEAN

    def test_object_dtype_numeric_strings(self) -> None:
        """Object dtype with numeric strings should detect number type."""
        series = pd.Series(["1", "2", "3"])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.INTEGER

    def test_object_dtype_date_strings(self) -> None:
        """Object dtype with date strings should detect DATE."""
        series = pd.Series(["2020-01-01", "2020-02-15", "2020-03-20"])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.DATE


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
        assert float(result["success_rate"]) > 0.9  # type: ignore[arg-type]
        assert result["has_time"] is False

    def test_datetime_with_time(self) -> None:
        series = pd.Series(["2020-01-15 10:30:00", "2020-02-20 14:45:00"])
        result = attempt_date_parse(series)
        assert float(result["success_rate"]) > 0.9  # type: ignore[arg-type]
        assert result["has_time"] is True

    def test_iso_datetime(self) -> None:
        series = pd.Series(["2020-01-15T10:30:00", "2020-02-20T14:45:00"])
        result = attempt_date_parse(series)
        assert float(result["success_rate"]) > 0.9  # type: ignore[arg-type]
        assert result["has_time"] is True

    def test_non_date_strings(self) -> None:
        series = pd.Series(["apple", "banana", "cherry"])
        result = attempt_date_parse(series)
        assert float(result["success_rate"]) < 0.1  # type: ignore[arg-type]

    def test_empty_series(self) -> None:
        series = pd.Series([], dtype=object)
        result = attempt_date_parse(series)
        assert result["success_rate"] == 0.0
        assert result["best_format"] is None

    def test_failure_examples_returned(self) -> None:
        """Tests that failure examples are included in result."""
        series = pd.Series(["2020-01-15", "not-a-date", "2020-03-25"])
        result = attempt_date_parse(series)
        assert "failure_examples" in result
        failures = result["failure_examples"]
        assert isinstance(failures, list)


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
        assert evidence.parse_results.json_array_rate > 0.9
        assert "Contains JSON arrays" in evidence.notes

    def test_header_date_like_flag(self) -> None:
        """Column named like a date should have header_date_like=True."""
        series = pd.Series([100, 200, 300])
        evidence = extract_column_evidence(series, "2024-01")

        assert evidence.header_date_like is True
        assert "Column name appears to be a date" in evidence.notes

    def test_unique_count(self) -> None:
        """unique_count should be computed correctly."""
        series = pd.Series([1, 1, 2, 2, 3, None])
        evidence = extract_column_evidence(series, "col")

        assert evidence.unique_count == 3


class TestValueProfile:
    """Tests for value_profile computation."""

    def test_numeric_value_profile(self) -> None:
        """Value profile should contain numeric statistics."""
        series = pd.Series([0.0, 0.5, 1.0, 50.0, 100.0])
        evidence = extract_column_evidence(series, "values")

        assert evidence.value_profile.min_value == 0.0
        assert evidence.value_profile.max_value == 100.0
        assert evidence.value_profile.mean == pytest.approx(30.3)

    def test_bounded_0_1_detection(self) -> None:
        """Should detect values bounded in [0, 1]."""
        series = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
        evidence = extract_column_evidence(series, "probability")

        assert evidence.value_profile.bounded_0_1_ratio == 1.0

    def test_bounded_0_100_detection(self) -> None:
        """Should detect values bounded in [0, 100]."""
        series = pd.Series([0, 25, 50, 75, 100])
        evidence = extract_column_evidence(series, "percentage")

        assert evidence.value_profile.bounded_0_100_ratio == 1.0

    def test_integer_ratio_detection(self) -> None:
        """Should detect ratio of integer values."""
        series = pd.Series([1.0, 2.5, 3.0, 4.0, 5.5])
        evidence = extract_column_evidence(series, "mixed")

        # 3 out of 5 are integers (1.0, 3.0, 4.0)
        assert evidence.value_profile.integer_ratio == pytest.approx(0.6)

    def test_low_cardinality_flag(self) -> None:
        """Should set low_cardinality flag for <=5 unique values."""
        series = pd.Series([1, 2, 3, 1, 2, 3])
        evidence = extract_column_evidence(series, "codes")

        assert evidence.value_profile.low_cardinality is True

    def test_mostly_null_flag(self) -> None:
        """Should set mostly_null flag for >80% nulls."""
        series = pd.Series([None, None, None, None, None, 1])  # 5/6 = 83% nulls
        evidence = extract_column_evidence(series, "sparse")

        assert evidence.value_profile.mostly_null is True


class TestArrayProfile:
    """Tests for array_profile computation."""

    def test_array_profile_computed(self) -> None:
        """Array profile should be computed for ARRAY structural type."""
        series = pd.Series(['[1, 2, 3]', '[4, 5, 6]', '[7, 8, 9]'])
        evidence = extract_column_evidence(series, "arrays")

        assert evidence.array_profile is not None
        assert evidence.array_profile.avg_length == 3.0
        assert evidence.array_profile.min_length == 3
        assert evidence.array_profile.max_length == 3
        assert evidence.array_profile.consistent_length is True

    def test_array_profile_inconsistent_length(self) -> None:
        """Should detect inconsistent array lengths."""
        series = pd.Series(['[1, 2]', '[3, 4, 5, 6]', '[7]'])
        evidence = extract_column_evidence(series, "arrays")

        assert evidence.array_profile is not None
        assert evidence.array_profile.min_length == 1
        assert evidence.array_profile.max_length == 4
        assert evidence.array_profile.consistent_length is False


class TestStricterBooleanDetection:
    """Tests for stricter boolean detection rules."""

    def test_boolean_requires_two_or_fewer_unique(self) -> None:
        """Boolean detection requires <=2 unique values."""
        # 3 unique values should NOT be boolean even if all in BOOL_VALUES
        series = pd.Series(["yes", "no", "true"])
        ptype, notes = detect_primitive_type(series)
        # With 3 unique values, should not detect as boolean
        assert ptype == PrimitiveType.STRING

    def test_0_1_with_more_values_not_boolean(self) -> None:
        """Numeric 0/1 columns with >2 values should not be boolean."""
        series = pd.Series(["0", "1", "0", "1", "2"])  # Has 0, 1, 2
        ptype, notes = detect_primitive_type(series)
        # Should be INTEGER not BOOLEAN because of "2"
        assert ptype == PrimitiveType.INTEGER

    def test_valid_boolean_true_false(self) -> None:
        """Standard true/false strings should be BOOLEAN."""
        series = pd.Series(["true", "false", "true", "false"])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.BOOLEAN

    def test_valid_boolean_yes_no(self) -> None:
        """Standard yes/no strings should be BOOLEAN."""
        series = pd.Series(["yes", "no", "yes", "no"])
        ptype, notes = detect_primitive_type(series)
        assert ptype == PrimitiveType.BOOLEAN
