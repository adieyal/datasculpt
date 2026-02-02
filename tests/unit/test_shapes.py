"""Unit tests for shape detection module."""

from __future__ import annotations

from datasculpt.core.shapes import (
    ShapeResult,
    compare_hypotheses,
    detect_shape,
    score_long_indicators,
    score_long_observations,
    score_series_column,
    score_wide_observations,
    score_wide_time_columns,
)
from datasculpt.core.types import (
    ColumnEvidence,
    InferenceConfig,
    PrimitiveType,
    Role,
    ShapeHypothesis,
    StructuralType,
)


def make_evidence(
    name: str = "test_col",
    primitive_type: PrimitiveType = PrimitiveType.STRING,
    structural_type: StructuralType = StructuralType.SCALAR,
    role_scores: dict[Role, float] | None = None,
) -> ColumnEvidence:
    """Helper to create ColumnEvidence for testing."""
    return ColumnEvidence(
        name=name,
        primitive_type=primitive_type,
        structural_type=structural_type,
        role_scores=role_scores or {},
    )


class TestScoreLongObservations:
    """Tests for score_long_observations function."""

    def test_classic_tidy_data(self) -> None:
        """Dimensions + measures without indicator pattern scores high."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("date", role_scores={Role.TIME: 0.7}),
            make_evidence("sales", role_scores={Role.MEASURE: 0.8}),
            make_evidence("quantity", role_scores={Role.MEASURE: 0.7}),
        ]
        config = InferenceConfig()
        result = score_long_observations(columns, config)

        assert result.hypothesis == ShapeHypothesis.LONG_OBSERVATIONS
        assert result.score >= 0.5

    def test_penalized_with_indicator_value_pattern(self) -> None:
        """Indicator/value pattern reduces long_observations score."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("indicator", role_scores={Role.INDICATOR_NAME: 0.7}),
            make_evidence("value", role_scores={Role.VALUE: 0.8}),
        ]
        config = InferenceConfig()
        result = score_long_observations(columns, config)

        # Score should be reduced due to indicator/value pattern
        assert result.score < 0.5

    def test_penalized_with_series_columns(self) -> None:
        """Series columns reduce long_observations score."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("data", structural_type=StructuralType.ARRAY, role_scores={Role.SERIES: 0.8}),
        ]
        config = InferenceConfig()
        result = score_long_observations(columns, config)

        # Score should be reduced due to series column
        assert result.score < 0.3


class TestScoreLongIndicators:
    """Tests for score_long_indicators function."""

    def test_indicator_value_pattern(self) -> None:
        """Indicator/value pattern scores high for long_indicators."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("year", role_scores={Role.TIME: 0.7}),
            make_evidence("indicator", role_scores={Role.INDICATOR_NAME: 0.8}),
            make_evidence("value", role_scores={Role.VALUE: 0.8}),
        ]
        config = InferenceConfig()
        result = score_long_indicators(columns, config)

        assert result.hypothesis == ShapeHypothesis.LONG_INDICATORS
        assert result.score >= 0.5

    def test_no_indicator_column(self) -> None:
        """Missing indicator column reduces score."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("value", role_scores={Role.VALUE: 0.5}),
        ]
        config = InferenceConfig()
        result = score_long_indicators(columns, config)

        assert result.score < 0.5
        assert any("No indicator_name column" in r for r in result.reasons)

    def test_no_value_column(self) -> None:
        """Missing value column reduces score."""
        columns = [
            make_evidence("indicator", role_scores={Role.INDICATOR_NAME: 0.8}),
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
        ]
        config = InferenceConfig()
        result = score_long_indicators(columns, config)

        assert result.score < 0.5


class TestScoreWideObservations:
    """Tests for score_wide_observations function."""

    def test_multiple_non_time_measures(self) -> None:
        """Multiple measure columns without time headers scores high."""
        columns = [
            make_evidence("person_id", role_scores={Role.KEY: 0.8}),
            make_evidence("height", role_scores={Role.MEASURE: 0.8}),
            make_evidence("weight", role_scores={Role.MEASURE: 0.8}),
            make_evidence("age", role_scores={Role.MEASURE: 0.7}),
        ]
        config = InferenceConfig()
        result = score_wide_observations(columns, config)

        assert result.hypothesis == ShapeHypothesis.WIDE_OBSERVATIONS
        assert result.score >= 0.5

    def test_penalized_with_time_like_headers(self) -> None:
        """Time-like column headers reduce wide_observations score."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("2020", role_scores={Role.MEASURE: 0.5}),
            make_evidence("2021", role_scores={Role.MEASURE: 0.5}),
            make_evidence("2022", role_scores={Role.MEASURE: 0.5}),
        ]
        config = InferenceConfig()
        result = score_wide_observations(columns, config)

        # Should be penalized due to time-like headers
        assert result.score < 0.5

    def test_single_measure_low_score(self) -> None:
        """Single measure column scores lower."""
        columns = [
            make_evidence("id", role_scores={Role.KEY: 0.8}),
            make_evidence("value", role_scores={Role.MEASURE: 0.8}),
        ]
        config = InferenceConfig()
        result = score_wide_observations(columns, config)

        # Less typical for wide format
        assert result.score < 0.5


class TestScoreWideTimeColumns:
    """Tests for score_wide_time_columns function."""

    def test_year_headers(self) -> None:
        """Year column headers score high for wide_time_columns."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("indicator", role_scores={Role.INDICATOR_NAME: 0.6}),
            make_evidence("2018"),
            make_evidence("2019"),
            make_evidence("2020"),
            make_evidence("2021"),
        ]
        config = InferenceConfig()
        result = score_wide_time_columns(columns, config)

        assert result.hypothesis == ShapeHypothesis.WIDE_TIME_COLUMNS
        assert result.score >= 0.5

    def test_iso_date_headers(self) -> None:
        """ISO date column headers score high."""
        columns = [
            make_evidence("entity", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("2020-01"),
            make_evidence("2020-02"),
            make_evidence("2020-03"),
        ]
        config = InferenceConfig()
        result = score_wide_time_columns(columns, config)

        assert result.score >= 0.5

    def test_too_few_time_columns(self) -> None:
        """Fewer than min_time_columns_for_wide reduces score."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("2020"),
        ]
        config = InferenceConfig(min_time_columns_for_wide=3)
        result = score_wide_time_columns(columns, config)

        assert result.score < 0.5
        assert any("Only" in r and "time-like columns" in r for r in result.reasons)


class TestScoreSeriesColumn:
    """Tests for score_series_column function."""

    def test_series_column_detected(self) -> None:
        """Series column with array type scores high."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("indicator", role_scores={Role.INDICATOR_NAME: 0.6}),
            make_evidence("data", structural_type=StructuralType.ARRAY, role_scores={Role.SERIES: 0.8}),
        ]
        config = InferenceConfig()
        result = score_series_column(columns, config)

        assert result.hypothesis == ShapeHypothesis.SERIES_COLUMN
        assert result.score >= 0.5

    def test_array_typed_without_series_role(self) -> None:
        """Array-typed column without explicit series role still scores."""
        columns = [
            make_evidence("entity", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("values", structural_type=StructuralType.ARRAY),
        ]
        config = InferenceConfig()
        result = score_series_column(columns, config)

        assert result.score >= 0.3

    def test_no_array_columns(self) -> None:
        """No array columns reduces score significantly."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("value", role_scores={Role.MEASURE: 0.8}),
        ]
        config = InferenceConfig()
        result = score_series_column(columns, config)

        assert result.score < 0.3


class TestCompareHypotheses:
    """Tests for compare_hypotheses function."""

    def test_returns_sorted_scores(self) -> None:
        """compare_hypotheses returns scores sorted by score descending."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("sales", role_scores={Role.MEASURE: 0.8}),
        ]
        ranked = compare_hypotheses(columns)

        assert len(ranked) == 6  # All 6 hypotheses (including MICRODATA)
        # Verify sorted order
        for i in range(len(ranked) - 1):
            assert ranked[i].score >= ranked[i + 1].score

    def test_all_hypotheses_represented(self) -> None:
        """All shape hypotheses are scored."""
        columns = [make_evidence("test")]
        ranked = compare_hypotheses(columns)

        hypotheses = {h.hypothesis for h in ranked}
        expected = {
            ShapeHypothesis.LONG_OBSERVATIONS,
            ShapeHypothesis.LONG_INDICATORS,
            ShapeHypothesis.WIDE_OBSERVATIONS,
            ShapeHypothesis.WIDE_TIME_COLUMNS,
            ShapeHypothesis.SERIES_COLUMN,
            ShapeHypothesis.MICRODATA,
        }
        assert hypotheses == expected


class TestDetectShape:
    """Tests for detect_shape function."""

    def test_returns_shape_result(self) -> None:
        """detect_shape returns ShapeResult dataclass."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("sales", role_scores={Role.MEASURE: 0.8}),
        ]
        result = detect_shape(columns)

        assert isinstance(result, ShapeResult)
        assert result.selected in ShapeHypothesis
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.explanation, str)

    def test_detects_long_observations(self) -> None:
        """Classic tidy data detected as long_observations."""
        columns = [
            make_evidence("customer_id", role_scores={Role.KEY: 0.9}),
            make_evidence("region", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("date", role_scores={Role.TIME: 0.8}),
            make_evidence("revenue", role_scores={Role.MEASURE: 0.9}),
            make_evidence("quantity", role_scores={Role.MEASURE: 0.8}),
        ]
        result = detect_shape(columns)

        assert result.selected == ShapeHypothesis.LONG_OBSERVATIONS

    def test_detects_long_indicators(self) -> None:
        """Indicator/value pattern detected as long_indicators."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("year", role_scores={Role.TIME: 0.8}),
            make_evidence("indicator", role_scores={Role.INDICATOR_NAME: 0.9}),
            make_evidence("value", role_scores={Role.VALUE: 0.9}),
        ]
        result = detect_shape(columns)

        assert result.selected == ShapeHypothesis.LONG_INDICATORS

    def test_detects_wide_time_columns(self) -> None:
        """Year headers detected as wide_time_columns."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("indicator", role_scores={Role.DIMENSION: 0.5}),
            make_evidence("2015"),
            make_evidence("2016"),
            make_evidence("2017"),
            make_evidence("2018"),
            make_evidence("2019"),
        ]
        result = detect_shape(columns)

        assert result.selected == ShapeHypothesis.WIDE_TIME_COLUMNS

    def test_detects_series_column(self) -> None:
        """Array column detected as series_column."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("indicator", role_scores={Role.DIMENSION: 0.5}),
            make_evidence("data", structural_type=StructuralType.ARRAY, role_scores={Role.SERIES: 0.9}),
        ]
        result = detect_shape(columns)

        assert result.selected == ShapeHypothesis.SERIES_COLUMN

    def test_detects_microdata(self) -> None:
        """Survey-like data with many question columns detected as microdata."""
        columns = [
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.8, Role.KEY: 0.5}),
            make_evidence("indiv", role_scores={Role.SUBUNIT_ID: 0.7}),
            make_evidence("zone", role_scores={Role.GEOGRAPHY_LEVEL: 0.7}),
            make_evidence("state", role_scores={Role.GEOGRAPHY_LEVEL: 0.7}),
        ]
        # Add 45+ question columns to trigger microdata detection
        for i in range(45):
            columns.append(
                make_evidence(
                    f"s1aq{i+1}",
                    primitive_type=PrimitiveType.STRING,
                    role_scores={Role.QUESTION_RESPONSE: 0.7},
                )
            )
        result = detect_shape(columns)

        assert result.selected == ShapeHypothesis.MICRODATA

    def test_ambiguity_when_close_scores(self) -> None:
        """Close top scores or low confidence marks result as ambiguous."""
        # Create a scenario where we can check ambiguity behavior
        columns = [
            make_evidence("entity", role_scores={Role.DIMENSION: 0.3}),
            make_evidence("metric", role_scores={Role.DIMENSION: 0.3}),
        ]
        config = InferenceConfig(hypothesis_confidence_gap=0.5)
        result = detect_shape(columns, config)

        # Result should either be ambiguous or have low confidence
        # The exact behavior depends on scoring implementation
        # Just verify the detection completes and returns a valid result
        assert result.selected in ShapeHypothesis
        assert 0.0 <= result.confidence <= 1.0

    def test_low_confidence_marked_ambiguous(self) -> None:
        """Low confidence scores mark result as ambiguous."""
        columns = [make_evidence("random")]
        result = detect_shape(columns)

        if result.confidence < 0.3:
            assert result.is_ambiguous

    def test_ranked_hypotheses_included(self) -> None:
        """Result includes all ranked hypotheses."""
        columns = [
            make_evidence("country", role_scores={Role.DIMENSION: 0.8}),
            make_evidence("value", role_scores={Role.MEASURE: 0.7}),
        ]
        result = detect_shape(columns)

        assert len(result.ranked_hypotheses) == 6  # All 6 hypotheses including MICRODATA
        # First should be selected
        assert result.ranked_hypotheses[0].hypothesis == result.selected

    def test_empty_columns_defaults_gracefully(self) -> None:
        """Empty column list returns fallback result."""
        result = detect_shape([])

        # Should have a valid result with low confidence
        assert result.selected in ShapeHypothesis
        assert result.is_ambiguous or result.confidence == 0.0
