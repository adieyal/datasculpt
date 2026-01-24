"""Unit tests for role scoring module."""

from __future__ import annotations

import pytest

from datasculpt.core.roles import (
    RoleAssignment,
    assign_roles,
    calculate_confidence,
    compute_role_scores,
    resolve_role,
    score_dimension_role,
    score_indicator_name_role,
    score_key_role,
    score_measure_role,
    score_series_role,
    score_time_role,
    score_value_role,
)
from datasculpt.core.types import (
    ColumnEvidence,
    InferenceConfig,
    PrimitiveType,
    Role,
    StructuralType,
)


def make_evidence(
    name: str = "test_col",
    primitive_type: PrimitiveType = PrimitiveType.STRING,
    structural_type: StructuralType = StructuralType.SCALAR,
    null_rate: float = 0.0,
    distinct_ratio: float = 0.5,
    parse_results: dict[str, float] | None = None,
    role_scores: dict[Role, float] | None = None,
) -> ColumnEvidence:
    """Helper to create ColumnEvidence for testing."""
    return ColumnEvidence(
        name=name,
        primitive_type=primitive_type,
        structural_type=structural_type,
        null_rate=null_rate,
        distinct_ratio=distinct_ratio,
        parse_results=parse_results or {},
        role_scores=role_scores or {},
    )


class TestScoreKeyRole:
    """Tests for score_key_role function."""

    def test_high_cardinality_low_null_id_name(self) -> None:
        """Perfect key candidate: high cardinality, no nulls, _id suffix."""
        evidence = make_evidence(
            name="user_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.95,
            null_rate=0.0,
        )
        score = score_key_role(evidence)
        assert score >= 0.8  # High score for ideal key

    def test_low_cardinality(self) -> None:
        """Low cardinality columns are poor key candidates."""
        evidence = make_evidence(
            name="status",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.05,
            null_rate=0.0,
        )
        score = score_key_role(evidence)
        assert score < 0.5

    def test_high_null_rate(self) -> None:
        """High null rate reduces key score."""
        evidence = make_evidence(
            name="optional_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.95,
            null_rate=0.5,
        )
        score = score_key_role(evidence)
        # Should be lower than perfect candidate
        perfect = make_evidence(
            name="optional_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.95,
            null_rate=0.0,
        )
        assert score < score_key_role(perfect)

    def test_key_naming_patterns(self) -> None:
        """Various key naming patterns should boost score."""
        patterns = ["user_id", "id", "country_code", "pk", "item_key"]
        for name in patterns:
            evidence = make_evidence(name=name, distinct_ratio=0.8)
            score = score_key_role(evidence)
            assert score > 0.3, f"Pattern {name} should boost key score"


class TestScoreDimensionRole:
    """Tests for score_dimension_role function."""

    def test_low_cardinality_string(self) -> None:
        """Low cardinality string is ideal dimension."""
        evidence = make_evidence(
            name="country",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.05,
            null_rate=0.0,
        )
        score = score_dimension_role(evidence)
        assert score >= 0.7

    def test_high_cardinality_penalty(self) -> None:
        """High cardinality columns are penalized as dimensions."""
        evidence = make_evidence(
            name="description",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.9,
        )
        score = score_dimension_role(evidence)
        assert score < 0.5

    def test_numeric_type_lower_score(self) -> None:
        """Numeric types are less typical for dimensions."""
        string_evidence = make_evidence(
            name="category",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.05,
        )
        int_evidence = make_evidence(
            name="category",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.05,
        )
        assert score_dimension_role(string_evidence) > score_dimension_role(int_evidence)


class TestScoreMeasureRole:
    """Tests for score_measure_role function."""

    def test_numeric_type_required(self) -> None:
        """Non-numeric types should score 0 for measure role."""
        string_evidence = make_evidence(
            name="amount",
            primitive_type=PrimitiveType.STRING,
        )
        assert score_measure_role(string_evidence) == 0.0

    def test_integer_type(self) -> None:
        """Integer type is valid measure."""
        evidence = make_evidence(
            name="count",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.5,
        )
        score = score_measure_role(evidence)
        assert score > 0.5

    def test_number_type(self) -> None:
        """Number (float) type is valid measure."""
        evidence = make_evidence(
            name="price",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.8,
        )
        score = score_measure_role(evidence)
        assert score > 0.5

    def test_constant_value_lower_score(self) -> None:
        """Constant values are less useful as measures."""
        varying = make_evidence(
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.5,
        )
        constant = make_evidence(
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.001,  # Nearly constant
        )
        assert score_measure_role(varying) > score_measure_role(constant)


class TestScoreTimeRole:
    """Tests for score_time_role function."""

    def test_date_type(self) -> None:
        """DATE primitive type scores high for time role."""
        evidence = make_evidence(
            name="created_at",
            primitive_type=PrimitiveType.DATE,
        )
        score = score_time_role(evidence)
        assert score >= 0.5

    def test_datetime_type(self) -> None:
        """DATETIME primitive type scores high for time role."""
        evidence = make_evidence(
            name="timestamp",
            primitive_type=PrimitiveType.DATETIME,
        )
        score = score_time_role(evidence)
        assert score >= 0.5

    def test_time_naming_patterns(self) -> None:
        """Time naming patterns boost score."""
        patterns = ["date", "created_time", "period", "year", "month", "day", "dt"]
        for name in patterns:
            evidence = make_evidence(name=name)
            score = score_time_role(evidence)
            assert score > 0.1, f"Pattern {name} should boost time score"

    def test_string_with_no_date_parse(self) -> None:
        """String without date parse success scores low."""
        evidence = make_evidence(
            name="random",
            primitive_type=PrimitiveType.STRING,
        )
        score = score_time_role(evidence)
        assert score < 0.3


class TestScoreIndicatorNameRole:
    """Tests for score_indicator_name_role function."""

    def test_string_type_required(self) -> None:
        """Non-string types should score 0."""
        evidence = make_evidence(
            name="indicator",
            primitive_type=PrimitiveType.INTEGER,
        )
        assert score_indicator_name_role(evidence) == 0.0

    def test_low_cardinality_string(self) -> None:
        """Low cardinality string with indicator name scores high."""
        evidence = make_evidence(
            name="indicator",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.02,
            null_rate=0.0,
        )
        score = score_indicator_name_role(evidence)
        assert score >= 0.6

    def test_naming_patterns(self) -> None:
        """Indicator naming patterns boost score."""
        patterns = ["indicator", "metric_name", "measure", "variable", "series_name"]
        for name in patterns:
            evidence = make_evidence(
                name=name,
                primitive_type=PrimitiveType.STRING,
                distinct_ratio=0.05,
            )
            score = score_indicator_name_role(evidence)
            assert score > 0.4, f"Pattern {name} should boost indicator score"


class TestScoreValueRole:
    """Tests for score_value_role function."""

    def test_numeric_type_required(self) -> None:
        """Non-numeric types should score 0."""
        evidence = make_evidence(
            name="value",
            primitive_type=PrimitiveType.STRING,
        )
        assert score_value_role(evidence) == 0.0

    def test_numeric_with_indicator(self) -> None:
        """Numeric with indicator column present scores high."""
        evidence = make_evidence(
            name="value",
            primitive_type=PrimitiveType.NUMBER,
        )
        score = score_value_role(evidence, has_indicator_column=True)
        assert score >= 0.6

    def test_numeric_without_indicator(self) -> None:
        """Numeric without indicator column present scores lower."""
        evidence = make_evidence(
            name="amount",
            primitive_type=PrimitiveType.NUMBER,
        )
        score_with = score_value_role(evidence, has_indicator_column=True)
        score_without = score_value_role(evidence, has_indicator_column=False)
        assert score_with > score_without

    def test_value_naming_patterns(self) -> None:
        """Value naming patterns boost score."""
        patterns = ["value", "val", "amount", "obs", "observation"]
        for name in patterns:
            evidence = make_evidence(
                name=name,
                primitive_type=PrimitiveType.NUMBER,
            )
            score = score_value_role(evidence)
            assert score > 0.5, f"Pattern {name} should boost value score"


class TestScoreSeriesRole:
    """Tests for score_series_role function."""

    def test_array_structural_type(self) -> None:
        """Array structural type scores high for series role."""
        evidence = make_evidence(
            name="data",
            structural_type=StructuralType.ARRAY,
        )
        score = score_series_role(evidence)
        assert score >= 0.5

    def test_json_array_parse_success(self) -> None:
        """High JSON array parse success boosts score."""
        evidence = make_evidence(
            name="values",
            parse_results={"json_array": 0.95},
        )
        score = score_series_role(evidence)
        assert score >= 0.4

    def test_scalar_no_array_parse(self) -> None:
        """Scalar without array parse scores low."""
        evidence = make_evidence(
            name="regular_column",
            structural_type=StructuralType.SCALAR,
        )
        score = score_series_role(evidence)
        assert score < 0.3


class TestCalculateConfidence:
    """Tests for calculate_confidence function."""

    def test_large_gap_high_confidence(self) -> None:
        """Large gap between top scores yields high confidence."""
        scores = {
            Role.KEY: 0.9,
            Role.DIMENSION: 0.3,
            Role.MEASURE: 0.2,
        }
        confidence = calculate_confidence(scores)
        assert confidence > 0.5

    def test_small_gap_low_confidence(self) -> None:
        """Small gap between top scores yields low confidence."""
        scores = {
            Role.KEY: 0.5,
            Role.DIMENSION: 0.48,
            Role.MEASURE: 0.2,
        }
        confidence = calculate_confidence(scores)
        assert confidence < 0.2

    def test_empty_scores(self) -> None:
        """Empty scores dict returns 0 confidence."""
        assert calculate_confidence({}) == 0.0

    def test_single_score(self) -> None:
        """Single score with positive value returns 1.0 confidence."""
        scores = {Role.KEY: 0.8}
        assert calculate_confidence(scores) == 1.0

    def test_all_zero_scores(self) -> None:
        """All zero scores return 0 confidence."""
        scores = {
            Role.KEY: 0.0,
            Role.DIMENSION: 0.0,
        }
        assert calculate_confidence(scores) == 0.0


class TestResolveRole:
    """Tests for resolve_role function."""

    def test_returns_role_assignment(self) -> None:
        """resolve_role returns RoleAssignment dataclass."""
        evidence = make_evidence(
            name="user_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.95,
        )
        result = resolve_role(evidence)
        assert isinstance(result, RoleAssignment)
        assert result.role in Role
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0

    def test_key_column_resolved_correctly(self) -> None:
        """High cardinality integer with _id name resolves to KEY."""
        evidence = make_evidence(
            name="user_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.98,
            null_rate=0.0,
        )
        result = resolve_role(evidence)
        assert result.role == Role.KEY

    def test_measure_column_resolved_correctly(self) -> None:
        """Numeric column with high cardinality resolves to MEASURE."""
        evidence = make_evidence(
            name="revenue",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.7,
        )
        result = resolve_role(evidence)
        assert result.role == Role.MEASURE

    def test_all_scores_populated(self) -> None:
        """all_scores dict contains all roles."""
        evidence = make_evidence()
        result = resolve_role(evidence)
        assert Role.KEY in result.all_scores
        assert Role.DIMENSION in result.all_scores
        assert Role.MEASURE in result.all_scores
        assert Role.TIME in result.all_scores


class TestComputeRoleScores:
    """Tests for compute_role_scores function."""

    def test_returns_all_roles(self) -> None:
        """compute_role_scores returns scores for all roles."""
        evidence = make_evidence()
        scores = compute_role_scores(evidence)

        expected_roles = {
            Role.KEY,
            Role.DIMENSION,
            Role.MEASURE,
            Role.TIME,
            Role.INDICATOR_NAME,
            Role.VALUE,
            Role.SERIES,
            Role.METADATA,
        }
        assert set(scores.keys()) == expected_roles

    def test_all_scores_in_valid_range(self) -> None:
        """All scores should be between 0 and 1."""
        evidence = make_evidence()
        scores = compute_role_scores(evidence)

        for role, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{role} score {score} out of range"


class TestAssignRoles:
    """Tests for assign_roles function."""

    def test_assigns_all_columns(self) -> None:
        """assign_roles returns assignment for each input column."""
        evidences = [
            make_evidence(name="id", primitive_type=PrimitiveType.INTEGER, distinct_ratio=0.99),
            make_evidence(name="name", primitive_type=PrimitiveType.STRING, distinct_ratio=0.5),
            make_evidence(name="amount", primitive_type=PrimitiveType.NUMBER, distinct_ratio=0.7),
        ]
        assignments = assign_roles(evidences)

        assert len(assignments) == 3
        assert "id" in assignments
        assert "name" in assignments
        assert "amount" in assignments

    def test_indicator_context_affects_value_scoring(self) -> None:
        """Presence of indicator column affects value column scoring."""
        indicator = make_evidence(
            name="indicator",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.02,
        )
        value_col = make_evidence(
            name="value",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.5,
        )

        # With indicator column
        evidences_with = [indicator, value_col]
        assignments_with = assign_roles(evidences_with)

        # Without indicator column
        evidences_without = [value_col]
        assignments_without = assign_roles(evidences_without)

        # Value column should score higher for VALUE role when indicator present
        value_score_with = assignments_with["value"].all_scores.get(Role.VALUE, 0)
        value_score_without = assignments_without["value"].all_scores.get(Role.VALUE, 0)
        assert value_score_with > value_score_without
