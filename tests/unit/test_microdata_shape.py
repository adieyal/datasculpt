"""Unit tests for microdata shape scoring.

Tests for the score_microdata function which detects survey/observation
microdata patterns in datasets.
"""

from __future__ import annotations

from datasculpt.core.shapes import (
    compare_hypotheses,
    detect_shape,
    score_microdata,
)
from datasculpt.core.types import (
    ColumnEvidence,
    InferenceConfig,
    PrimitiveType,
    Role,
    ShapeHypothesis,
    StructuralType,
    ValueProfile,
)


def make_evidence(
    name: str = "test_col",
    primitive_type: PrimitiveType = PrimitiveType.STRING,
    structural_type: StructuralType = StructuralType.SCALAR,
    role_scores: dict[Role, float] | None = None,
    distinct_ratio: float = 0.5,
    null_rate: float = 0.0,
    unique_count: int = 100,
    value_profile: ValueProfile | None = None,
) -> ColumnEvidence:
    """Helper to create ColumnEvidence for testing."""
    return ColumnEvidence(
        name=name,
        primitive_type=primitive_type,
        structural_type=structural_type,
        role_scores=role_scores or {},
        distinct_ratio=distinct_ratio,
        null_rate=null_rate,
        unique_count=unique_count,
        value_profile=value_profile or ValueProfile(),
    )


def make_nlss_like_columns() -> list[ColumnEvidence]:
    """Create NLSS-like survey microdata columns.

    Nigerian Living Standards Survey (NLSS) style data with:
    - Hierarchical IDs (hhid, indiv)
    - Geography hierarchy (zone, state, lga)
    - Enumeration area (ea)
    - Survey weight
    - Coded question columns (s1aq1, s1aq2, ...)
    """
    columns = [
        # ID columns
        make_evidence(
            "hhid",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.7,
            role_scores={Role.RESPONDENT_ID: 0.8, Role.KEY: 0.6},
        ),
        make_evidence(
            "indiv",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
            role_scores={Role.SUBUNIT_ID: 0.7},
        ),
        # Geography hierarchy
        make_evidence(
            "zone",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.01,
            role_scores={Role.GEOGRAPHY_LEVEL: 0.8, Role.DIMENSION: 0.5},
        ),
        make_evidence(
            "state",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.05,
            role_scores={Role.GEOGRAPHY_LEVEL: 0.8, Role.DIMENSION: 0.5},
        ),
        make_evidence(
            "lga",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.1,
            role_scores={Role.GEOGRAPHY_LEVEL: 0.8, Role.DIMENSION: 0.4},
        ),
        # Cluster
        make_evidence(
            "ea",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.15,
            role_scores={Role.CLUSTER_ID: 0.7},
        ),
        # Weight
        make_evidence(
            "weight",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.3,
            role_scores={Role.SURVEY_WEIGHT: 0.7, Role.MEASURE: 0.3},
            value_profile=ValueProfile(non_negative_ratio=1.0),
        ),
    ]

    # Add 30+ question columns (s1aq1 through s1aq35)
    for i in range(1, 36):
        columns.append(
            make_evidence(
                f"s1aq{i}",
                primitive_type=PrimitiveType.STRING,
                distinct_ratio=0.03,
                unique_count=10,
                role_scores={Role.QUESTION_RESPONSE: 0.6, Role.DIMENSION: 0.2},
            )
        )

    return columns


def make_dhs_like_columns() -> list[ColumnEvidence]:
    """Create DHS-like survey microdata columns.

    Demographic and Health Survey (DHS) style data with:
    - Case ID (caseid)
    - Cluster (v001)
    - Household number (v002)
    - DHS-style question codes (v101, v102, hv001, mv001)
    """
    columns = [
        # ID columns
        make_evidence(
            "caseid",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.9,
            role_scores={Role.RESPONDENT_ID: 0.7, Role.KEY: 0.7},
        ),
        make_evidence(
            "v001",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
            role_scores={Role.CLUSTER_ID: 0.6},
        ),
        make_evidence(
            "v002",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.2,
        ),
        # Region
        make_evidence(
            "region",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.02,
            role_scores={Role.GEOGRAPHY_LEVEL: 0.7},
        ),
        make_evidence(
            "district",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.08,
            role_scores={Role.GEOGRAPHY_LEVEL: 0.7},
        ),
        # Weight
        make_evidence(
            "v005",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.25,
            role_scores={Role.SURVEY_WEIGHT: 0.5},
            value_profile=ValueProfile(non_negative_ratio=1.0),
        ),
    ]

    # Add DHS-style question columns (v101 through v140)
    for i in range(101, 141):
        columns.append(
            make_evidence(
                f"v{i}",
                primitive_type=PrimitiveType.STRING,
                distinct_ratio=0.02,
                unique_count=8,
                role_scores={Role.QUESTION_RESPONSE: 0.5},
            )
        )

    # Add household variables (hv001 through hv020)
    for i in range(1, 21):
        columns.append(
            make_evidence(
                f"hv{i:03d}",
                primitive_type=PrimitiveType.STRING,
                distinct_ratio=0.04,
                unique_count=12,
                role_scores={Role.QUESTION_RESPONSE: 0.5},
            )
        )

    return columns


def make_indicator_data_columns() -> list[ColumnEvidence]:
    """Create indicator/value pair data (NOT microdata).

    Classic long-format indicator data with country, year, indicator name, and value.
    """
    return [
        make_evidence(
            "country",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.05,
            role_scores={Role.DIMENSION: 0.8},
        ),
        make_evidence(
            "year",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.02,
            role_scores={Role.TIME: 0.8},
        ),
        make_evidence(
            "indicator_name",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.03,
            role_scores={Role.INDICATOR_NAME: 0.9},
        ),
        make_evidence(
            "value",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.9,
            role_scores={Role.VALUE: 0.9, Role.MEASURE: 0.4},
        ),
    ]


class TestScoreMicrodata:
    """Tests for score_microdata function."""

    def test_nlss_like_data_scores_high(self) -> None:
        """NLSS-style microdata should score > 0.7."""
        columns = make_nlss_like_columns()
        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert result.hypothesis == ShapeHypothesis.MICRODATA
        assert result.score > 0.7
        # Check for expected positive signals in reasons
        assert any("column count" in r.lower() for r in result.reasons)
        assert any("question column" in r.lower() for r in result.reasons)
        assert any("respondent id" in r.lower() for r in result.reasons)

    def test_dhs_like_data_scores_high(self) -> None:
        """DHS-style microdata should score > 0.6."""
        columns = make_dhs_like_columns()
        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert result.hypothesis == ShapeHypothesis.MICRODATA
        assert result.score > 0.6

    def test_indicator_data_scores_low(self) -> None:
        """Indicator/value data should score < 0.3 for microdata."""
        columns = make_indicator_data_columns()
        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert result.score < 0.3
        # Should have penalty reason for indicator/value pattern
        assert any("indicator/value" in r.lower() for r in result.reasons)

    def test_high_column_count_increases_score(self) -> None:
        """50+ columns should increase microdata score by 0.25."""
        # Create 60 generic columns (no other microdata signals)
        columns = [
            make_evidence(f"col_{i}", distinct_ratio=0.5)
            for i in range(60)
        ]
        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("high column count" in r.lower() for r in result.reasons)

    def test_moderate_column_count_increases_score(self) -> None:
        """30-49 columns should increase microdata score by 0.15."""
        # Create 35 columns
        columns = [
            make_evidence(f"col_{i}", distinct_ratio=0.5)
            for i in range(35)
        ]
        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("moderate column count" in r.lower() for r in result.reasons)

    def test_low_column_count_penalizes(self) -> None:
        """<15 columns should penalize microdata score."""
        columns = [
            make_evidence(f"col_{i}", distinct_ratio=0.5)
            for i in range(10)
        ]
        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("low column count" in r.lower() for r in result.reasons)

    def test_question_pattern_detection_lsms(self) -> None:
        """LSMS-style question patterns (s1aq1) should be detected."""
        columns = [
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.8}),
            make_evidence("s1aq1", distinct_ratio=0.02),
            make_evidence("s1aq2", distinct_ratio=0.02),
            make_evidence("s1aq3", distinct_ratio=0.02),
            make_evidence("s2bq1", distinct_ratio=0.02),
            make_evidence("s2bq2", distinct_ratio=0.02),
        ]

        # Add more columns to avoid low column count penalty
        for i in range(20):
            columns.append(make_evidence(f"extra_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        # Should detect question patterns
        assert any("question column" in r.lower() for r in result.reasons)

    def test_question_pattern_detection_dhs(self) -> None:
        """DHS-style question patterns (v101, hv001) should be detected."""
        columns = [
            make_evidence("caseid", role_scores={Role.RESPONDENT_ID: 0.7}),
            make_evidence("v101", distinct_ratio=0.02),
            make_evidence("v102", distinct_ratio=0.02),
            make_evidence("hv001", distinct_ratio=0.02),
            make_evidence("hv002", distinct_ratio=0.02),
            make_evidence("mv001", distinct_ratio=0.02),
        ]

        # Add more columns
        for i in range(20):
            columns.append(make_evidence(f"extra_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        # Question patterns should contribute to score
        assert result.score > 0.0

    def test_respondent_id_detection(self) -> None:
        """Respondent ID columns (hhid) should be detected."""
        columns = [
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.8}),
            make_evidence("s1aq1", role_scores={Role.QUESTION_RESPONSE: 0.6}),
        ]

        # Add padding columns
        for i in range(20):
            columns.append(make_evidence(f"col_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("respondent id" in r.lower() for r in result.reasons)

    def test_subunit_id_detection(self) -> None:
        """Subunit ID columns (indiv) should be detected."""
        columns = [
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.7}),
            make_evidence("indiv", role_scores={Role.SUBUNIT_ID: 0.7}),
        ]

        # Add padding
        for i in range(20):
            columns.append(make_evidence(f"col_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("subunit id" in r.lower() for r in result.reasons)

    def test_cluster_id_detection(self) -> None:
        """Cluster ID columns (ea) should be detected."""
        columns = [
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.7}),
            make_evidence("ea", role_scores={Role.CLUSTER_ID: 0.7}),
        ]

        # Add padding
        for i in range(20):
            columns.append(make_evidence(f"col_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("cluster" in r.lower() for r in result.reasons)

    def test_geography_hierarchy_detection(self) -> None:
        """Geography hierarchy (2+ levels) should be detected."""
        columns = [
            make_evidence("zone", role_scores={Role.GEOGRAPHY_LEVEL: 0.8}),
            make_evidence("state", role_scores={Role.GEOGRAPHY_LEVEL: 0.8}),
            make_evidence("lga", role_scores={Role.GEOGRAPHY_LEVEL: 0.7}),
        ]

        # Add padding
        for i in range(20):
            columns.append(make_evidence(f"col_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("geography hierarchy" in r.lower() for r in result.reasons)

    def test_weight_column_detection(self) -> None:
        """Survey weight column should be detected."""
        columns = [
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.7}),
            make_evidence(
                "weight",
                primitive_type=PrimitiveType.NUMBER,
                role_scores={Role.SURVEY_WEIGHT: 0.7},
            ),
        ]

        # Add padding
        for i in range(20):
            columns.append(make_evidence(f"col_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("weight" in r.lower() for r in result.reasons)

    def test_low_cardinality_categoricals(self) -> None:
        """Many low-cardinality string columns should boost score."""
        columns = []

        # Add low-cardinality categorical columns (distinct_ratio < 0.05)
        for i in range(15):
            columns.append(
                make_evidence(
                    f"cat_{i}",
                    primitive_type=PrimitiveType.STRING,
                    distinct_ratio=0.02,
                )
            )

        # Add some other columns
        for i in range(10):
            columns.append(make_evidence(f"other_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("low-cardinality" in r.lower() for r in result.reasons)

    def test_no_question_columns_penalizes(self) -> None:
        """No coded question columns should penalize score."""
        columns = [
            make_evidence("id"),
            make_evidence("name"),
            make_evidence("value"),
        ]

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("no coded question" in r.lower() for r in result.reasons)

    def test_no_id_columns_penalizes(self) -> None:
        """No respondent/unit ID columns should penalize score."""
        # Create columns without any ID patterns
        columns = [
            make_evidence("col1"),
            make_evidence("col2"),
            make_evidence("col3"),
        ]

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert any("no respondent/unit id" in r.lower() for r in result.reasons)

    def test_indicator_value_pattern_strongly_penalizes(self) -> None:
        """Indicator/value pattern should apply -0.35 penalty."""
        # Create data with indicator/value pattern
        columns = [
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.7}),
            make_evidence(
                "indicator",
                role_scores={Role.INDICATOR_NAME: 0.8},
            ),
            make_evidence(
                "value",
                primitive_type=PrimitiveType.NUMBER,
                role_scores={Role.VALUE: 0.8},
            ),
        ]

        # Add some question-like columns
        for i in range(20):
            columns.append(
                make_evidence(f"s1aq{i}", role_scores={Role.QUESTION_RESPONSE: 0.5})
            )

        config = InferenceConfig()
        result = score_microdata(columns, config)

        # Should have significant penalty
        assert any("indicator/value" in r.lower() for r in result.reasons)

    def test_direct_pattern_matching_fallback(self) -> None:
        """Direct pattern matching should work without role_scores."""
        # Create columns with no role_scores but matching patterns
        columns = [
            make_evidence("hhid"),  # Should match RESPONDENT_ID pattern
            make_evidence("indiv"),  # Should match SUBUNIT_ID pattern
            make_evidence("zone"),  # Should match GEOGRAPHY_LEVEL pattern
            make_evidence("state"),  # Should match GEOGRAPHY_LEVEL pattern
            make_evidence("ea"),  # Should match CLUSTER_ID pattern
            make_evidence("weight", primitive_type=PrimitiveType.NUMBER),  # Should match WEIGHT pattern
        ]

        # Add question patterns
        for i in range(30):
            columns.append(make_evidence(f"s1aq{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        # Should still detect microdata patterns via direct pattern matching
        assert result.score > 0.5


class TestMicrodataDetection:
    """Tests for microdata detection in full detection pipeline."""

    def test_microdata_detected_over_other_shapes(self) -> None:
        """NLSS-like data should be detected as MICRODATA shape."""
        columns = make_nlss_like_columns()
        result = detect_shape(columns)

        assert result.selected == ShapeHypothesis.MICRODATA
        assert result.confidence > 0.5

    def test_indicator_data_not_detected_as_microdata(self) -> None:
        """Indicator data should NOT be detected as MICRODATA."""
        columns = make_indicator_data_columns()
        result = detect_shape(columns)

        assert result.selected != ShapeHypothesis.MICRODATA
        # Should be LONG_INDICATORS instead
        assert result.selected == ShapeHypothesis.LONG_INDICATORS

    def test_microdata_in_compare_hypotheses(self) -> None:
        """MICRODATA should be included in compare_hypotheses results."""
        columns = make_nlss_like_columns()
        ranked = compare_hypotheses(columns)

        hypotheses = [h.hypothesis for h in ranked]
        assert ShapeHypothesis.MICRODATA in hypotheses

    def test_microdata_ranked_first_for_survey_data(self) -> None:
        """For survey data, MICRODATA should rank first."""
        columns = make_nlss_like_columns()
        ranked = compare_hypotheses(columns)

        assert ranked[0].hypothesis == ShapeHypothesis.MICRODATA


class TestEdgeCases:
    """Edge case tests for microdata detection."""

    def test_empty_columns(self) -> None:
        """Empty column list should not crash."""
        config = InferenceConfig()
        result = score_microdata([], config)

        assert result.score == 0.0

    def test_single_column(self) -> None:
        """Single column should handle gracefully."""
        columns = [make_evidence("hhid")]
        config = InferenceConfig()
        result = score_microdata(columns, config)

        # Should have low column count penalty
        assert result.score < 0.3

    def test_mixed_signals(self) -> None:
        """Mixed microdata + indicator signals should be handled."""
        columns = [
            # Microdata signals
            make_evidence("hhid", role_scores={Role.RESPONDENT_ID: 0.8}),
            make_evidence("zone", role_scores={Role.GEOGRAPHY_LEVEL: 0.7}),
            make_evidence("state", role_scores={Role.GEOGRAPHY_LEVEL: 0.7}),
            # Indicator signals
            make_evidence("indicator", role_scores={Role.INDICATOR_NAME: 0.6}),
            make_evidence("value", role_scores={Role.VALUE: 0.6}),
        ]

        # Add more columns
        for i in range(20):
            columns.append(make_evidence(f"col_{i}"))

        config = InferenceConfig()
        result = score_microdata(columns, config)

        # Should have both positive and negative signals
        assert any("indicator/value" in r.lower() for r in result.reasons)

    def test_score_clamping(self) -> None:
        """Score should be clamped to [0, 1] range."""
        # Very poor candidate (many penalties)
        columns = [
            make_evidence("col1", role_scores={Role.INDICATOR_NAME: 0.8}),
            make_evidence("col2", role_scores={Role.VALUE: 0.8}),
        ]

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert 0.0 <= result.score <= 1.0

    def test_very_strong_candidate(self) -> None:
        """Very strong microdata candidate should not exceed 1.0."""
        # Create ideal microdata with all positive signals
        columns = make_nlss_like_columns()

        # Add even more question columns to push the score higher
        for i in range(50, 100):
            columns.append(
                make_evidence(
                    f"s2aq{i}",
                    primitive_type=PrimitiveType.STRING,
                    distinct_ratio=0.02,
                    role_scores={Role.QUESTION_RESPONSE: 0.7},
                )
            )

        config = InferenceConfig()
        result = score_microdata(columns, config)

        assert result.score <= 1.0
