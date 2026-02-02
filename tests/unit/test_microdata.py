"""Unit tests for microdata detection and metadata extraction.

This module tests:
- T4: Question column detection (score_question_response_role)
- T5: ID column detection (respondent, subunit, cluster, weight scoring)
- T6: Survey metadata extraction (infer_survey_type, infer_section, etc.)
"""

from __future__ import annotations

import pytest

from datasculpt.core.microdata import (
    extract_question_prefix,
    infer_section,
    infer_survey_type,
    infer_survey_type_from_columns,
    is_other_specify_column,
    order_geography_hierarchy,
    parse_question_column,
)
from datasculpt.core.roles import (
    score_cluster_id_role,
    score_question_response_role,
    score_respondent_id_role,
    score_subunit_id_role,
    score_survey_weight_role,
)
from datasculpt.core.types import (
    ColumnEvidence,
    ParseResults,
    PrimitiveType,
    ShapeHypothesis,
    StructuralType,
    ValueProfile,
)


def make_evidence(
    name: str = "test_col",
    primitive_type: PrimitiveType = PrimitiveType.STRING,
    structural_type: StructuralType = StructuralType.SCALAR,
    null_rate: float = 0.0,
    distinct_ratio: float = 0.5,
    unique_count: int = 10,
    non_negative_ratio: float = 1.0,
) -> ColumnEvidence:
    """Helper to create ColumnEvidence for testing."""
    return ColumnEvidence(
        name=name,
        primitive_type=primitive_type,
        structural_type=structural_type,
        null_rate=null_rate,
        distinct_ratio=distinct_ratio,
        unique_count=unique_count,
        value_profile=ValueProfile(non_negative_ratio=non_negative_ratio),
        parse_results=ParseResults(),
        parse_results_dict={},
        role_scores={},
    )


# ============ T6: Survey Metadata Extraction Tests ============


class TestInferSurveyType:
    """Tests for infer_survey_type function."""

    def test_dhs_hv_pattern(self) -> None:
        """hv* patterns indicate DHS household survey."""
        assert infer_survey_type("hv001") == "DHS"
        assert infer_survey_type("hv103") == "DHS"
        assert infer_survey_type("HV001") == "DHS"  # Case insensitive

    def test_dhs_mv_pattern(self) -> None:
        """mv* patterns indicate DHS men's survey."""
        assert infer_survey_type("mv001") == "DHS"
        assert infer_survey_type("mv103") == "DHS"
        assert infer_survey_type("MV001") == "DHS"

    def test_dhs_v_pattern(self) -> None:
        """v* patterns indicate DHS (generic)."""
        assert infer_survey_type("v101") == "DHS"
        assert infer_survey_type("v0401") == "DHS"
        assert infer_survey_type("V101") == "DHS"

    def test_lsms_pattern(self) -> None:
        """s#q# patterns indicate LSMS survey."""
        assert infer_survey_type("s1q1") == "LSMS"
        assert infer_survey_type("s1aq1") == "LSMS"
        assert infer_survey_type("s2bq3") == "LSMS"
        assert infer_survey_type("s01aq01") == "LSMS"
        assert infer_survey_type("S1AQ1") == "LSMS"

    def test_none_for_unrecognized(self) -> None:
        """Unrecognized patterns return None."""
        # Note: "hhid" matches MICS since it starts with "hh"
        assert infer_survey_type("weight") is None
        assert infer_survey_type("country") is None
        assert infer_survey_type("age") is None
        assert infer_survey_type("income") is None

    def test_none_for_empty(self) -> None:
        """Empty or None input returns None."""
        assert infer_survey_type(None) is None
        assert infer_survey_type("") is None


class TestInferSection:
    """Tests for infer_section function."""

    def test_lsms_section_extraction(self) -> None:
        """Extract section from LSMS-style columns."""
        columns = ["s1aq1", "s1aq2", "s1aq3"]
        result = infer_section(columns)
        # Should return section name or number
        assert result is not None
        assert "1" in result or "demo" in result.lower()

    def test_lsms_multiple_sections(self) -> None:
        """Returns most common section when multiple present."""
        columns = ["s1aq1", "s1aq2", "s1aq3", "s2bq1"]
        result = infer_section(columns)
        # Section 1 should win (3 vs 1)
        assert result is not None
        assert "1" in result or "demo" in result.lower()

    def test_dhs_hv_section(self) -> None:
        """Extract section from DHS household columns."""
        columns = ["hv001", "hv002", "hv103"]
        result = infer_section(columns)
        assert result is not None
        assert "household" in result.lower() or "hv" in result.lower()

    def test_handles_os_suffix(self) -> None:
        """Correctly handles _os suffix."""
        columns = ["s1aq1", "s1aq1_os", "s1aq2"]
        result = infer_section(columns)
        assert result is not None

    def test_empty_returns_none(self) -> None:
        """Empty list returns None."""
        assert infer_section([]) is None

    def test_no_pattern_returns_none(self) -> None:
        """Columns without patterns return None."""
        # Use columns that definitely don't match any survey pattern
        columns = ["age", "income", "status"]
        assert infer_section(columns) is None


class TestExtractQuestionPrefix:
    """Tests for extract_question_prefix function."""

    def test_lsms_prefix(self) -> None:
        """Extract LSMS prefix."""
        columns = ["s1aq1", "s1aq2", "s1aq3"]
        result = extract_question_prefix(columns)
        assert result is not None
        assert "s1a" in result or result.startswith("s")

    def test_dhs_hv_prefix(self) -> None:
        """Extract DHS hv prefix."""
        columns = ["hv001", "hv002", "hv103"]
        assert extract_question_prefix(columns) == "hv"

    def test_dhs_mv_prefix(self) -> None:
        """Extract DHS mv prefix."""
        columns = ["mv001", "mv002"]
        assert extract_question_prefix(columns) == "mv"

    def test_dhs_v_prefix(self) -> None:
        """Extract DHS v prefix."""
        columns = ["v101", "v102", "v103"]
        assert extract_question_prefix(columns) == "v"

    def test_handles_os_suffix(self) -> None:
        """Handles _os suffix correctly."""
        columns = ["s1aq1_os", "s1aq2", "s1aq3"]
        result = extract_question_prefix(columns)
        assert result is not None

    def test_empty_returns_none(self) -> None:
        """Empty list returns None."""
        assert extract_question_prefix([]) is None


class TestIsOtherSpecifyColumn:
    """Tests for is_other_specify_column function."""

    def test_detects_os_suffix(self) -> None:
        """Detects _os suffix."""
        assert is_other_specify_column("s1aq1_os") is True
        assert is_other_specify_column("hv101_os") is True
        assert is_other_specify_column("q5_os") is True

    def test_detects_other_suffix(self) -> None:
        """Detects _other suffix."""
        assert is_other_specify_column("s1aq1_other") is True

    def test_case_insensitive(self) -> None:
        """Case insensitive detection."""
        assert is_other_specify_column("s1aq1_OS") is True
        assert is_other_specify_column("S1AQ1_os") is True

    def test_rejects_non_os(self) -> None:
        """Rejects columns without _os suffix."""
        assert is_other_specify_column("s1aq1") is False
        assert is_other_specify_column("hv101") is False
        assert is_other_specify_column("q5") is False


class TestParseQuestionColumn:
    """Tests for parse_question_column function."""

    def test_parses_lsms_pattern(self) -> None:
        """Parses full LSMS pattern."""
        result = parse_question_column("s1aq1")
        assert result["section_code"] == "s1a"
        assert result["question_number"] == "q1"
        assert result["subquestion"] is None

    def test_parses_with_os(self) -> None:
        """Parses pattern with _os suffix."""
        result = parse_question_column("s1aq1_os")
        assert result["section_code"] == "s1a"
        assert result["question_number"] == "q1"
        assert result["subquestion"] is not None
        assert "os" in result["subquestion"].lower()

    def test_parses_dhs_hv_pattern(self) -> None:
        """Parses DHS household pattern."""
        result = parse_question_column("hv001")
        assert result["section_code"] == "hv"
        assert result["question_number"] == "001"

    def test_parses_dhs_v_pattern(self) -> None:
        """Parses DHS generic pattern."""
        result = parse_question_column("v101")
        assert result["section_code"] == "v"
        assert result["question_number"] == "101"


class TestInferSurveyTypeFromColumns:
    """Tests for infer_survey_type_from_columns function."""

    def test_infers_lsms(self) -> None:
        """Infers LSMS from column patterns."""
        columns = ["s1aq1", "s1aq2", "s2bq1"]
        result = infer_survey_type_from_columns(columns)
        assert result == "LSMS"

    def test_infers_dhs(self) -> None:
        """Infers DHS from column patterns."""
        columns = ["v101", "v102", "v103"]
        result = infer_survey_type_from_columns(columns)
        assert result == "DHS"


class TestOrderGeographyHierarchy:
    """Tests for order_geography_hierarchy function."""

    def test_orders_from_broad_to_narrow(self) -> None:
        """Orders geography from broad to narrow."""
        columns = ["ward", "state", "lga", "region"]
        result = order_geography_hierarchy(columns)
        # region > state > lga > ward
        assert result.index("region") < result.index("state")
        assert result.index("state") < result.index("lga")
        assert result.index("lga") < result.index("ward")

    def test_empty_returns_empty(self) -> None:
        """Empty input returns empty list."""
        assert order_geography_hierarchy([]) == []


# ============ T4: Question Column Detection (Role Scoring) Tests ============


class TestScoreQuestionResponseRole:
    """Tests for score_question_response_role function."""

    def test_lsms_patterns_score_high(self) -> None:
        """LSMS patterns should score >0.5 for question role."""
        patterns = ["s1aq1", "s2bq3", "s01aq01", "s10q5"]
        for name in patterns:
            evidence = make_evidence(
                name=name,
                primitive_type=PrimitiveType.INTEGER,
                distinct_ratio=0.05,
                unique_count=10,
            )
            score = score_question_response_role(
                evidence, detected_shape=ShapeHypothesis.MICRODATA
            )
            assert score > 0.5, f"LSMS pattern {name} should score >0.5, got {score}"

    def test_dhs_patterns_score_high(self) -> None:
        """DHS patterns should score >0.5 for question role."""
        patterns = ["v101", "hv001", "mv001", "v0401"]
        for name in patterns:
            evidence = make_evidence(
                name=name,
                primitive_type=PrimitiveType.INTEGER,
                distinct_ratio=0.05,
                unique_count=10,
            )
            score = score_question_response_role(
                evidence, detected_shape=ShapeHypothesis.MICRODATA
            )
            assert score > 0.5, f"DHS pattern {name} should score >0.5, got {score}"

    def test_other_specify_patterns(self) -> None:
        """Patterns with _os suffix should still score as questions."""
        evidence = make_evidence(
            name="s1aq1_os",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.3,
            unique_count=50,
        )
        score = score_question_response_role(
            evidence, detected_shape=ShapeHypothesis.MICRODATA
        )
        # _os columns still match patterns
        assert score >= 0.3

    def test_generic_question_patterns(self) -> None:
        """Generic q# patterns should score as questions."""
        patterns = ["q1", "q01", "question_1"]
        for name in patterns:
            evidence = make_evidence(
                name=name,
                primitive_type=PrimitiveType.INTEGER,
                distinct_ratio=0.05,
                unique_count=5,
            )
            score = score_question_response_role(
                evidence, detected_shape=ShapeHypothesis.MICRODATA
            )
            assert score > 0.5, f"Pattern {name} should score >0.5, got {score}"

    def test_categorical_cardinality_boost(self) -> None:
        """Low cardinality (typical categorical responses) should boost score."""
        low_card = make_evidence(
            name="s1aq1",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.02,
            unique_count=5,
        )
        high_card = make_evidence(
            name="s1aq1",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.5,
            unique_count=500,
        )
        low_score = score_question_response_role(
            low_card, detected_shape=ShapeHypothesis.MICRODATA
        )
        high_score = score_question_response_role(
            high_card, detected_shape=ShapeHypothesis.MICRODATA
        )
        assert low_score > high_score

    def test_non_microdata_shape_lower_score(self) -> None:
        """Question patterns should score lower in non-microdata shapes."""
        evidence = make_evidence(
            name="s1aq1",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.05,
            unique_count=10,
        )
        microdata_score = score_question_response_role(
            evidence, detected_shape=ShapeHypothesis.MICRODATA
        )
        wide_score = score_question_response_role(
            evidence, detected_shape=ShapeHypothesis.WIDE_OBSERVATIONS
        )
        assert microdata_score > wide_score

    def test_id_patterns_penalized(self) -> None:
        """ID-like patterns should be penalized."""
        evidence = make_evidence(
            name="respondent_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.05,
            unique_count=10,
        )
        score = score_question_response_role(
            evidence, detected_shape=ShapeHypothesis.MICRODATA
        )
        assert score < 0.5


# ============ T5: ID Column Detection Tests ============


class TestScoreRespondentIdRole:
    """Tests for score_respondent_id_role function."""

    def test_hhid_scores_high(self) -> None:
        """hhid should score >0.5."""
        evidence = make_evidence(
            name="hhid",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.4,
            null_rate=0.0,
        )
        score = score_respondent_id_role(evidence)
        assert score > 0.5, f"hhid should score >0.5, got {score}"

    def test_hh_id_scores_high(self) -> None:
        """hh_id should score >0.5."""
        evidence = make_evidence(
            name="hh_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.4,
            null_rate=0.0,
        )
        score = score_respondent_id_role(evidence)
        assert score > 0.5, f"hh_id should score >0.5, got {score}"

    def test_person_id_scores_high(self) -> None:
        """person_id should score >0.5."""
        evidence = make_evidence(
            name="person_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.5,
            null_rate=0.0,
        )
        score = score_respondent_id_role(evidence)
        assert score > 0.5, f"person_id should score >0.5, got {score}"

    def test_case_id_scores_high(self) -> None:
        """case_id should score >0.5."""
        evidence = make_evidence(
            name="case_id",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.5,
            null_rate=0.0,
        )
        score = score_respondent_id_role(evidence)
        assert score > 0.5, f"case_id should score >0.5, got {score}"

    def test_caseid_scores_high(self) -> None:
        """caseid (no underscore) should score >0.5."""
        evidence = make_evidence(
            name="caseid",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.5,
            null_rate=0.0,
        )
        score = score_respondent_id_role(evidence)
        assert score > 0.5, f"caseid should score >0.5, got {score}"

    def test_high_cardinality_boost(self) -> None:
        """High cardinality boosts score (one per household)."""
        high_card = make_evidence(
            name="hhid",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.5,
        )
        low_card = make_evidence(
            name="hhid",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.05,
        )
        assert score_respondent_id_role(high_card) > score_respondent_id_role(low_card)

    def test_microdata_shape_boost(self) -> None:
        """MICRODATA shape boosts score."""
        evidence = make_evidence(
            name="hhid",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.4,
        )
        microdata_score = score_respondent_id_role(
            evidence, detected_shape=ShapeHypothesis.MICRODATA
        )
        no_shape_score = score_respondent_id_role(evidence, detected_shape=None)
        assert microdata_score > no_shape_score


class TestScoreSubunitIdRole:
    """Tests for score_subunit_id_role function."""

    def test_indiv_scores_high(self) -> None:
        """indiv should score >0.5."""
        evidence = make_evidence(
            name="indiv",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_subunit_id_role(evidence)
        assert score > 0.5, f"indiv should score >0.5, got {score}"

    def test_individual_scores_high(self) -> None:
        """individual should score >0.5."""
        evidence = make_evidence(
            name="individual",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_subunit_id_role(evidence)
        assert score > 0.5, f"individual should score >0.5, got {score}"

    def test_member_num_scores_high(self) -> None:
        """member_num should score >0.5."""
        evidence = make_evidence(
            name="member_num",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_subunit_id_role(evidence)
        assert score > 0.5, f"member_num should score >0.5, got {score}"

    def test_member_scores_high(self) -> None:
        """member should score >0.5."""
        evidence = make_evidence(
            name="member",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_subunit_id_role(evidence)
        assert score > 0.5, f"member should score >0.5, got {score}"

    def test_line_num_scores_high(self) -> None:
        """line_num (roster line) should score >0.5."""
        evidence = make_evidence(
            name="line_num",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_subunit_id_role(evidence)
        assert score > 0.5, f"line_num should score >0.5, got {score}"

    def test_low_moderate_cardinality(self) -> None:
        """Low-moderate cardinality (1-N within unit) boosts score."""
        low_card = make_evidence(
            name="indiv",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.05,
        )
        high_card = make_evidence(
            name="indiv",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.5,
        )
        assert score_subunit_id_role(low_card) >= score_subunit_id_role(high_card)


class TestScoreClusterIdRole:
    """Tests for score_cluster_id_role function."""

    def test_ea_scores_high(self) -> None:
        """ea (Enumeration Area) should score >0.5."""
        evidence = make_evidence(
            name="ea",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_cluster_id_role(evidence)
        assert score > 0.5, f"ea should score >0.5, got {score}"

    def test_psu_scores_high(self) -> None:
        """psu (Primary Sampling Unit) should score >0.5."""
        evidence = make_evidence(
            name="psu",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_cluster_id_role(evidence)
        assert score > 0.5, f"psu should score >0.5, got {score}"

    def test_cluster_scores_high(self) -> None:
        """cluster should score >0.5."""
        evidence = make_evidence(
            name="cluster",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_cluster_id_role(evidence)
        assert score > 0.5, f"cluster should score >0.5, got {score}"

    def test_cluster_id_scores_high(self) -> None:
        """cluster_id should score >0.5."""
        evidence = make_evidence(
            name="cluster_id",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.1,
        )
        score = score_cluster_id_role(evidence)
        assert score > 0.5, f"cluster_id should score >0.5, got {score}"

    def test_stratum_scores_high(self) -> None:
        """stratum should score >0.5."""
        evidence = make_evidence(
            name="stratum",
            primitive_type=PrimitiveType.INTEGER,
            distinct_ratio=0.05,
        )
        score = score_cluster_id_role(evidence)
        assert score > 0.5, f"stratum should score >0.5, got {score}"


class TestScoreSurveyWeightRole:
    """Tests for score_survey_weight_role function."""

    def test_weight_scores_high(self) -> None:
        """weight should score >0.5."""
        evidence = make_evidence(
            name="weight",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.3,
            non_negative_ratio=1.0,
        )
        score = score_survey_weight_role(evidence)
        assert score > 0.5, f"weight should score >0.5, got {score}"

    def test_wgt_scores_high(self) -> None:
        """wgt should score >0.5."""
        evidence = make_evidence(
            name="wgt",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.3,
            non_negative_ratio=1.0,
        )
        score = score_survey_weight_role(evidence)
        assert score > 0.5, f"wgt should score >0.5, got {score}"

    def test_hh_weight_scores_high(self) -> None:
        """hh_weight should score >0.5."""
        evidence = make_evidence(
            name="hh_weight",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.3,
            non_negative_ratio=1.0,
        )
        score = score_survey_weight_role(evidence)
        assert score > 0.5, f"hh_weight should score >0.5, got {score}"

    def test_person_weight_scores_high(self) -> None:
        """person_weight should score >0.5."""
        evidence = make_evidence(
            name="person_weight",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.3,
            non_negative_ratio=1.0,
        )
        score = score_survey_weight_role(evidence)
        assert score > 0.5, f"person_weight should score >0.5, got {score}"

    def test_wt_scores_high(self) -> None:
        """wt should score >0.5."""
        evidence = make_evidence(
            name="wt",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.3,
            non_negative_ratio=1.0,
        )
        score = score_survey_weight_role(evidence)
        assert score > 0.5, f"wt should score >0.5, got {score}"

    def test_pweight_scores_high(self) -> None:
        """pweight (Stata convention) should score >0.5."""
        evidence = make_evidence(
            name="pweight",
            primitive_type=PrimitiveType.NUMBER,
            distinct_ratio=0.3,
            non_negative_ratio=1.0,
        )
        score = score_survey_weight_role(evidence)
        assert score > 0.5, f"pweight should score >0.5, got {score}"

    def test_non_numeric_zero(self) -> None:
        """Non-numeric types should return 0."""
        evidence = make_evidence(
            name="weight",
            primitive_type=PrimitiveType.STRING,
            distinct_ratio=0.3,
        )
        score = score_survey_weight_role(evidence)
        assert score == 0.0

    def test_positive_values_boost(self) -> None:
        """Positive values (non-negative ratio ≈ 1) boost score."""
        positive = make_evidence(
            name="weight",
            primitive_type=PrimitiveType.NUMBER,
            non_negative_ratio=1.0,
        )
        mixed = make_evidence(
            name="weight",
            primitive_type=PrimitiveType.NUMBER,
            non_negative_ratio=0.5,
        )
        assert score_survey_weight_role(positive) > score_survey_weight_role(mixed)


# ============ Integration Tests ============


class TestMicrodataRoleIntegration:
    """Integration tests for microdata role detection."""

    def test_typical_lsms_columns(self) -> None:
        """Test role scoring for typical LSMS survey columns."""
        columns = {
            "hhid": (PrimitiveType.STRING, 0.4),
            "indiv": (PrimitiveType.INTEGER, 0.05),
            "s1aq1": (PrimitiveType.INTEGER, 0.02),
            "s1aq2": (PrimitiveType.INTEGER, 0.03),
            "s1aq3_os": (PrimitiveType.STRING, 0.3),
            "weight": (PrimitiveType.NUMBER, 0.2),
            "ea": (PrimitiveType.INTEGER, 0.08),
        }

        for name, (ptype, distinct_ratio) in columns.items():
            evidence = make_evidence(
                name=name,
                primitive_type=ptype,
                distinct_ratio=distinct_ratio,
                unique_count=int(distinct_ratio * 1000),
                non_negative_ratio=1.0,
            )

            if name == "hhid":
                assert score_respondent_id_role(evidence) > 0.5
            elif name == "indiv":
                assert score_subunit_id_role(evidence) > 0.5
            elif name.startswith("s1aq"):
                score = score_question_response_role(
                    evidence, detected_shape=ShapeHypothesis.MICRODATA
                )
                assert score > 0.3
            elif name == "weight":
                assert score_survey_weight_role(evidence) > 0.5
            elif name == "ea":
                assert score_cluster_id_role(evidence) > 0.5

    def test_typical_dhs_columns(self) -> None:
        """Test role scoring for typical DHS survey columns."""
        columns = {
            "caseid": (PrimitiveType.STRING, 0.4),
            "v101": (PrimitiveType.INTEGER, 0.02),
            "v102": (PrimitiveType.INTEGER, 0.02),
            "hv001": (PrimitiveType.INTEGER, 0.08),
            "wt": (PrimitiveType.NUMBER, 0.15),
        }

        for name, (ptype, distinct_ratio) in columns.items():
            evidence = make_evidence(
                name=name,
                primitive_type=ptype,
                distinct_ratio=distinct_ratio,
                unique_count=int(distinct_ratio * 1000),
                non_negative_ratio=1.0,
            )

            if name == "caseid":
                assert score_respondent_id_role(evidence) > 0.5
            elif name.startswith("v"):
                score = score_question_response_role(
                    evidence, detected_shape=ShapeHypothesis.MICRODATA
                )
                assert score > 0.5
            elif name == "hv001":
                score = score_question_response_role(
                    evidence, detected_shape=ShapeHypothesis.MICRODATA
                )
                assert score > 0.3
            elif name == "wt":
                assert score_survey_weight_role(evidence) > 0.5
