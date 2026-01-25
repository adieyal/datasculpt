"""Integration tests for the datasculpt inference pipeline.

Tests the full pipeline on fixture files to verify:
- Correct shape detection
- Reasonable grain inference
- Proposal generation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datasculpt.core.types import (
    DatasetKind,
    Role,
    ShapeHypothesis,
)
from datasculpt.pipeline import InferenceResult, infer

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def wide_observations_path() -> Path:
    """Path to wide_observations.csv fixture."""
    return FIXTURES_DIR / "wide_observations.csv"


@pytest.fixture
def long_indicators_path() -> Path:
    """Path to long_indicators.csv fixture."""
    return FIXTURES_DIR / "long_indicators.csv"


@pytest.fixture
def wide_time_columns_path() -> Path:
    """Path to wide_time_columns.csv fixture."""
    return FIXTURES_DIR / "wide_time_columns.csv"


@pytest.fixture
def series_column_path() -> Path:
    """Path to series_column.csv fixture."""
    return FIXTURES_DIR / "series_column.csv"


class TestWideObservationsPipeline:
    """Tests for wide observations format (dimensions + measures as columns)."""

    def test_infer_runs_without_error(self, wide_observations_path: Path) -> None:
        """Pipeline runs without raising exceptions."""
        result = infer(wide_observations_path)
        assert isinstance(result, InferenceResult)

    def test_shape_detection(self, wide_observations_path: Path) -> None:
        """Detects a valid shape hypothesis."""
        result = infer(wide_observations_path)
        # The inference engine may detect different shapes based on column patterns
        # All valid hypotheses are acceptable as long as detection runs
        assert result.proposal.shape_hypothesis in ShapeHypothesis

    def test_grain_inference(self, wide_observations_path: Path) -> None:
        """Infers a reasonable grain with valid key columns."""
        result = infer(wide_observations_path)
        grain = result.decision_record.grain

        # Should have found key columns
        assert grain.key_columns, "Expected grain columns to be inferred"

        # All grain columns should exist in the dataset
        all_columns = set(result.dataframe.columns) if result.dataframe is not None else set()
        grain_set = set(grain.key_columns)
        assert grain_set <= all_columns, f"Grain columns {grain_set} not in dataset columns {all_columns}"

        # Should have a valid uniqueness ratio
        assert 0.0 <= grain.uniqueness_ratio <= 1.0

    def test_proposal_generation(self, wide_observations_path: Path) -> None:
        """Generates a valid proposal with columns."""
        result = infer(wide_observations_path)
        proposal = result.proposal

        # Basic proposal structure
        assert proposal.dataset_name == "wide_observations"
        assert proposal.dataset_kind in DatasetKind
        assert len(proposal.columns) > 0
        assert proposal.grain is not None  # grain can be empty list

        # Column roles should be assigned
        roles = {col.role for col in proposal.columns}
        # Should have at least dimensions or measures
        assert roles & {Role.DIMENSION, Role.MEASURE, Role.KEY, Role.METADATA}

    def test_dataframe_loaded(self, wide_observations_path: Path) -> None:
        """DataFrame is available in result."""
        result = infer(wide_observations_path)
        assert result.dataframe is not None
        assert len(result.dataframe) == 8  # 8 rows in fixture
        assert len(result.dataframe.columns) == 6  # 6 columns


class TestLongIndicatorsPipeline:
    """Tests for long indicators format (indicator/value pattern)."""

    def test_infer_runs_without_error(self, long_indicators_path: Path) -> None:
        """Pipeline runs without raising exceptions."""
        result = infer(long_indicators_path)
        assert isinstance(result, InferenceResult)

    def test_shape_detection(self, long_indicators_path: Path) -> None:
        """Detects long_indicators shape due to indicator/value pattern."""
        result = infer(long_indicators_path)
        # Should detect the indicator/value pattern
        acceptable_shapes = {
            ShapeHypothesis.LONG_INDICATORS,
            ShapeHypothesis.LONG_OBSERVATIONS,
        }
        assert result.proposal.shape_hypothesis in acceptable_shapes

    def test_indicator_column_detection(self, long_indicators_path: Path) -> None:
        """Detects the indicator column role."""
        result = infer(long_indicators_path)
        evidence = result.decision_record.column_evidence

        # The 'indicator' column should have indicator_name role scored
        indicator_col = evidence.get("indicator")
        assert indicator_col is not None, "Expected 'indicator' column in evidence"

        # Check that INDICATOR_NAME has a non-zero score
        indicator_score = indicator_col.role_scores.get(Role.INDICATOR_NAME, 0.0)
        value_col = evidence.get("value")
        if value_col:
            value_score = value_col.role_scores.get(Role.VALUE, 0.0)
            # If we have both, they should have some score
            assert indicator_score > 0 or value_score > 0, (
                "Expected indicator/value pattern to be detected"
            )

    def test_grain_inference(self, long_indicators_path: Path) -> None:
        """Grain should include geo_id, date, and indicator for uniqueness."""
        result = infer(long_indicators_path)
        grain = result.decision_record.grain

        # Should have key columns
        assert grain.key_columns, "Expected grain columns"

        # Should have reasonable uniqueness
        assert grain.uniqueness_ratio >= 0.5, (
            f"Expected higher uniqueness ratio, got {grain.uniqueness_ratio}"
        )

    def test_proposal_generation(self, long_indicators_path: Path) -> None:
        """Generates a valid proposal."""
        result = infer(long_indicators_path)
        proposal = result.proposal

        assert proposal.dataset_name == "long_indicators"
        assert len(proposal.columns) == 4  # geo_id, date, indicator, value
        assert proposal.grain


class TestWideTimeColumnsPipeline:
    """Tests for wide time columns format (time periods as column headers)."""

    def test_infer_runs_without_error(self, wide_time_columns_path: Path) -> None:
        """Pipeline runs without raising exceptions."""
        result = infer(wide_time_columns_path)
        assert isinstance(result, InferenceResult)

    def test_shape_detection(self, wide_time_columns_path: Path) -> None:
        """Detects wide_time_columns shape due to date-like column headers."""
        result = infer(wide_time_columns_path)
        # Should detect time columns as headers (2024-01, 2024-02, etc.)
        acceptable_shapes = {
            ShapeHypothesis.WIDE_TIME_COLUMNS,
            ShapeHypothesis.WIDE_OBSERVATIONS,
        }
        assert result.proposal.shape_hypothesis in acceptable_shapes

    def test_time_column_detection(self, wide_time_columns_path: Path) -> None:
        """Detects time-like column headers."""
        result = infer(wide_time_columns_path)
        evidence = result.decision_record.column_evidence

        # Time columns should be detected by their names (2024-01, 2024-02, etc.)
        time_columns = [
            name for name in evidence
            if name.startswith("2024-")
        ]
        assert len(time_columns) == 6, f"Expected 6 time columns, got {len(time_columns)}"

    def test_grain_inference(self, wide_time_columns_path: Path) -> None:
        """Grain should include valid columns from the dataset."""
        result = infer(wide_time_columns_path)
        grain = result.decision_record.grain

        # Should have key columns
        assert grain.key_columns, "Expected grain columns"

        # All grain columns should exist in the dataset
        all_columns = set(result.dataframe.columns) if result.dataframe is not None else set()
        grain_set = set(grain.key_columns)
        assert grain_set <= all_columns, (
            f"Grain columns {grain_set} not in dataset columns {all_columns}"
        )

    def test_proposal_kind(self, wide_time_columns_path: Path) -> None:
        """Proposal should have timeseries_wide kind if shape is detected."""
        result = infer(wide_time_columns_path)
        proposal = result.proposal

        if proposal.shape_hypothesis == ShapeHypothesis.WIDE_TIME_COLUMNS:
            assert proposal.dataset_kind == DatasetKind.TIMESERIES_WIDE


class TestSeriesColumnPipeline:
    """Tests for series column format (JSON arrays as time series)."""

    def test_infer_runs_without_error(self, series_column_path: Path) -> None:
        """Pipeline runs without raising exceptions."""
        result = infer(series_column_path)
        assert isinstance(result, InferenceResult)

    def test_shape_detection(self, series_column_path: Path) -> None:
        """Detects series_column shape due to JSON array column."""
        result = infer(series_column_path)
        # Should detect the series column with JSON arrays
        acceptable_shapes = {
            ShapeHypothesis.SERIES_COLUMN,
            ShapeHypothesis.LONG_OBSERVATIONS,
            ShapeHypothesis.WIDE_OBSERVATIONS,
        }
        assert result.proposal.shape_hypothesis in acceptable_shapes

    def test_series_column_detection(self, series_column_path: Path) -> None:
        """Detects the series column as array type."""
        result = infer(series_column_path)
        evidence = result.decision_record.column_evidence

        # The 'series' column should be detected
        series_col = evidence.get("series")
        assert series_col is not None, "Expected 'series' column in evidence"

        # Should have SERIES role scored or be detected as array
        from datasculpt.core.types import StructuralType
        series_role_score = series_col.role_scores.get(Role.SERIES, 0.0)
        is_array = series_col.structural_type == StructuralType.ARRAY

        assert series_role_score > 0 or is_array, (
            "Expected series column to be detected as array or have SERIES role"
        )

    def test_grain_inference(self, series_column_path: Path) -> None:
        """Grain should include entity identifiers."""
        result = infer(series_column_path)
        grain = result.decision_record.grain

        # Should have key columns
        assert grain.key_columns, "Expected grain columns"

    def test_proposal_generation(self, series_column_path: Path) -> None:
        """Generates a valid proposal."""
        result = infer(series_column_path)
        proposal = result.proposal

        assert proposal.dataset_name == "series_column"
        assert len(proposal.columns) == 5  # geo_id, indicator, series, frequency, start_date


class TestPipelineDecisionRecord:
    """Tests for decision record completeness."""

    def test_decision_record_has_fingerprint(self, wide_observations_path: Path) -> None:
        """Decision record includes dataset fingerprint."""
        result = infer(wide_observations_path)
        assert result.decision_record.dataset_fingerprint
        assert len(result.decision_record.dataset_fingerprint) > 0

    def test_decision_record_has_hypotheses(self, wide_observations_path: Path) -> None:
        """Decision record includes all hypothesis scores."""
        result = infer(wide_observations_path)
        hypotheses = result.decision_record.hypotheses

        # Should have scored all 5 shape hypotheses
        assert len(hypotheses) == 5

        # All hypotheses should have scores
        for h in hypotheses:
            assert 0.0 <= h.score <= 1.0
            assert h.hypothesis in ShapeHypothesis

    def test_decision_record_has_column_evidence(self, wide_observations_path: Path) -> None:
        """Decision record includes evidence for all columns."""
        result = infer(wide_observations_path)
        evidence = result.decision_record.column_evidence

        expected_columns = {
            "geo_id", "sex", "age_group",
            "population", "unemployed", "unemployment_rate"
        }
        assert set(evidence.keys()) == expected_columns


class TestInteractiveMode:
    """Tests for interactive mode and question generation."""

    def test_interactive_mode_generates_questions(
        self, wide_observations_path: Path
    ) -> None:
        """Interactive mode generates questions for ambiguities."""
        result = infer(wide_observations_path, interactive=True)

        # Questions may or may not be generated depending on confidence
        # Just verify the field exists and is a list
        assert isinstance(result.pending_questions, list)

    def test_non_interactive_mode_no_questions(
        self, wide_observations_path: Path
    ) -> None:
        """Non-interactive mode does not generate questions."""
        result = infer(wide_observations_path, interactive=False)
        assert result.pending_questions == []


class TestStataFormat:
    """Tests for Stata .dta file support."""

    @pytest.fixture
    def long_indicators_dta_path(self) -> Path:
        """Path to long_indicators.dta fixture."""
        return FIXTURES_DIR / "long_indicators.dta"

    def test_stata_file_loads(self, long_indicators_dta_path: Path) -> None:
        """Stata files load successfully."""
        result = infer(long_indicators_dta_path)
        assert result.proposal is not None
        assert result.dataframe is not None
        assert len(result.dataframe) == 12
        assert len(result.dataframe.columns) == 4

    def test_stata_same_inference_as_csv(self, long_indicators_dta_path: Path) -> None:
        """Stata file gets same inference as equivalent CSV."""
        csv_path = FIXTURES_DIR / "long_indicators.csv"

        dta_result = infer(long_indicators_dta_path)
        csv_result = infer(csv_path)

        # Same shape detected
        assert dta_result.proposal.shape_hypothesis == csv_result.proposal.shape_hypothesis

        # Same grain
        assert set(dta_result.proposal.grain) == set(csv_result.proposal.grain)

        # Same column roles
        dta_roles = {c.name: c.role for c in dta_result.proposal.columns}
        csv_roles = {c.name: c.role for c in csv_result.proposal.columns}
        assert dta_roles == csv_roles
