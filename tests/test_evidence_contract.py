"""Tests for the stable evidence contract."""

import dataclasses

import pandas as pd
import pytest

from datasculpt.core.types import ColumnEvidence, ColumnSample
from datasculpt.core.evidence import extract_column_evidence, extract_column_sample


class TestColumnEvidenceContract:
    """Verify ColumnEvidence has all required fields."""

    def test_column_evidence_fields_exist(self):
        """Verify stable contract fields."""
        fields = {f.name for f in dataclasses.fields(ColumnEvidence)}
        required = {
            "name", "n_rows", "n_non_null", "null_rate", "distinct_ratio",
            "unique_count", "top_values", "value_length_stats"
        }
        assert required.issubset(fields), f"Missing fields: {required - fields}"


class TestColumnSampleContract:
    """Verify ColumnSample has all required fields."""

    def test_column_sample_fields_exist(self):
        """Verify stable contract fields."""
        fields = {f.name for f in dataclasses.fields(ColumnSample)}
        required = {"values", "sample_size", "sampling_method", "seed"}
        assert required.issubset(fields), f"Missing fields: {required - fields}"


class TestSamplingDeterminism:
    """Verify same seed produces same sample."""

    def test_sampling_deterministic(self):
        """Verify same seed produces same sample."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result1 = extract_column_sample(series, sample_size=100, seed=42)
        result2 = extract_column_sample(series, sample_size=100, seed=42)

        assert result1.values == result2.values
        assert result1.seed == result2.seed

    def test_different_seeds_produce_different_samples(self):
        """Different seeds should produce different samples."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result1 = extract_column_sample(series, sample_size=100, seed=42)
        result2 = extract_column_sample(series, sample_size=100, seed=123)

        assert result1.values != result2.values


class TestEvidenceSnapshot:
    """Golden JSON snapshot test for evidence contract."""

    def test_evidence_has_expected_structure(self):
        """Verify evidence JSON structure."""
        series = pd.Series(["a", "b", "c", "a", "b"])
        evidence = extract_column_evidence(series, "test_col")

        # Convert to dict and verify structure
        evidence_dict = dataclasses.asdict(evidence)

        assert "name" in evidence_dict
        assert "n_rows" in evidence_dict
        assert "n_non_null" in evidence_dict
        assert "top_values" in evidence_dict
        assert "value_length_stats" in evidence_dict

        # Verify values
        assert evidence_dict["name"] == "test_col"
        assert evidence_dict["n_rows"] == 5
        assert evidence_dict["n_non_null"] == 5
