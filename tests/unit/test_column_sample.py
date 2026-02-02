"""Unit tests for ColumnSample and extract_column_sample."""

from __future__ import annotations

import pandas as pd
import pytest

from datasculpt.core.evidence import extract_column_sample
from datasculpt.core.types import ColumnSample


class TestColumnSampleDataclass:
    """Tests for ColumnSample dataclass structure."""

    def test_column_sample_creation_with_all_fields(self) -> None:
        """ColumnSample should be creatable with all fields."""
        sample = ColumnSample(
            values=["a", "b", "c"],
            sample_size=3,
            sampling_method="random",
            seed=42,
        )

        assert sample.values == ["a", "b", "c"]
        assert sample.sample_size == 3
        assert sample.sampling_method == "random"
        assert sample.seed == 42

    def test_column_sample_seed_none(self) -> None:
        """ColumnSample should accept None seed."""
        sample = ColumnSample(
            values=["a"],
            sample_size=1,
            sampling_method="full",
            seed=None,
        )

        assert sample.seed is None

    def test_column_sample_values_are_strings(self) -> None:
        """ColumnSample values should be stringified."""
        sample = ColumnSample(
            values=["1", "2", "3.14"],
            sample_size=3,
            sampling_method="random",
            seed=42,
        )

        assert all(isinstance(v, str) for v in sample.values)


class TestExtractColumnSample:
    """Tests for extract_column_sample function."""

    def test_returns_column_sample(self) -> None:
        """extract_column_sample should return a ColumnSample instance."""
        series = pd.Series(["a", "b", "c"])
        result = extract_column_sample(series)

        assert isinstance(result, ColumnSample)

    def test_sampling_is_deterministic_with_same_seed(self) -> None:
        """Same seed should produce identical samples."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result1 = extract_column_sample(series, sample_size=100, seed=42)
        result2 = extract_column_sample(series, sample_size=100, seed=42)

        assert result1.values == result2.values

    def test_different_seeds_produce_different_samples(self) -> None:
        """Different seeds should produce different samples."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result1 = extract_column_sample(series, sample_size=100, seed=42)
        result2 = extract_column_sample(series, sample_size=100, seed=123)

        assert result1.values != result2.values

    def test_sample_size_is_respected(self) -> None:
        """Sample should be limited to sample_size."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result = extract_column_sample(series, sample_size=50, seed=42)

        assert len(result.values) == 50
        assert result.sample_size == 50

    def test_small_series_returns_all_values(self) -> None:
        """Series smaller than sample_size should return all values."""
        series = pd.Series(["a", "b", "c"])

        result = extract_column_sample(series, sample_size=200, seed=42)

        assert len(result.values) == 3
        assert result.sample_size == 3
        assert set(result.values) == {"a", "b", "c"}

    def test_small_series_uses_full_method(self) -> None:
        """Small series should use full sampling method."""
        series = pd.Series(["a", "b", "c"])

        result = extract_column_sample(series, sample_size=200)

        assert result.sampling_method == "full"

    def test_large_series_uses_random_method(self) -> None:
        """Large series should use random sampling method."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result = extract_column_sample(series, sample_size=100)

        assert result.sampling_method == "random"

    def test_null_values_are_excluded(self) -> None:
        """Null values should not appear in sample."""
        series = pd.Series(["a", None, "b", None, "c", None])

        result = extract_column_sample(series, sample_size=200)

        assert None not in result.values
        assert "None" not in result.values
        assert len(result.values) == 3

    def test_nan_values_are_excluded(self) -> None:
        """NaN values should not appear in sample."""
        series = pd.Series(["a", float("nan"), "b", float("nan")])

        result = extract_column_sample(series)

        assert "nan" not in [v.lower() for v in result.values]
        assert len(result.values) == 2

    def test_empty_series_returns_empty_sample(self) -> None:
        """Empty series should return empty sample."""
        series = pd.Series([], dtype=object)

        result = extract_column_sample(series)

        assert result.values == []
        assert result.sample_size == 0
        assert result.sampling_method == "full"

    def test_all_null_series_returns_empty_sample(self) -> None:
        """Series with all nulls should return empty sample."""
        series = pd.Series([None, None, None])

        result = extract_column_sample(series)

        assert result.values == []
        assert result.sample_size == 0

    def test_numeric_values_are_stringified(self) -> None:
        """Numeric values should be converted to strings."""
        series = pd.Series([1, 2, 3, 4, 5])

        result = extract_column_sample(series)

        assert all(isinstance(v, str) for v in result.values)
        assert "1" in result.values

    def test_seed_is_recorded(self) -> None:
        """The seed used should be recorded in the result."""
        series = pd.Series(["a", "b", "c"])

        result = extract_column_sample(series, seed=99)

        assert result.seed == 99

    def test_default_sample_size_is_200(self) -> None:
        """Default sample size should be 200."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result = extract_column_sample(series)

        assert len(result.values) == 200

    def test_default_seed_is_42(self) -> None:
        """Default seed should be 42 for determinism."""
        series = pd.Series([f"val_{i}" for i in range(500)])

        result = extract_column_sample(series)

        assert result.seed == 42


class TestInferenceConfigSampling:
    """Tests for InferenceConfig sampling fields."""

    def test_inference_config_has_return_samples_field(self) -> None:
        """InferenceConfig should have return_samples field."""
        from datasculpt.core.types import InferenceConfig

        config = InferenceConfig()

        assert hasattr(config, "return_samples")
        assert config.return_samples is False

    def test_inference_config_has_sample_size_field(self) -> None:
        """InferenceConfig should have sample_size field."""
        from datasculpt.core.types import InferenceConfig

        config = InferenceConfig()

        assert hasattr(config, "sample_size")
        assert config.sample_size == 200

    def test_inference_config_has_sample_seed_field(self) -> None:
        """InferenceConfig should have sample_seed field."""
        from datasculpt.core.types import InferenceConfig

        config = InferenceConfig()

        assert hasattr(config, "sample_seed")
        assert config.sample_seed == 42

    def test_inference_config_sampling_fields_customizable(self) -> None:
        """InferenceConfig sampling fields should be customizable."""
        from datasculpt.core.types import InferenceConfig

        config = InferenceConfig(
            return_samples=True,
            sample_size=500,
            sample_seed=123,
        )

        assert config.return_samples is True
        assert config.sample_size == 500
        assert config.sample_seed == 123
