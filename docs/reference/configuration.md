# Configuration Reference

All configuration options for `InferenceConfig`.

## Usage

```python
from datasculpt import infer
from datasculpt.core.types import InferenceConfig

config = InferenceConfig(
    max_grain_columns=3,
    hypothesis_confidence_gap=0.15,
)

result = infer("data.csv", config=config)
```

## Role Scoring Options

### `key_cardinality_threshold`

**Type:** `float`
**Default:** `0.9`

Minimum distinct ratio for a column to be considered a key candidate.

```python
# More strict (requires near-unique columns)
config = InferenceConfig(key_cardinality_threshold=0.95)

# Less strict (allows columns with some duplicates)
config = InferenceConfig(key_cardinality_threshold=0.8)
```

### `dimension_cardinality_max`

**Type:** `float`
**Default:** `0.1`

Maximum distinct ratio for a column to be considered a dimension.

```python
# Only very low cardinality columns (< 5% unique)
config = InferenceConfig(dimension_cardinality_max=0.05)

# Allow medium cardinality dimensions (< 20% unique)
config = InferenceConfig(dimension_cardinality_max=0.2)
```

### `null_rate_threshold`

**Type:** `float`
**Default:** `0.01`

Maximum null rate for key column candidates.

```python
# Strict: no nulls in keys
config = InferenceConfig(null_rate_threshold=0.0)

# Lenient: up to 5% nulls allowed
config = InferenceConfig(null_rate_threshold=0.05)
```

## Shape Detection Options

### `min_time_columns_for_wide`

**Type:** `int`
**Default:** `3`

Minimum number of date-like column headers required to detect `wide_time_columns` shape.

```python
# Require more evidence (5+ time columns)
config = InferenceConfig(min_time_columns_for_wide=5)

# Accept fewer (2+ time columns)
config = InferenceConfig(min_time_columns_for_wide=2)
```

## Grain Inference Options

### `max_grain_columns`

**Type:** `int`
**Default:** `4`

Maximum number of columns to include in grain combinations.

Testing combinations grows exponentially with column count. Limit for performance.

```python
# Faster, may miss complex grains
config = InferenceConfig(max_grain_columns=3)

# Slower, catches complex grains
config = InferenceConfig(max_grain_columns=5)
```

### `min_uniqueness_confidence`

**Type:** `float`
**Default:** `0.95`

Minimum uniqueness ratio to accept a grain without warning.

```python
# Strict: require 99%+ unique
config = InferenceConfig(min_uniqueness_confidence=0.99)

# Lenient: accept 90%+ unique
config = InferenceConfig(min_uniqueness_confidence=0.90)
```

## Ambiguity Options

### `hypothesis_confidence_gap`

**Type:** `float`
**Default:** `0.1`

Minimum score gap between top hypotheses to avoid ambiguity.

If the gap between the top two shape scores is below this threshold, the inference is considered ambiguous. In interactive mode, this generates a question.

```python
# More sensitive (more questions)
config = InferenceConfig(hypothesis_confidence_gap=0.15)

# Less sensitive (fewer questions)
config = InferenceConfig(hypothesis_confidence_gap=0.05)
```

## Adapter Options

### `use_frictionless`

**Type:** `bool`
**Default:** `False`

Enable Frictionless adapter for schema inference.

Requires: `pip install datasculpt[frictionless]`

```python
config = InferenceConfig(use_frictionless=True)
```

### `use_dataprofiler`

**Type:** `bool`
**Default:** `False`

Enable DataProfiler adapter for statistical analysis.

Requires: `pip install datasculpt[dataprofiler]`

```python
config = InferenceConfig(use_dataprofiler=True)
```

## Configuration Presets

### Conservative (Strict)

Prioritizes accuracy over convenience:

```python
conservative = InferenceConfig(
    key_cardinality_threshold=0.95,
    dimension_cardinality_max=0.05,
    null_rate_threshold=0.0,
    min_time_columns_for_wide=4,
    max_grain_columns=3,
    min_uniqueness_confidence=0.99,
    hypothesis_confidence_gap=0.15,
)
```

### Permissive (Lenient)

Prioritizes convenience, accepts more uncertainty:

```python
permissive = InferenceConfig(
    key_cardinality_threshold=0.8,
    dimension_cardinality_max=0.2,
    null_rate_threshold=0.05,
    min_time_columns_for_wide=2,
    max_grain_columns=5,
    min_uniqueness_confidence=0.90,
    hypothesis_confidence_gap=0.05,
)
```

### High-Performance

Optimized for speed on large datasets:

```python
fast = InferenceConfig(
    max_grain_columns=2,
    use_frictionless=False,
    use_dataprofiler=False,
)
```

### Full Analysis

Maximum analysis with all adapters:

```python
full = InferenceConfig(
    use_frictionless=True,
    use_dataprofiler=True,
    max_grain_columns=4,
)
```
