# Minimal Integration

Get started with core inference only — no optional dependencies.

## Installation

```bash
pip install datasculpt
```

This installs only the core package with pandas.

## Basic Usage

```python
from datasculpt import infer

# From file
result = infer("data.csv")

# From DataFrame
import pandas as pd
df = pd.read_csv("data.csv")
result = infer(df)
```

## Extracting Results

### Shape

```python
shape = result.proposal.shape_hypothesis
print(f"Shape: {shape.value}")
# Shape: wide_observations
```

### Grain

```python
grain = result.decision_record.grain
print(f"Grain columns: {grain.key_columns}")
print(f"Uniqueness: {grain.uniqueness_ratio:.1%}")
print(f"Confidence: {grain.confidence:.2f}")
```

### Column Roles

```python
for col in result.proposal.columns:
    print(f"{col.name}: {col.role.value}")
```

### Warnings

```python
for warning in result.proposal.warnings:
    print(f"Warning: {warning}")
```

## Configuration

Customize inference behavior:

```python
from datasculpt.core.types import InferenceConfig

config = InferenceConfig(
    # Role scoring
    key_cardinality_threshold=0.9,
    dimension_cardinality_max=0.1,
    null_rate_threshold=0.01,

    # Shape detection
    min_time_columns_for_wide=3,

    # Grain inference
    max_grain_columns=4,
    min_uniqueness_confidence=0.95,

    # Ambiguity
    hypothesis_confidence_gap=0.1,
)

result = infer("data.csv", config=config)
```

## Error Handling

```python
from datasculpt import infer

try:
    result = infer("data.csv")
except FileNotFoundError:
    print("File not found")
except pd.errors.EmptyDataError:
    print("File is empty")
except Exception as e:
    print(f"Inference failed: {e}")
```

## Accessing Raw Evidence

```python
# Get evidence for a specific column
evidence = result.decision_record.column_evidence["population"]

print(f"Type: {evidence.primitive_type.value}")
print(f"Null rate: {evidence.null_rate:.1%}")
print(f"Distinct ratio: {evidence.distinct_ratio:.2f}")
print(f"Role scores: {evidence.role_scores}")
```

## Pipeline Integration

### As a Pre-Processing Step

```python
def process_dataset(filepath: str) -> pd.DataFrame:
    # Infer structure
    result = infer(filepath)

    # Extract grain for downstream use
    grain_columns = result.decision_record.grain.key_columns

    # Load data
    df = result.dataframe

    # Validate grain
    if df[grain_columns].duplicated().any():
        raise ValueError(f"Duplicates in grain: {grain_columns}")

    return df
```

### As a Validation Step

```python
def validate_dataset(df: pd.DataFrame, expected_shape: str) -> bool:
    result = infer(df)

    actual_shape = result.proposal.shape_hypothesis.value
    if actual_shape != expected_shape:
        print(f"Expected {expected_shape}, got {actual_shape}")
        return False

    if result.decision_record.grain.uniqueness_ratio < 1.0:
        print("Dataset has duplicate rows")
        return False

    return True
```

### As a Metadata Producer

```python
def produce_metadata(filepath: str) -> dict:
    result = infer(filepath)

    return {
        "shape": result.proposal.shape_hypothesis.value,
        "grain": result.decision_record.grain.key_columns,
        "columns": [
            {
                "name": col.name,
                "role": col.role.value,
                "type": col.primitive_type.value,
            }
            for col in result.proposal.columns
        ],
        "warnings": result.proposal.warnings,
    }
```

## DataFrame Access

The result includes the loaded DataFrame:

```python
result = infer("data.csv")

# Access the DataFrame directly
df = result.dataframe

# Use for further processing
print(df.head())
print(df.shape)
```

## Next Steps

- [Interactive Mode](interactive-mode.md) — Handle ambiguous datasets
- [Optional Adapters](optional-adapters.md) — Add profiling capabilities
- [API Reference](../reference/api.md) — Full function signatures
