# API Reference

## Main Functions

### `infer()`

Main entry point for dataset inference.

```python
def infer(
    source: str | Path | pd.DataFrame,
    config: InferenceConfig | None = None,
    interactive: bool = False,
    answers: dict[str, Any] | None = None,
) -> InferenceResult:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str \| Path \| DataFrame` | required | File path or DataFrame to analyze |
| `config` | `InferenceConfig \| None` | `None` | Inference configuration |
| `interactive` | `bool` | `False` | Generate questions for ambiguous aspects |
| `answers` | `dict[str, Any] \| None` | `None` | Pre-provided answers from previous run |

**Returns:** `InferenceResult`

**Example:**

```python
from datasculpt import infer

# From file
result = infer("data.csv")

# From DataFrame
result = infer(df)

# With configuration
from datasculpt.core.types import InferenceConfig
config = InferenceConfig(max_grain_columns=3)
result = infer("data.csv", config=config)

# Interactive mode
result = infer("data.csv", interactive=True)
```

---

### `apply_answers()`

Re-run inference with user-provided answers applied.

```python
def apply_answers(
    previous_result: InferenceResult,
    answers: dict[str, Any],
    config: InferenceConfig | None = None,
) -> InferenceResult:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `previous_result` | `InferenceResult` | required | Result from previous `infer()` call |
| `answers` | `dict[str, Any]` | required | Question ID → answer mapping |
| `config` | `InferenceConfig \| None` | `None` | Optional override configuration |

**Returns:** `InferenceResult`

**Raises:** `ValueError` if `previous_result.dataframe` is `None`

**Example:**

```python
from datasculpt import infer, apply_answers

result = infer("data.csv", interactive=True)

if result.pending_questions:
    answers = {
        result.pending_questions[0].id: "long_indicators"
    }
    result = apply_answers(result, answers)
```

---

## Evidence Extraction

### `extract_dataframe_evidence()`

Extract evidence for all columns in a DataFrame.

```python
from datasculpt.core.evidence import extract_dataframe_evidence

def extract_dataframe_evidence(
    df: pd.DataFrame,
) -> dict[str, ColumnEvidence]:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | DataFrame to analyze |

**Returns:** `dict[str, ColumnEvidence]` — Column name → evidence mapping

**Example:**

```python
from datasculpt.core.evidence import extract_dataframe_evidence

evidence = extract_dataframe_evidence(df)
for col_name, ev in evidence.items():
    print(f"{col_name}: {ev.primitive_type.value}")
```

---

## Shape Detection

### `detect_shape()`

Detect dataset shape from column evidence.

```python
from datasculpt.core.shapes import detect_shape

def detect_shape(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> ShapeResult:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `evidences` | `list[ColumnEvidence]` | Evidence for all columns |
| `config` | `InferenceConfig` | Inference configuration |

**Returns:** `ShapeResult`

**Example:**

```python
from datasculpt.core.shapes import detect_shape
from datasculpt.core.evidence import extract_dataframe_evidence
from datasculpt.core.types import InferenceConfig

evidence = extract_dataframe_evidence(df)
config = InferenceConfig()
shape_result = detect_shape(list(evidence.values()), config)

print(f"Shape: {shape_result.selected.value}")
print(f"Ambiguous: {shape_result.is_ambiguous}")
```

---

## Grain Inference

### `infer_grain()`

Infer the grain (unique key) from a DataFrame.

```python
from datasculpt.core.grain import infer_grain

def infer_grain(
    df: pd.DataFrame,
    column_evidence: dict[str, ColumnEvidence],
    config: InferenceConfig,
    shape: ShapeHypothesis | None = None,
) -> GrainInference:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | DataFrame to analyze |
| `column_evidence` | `dict[str, ColumnEvidence]` | Evidence for all columns |
| `config` | `InferenceConfig` | Inference configuration |
| `shape` | `ShapeHypothesis \| None` | Detected shape (for shape-aware inference) |

**Returns:** `GrainInference`

**Example:**

```python
from datasculpt.core.grain import infer_grain

grain = infer_grain(df, evidence, config, shape_result.selected)
print(f"Grain: {grain.key_columns}")
print(f"Uniqueness: {grain.uniqueness_ratio:.1%}")
```

---

## Role Assignment

### `assign_roles()`

Assign roles to columns based on evidence.

```python
from datasculpt.core.roles import assign_roles

def assign_roles(
    evidences: list[ColumnEvidence],
    config: InferenceConfig,
) -> dict[str, RoleAssignment]:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `evidences` | `list[ColumnEvidence]` | Evidence for all columns |
| `config` | `InferenceConfig` | Inference configuration |

**Returns:** `dict[str, RoleAssignment]` — Column name → role assignment

---

## Decision Record Serialization

### `serialize_decision_record()`

Serialize a decision record to a JSON-compatible dictionary.

```python
from datasculpt.decision import serialize_decision_record

def serialize_decision_record(
    record: DecisionRecord,
) -> dict[str, Any]:
```

### `deserialize_decision_record()`

Deserialize a decision record from a dictionary.

```python
from datasculpt.decision import deserialize_decision_record

def deserialize_decision_record(
    data: dict[str, Any],
) -> DecisionRecord:
```

**Example:**

```python
from datasculpt.decision import serialize_decision_record, deserialize_decision_record
import json

# Serialize
data = serialize_decision_record(result.decision_record)
json_str = json.dumps(data, indent=2)

# Deserialize
data = json.loads(json_str)
record = deserialize_decision_record(data)
```

---

## File Intake

### `intake_file()`

Load a file and extract metadata.

```python
from datasculpt.intake import intake_file

def intake_file(
    source: str | Path,
) -> IntakeResult:
```

**Returns:** `IntakeResult` with `dataframe`, `fingerprint`, and `preview`

---

## Error Handling

### Exceptions

Datasculpt defines one custom exception and uses standard Python exceptions for error handling:

#### `IntakeError`

Raised when file loading or processing fails during intake.

```python
from datasculpt.intake import IntakeError
```

**Common causes:**

| Scenario | Message Pattern |
|----------|-----------------|
| File does not exist | `"File not found: {path}"` |
| Unsupported file format | `"Unsupported file format: {suffix}. Supported: .csv, .xlsx, .xls, .parquet, .dta"` |
| File parsing failure | `"Failed to load {path}: {underlying_error}"` |

#### `ValueError`

Raised for invalid function arguments or state.

**Common causes:**

| Function | Scenario | Message |
|----------|----------|---------|
| `apply_answers()` | Called with result that has no DataFrame | `"Cannot apply answers: previous result has no DataFrame. Re-run infer() with the original source."` |

### Error Handling Examples

#### Handling File Loading Errors

```python
from datasculpt import infer
from datasculpt.intake import IntakeError

try:
    result = infer("data.csv")
except IntakeError as e:
    if "File not found" in str(e):
        print(f"File does not exist: {e}")
    elif "Unsupported file format" in str(e):
        print(f"Use a supported format (.csv, .xlsx, .parquet, .dta): {e}")
    else:
        print(f"Failed to load file: {e}")
```

#### Handling apply_answers() Errors

```python
from datasculpt import infer, apply_answers

# First inference run
result = infer("data.csv", interactive=True)

# Simulate losing the DataFrame (e.g., after serialization)
result.dataframe = None

try:
    # This will fail because dataframe is None
    updated_result = apply_answers(result, {"q_123": "long_observations"})
except ValueError as e:
    # Re-run inference with original source
    result = infer("data.csv", interactive=True, answers={"q_123": "long_observations"})
```

#### Safe Inference Pattern

A robust pattern that handles common error scenarios:

```python
from pathlib import Path
from datasculpt import infer
from datasculpt.intake import IntakeError

def safe_infer(source: str | Path):
    """Safely infer dataset metadata with comprehensive error handling."""
    path = Path(source)

    # Pre-flight checks
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() not in {".csv", ".xlsx", ".xls", ".parquet", ".dta"}:
        raise ValueError(f"Unsupported format: {path.suffix}")

    try:
        result = infer(source)
        return result
    except IntakeError as e:
        # Log the error for debugging
        print(f"Intake failed: {e}")
        raise
    except Exception as e:
        # Unexpected errors
        print(f"Unexpected error during inference: {type(e).__name__}: {e}")
        raise

# Usage
try:
    result = safe_infer("my_data.csv")
    print(f"Shape: {result.proposal.shape_hypothesis.value}")
    print(f"Grain: {result.proposal.grain}")
except (FileNotFoundError, ValueError, IntakeError) as e:
    print(f"Could not process dataset: {e}")
```

#### Handling Empty or Malformed DataFrames

```python
import pandas as pd
from datasculpt import infer

# Empty DataFrame handling
df = pd.DataFrame()
result = infer(df)

# Check for warnings in the result
if result.decision_record.grain.confidence == 0.0:
    print("Warning: Could not determine grain (dataset may be empty)")
    print(f"Evidence: {result.decision_record.grain.evidence}")

# DataFrame with issues
df = pd.DataFrame({"col1": [None, None], "col2": [None, None]})
result = infer(df)

# Check grain diagnostics for quality issues
if result.decision_record.grain.diagnostics:
    diag = result.decision_record.grain.diagnostics
    if diag.rows_with_null_in_key > 0:
        print(f"Warning: {diag.rows_with_null_in_key} rows have nulls in key columns")
```

### Supported File Formats

The `infer()` function supports these file formats through `intake_file()`:

| Extension | Format | Notes |
|-----------|--------|-------|
| `.csv` | CSV | Uses `pandas.read_csv()` |
| `.xlsx` | Excel | Uses `pandas.read_excel()` |
| `.xls` | Excel (legacy) | Uses `pandas.read_excel()` |
| `.parquet` | Parquet | Uses `pandas.read_parquet()` |
| `.dta` | Stata | Uses `pandas.read_stata()` |

Attempting to load other formats will raise `IntakeError`.
