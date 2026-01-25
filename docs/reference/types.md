# Type Reference

Complete type definitions for Datasculpt.

## Enums

### PrimitiveType

```python
class PrimitiveType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UNKNOWN = "unknown"
```

### StructuralType

```python
class StructuralType(str, Enum):
    SCALAR = "scalar"
    ARRAY = "array"
    OBJECT = "object"
    UNKNOWN = "unknown"
```

### Role

```python
class Role(str, Enum):
    KEY = "key"
    DIMENSION = "dimension"
    MEASURE = "measure"
    TIME = "time"
    INDICATOR_NAME = "indicator_name"
    VALUE = "value"
    SERIES = "series"
    METADATA = "metadata"
```

### ShapeHypothesis

```python
class ShapeHypothesis(str, Enum):
    LONG_OBSERVATIONS = "long_observations"
    LONG_INDICATORS = "long_indicators"
    WIDE_OBSERVATIONS = "wide_observations"
    WIDE_TIME_COLUMNS = "wide_time_columns"
    SERIES_COLUMN = "series_column"
```

### DatasetKind

```python
class DatasetKind(str, Enum):
    OBSERVATIONS = "observations"
    INDICATORS_LONG = "indicators_long"
    TIMESERIES_WIDE = "timeseries_wide"
    TIMESERIES_SERIES = "timeseries_series"
```

### QuestionType

```python
class QuestionType(str, Enum):
    CHOOSE_ONE = "choose_one"
    CHOOSE_MANY = "choose_many"
    CONFIRM = "confirm"
    FREE_TEXT = "free_text"
```

---

## Dataclasses

### InferenceResult

Main result from `infer()`.

```python
@dataclass
class InferenceResult:
    proposal: InvariantProposal
    decision_record: DecisionRecord
    pending_questions: list[Question]
    dataframe: pd.DataFrame | None = None
```

### InvariantProposal

Proposal for dataset registration.

```python
@dataclass
class InvariantProposal:
    dataset_name: str
    dataset_kind: DatasetKind
    shape_hypothesis: ShapeHypothesis
    grain: list[str]
    columns: list[ColumnSpec]

    warnings: list[str] = field(default_factory=list)
    required_user_confirmations: list[str] = field(default_factory=list)
    decision_record_id: str = ""
```

### ColumnSpec

Column specification for proposals.

```python
@dataclass
class ColumnSpec:
    name: str
    role: Role
    primitive_type: PrimitiveType
    structural_type: StructuralType
    reference_system_hint: str | None = None
    concept_hint: str | None = None
    unit_hint: str | None = None
    time_granularity: str | None = None
    notes: list[str] = field(default_factory=list)
```

### DecisionRecord

Complete audit trail for inference.

```python
@dataclass
class DecisionRecord:
    decision_id: str
    dataset_fingerprint: str
    timestamp: datetime

    selected_hypothesis: ShapeHypothesis
    hypotheses: list[HypothesisScore]

    grain: GrainInference
    column_evidence: dict[str, ColumnEvidence]

    questions: list[Question] = field(default_factory=list)
    answers: dict[str, Any] = field(default_factory=dict)
```

### ColumnEvidence

Evidence about a column.

```python
@dataclass
class ColumnEvidence:
    name: str
    primitive_type: PrimitiveType
    structural_type: StructuralType

    null_rate: float = 0.0
    distinct_ratio: float = 0.0
    unique_count: int = 0

    value_profile: ValueProfile = field(default_factory=ValueProfile)
    array_profile: ArrayProfile | None = None

    header_date_like: bool = False
    parse_results: ParseResults = field(default_factory=ParseResults)

    role_scores: dict[Role, float] = field(default_factory=dict)
    external: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
```

### ValueProfile

Distribution profile for numeric columns.

```python
@dataclass
class ValueProfile:
    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None

    integer_ratio: float = 0.0
    non_negative_ratio: float = 0.0
    bounded_0_1_ratio: float = 0.0
    bounded_0_100_ratio: float = 0.0

    low_cardinality: bool = False
    mostly_null: bool = False
```

### ArrayProfile

Profile for array-type columns.

```python
@dataclass
class ArrayProfile:
    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    consistent_length: bool = False
```

### ParseResults

Results from parsing attempts.

```python
@dataclass
class ParseResults:
    date_parse_rate: float = 0.0
    has_time: bool = False
    best_date_format: str | None = None
    date_failure_examples: list[str] = field(default_factory=list)

    json_array_rate: float = 0.0
```

### HypothesisScore

Score for a shape hypothesis.

```python
@dataclass
class HypothesisScore:
    hypothesis: ShapeHypothesis
    score: float
    reasons: list[str] = field(default_factory=list)
```

### GrainInference

Inferred grain for a dataset.

```python
@dataclass
class GrainInference:
    key_columns: list[str]
    confidence: float
    uniqueness_ratio: float
    evidence: list[str] = field(default_factory=list)
    diagnostics: GrainDiagnostics | None = None
```

### GrainDiagnostics

Diagnostics about grain quality.

```python
@dataclass
class GrainDiagnostics:
    duplicate_groups: int = 0
    max_group_size: int = 0
    rows_with_null_in_key: int = 0
    null_columns: list[str] = field(default_factory=list)
    example_duplicate_keys: list[tuple] = field(default_factory=list)
```

### PseudoKeySignals

Signals indicating a pseudo-key.

```python
@dataclass
class PseudoKeySignals:
    is_monotonic_sequence: bool = False
    is_uuid_like: bool = False
    is_ingestion_timestamp: bool = False
    name_signal_penalty: float = 0.0
    total_penalty: float = 0.0
```

### Question

Interactive question for ambiguity resolution.

```python
@dataclass
class Question:
    id: str
    type: QuestionType
    prompt: str
    choices: list[dict[str, Any]] = field(default_factory=list)
    default: Any = None
    rationale: str | None = None
```

### InferenceConfig

Configuration for inference behavior.

```python
@dataclass
class InferenceConfig:
    # Role scoring
    key_cardinality_threshold: float = 0.9
    dimension_cardinality_max: float = 0.1
    null_rate_threshold: float = 0.01

    # Shape detection
    min_time_columns_for_wide: int = 3

    # Grain inference
    max_grain_columns: int = 4
    min_uniqueness_confidence: float = 0.95

    # Ambiguity
    hypothesis_confidence_gap: float = 0.1

    # Adapters
    use_frictionless: bool = False
    use_dataprofiler: bool = False
```
