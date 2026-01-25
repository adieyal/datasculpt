"""Core type definitions for Datasculpt (browser bundle)."""


from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PrimitiveType(str, Enum):
    """Primitive data types for columns."""

    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    UNKNOWN = "unknown"


class StructuralType(str, Enum):
    """Structural types for column values."""

    SCALAR = "scalar"
    ARRAY = "array"
    OBJECT = "object"
    UNKNOWN = "unknown"


class Role(str, Enum):
    """Column roles in dataset structure."""

    KEY = "key"
    DIMENSION = "dimension"
    MEASURE = "measure"
    TIME = "time"
    INDICATOR_NAME = "indicator_name"
    VALUE = "value"
    SERIES = "series"
    METADATA = "metadata"


class ShapeHypothesis(str, Enum):
    """Dataset shape hypotheses."""

    LONG_OBSERVATIONS = "long_observations"
    LONG_INDICATORS = "long_indicators"
    WIDE_OBSERVATIONS = "wide_observations"
    WIDE_TIME_COLUMNS = "wide_time_columns"
    SERIES_COLUMN = "series_column"


class DatasetKind(str, Enum):
    """Dataset kinds for Invariant registration."""

    OBSERVATIONS = "observations"
    INDICATORS_LONG = "indicators_long"
    TIMESERIES_WIDE = "timeseries_wide"
    TIMESERIES_SERIES = "timeseries_series"


class QuestionType(str, Enum):
    """Types of interactive questions."""

    CHOOSE_ONE = "choose_one"
    CHOOSE_MANY = "choose_many"
    CONFIRM = "confirm"
    FREE_TEXT = "free_text"


@dataclass
class ValueProfile:
    """Distribution shape information for a column."""

    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None
    integer_ratio: float = 0.0
    non_negative_ratio: float = 0.0
    bounded_0_1_ratio: float = 0.0
    bounded_0_100_ratio: float = 0.0
    low_cardinality: bool = False
    mostly_null: bool = False


@dataclass
class ArrayProfile:
    """Profile for array-type columns."""

    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    consistent_length: bool = False


@dataclass
class ParseResults:
    """Results from parsing attempts on a column."""

    date_parse_rate: float = 0.0
    has_time: bool = False
    best_date_format: str | None = None
    date_failure_examples: list[str] = field(default_factory=list)
    json_array_rate: float = 0.0


@dataclass
class ColumnEvidence:
    """Normalized evidence about a column."""

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
    parse_results_dict: dict[str, float] = field(default_factory=dict)
    role_scores: dict[Role, float] = field(default_factory=dict)
    external: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class HypothesisScore:
    """Score for a shape hypothesis."""

    hypothesis: ShapeHypothesis
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class PseudoKeySignals:
    """Signals indicating a column may be a pseudo-key."""

    is_monotonic_sequence: bool = False
    is_uuid_like: bool = False
    is_ingestion_timestamp: bool = False
    name_signal_penalty: float = 0.0
    total_penalty: float = 0.0


@dataclass
class GrainDiagnostics:
    """Diagnostics for user feedback about grain quality."""

    duplicate_groups: int = 0
    max_group_size: int = 0
    rows_with_null_in_key: int = 0
    null_columns: list[str] = field(default_factory=list)
    example_duplicate_keys: list[tuple] = field(default_factory=list)


@dataclass
class GrainInference:
    """Inferred grain (unique key) for a dataset."""

    key_columns: list[str]
    confidence: float
    uniqueness_ratio: float
    evidence: list[str] = field(default_factory=list)
    diagnostics: GrainDiagnostics | None = None


@dataclass
class Question:
    """Interactive question for ambiguity resolution."""

    id: str
    type: QuestionType
    prompt: str
    choices: list[dict[str, Any]] = field(default_factory=list)
    default: Any = None
    rationale: str | None = None


@dataclass
class ColumnSpec:
    """Column specification for Invariant proposal."""

    name: str
    role: Role
    primitive_type: PrimitiveType
    structural_type: StructuralType
    reference_system_hint: str | None = None
    concept_hint: str | None = None
    unit_hint: str | None = None
    time_granularity: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DecisionRecord:
    """Complete audit trail for an inference run."""

    decision_id: str
    dataset_fingerprint: str
    timestamp: datetime
    selected_hypothesis: ShapeHypothesis
    hypotheses: list[HypothesisScore]
    grain: GrainInference
    column_evidence: dict[str, ColumnEvidence]
    questions: list[Question] = field(default_factory=list)
    answers: dict[str, Any] = field(default_factory=dict)


@dataclass
class InvariantProposal:
    """Proposal for Invariant dataset registration."""

    dataset_name: str
    dataset_kind: DatasetKind
    shape_hypothesis: ShapeHypothesis
    grain: list[str]
    columns: list[ColumnSpec]
    warnings: list[str] = field(default_factory=list)
    required_user_confirmations: list[str] = field(default_factory=list)
    decision_record_id: str = ""


@dataclass
class InferenceConfig:
    """Configuration for inference behavior."""

    key_cardinality_threshold: float = 0.9
    dimension_cardinality_max: float = 0.1
    null_rate_threshold: float = 0.01
    min_time_columns_for_wide: int = 3
    max_grain_columns: int = 4
    min_uniqueness_confidence: float = 0.95
    hypothesis_confidence_gap: float = 0.1
    use_frictionless: bool = False
    use_dataprofiler: bool = False
