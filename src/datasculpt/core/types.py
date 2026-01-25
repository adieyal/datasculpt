"""Core type definitions for Datasculpt."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
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
    """Distribution shape information for a column.

    Provides discriminators beyond distinct_ratio that help distinguish
    codes vs measures, flags, weights vs denominators, probabilities vs counts.
    """

    min_value: float | None = None
    max_value: float | None = None
    mean: float | None = None  # Numeric columns only

    # Ratio of values close to integers (within 1e-9)
    integer_ratio: float = 0.0

    # Ratio of values >= 0
    non_negative_ratio: float = 0.0

    # Ratio of values in [0, 1]
    bounded_0_1_ratio: float = 0.0

    # Ratio of values in [0, 100]
    bounded_0_100_ratio: float = 0.0

    # True if unique_count <= 5
    low_cardinality: bool = False

    # True if null_rate > 0.8
    mostly_null: bool = False


@dataclass
class ArrayProfile:
    """Profile for array-type columns.

    Captures array length statistics to inform series detection.
    """

    avg_length: float = 0.0
    min_length: int = 0
    max_length: int = 0

    # True if max_length - min_length <= 1
    consistent_length: bool = False


@dataclass
class ParseResults:
    """Results from parsing attempts on a column."""

    # Date parsing
    date_parse_rate: float = 0.0
    has_time: bool = False
    best_date_format: str | None = None
    date_failure_examples: list[str] = field(default_factory=list)

    # JSON array detection
    json_array_rate: float = 0.0


@dataclass
class ColumnEvidence:
    """Normalized evidence about a column."""

    name: str
    primitive_type: PrimitiveType
    structural_type: StructuralType

    # Statistics
    null_rate: float = 0.0
    distinct_ratio: float = 0.0
    unique_count: int = 0

    # Value distribution profile
    value_profile: ValueProfile = field(default_factory=ValueProfile)

    # Array profile (only populated if structural_type is ARRAY)
    array_profile: ArrayProfile | None = None

    # Header signals
    header_date_like: bool = False

    # Parse attempt results
    parse_results: ParseResults = field(default_factory=ParseResults)

    # Legacy parse_results dict (for backwards compatibility during transition)
    parse_results_dict: dict[str, float] = field(default_factory=dict)

    # Role likelihoods (Role -> score)
    role_scores: dict[Role, float] = field(default_factory=dict)

    # External profiler data
    external: dict[str, Any] = field(default_factory=dict)

    # Explanatory notes
    notes: list[str] = field(default_factory=list)


@dataclass
class HypothesisScore:
    """Score for a shape hypothesis."""

    hypothesis: ShapeHypothesis
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class PseudoKeySignals:
    """Signals indicating a column may be a pseudo-key.

    Pseudo-keys are columns that appear unique but don't represent meaningful
    business keys (e.g., row indices, UUIDs, auto-increment IDs).
    """

    is_monotonic_sequence: bool = False  # 1, 2, 3, ... N pattern
    is_uuid_like: bool = False  # High entropy, no repeats, hex-like
    is_ingestion_timestamp: bool = False  # created_at with tiny deltas
    name_signal_penalty: float = 0.0  # Penalty from name patterns (row_id, index, etc.)
    total_penalty: float = 0.0  # Combined penalty (capped at 0.5)


@dataclass
class GrainDiagnostics:
    """Diagnostics for user feedback about grain quality."""

    duplicate_groups: int = 0  # Number of key combinations with duplicates
    max_group_size: int = 0  # Largest duplicate group size
    rows_with_null_in_key: int = 0  # Rows with nulls in key columns
    null_columns: list[str] = field(default_factory=list)  # Key columns with nulls
    example_duplicate_keys: list[tuple] = field(default_factory=list)  # Sample duplicates (max 3)


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
class DecisionRecordSummary:
    """Summary of a decision record for listing purposes."""

    decision_id: str
    dataset_fingerprint: str
    timestamp: datetime
    path: Path
    selected_hypothesis: str


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

    # Role scoring thresholds
    key_cardinality_threshold: float = 0.9
    dimension_cardinality_max: float = 0.1
    null_rate_threshold: float = 0.01

    # Shape detection
    min_time_columns_for_wide: int = 3

    # Grain inference
    max_grain_columns: int = 4
    min_uniqueness_confidence: float = 0.95

    # Ambiguity thresholds
    hypothesis_confidence_gap: float = 0.1

    # Optional adapters
    use_frictionless: bool = False
    use_dataprofiler: bool = False
