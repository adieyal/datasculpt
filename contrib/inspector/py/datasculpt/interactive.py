"""Interactive clarification module for Datasculpt (browser bundle)."""


import uuid
from dataclasses import dataclass, field
from typing import Any

from datasculpt.time_axis import GranularityResult, TimeGranularity
from datasculpt.types import (
    ColumnEvidence,
    GrainInference,
    Question,
    QuestionType,
    Role,
    ShapeHypothesis,
)


DEFAULT_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class InferenceConstraints:
    """User-provided constraints that lock specific inference outcomes."""

    shape: ShapeHypothesis | None = None
    column_roles: dict[str, Role] = field(default_factory=dict)
    grain_columns: list[str] | None = None
    time_granularities: dict[str, TimeGranularity] = field(default_factory=dict)

    def is_shape_locked(self) -> bool:
        """Check if shape hypothesis is locked by user."""
        return self.shape is not None

    def is_role_locked(self, column_name: str) -> bool:
        """Check if a column's role is locked by user."""
        return column_name in self.column_roles

    def is_grain_locked(self) -> bool:
        """Check if grain columns are locked by user."""
        return self.grain_columns is not None

    def is_time_granularity_locked(self, column_name: str) -> bool:
        """Check if a column's time granularity is locked by user."""
        return column_name in self.time_granularities


@dataclass
class Override:
    """Record of a single user override."""

    question_id: str
    question_type: QuestionType
    original_value: Any
    user_value: Any
    column_name: str | None = None


@dataclass
class ClarificationSession:
    """Tracks questions posed and answers received during inference."""

    questions: list[Question] = field(default_factory=list)
    answers: dict[str, Any] = field(default_factory=dict)
    overrides: list[Override] = field(default_factory=list)
    constraints: InferenceConstraints = field(default_factory=InferenceConstraints)

    def add_question(self, question: Question) -> None:
        """Add a question to the session."""
        self.questions.append(question)

    def add_answer(self, question_id: str, value: Any) -> None:
        """Record an answer for a question."""
        self.answers[question_id] = value

    def add_override(self, override: Override) -> None:
        """Record an override for audit trail."""
        self.overrides.append(override)

    def get_answer(self, question_id: str) -> Any | None:
        """Get the answer for a question, if provided."""
        return self.answers.get(question_id)

    def has_answer(self, question_id: str) -> bool:
        """Check if a question has been answered."""
        return question_id in self.answers

    def pending_questions(self) -> list[Question]:
        """Get questions that have not been answered."""
        return [q for q in self.questions if q.id not in self.answers]


def generate_question_id() -> str:
    """Generate a unique question ID."""
    return f"q_{uuid.uuid4().hex[:8]}"


def _format_hypothesis(hypothesis: ShapeHypothesis) -> str:
    """Format hypothesis enum as human-readable string."""
    return hypothesis.value.replace("_", " ").title()


# Question generators

def generate_shape_question(
    current_hypothesis: ShapeHypothesis,
    confidence: float,
    alternatives: list[tuple[ShapeHypothesis, float]] | None = None,
) -> Question:
    """Generate a question asking user to confirm or select dataset shape."""
    choices = [
        {
            "value": ShapeHypothesis.LONG_OBSERVATIONS.value,
            "label": "Long format (observations as rows)",
            "description": "Each row is one observation with dimension columns and measure columns.",
        },
        {
            "value": ShapeHypothesis.LONG_INDICATORS.value,
            "label": "Long format (indicator/value pairs)",
            "description": "One column has indicator names, another has values.",
        },
        {
            "value": ShapeHypothesis.WIDE_OBSERVATIONS.value,
            "label": "Wide format (measures as columns)",
            "description": "Multiple measure columns per row (e.g., height, weight, age).",
        },
        {
            "value": ShapeHypothesis.WIDE_TIME_COLUMNS.value,
            "label": "Wide format (time periods as columns)",
            "description": "Column headers are years, months, or dates.",
        },
        {
            "value": ShapeHypothesis.SERIES_COLUMN.value,
            "label": "Series column (JSON arrays)",
            "description": "One column contains JSON arrays representing time series.",
        },
    ]

    rationale_parts = [
        f"Inferred shape: {_format_hypothesis(current_hypothesis)} "
        f"(confidence: {confidence:.0%})."
    ]

    if alternatives:
        alt_strs = [f"{_format_hypothesis(h)} ({s:.0%})" for h, s in alternatives[:2]]
        rationale_parts.append(f"Alternatives considered: {', '.join(alt_strs)}.")

    return Question(
        id=generate_question_id(),
        type=QuestionType.CHOOSE_ONE,
        prompt="Is this dataset in wide or long format?",
        choices=choices,
        default=current_hypothesis.value,
        rationale=" ".join(rationale_parts),
    )


def generate_role_question(
    column_name: str,
    current_role: Role,
    confidence: float,
    evidence: ColumnEvidence | None = None,
) -> Question:
    """Generate a question asking user to confirm or override a column's role."""
    choices = [
        {"value": Role.KEY.value, "label": "Key", "description": "Primary or foreign key identifier."},
        {"value": Role.DIMENSION.value, "label": "Dimension", "description": "Categorical grouping variable."},
        {"value": Role.MEASURE.value, "label": "Measure", "description": "Numeric fact or metric to aggregate."},
        {"value": Role.TIME.value, "label": "Time", "description": "Date or timestamp column."},
        {"value": Role.INDICATOR_NAME.value, "label": "Indicator Name", "description": "Column containing indicator names."},
        {"value": Role.VALUE.value, "label": "Value", "description": "Column containing indicator values."},
        {"value": Role.SERIES.value, "label": "Series", "description": "JSON array containing time series data."},
        {"value": Role.METADATA.value, "label": "Metadata", "description": "Descriptive information, not used for analysis."},
    ]

    rationale_parts = [
        f"Inferred role: {current_role.value} (confidence: {confidence:.0%})."
    ]

    if evidence and evidence.notes:
        rationale_parts.append(f"Notes: {'; '.join(evidence.notes[:2])}.")

    return Question(
        id=generate_question_id(),
        type=QuestionType.CHOOSE_ONE,
        prompt=f"What role should column '{column_name}' have?",
        choices=choices,
        default=current_role.value,
        rationale=" ".join(rationale_parts),
    )


def generate_grain_question(
    key_columns: list[str],
    confidence: float,
    uniqueness_ratio: float,
    alternative_columns: list[str] | None = None,
) -> Question:
    """Generate a question asking user to confirm or specify grain columns."""
    grain_str = f"[{', '.join(key_columns)}]" if key_columns else "(none)"

    choices = [
        {"value": "yes", "label": "Yes, this is correct", "description": f"Grain is {grain_str}."},
        {"value": "no", "label": "No, let me specify", "description": "I will provide the correct grain columns."},
    ]

    if alternative_columns:
        alt_str = f"[{', '.join(alternative_columns)}]"
        choices.append({
            "value": "alternative",
            "label": f"Use alternative: {alt_str}",
            "description": "Use these columns instead.",
            "columns": alternative_columns,
        })

    rationale = (
        f"Inferred grain: {grain_str} with {uniqueness_ratio:.0%} uniqueness "
        f"(confidence: {confidence:.0%})."
    )

    return Question(
        id=generate_question_id(),
        type=QuestionType.CHOOSE_ONE,
        prompt=f"Is {grain_str} the correct grain (unique key) for this dataset?",
        choices=choices,
        default="yes" if confidence >= DEFAULT_CONFIDENCE_THRESHOLD else None,
        rationale=rationale,
    )


# Condition checkers

def needs_shape_question(
    confidence: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a shape confirmation question is needed."""
    return confidence < threshold


def needs_role_question(
    confidence: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a role confirmation question is needed."""
    return confidence < threshold


def needs_grain_question(
    grain: GrainInference,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a grain confirmation question is needed."""
    return grain.confidence < threshold


# Session management

def create_session() -> ClarificationSession:
    """Create a new clarification session."""
    return ClarificationSession()


def session_to_dict(session: ClarificationSession) -> dict[str, Any]:
    """Serialize session to dictionary for storage."""
    return {
        "questions": [
            {
                "id": q.id,
                "type": q.type.value,
                "prompt": q.prompt,
                "choices": q.choices,
                "default": q.default,
                "rationale": q.rationale,
            }
            for q in session.questions
        ],
        "answers": session.answers,
        "overrides": [
            {
                "question_id": o.question_id,
                "question_type": o.question_type.value,
                "original_value": o.original_value,
                "user_value": o.user_value,
                "column_name": o.column_name,
            }
            for o in session.overrides
        ],
        "constraints": {
            "shape": session.constraints.shape.value if session.constraints.shape else None,
            "column_roles": {
                col: role.value
                for col, role in session.constraints.column_roles.items()
            },
            "grain_columns": session.constraints.grain_columns,
            "time_granularities": {
                col: granularity.value
                for col, granularity in session.constraints.time_granularities.items()
            },
        },
    }


def session_from_dict(data: dict[str, Any]) -> ClarificationSession:
    """Deserialize session from dictionary."""
    session = ClarificationSession()

    for q_data in data.get("questions", []):
        session.questions.append(Question(
            id=q_data["id"],
            type=QuestionType(q_data["type"]),
            prompt=q_data["prompt"],
            choices=q_data.get("choices", []),
            default=q_data.get("default"),
            rationale=q_data.get("rationale"),
        ))

    session.answers = data.get("answers", {})

    for o_data in data.get("overrides", []):
        session.overrides.append(Override(
            question_id=o_data["question_id"],
            question_type=QuestionType(o_data["question_type"]),
            original_value=o_data["original_value"],
            user_value=o_data["user_value"],
            column_name=o_data.get("column_name"),
        ))

    constraints_data = data.get("constraints", {})
    if constraints_data.get("shape"):
        session.constraints.shape = ShapeHypothesis(constraints_data["shape"])

    for col, role_value in constraints_data.get("column_roles", {}).items():
        session.constraints.column_roles[col] = Role(role_value)

    session.constraints.grain_columns = constraints_data.get("grain_columns")

    for col, granularity_value in constraints_data.get("time_granularities", {}).items():
        session.constraints.time_granularities[col] = TimeGranularity(granularity_value)

    return session
