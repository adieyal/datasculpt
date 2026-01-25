"""Interactive clarification module for ambiguity resolution.

This module handles question generation and answer application for cases
where automated inference has low confidence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from datasculpt.core.time import GranularityResult, TimeGranularity
from datasculpt.core.types import (
    ColumnEvidence,
    GrainInference,
    Question,
    QuestionType,
    Role,
    ShapeHypothesis,
)

# Default confidence threshold for generating questions
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class InferenceConstraints:
    """User-provided constraints that lock specific inference outcomes.

    These constraints override automated inference results when set.
    """

    # Shape constraint (if set, locks the shape hypothesis)
    shape: ShapeHypothesis | None = None

    # Role constraints (column_name -> locked role)
    column_roles: dict[str, Role] = field(default_factory=dict)

    # Grain constraint (if set, locks the grain columns)
    grain_columns: list[str] | None = None

    # Time granularity constraints (column_name -> locked granularity)
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

    def get_locked_time_granularity(self, column_name: str) -> TimeGranularity | None:
        """Get the locked time granularity for a column, if any."""
        return self.time_granularities.get(column_name)

    def get_locked_role(self, column_name: str) -> Role | None:
        """Get the locked role for a column, if any."""
        return self.column_roles.get(column_name)


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
    """Tracks questions posed and answers received during inference.

    Maintains the full history of user interactions and resulting constraints.
    """

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


# -----------------------------------------------------------------------------
# Question generators
# -----------------------------------------------------------------------------


def generate_shape_question(
    current_hypothesis: ShapeHypothesis,
    confidence: float,
    alternatives: list[tuple[ShapeHypothesis, float]] | None = None,
) -> Question:
    """Generate a question asking user to confirm or select dataset shape.

    Args:
        current_hypothesis: The currently inferred shape hypothesis.
        confidence: Confidence score for current hypothesis.
        alternatives: Optional list of (hypothesis, score) tuples for alternatives.

    Returns:
        Question with shape choices.
    """
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
    """Generate a question asking user to confirm or override a column's role.

    Args:
        column_name: Name of the column.
        current_role: Currently assigned role.
        confidence: Confidence score for current assignment.
        evidence: Optional column evidence for rationale.

    Returns:
        Question with role choices.
    """
    choices = [
        {
            "value": Role.KEY.value,
            "label": "Key",
            "description": "Primary or foreign key identifier.",
        },
        {
            "value": Role.DIMENSION.value,
            "label": "Dimension",
            "description": "Categorical grouping variable (e.g., country, category).",
        },
        {
            "value": Role.MEASURE.value,
            "label": "Measure",
            "description": "Numeric fact or metric to aggregate.",
        },
        {
            "value": Role.TIME.value,
            "label": "Time",
            "description": "Date or timestamp column.",
        },
        {
            "value": Role.INDICATOR_NAME.value,
            "label": "Indicator Name",
            "description": "Column containing indicator/variable names (long format).",
        },
        {
            "value": Role.VALUE.value,
            "label": "Value",
            "description": "Column containing indicator values (paired with indicator name).",
        },
        {
            "value": Role.SERIES.value,
            "label": "Series",
            "description": "JSON array containing time series data.",
        },
        {
            "value": Role.METADATA.value,
            "label": "Metadata",
            "description": "Descriptive information, not used for analysis.",
        },
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
    """Generate a question asking user to confirm or specify grain columns.

    Args:
        key_columns: Currently inferred key columns.
        confidence: Confidence score for grain inference.
        uniqueness_ratio: Uniqueness ratio achieved by key columns.
        alternative_columns: Optional list of other candidate columns.

    Returns:
        Question for grain confirmation.
    """
    grain_str = f"[{', '.join(key_columns)}]" if key_columns else "(none)"

    choices = [
        {
            "value": "yes",
            "label": "Yes, this is correct",
            "description": f"Grain is {grain_str}.",
        },
        {
            "value": "no",
            "label": "No, let me specify",
            "description": "I will provide the correct grain columns.",
        },
    ]

    # Add alternative suggestions if available
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


def generate_grain_columns_question(
    available_columns: list[str],
    current_columns: list[str] | None = None,
) -> Question:
    """Generate a question for user to select grain columns.

    This is a follow-up question when user indicates inferred grain is wrong.

    Args:
        available_columns: All columns available in the dataset.
        current_columns: Currently selected grain columns (for defaults).

    Returns:
        Question with multi-select column choices.
    """
    choices = [
        {
            "value": col,
            "label": col,
            "description": "",
        }
        for col in available_columns
    ]

    return Question(
        id=generate_question_id(),
        type=QuestionType.CHOOSE_MANY,
        prompt="Which columns form the grain (unique key) of this dataset?",
        choices=choices,
        default=current_columns if current_columns else [],
        rationale="Select the minimum set of columns that uniquely identifies each row.",
    )


def generate_time_granularity_question(
    column_name: str,
    detected_granularity: TimeGranularity | None = None,
    confidence: float = 0.0,
) -> Question:
    """Generate a question asking user to confirm or specify time granularity.

    Args:
        column_name: Name of the time column.
        detected_granularity: Currently detected granularity (if any).
        confidence: Confidence score for detected granularity.

    Returns:
        Question with granularity choices.
    """
    choices = [
        {
            "value": TimeGranularity.DAILY.value,
            "label": "Daily",
            "description": "One observation per day.",
        },
        {
            "value": TimeGranularity.WEEKLY.value,
            "label": "Weekly",
            "description": "One observation per week.",
        },
        {
            "value": TimeGranularity.MONTHLY.value,
            "label": "Monthly",
            "description": "One observation per month.",
        },
        {
            "value": TimeGranularity.QUARTERLY.value,
            "label": "Quarterly",
            "description": "One observation per quarter (Q1, Q2, Q3, Q4).",
        },
        {
            "value": TimeGranularity.ANNUAL.value,
            "label": "Annual",
            "description": "One observation per year.",
        },
    ]

    rationale_parts = []
    if detected_granularity and detected_granularity != TimeGranularity.UNKNOWN:
        rationale_parts.append(
            f"Detected granularity: {detected_granularity.value} "
            f"(confidence: {confidence:.0%})."
        )
    else:
        rationale_parts.append("Could not automatically detect time granularity.")

    # Set default if confidence is high enough
    default_value = None
    if (
        detected_granularity
        and detected_granularity != TimeGranularity.UNKNOWN
        and confidence >= DEFAULT_CONFIDENCE_THRESHOLD
    ):
        default_value = detected_granularity.value

    return Question(
        id=generate_question_id(),
        type=QuestionType.CHOOSE_ONE,
        prompt=f"What is the time granularity of column '{column_name}'?",
        choices=choices,
        default=default_value,
        rationale=" ".join(rationale_parts),
    )


# -----------------------------------------------------------------------------
# Conditional question generation
# -----------------------------------------------------------------------------


def needs_shape_question(
    confidence: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a shape confirmation question is needed.

    Args:
        confidence: Shape detection confidence score.
        threshold: Confidence threshold below which to ask.

    Returns:
        True if question should be generated.
    """
    return confidence < threshold


def needs_role_question(
    confidence: float,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a role confirmation question is needed.

    Args:
        confidence: Role assignment confidence score.
        threshold: Confidence threshold below which to ask.

    Returns:
        True if question should be generated.
    """
    return confidence < threshold


def needs_grain_question(
    grain: GrainInference,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a grain confirmation question is needed.

    Args:
        grain: Grain inference result.
        threshold: Confidence threshold below which to ask.

    Returns:
        True if question should be generated.
    """
    return grain.confidence < threshold


def needs_time_question(
    granularity_result: GranularityResult,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a time granularity question is needed.

    Args:
        granularity_result: Time granularity detection result.
        threshold: Confidence threshold below which to ask.

    Returns:
        True if question should be generated.
    """
    # Always ask if granularity is unknown
    if granularity_result.granularity == TimeGranularity.UNKNOWN:
        return True

    # Ask if confidence is below threshold
    return granularity_result.confidence < threshold


# -----------------------------------------------------------------------------
# Answer application
# -----------------------------------------------------------------------------


def apply_shape_answer(
    session: ClarificationSession,
    question_id: str,
    answer: str,
    original_hypothesis: ShapeHypothesis,
) -> None:
    """Apply user's shape answer to session constraints.

    Args:
        session: Current clarification session.
        question_id: ID of the answered question.
        answer: User's answer (shape hypothesis value).
        original_hypothesis: The originally inferred hypothesis.
    """
    session.add_answer(question_id, answer)

    new_shape = ShapeHypothesis(answer)
    session.constraints.shape = new_shape

    if new_shape != original_hypothesis:
        session.add_override(Override(
            question_id=question_id,
            question_type=QuestionType.CHOOSE_ONE,
            original_value=original_hypothesis.value,
            user_value=answer,
        ))


def apply_role_answer(
    session: ClarificationSession,
    question_id: str,
    answer: str,
    column_name: str,
    original_role: Role,
) -> None:
    """Apply user's role answer to session constraints.

    Args:
        session: Current clarification session.
        question_id: ID of the answered question.
        answer: User's answer (role value).
        column_name: Name of the column.
        original_role: The originally inferred role.
    """
    session.add_answer(question_id, answer)

    new_role = Role(answer)
    session.constraints.column_roles[column_name] = new_role

    if new_role != original_role:
        session.add_override(Override(
            question_id=question_id,
            question_type=QuestionType.CHOOSE_ONE,
            original_value=original_role.value,
            user_value=answer,
            column_name=column_name,
        ))


def apply_grain_answer(
    session: ClarificationSession,
    question_id: str,
    answer: str | list[str],
    original_columns: list[str],
) -> None:
    """Apply user's grain answer to session constraints.

    Args:
        session: Current clarification session.
        question_id: ID of the answered question.
        answer: User's answer ('yes', 'no', 'alternative', or list of columns).
        original_columns: The originally inferred grain columns.
    """
    session.add_answer(question_id, answer)

    if answer == "yes":
        # User confirmed original grain
        session.constraints.grain_columns = original_columns
    elif isinstance(answer, list):
        # User provided specific columns
        session.constraints.grain_columns = answer
        if answer != original_columns:
            session.add_override(Override(
                question_id=question_id,
                question_type=QuestionType.CHOOSE_MANY,
                original_value=original_columns,
                user_value=answer,
            ))


def apply_grain_columns_answer(
    session: ClarificationSession,
    question_id: str,
    columns: list[str],
    original_columns: list[str],
) -> None:
    """Apply user's grain columns selection to session constraints.

    Args:
        session: Current clarification session.
        question_id: ID of the answered question.
        columns: User-selected grain columns.
        original_columns: The originally inferred grain columns.
    """
    session.add_answer(question_id, columns)
    session.constraints.grain_columns = columns

    if columns != original_columns:
        session.add_override(Override(
            question_id=question_id,
            question_type=QuestionType.CHOOSE_MANY,
            original_value=original_columns,
            user_value=columns,
        ))


def apply_time_answer(
    session: ClarificationSession,
    question_id: str,
    answer: str,
    column_name: str,
    original_granularity: TimeGranularity | None = None,
) -> None:
    """Apply user's time granularity answer to session constraints.

    Args:
        session: Current clarification session.
        question_id: ID of the answered question.
        answer: User's answer (granularity value).
        column_name: Name of the time column.
        original_granularity: The originally detected granularity.
    """
    session.add_answer(question_id, answer)

    new_granularity = TimeGranularity(answer)
    session.constraints.time_granularities[column_name] = new_granularity

    if original_granularity is None or new_granularity != original_granularity:
        session.add_override(Override(
            question_id=question_id,
            question_type=QuestionType.CHOOSE_ONE,
            original_value=original_granularity.value if original_granularity else None,
            user_value=answer,
            column_name=column_name,
        ))


# -----------------------------------------------------------------------------
# Constraint-based re-inference helpers
# -----------------------------------------------------------------------------


def apply_constraints_to_evidence(
    evidence: dict[str, ColumnEvidence],
    constraints: InferenceConstraints,
) -> dict[str, ColumnEvidence]:
    """Apply role constraints to column evidence for re-inference.

    When a user locks a column's role, we set that role's score to 1.0
    and all other roles to 0.0 to force the assignment.

    Args:
        evidence: Column evidence dictionary.
        constraints: User-provided constraints.

    Returns:
        Updated column evidence with locked roles applied.
    """
    for col_name, locked_role in constraints.column_roles.items():
        if col_name in evidence:
            col_evidence = evidence[col_name]
            # Set all role scores to 0.0
            col_evidence.role_scores = dict.fromkeys(Role, 0.0)
            # Set locked role to 1.0
            col_evidence.role_scores[locked_role] = 1.0
            # Add note about override
            col_evidence.notes.append(f"Role locked by user: {locked_role.value}")

    return evidence


def apply_constraints_to_grain(
    grain: GrainInference,
    constraints: InferenceConstraints,
) -> GrainInference:
    """Apply grain constraints to grain inference result.

    When a user locks the grain columns, we override the inference.

    Args:
        grain: Original grain inference result.
        constraints: User-provided constraints.

    Returns:
        Updated grain inference with locked columns applied.
    """
    if not constraints.is_grain_locked():
        return grain

    locked_columns = constraints.grain_columns
    if locked_columns is None:
        return grain

    return GrainInference(
        key_columns=locked_columns,
        confidence=1.0,  # User-confirmed
        uniqueness_ratio=grain.uniqueness_ratio,  # Keep original for audit
        evidence=grain.evidence + ["Grain columns locked by user"],
    )


# -----------------------------------------------------------------------------
# Session management
# -----------------------------------------------------------------------------


def create_session() -> ClarificationSession:
    """Create a new clarification session."""
    return ClarificationSession()


def session_to_dict(session: ClarificationSession) -> dict[str, Any]:
    """Serialize session to dictionary for storage.

    Args:
        session: Session to serialize.

    Returns:
        Dictionary representation.
    """
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
    """Deserialize session from dictionary.

    Args:
        data: Dictionary representation.

    Returns:
        Reconstructed session.
    """
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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _format_hypothesis(hypothesis: ShapeHypothesis) -> str:
    """Format hypothesis enum as human-readable string."""
    return hypothesis.value.replace("_", " ").title()
