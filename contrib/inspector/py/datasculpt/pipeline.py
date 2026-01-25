"""Main inference pipeline for Datasculpt (browser bundle)."""


import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from datasculpt.evidence import extract_dataframe_evidence
from datasculpt.grain import infer_grain
from datasculpt.roles import assign_roles, update_evidence_with_roles
from datasculpt.shapes import detect_shape
from datasculpt.types import (
    ColumnEvidence,
    ColumnSpec,
    DatasetKind,
    DecisionRecord,
    GrainInference,
    HypothesisScore,
    InferenceConfig,
    InvariantProposal,
    Question,
    QuestionType,
    Role,
    ShapeHypothesis,
)


@dataclass
class InferenceResult:
    """Complete result of an inference run."""

    proposal: InvariantProposal
    decision_record: DecisionRecord
    pending_questions: list[Question]
    dataframe: pd.DataFrame | None = None


# Shape to DatasetKind mapping
SHAPE_TO_KIND: dict[ShapeHypothesis, DatasetKind] = {
    ShapeHypothesis.LONG_OBSERVATIONS: DatasetKind.OBSERVATIONS,
    ShapeHypothesis.LONG_INDICATORS: DatasetKind.INDICATORS_LONG,
    ShapeHypothesis.WIDE_OBSERVATIONS: DatasetKind.OBSERVATIONS,
    ShapeHypothesis.WIDE_TIME_COLUMNS: DatasetKind.TIMESERIES_WIDE,
    ShapeHypothesis.SERIES_COLUMN: DatasetKind.TIMESERIES_SERIES,
}


def _generate_decision_id() -> str:
    """Generate a unique decision ID."""
    return f"dec_{uuid.uuid4().hex[:12]}"


def _generate_question_id() -> str:
    """Generate a unique question ID."""
    return f"q_{uuid.uuid4().hex[:8]}"


def _generate_shape_question(
    hypotheses: list[HypothesisScore],
    ambiguity_details: list[str],
) -> Question:
    """Generate a question about ambiguous shape detection."""
    choices = []
    for h in hypotheses[:3]:
        choices.append({
            "value": h.hypothesis.value,
            "label": h.hypothesis.value.replace("_", " ").title(),
            "score": h.score,
            "reasons": h.reasons[:2],
        })

    rationale = "; ".join(ambiguity_details) if ambiguity_details else None

    return Question(
        id=_generate_question_id(),
        type=QuestionType.CHOOSE_ONE,
        prompt="The dataset shape is ambiguous. Please select the most appropriate shape:",
        choices=choices,
        default=hypotheses[0].hypothesis.value if hypotheses else None,
        rationale=rationale,
    )


def _generate_grain_question(
    grain: GrainInference,
    all_columns: list[str],
) -> Question:
    """Generate a question about uncertain grain inference."""
    choices = []

    if grain.key_columns:
        choices.append({
            "value": grain.key_columns,
            "label": f"Inferred: {', '.join(grain.key_columns)} ({grain.uniqueness_ratio:.1%} unique)",
        })

    for col in all_columns[:10]:
        if col not in grain.key_columns:
            choices.append({
                "value": [col],
                "label": col,
            })

    evidence_text = "; ".join(grain.evidence) if grain.evidence else None

    return Question(
        id=_generate_question_id(),
        type=QuestionType.CHOOSE_MANY,
        prompt="Please confirm or select the grain (unique key columns) for this dataset:",
        choices=choices,
        default=grain.key_columns if grain.key_columns else None,
        rationale=evidence_text,
    )


def _generate_role_question(
    column_name: str,
    evidence: ColumnEvidence,
    current_role: Role,
    confidence: float,
) -> Question:
    """Generate a question about ambiguous column role."""
    sorted_roles = sorted(
        evidence.role_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    choices = []
    for role, score in sorted_roles[:5]:
        choices.append({
            "value": role.value,
            "label": f"{role.value} ({score:.2f})",
        })

    return Question(
        id=_generate_question_id(),
        type=QuestionType.CHOOSE_ONE,
        prompt=f"What is the role of column '{column_name}'?",
        choices=choices,
        default=current_role.value,
        rationale=f"Current assignment: {current_role.value} (confidence: {confidence:.2f})",
    )


def _generate_questions(
    shape_result: Any,
    grain: GrainInference,
    column_evidence: dict[str, ColumnEvidence],
    role_assignments: dict[str, Any],
    config: InferenceConfig,
) -> list[Question]:
    """Generate questions for ambiguous aspects of inference."""
    questions: list[Question] = []

    if shape_result.is_ambiguous:
        questions.append(
            _generate_shape_question(
                shape_result.ranked_hypotheses,
                shape_result.ambiguity_details,
            )
        )

    if grain.confidence < 0.8:
        all_columns = list(column_evidence.keys())
        questions.append(_generate_grain_question(grain, all_columns))

    for col_name, assignment in role_assignments.items():
        if assignment.confidence < config.hypothesis_confidence_gap * 2:
            evidence = column_evidence.get(col_name)
            if evidence:
                questions.append(
                    _generate_role_question(
                        col_name,
                        evidence,
                        assignment.role,
                        assignment.confidence,
                    )
                )

    return questions


def _create_decision_record(
    decision_id: str,
    fingerprint: str,
    shape_result: Any,
    grain: GrainInference,
    column_evidence: dict[str, ColumnEvidence],
    questions: list[Question],
    answers: dict[str, Any],
) -> DecisionRecord:
    """Create a complete decision record for the inference run."""
    return DecisionRecord(
        decision_id=decision_id,
        dataset_fingerprint=fingerprint,
        timestamp=datetime.now(),
        selected_hypothesis=shape_result.selected,
        hypotheses=shape_result.ranked_hypotheses,
        grain=grain,
        column_evidence=column_evidence,
        questions=questions,
        answers=answers,
    )


def _create_column_spec(
    col_name: str,
    evidence: ColumnEvidence,
    role: Role,
) -> ColumnSpec:
    """Create a column specification from evidence and role."""
    return ColumnSpec(
        name=col_name,
        role=role,
        primitive_type=evidence.primitive_type,
        structural_type=evidence.structural_type,
        notes=list(evidence.notes),
    )


def _create_proposal(
    dataset_name: str,
    shape_hypothesis: ShapeHypothesis,
    grain: GrainInference,
    column_evidence: dict[str, ColumnEvidence],
    role_assignments: dict[str, Any],
    decision_id: str,
    questions: list[Question],
) -> InvariantProposal:
    """Create an Invariant proposal from inference results."""
    dataset_kind = SHAPE_TO_KIND.get(shape_hypothesis, DatasetKind.OBSERVATIONS)

    columns: list[ColumnSpec] = []
    for col_name, evidence in column_evidence.items():
        role = role_assignments[col_name].role if col_name in role_assignments else Role.METADATA
        columns.append(_create_column_spec(col_name, evidence, role))

    warnings: list[str] = []
    if grain.confidence < 0.9:
        warnings.append(
            f"Grain confidence is {grain.confidence:.2f}. "
            "Consider verifying the unique key columns."
        )

    if grain.uniqueness_ratio < 1.0:
        warnings.append(
            f"Grain uniqueness is {grain.uniqueness_ratio:.1%}. "
            "Dataset may have duplicate rows."
        )

    required_confirmations = [q.prompt for q in questions]

    return InvariantProposal(
        dataset_name=dataset_name,
        dataset_kind=dataset_kind,
        shape_hypothesis=shape_hypothesis,
        grain=grain.key_columns,
        columns=columns,
        warnings=warnings,
        required_user_confirmations=required_confirmations,
        decision_record_id=decision_id,
    )


def infer(
    df: pd.DataFrame,
    dataset_name: str = "unnamed_dataset",
    config: InferenceConfig | None = None,
    interactive: bool = False,
    answers: dict[str, Any] | None = None,
) -> InferenceResult:
    """Main entry point for dataset inference.

    This is a browser-adapted version that takes a DataFrame directly
    instead of a file path.

    Args:
        df: DataFrame to analyze.
        dataset_name: Name for the dataset.
        config: Optional inference configuration.
        interactive: If True, generates questions for ambiguous aspects.
        answers: Pre-provided answers for questions.

    Returns:
        InferenceResult containing proposal, decision record, and questions.
    """
    if config is None:
        config = InferenceConfig()

    if answers is None:
        answers = {}

    # Generate fingerprint
    fingerprint = f"df_{hash(tuple(df.columns))}_{len(df)}"

    # Extract evidence for each column
    column_evidence = extract_dataframe_evidence(df)

    # Score roles for each column
    evidences = list(column_evidence.values())
    role_assignments = assign_roles(evidences, config)

    # Update evidence with role scores
    has_indicator = any(
        ra.role == Role.INDICATOR_NAME
        for ra in role_assignments.values()
    )

    for _col_name, evidence in column_evidence.items():
        update_evidence_with_roles(evidence, config, has_indicator)

    # Detect shape
    shape_result = detect_shape(evidences, config)

    # Apply shape answer if provided
    shape_question_ids = [
        q_id for q_id, ans in answers.items()
        if isinstance(ans, str) and ans in [h.value for h in ShapeHypothesis]
    ]
    if shape_question_ids:
        shape_value = answers[shape_question_ids[0]]
        shape_result.selected = ShapeHypothesis(shape_value)
        shape_result.is_ambiguous = False

    # Infer grain (shape-aware)
    grain = infer_grain(df, column_evidence, config, shape_result.selected)

    # Apply grain answer if provided
    grain_question_ids = [
        q_id for q_id, ans in answers.items()
        if isinstance(ans, list) and all(isinstance(x, str) for x in ans)
    ]
    if grain_question_ids:
        grain_columns = answers[grain_question_ids[0]]
        if grain_columns:
            grain = GrainInference(
                key_columns=grain_columns,
                confidence=1.0,
                uniqueness_ratio=grain.uniqueness_ratio,
                evidence=grain.evidence + ["User confirmed grain columns"],
            )

    # Generate questions (if interactive)
    questions: list[Question] = []
    if interactive:
        questions = _generate_questions(
            shape_result,
            grain,
            column_evidence,
            role_assignments,
            config,
        )
        questions = [q for q in questions if q.id not in answers]

    # Create decision record
    decision_id = _generate_decision_id()
    decision_record = _create_decision_record(
        decision_id=decision_id,
        fingerprint=fingerprint,
        shape_result=shape_result,
        grain=grain,
        column_evidence=column_evidence,
        questions=questions,
        answers=answers,
    )

    # Generate proposal
    proposal = _create_proposal(
        dataset_name=dataset_name,
        shape_hypothesis=shape_result.selected,
        grain=grain,
        column_evidence=column_evidence,
        role_assignments=role_assignments,
        decision_id=decision_id,
        questions=questions,
    )

    return InferenceResult(
        proposal=proposal,
        decision_record=decision_record,
        pending_questions=questions,
        dataframe=df,
    )


def apply_answers(
    previous_result: InferenceResult,
    answers: dict[str, Any],
    config: InferenceConfig | None = None,
) -> InferenceResult:
    """Re-run inference with user-provided answers applied.

    Args:
        previous_result: Result from a previous infer() call.
        answers: Dictionary mapping question IDs to user answers.
        config: Optional inference configuration.

    Returns:
        New InferenceResult with answers applied.
    """
    if previous_result.dataframe is None:
        raise ValueError("Cannot apply answers: previous result has no DataFrame.")

    merged_answers = dict(previous_result.decision_record.answers)
    merged_answers.update(answers)

    return infer(
        df=previous_result.dataframe,
        dataset_name=previous_result.proposal.dataset_name,
        config=config,
        interactive=True,
        answers=merged_answers,
    )
