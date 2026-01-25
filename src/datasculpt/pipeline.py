"""Main inference pipeline for Datasculpt.

This module provides the main entry point for dataset inference, orchestrating
the loading, evidence extraction, role scoring, shape detection, grain inference,
question generation, decision recording, and proposal generation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from datasculpt.core.evidence import extract_dataframe_evidence
from datasculpt.core.grain import infer_grain
from datasculpt.core.roles import assign_roles, update_evidence_with_roles
from datasculpt.core.shapes import detect_shape
from datasculpt.core.types import (
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
from datasculpt.intake import IntakeResult, intake_file


@dataclass
class InferenceResult:
    """Complete result of an inference run.

    Attributes:
        proposal: The generated InvariantProposal for dataset registration.
        decision_record: Complete audit trail of the inference process.
        pending_questions: Questions requiring user input for ambiguity resolution.
        dataframe: The loaded DataFrame, available for further processing.
    """

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


def _extract_dataset_name(source: str | Path | pd.DataFrame) -> str:
    """Extract a dataset name from the source.

    Args:
        source: File path or DataFrame.

    Returns:
        A suitable dataset name.
    """
    if isinstance(source, pd.DataFrame):
        return "unnamed_dataset"

    path = Path(source)
    return path.stem


def _generate_shape_question(
    hypotheses: list[HypothesisScore],
    ambiguity_details: list[str],
) -> Question:
    """Generate a question about ambiguous shape detection.

    Args:
        hypotheses: Ranked list of shape hypotheses.
        ambiguity_details: Details about why shape is ambiguous.

    Returns:
        Question for user to resolve shape ambiguity.
    """
    choices = []
    for h in hypotheses[:3]:  # Top 3 choices
        choices.append({
            "value": h.hypothesis.value,
            "label": h.hypothesis.value.replace("_", " ").title(),
            "score": h.score,
            "reasons": h.reasons[:2],  # First 2 reasons
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
    """Generate a question about uncertain grain inference.

    Args:
        grain: The inferred grain result.
        all_columns: All column names in the dataset.

    Returns:
        Question for user to confirm or correct grain.
    """
    choices = []

    # Add inferred grain as first option
    if grain.key_columns:
        choices.append({
            "value": grain.key_columns,
            "label": f"Inferred: {', '.join(grain.key_columns)} ({grain.uniqueness_ratio:.1%} unique)",
        })

    # Add individual columns as alternatives
    for col in all_columns[:10]:  # Limit choices
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
    """Generate a question about ambiguous column role.

    Args:
        column_name: Name of the column.
        evidence: Column evidence with role scores.
        current_role: Currently assigned role.
        confidence: Confidence in the assignment.

    Returns:
        Question for user to confirm or correct role.
    """
    # Sort roles by score
    sorted_roles = sorted(
        evidence.role_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    choices = []
    for role, score in sorted_roles[:5]:  # Top 5 roles
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
    shape_result: Any,  # ShapeResult from shapes module
    grain: GrainInference,
    column_evidence: dict[str, ColumnEvidence],
    role_assignments: dict[str, Any],  # RoleAssignment from roles module
    config: InferenceConfig,
) -> list[Question]:
    """Generate questions for ambiguous aspects of inference.

    Args:
        shape_result: Shape detection result.
        grain: Grain inference result.
        column_evidence: Evidence for each column.
        role_assignments: Role assignments for each column.
        config: Inference configuration.

    Returns:
        List of questions for user resolution.
    """
    questions: list[Question] = []

    # Question about shape if ambiguous
    if shape_result.is_ambiguous:
        questions.append(
            _generate_shape_question(
                shape_result.ranked_hypotheses,
                shape_result.ambiguity_details,
            )
        )

    # Question about grain if low confidence
    if grain.confidence < 0.8:
        all_columns = list(column_evidence.keys())
        questions.append(_generate_grain_question(grain, all_columns))

    # Questions about low-confidence role assignments
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
    shape_result: Any,  # ShapeResult
    grain: GrainInference,
    column_evidence: dict[str, ColumnEvidence],
    questions: list[Question],
    answers: dict[str, Any],
) -> DecisionRecord:
    """Create a complete decision record for the inference run.

    Args:
        decision_id: Unique ID for this decision.
        fingerprint: Dataset fingerprint hash.
        shape_result: Shape detection result.
        grain: Grain inference result.
        column_evidence: Evidence for each column.
        questions: Questions generated during inference.
        answers: User-provided answers (if any).

    Returns:
        Complete DecisionRecord.
    """
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
    """Create a column specification from evidence and role.

    Args:
        col_name: Column name.
        evidence: Column evidence.
        role: Assigned role.

    Returns:
        ColumnSpec for proposal.
    """
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
    role_assignments: dict[str, Any],  # RoleAssignment
    decision_id: str,
    questions: list[Question],
) -> InvariantProposal:
    """Create an Invariant proposal from inference results.

    Args:
        dataset_name: Name for the dataset.
        shape_hypothesis: Selected shape hypothesis.
        grain: Inferred grain.
        column_evidence: Evidence for each column.
        role_assignments: Role assignments for each column.
        decision_id: ID of the decision record.
        questions: Pending questions.

    Returns:
        InvariantProposal for dataset registration.
    """
    # Map shape to dataset kind
    dataset_kind = SHAPE_TO_KIND.get(shape_hypothesis, DatasetKind.OBSERVATIONS)

    # Build column specs
    columns: list[ColumnSpec] = []
    for col_name, evidence in column_evidence.items():
        role = role_assignments[col_name].role if col_name in role_assignments else Role.METADATA
        columns.append(_create_column_spec(col_name, evidence, role))

    # Generate warnings
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

    # Required confirmations from questions
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
    source: str | Path | pd.DataFrame,
    config: InferenceConfig | None = None,
    interactive: bool = False,
    answers: dict[str, Any] | None = None,
) -> InferenceResult:
    """Main entry point for dataset inference.

    Orchestrates the full inference pipeline:
    1. Load file (if path provided)
    2. Extract evidence for each column
    3. Score roles for each column
    4. Detect dataset shape
    5. Infer grain (unique key)
    6. Generate questions for ambiguous aspects (if interactive)
    7. Create decision record
    8. Generate proposal

    Args:
        source: File path (str/Path) or DataFrame to analyze.
        config: Optional inference configuration. Uses defaults if None.
        interactive: If True, generates questions for ambiguous aspects.
        answers: Pre-provided answers for questions (from previous run).

    Returns:
        InferenceResult containing proposal, decision record, questions,
        and optionally the DataFrame.

    Example:
        >>> result = infer("data.csv")
        >>> print(result.proposal.dataset_kind)
        >>> print(result.decision_record.grain.key_columns)

        >>> # With interactive mode
        >>> result = infer("data.csv", interactive=True)
        >>> if result.pending_questions:
        ...     # Present questions to user, collect answers
        ...     answers = {"q_abc123": "long_observations"}
        ...     result = apply_answers(result, answers)
    """
    if config is None:
        config = InferenceConfig()

    if answers is None:
        answers = {}

    # Step 1: Load data
    df: pd.DataFrame
    fingerprint: str
    dataset_name: str

    if isinstance(source, pd.DataFrame):
        df = source.copy()
        fingerprint = f"df_{hash(tuple(df.columns))}_{len(df)}"
        dataset_name = "unnamed_dataset"
    else:
        intake_result: IntakeResult = intake_file(source)
        df = intake_result.dataframe
        fingerprint = intake_result.fingerprint.hash
        dataset_name = _extract_dataset_name(source)

    # Step 2: Extract evidence for each column
    column_evidence = extract_dataframe_evidence(df)

    # Step 3: Score roles for each column
    # First pass: detect if there's an indicator column
    evidences = list(column_evidence.values())
    role_assignments = assign_roles(evidences, config)

    # Update evidence with role scores
    has_indicator = any(
        ra.role == Role.INDICATOR_NAME
        for ra in role_assignments.values()
    )

    for _col_name, evidence in column_evidence.items():
        update_evidence_with_roles(evidence, config, has_indicator)

    # Step 4: Detect shape
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

    # Step 5: Infer grain (shape-aware)
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
                confidence=1.0,  # User confirmed
                uniqueness_ratio=grain.uniqueness_ratio,
                evidence=grain.evidence + ["User confirmed grain columns"],
            )

    # Step 6: Generate questions (if interactive)
    questions: list[Question] = []
    if interactive:
        questions = _generate_questions(
            shape_result,
            grain,
            column_evidence,
            role_assignments,
            config,
        )
        # Remove questions that already have answers
        questions = [q for q in questions if q.id not in answers]

    # Step 7: Create decision record
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

    # Step 8: Generate proposal
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

    This function takes a previous inference result and user answers,
    then re-runs inference with those answers applied to resolve
    ambiguities.

    Args:
        previous_result: Result from a previous infer() call.
        answers: Dictionary mapping question IDs to user answers.
        config: Optional inference configuration.

    Returns:
        New InferenceResult with answers applied.

    Example:
        >>> result = infer("data.csv", interactive=True)
        >>> if result.pending_questions:
        ...     # User answers the questions
        ...     answers = {
        ...         result.pending_questions[0].id: "long_observations",
        ...     }
        ...     result = apply_answers(result, answers)
    """
    if previous_result.dataframe is None:
        raise ValueError(
            "Cannot apply answers: previous result has no DataFrame. "
            "Re-run infer() with the original source."
        )

    # Merge previous answers with new answers
    merged_answers = dict(previous_result.decision_record.answers)
    merged_answers.update(answers)

    # Re-run inference with merged answers
    return infer(
        source=previous_result.dataframe,
        config=config,
        interactive=True,
        answers=merged_answers,
    )
