"""Inference wrapper for browser execution.

This module provides a simplified interface for running datasculpt inference
in the browser via PyScript.
"""


from typing import Any

import pandas as pd

from datasculpt.pipeline import InferenceResult, apply_answers, infer
from datasculpt.proposal import inference_result_to_dict, proposal_to_dict
from datasculpt.types import InferenceConfig


def run_inference(
    df: pd.DataFrame,
    filename: str = "unnamed_dataset",
    interactive: bool = True,
) -> InferenceResult:
    """Run datasculpt inference on a DataFrame.

    Args:
        df: DataFrame to analyze.
        filename: Name of the source file (used as dataset name).
        interactive: If True, generates questions for ambiguous aspects.

    Returns:
        InferenceResult with proposal, decision record, and pending questions.
    """
    # Extract dataset name from filename
    dataset_name = filename.rsplit(".", 1)[0] if "." in filename else filename

    config = InferenceConfig()

    return infer(
        df=df,
        dataset_name=dataset_name,
        config=config,
        interactive=interactive,
    )


def rerun_with_answers(
    result: InferenceResult,
    answers: dict[str, Any],
) -> InferenceResult:
    """Re-run inference with user-provided answers.

    Args:
        result: Previous inference result.
        answers: Dictionary mapping question IDs to user answers.

    Returns:
        New InferenceResult with answers applied.
    """
    return apply_answers(result, answers)


def get_result_summary(result: InferenceResult) -> dict[str, Any]:
    """Get a summary of inference results for UI display.

    Args:
        result: Inference result to summarize.

    Returns:
        Dictionary with summary information.
    """
    proposal = result.proposal
    record = result.decision_record

    # Build shape hypothesis info
    shape_info = []
    for h in record.hypotheses:
        shape_info.append({
            "name": h.hypothesis.value,
            "label": h.hypothesis.value.replace("_", " ").title(),
            "score": h.score,
            "reasons": h.reasons,
            "is_winner": h.hypothesis == proposal.shape_hypothesis,
        })

    # Build column info
    column_info = []
    for col in proposal.columns:
        evidence = record.column_evidence.get(col.name)
        alternatives = []
        if evidence and evidence.role_scores:
            sorted_roles = sorted(
                evidence.role_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            alternatives = [
                {"role": r.value, "score": s}
                for r, s in sorted_roles[:4]
                if r != col.role
            ]

        column_info.append({
            "name": col.name,
            "role": col.role.value,
            "primitive_type": col.primitive_type.value,
            "structural_type": col.structural_type.value,
            "null_rate": evidence.null_rate if evidence else 0,
            "distinct_ratio": evidence.distinct_ratio if evidence else 0,
            "alternatives": alternatives,
        })

    # Build grain info
    grain_info = {
        "key_columns": record.grain.key_columns,
        "confidence": record.grain.confidence,
        "uniqueness_ratio": record.grain.uniqueness_ratio,
        "evidence": record.grain.evidence,
    }

    # Build questions info
    questions_info = [
        {
            "id": q.id,
            "type": q.type.value,
            "prompt": q.prompt,
            "choices": q.choices,
            "default": q.default,
            "rationale": q.rationale,
        }
        for q in result.pending_questions
    ]

    return {
        "dataset_name": proposal.dataset_name,
        "dataset_kind": proposal.dataset_kind.value,
        "row_count": len(result.dataframe) if result.dataframe is not None else 0,
        "column_count": len(proposal.columns),
        "shapes": shape_info,
        "columns": column_info,
        "grain": grain_info,
        "warnings": proposal.warnings,
        "questions": questions_info,
    }


def get_export_data(result: InferenceResult) -> dict[str, Any]:
    """Get full export data for JSON download.

    Args:
        result: Inference result to export.

    Returns:
        Dictionary suitable for JSON export.
    """
    return inference_result_to_dict(result)
