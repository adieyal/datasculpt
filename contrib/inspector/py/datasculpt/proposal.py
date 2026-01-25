"""Proposal generation and export for Datasculpt (browser bundle)."""


from typing import Any

from datasculpt.types import (
    ColumnSpec,
    DatasetKind,
    InvariantProposal,
    PrimitiveType,
    Role,
    ShapeHypothesis,
    StructuralType,
)


def proposal_to_dict(proposal: InvariantProposal) -> dict[str, Any]:
    """Convert an InvariantProposal to a dictionary for JSON export.

    Args:
        proposal: The proposal to convert.

    Returns:
        Dictionary representation of the proposal.
    """
    return {
        "dataset_name": proposal.dataset_name,
        "dataset_kind": proposal.dataset_kind.value,
        "shape_hypothesis": proposal.shape_hypothesis.value,
        "grain": proposal.grain,
        "columns": [
            {
                "name": col.name,
                "role": col.role.value,
                "primitive_type": col.primitive_type.value,
                "structural_type": col.structural_type.value,
                "reference_system_hint": col.reference_system_hint,
                "concept_hint": col.concept_hint,
                "unit_hint": col.unit_hint,
                "time_granularity": col.time_granularity,
                "notes": col.notes,
            }
            for col in proposal.columns
        ],
        "warnings": proposal.warnings,
        "required_user_confirmations": proposal.required_user_confirmations,
        "decision_record_id": proposal.decision_record_id,
    }


def decision_record_to_dict(record: Any) -> dict[str, Any]:
    """Convert a DecisionRecord to a dictionary for JSON export.

    Args:
        record: The decision record to convert.

    Returns:
        Dictionary representation of the decision record.
    """
    return {
        "decision_id": record.decision_id,
        "dataset_fingerprint": record.dataset_fingerprint,
        "timestamp": record.timestamp.isoformat(),
        "selected_hypothesis": record.selected_hypothesis.value,
        "hypotheses": [
            {
                "hypothesis": h.hypothesis.value,
                "score": h.score,
                "reasons": h.reasons,
            }
            for h in record.hypotheses
        ],
        "grain": {
            "key_columns": record.grain.key_columns,
            "confidence": record.grain.confidence,
            "uniqueness_ratio": record.grain.uniqueness_ratio,
            "evidence": record.grain.evidence,
        },
        "column_evidence": {
            name: {
                "name": ev.name,
                "primitive_type": ev.primitive_type.value,
                "structural_type": ev.structural_type.value,
                "null_rate": ev.null_rate,
                "distinct_ratio": ev.distinct_ratio,
                "unique_count": ev.unique_count,
                "role_scores": {
                    role.value: score
                    for role, score in ev.role_scores.items()
                },
                "notes": ev.notes,
            }
            for name, ev in record.column_evidence.items()
        },
        "questions": [
            {
                "id": q.id,
                "type": q.type.value,
                "prompt": q.prompt,
                "choices": q.choices,
                "default": q.default,
                "rationale": q.rationale,
            }
            for q in record.questions
        ],
        "answers": record.answers,
    }


def inference_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert an InferenceResult to a dictionary for JSON export.

    Args:
        result: The inference result to convert.

    Returns:
        Dictionary representation suitable for JSON export.
    """
    return {
        "proposal": proposal_to_dict(result.proposal),
        "decision_record": decision_record_to_dict(result.decision_record),
        "pending_questions": [
            {
                "id": q.id,
                "type": q.type.value,
                "prompt": q.prompt,
                "choices": q.choices,
                "default": q.default,
                "rationale": q.rationale,
            }
            for q in result.pending_questions
        ],
    }


# Validation support

VALID_ROLES = {r.value for r in Role}
VALID_PRIMITIVES = {p.value for p in PrimitiveType}
VALID_STRUCTURALS = {s.value for s in StructuralType}
VALID_KINDS = {k.value for k in DatasetKind}
VALID_SHAPES = {s.value for s in ShapeHypothesis}


def validate_proposal_dict(data: dict[str, Any]) -> list[str]:
    """Validate a proposal dictionary.

    Args:
        data: Proposal dictionary to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    required_fields = [
        "dataset_name",
        "dataset_kind",
        "shape_hypothesis",
        "grain",
        "columns",
    ]

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if "dataset_name" in data:
        if not isinstance(data["dataset_name"], str):
            errors.append("dataset_name must be a string")
        elif len(data["dataset_name"]) == 0:
            errors.append("dataset_name cannot be empty")

    if "dataset_kind" in data:
        if data["dataset_kind"] not in VALID_KINDS:
            errors.append(f"Invalid dataset_kind: {data['dataset_kind']}")

    if "shape_hypothesis" in data:
        if data["shape_hypothesis"] not in VALID_SHAPES:
            errors.append(f"Invalid shape_hypothesis: {data['shape_hypothesis']}")

    if "grain" in data:
        if not isinstance(data["grain"], list):
            errors.append("grain must be an array")

    if "columns" in data:
        if not isinstance(data["columns"], list):
            errors.append("columns must be an array")
        else:
            for i, col in enumerate(data["columns"]):
                if not isinstance(col, dict):
                    errors.append(f"columns[{i}] must be an object")
                    continue

                for field in ["name", "role", "primitive_type", "structural_type"]:
                    if field not in col:
                        errors.append(f"columns[{i}] missing required field: {field}")

                if "role" in col and col["role"] not in VALID_ROLES:
                    errors.append(f"columns[{i}].role invalid: {col['role']}")
                if "primitive_type" in col and col["primitive_type"] not in VALID_PRIMITIVES:
                    errors.append(f"columns[{i}].primitive_type invalid: {col['primitive_type']}")
                if "structural_type" in col and col["structural_type"] not in VALID_STRUCTURALS:
                    errors.append(f"columns[{i}].structural_type invalid: {col['structural_type']}")

    return errors
