"""Decision record module for audit trails and reproducibility.

This module provides functions to create, serialize, and export decision records
that capture the complete audit trail for inference runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

from datasculpt.core.types import (
    ArrayProfile,
    ColumnEvidence,
    DecisionRecord,
    DecisionRecordSummary,
    GrainInference,
    HypothesisScore,
    ParseResults,
    Question,
    ShapeHypothesis,
    ValueProfile,
)

# Standard location for storing decision records
DECISIONS_DIRECTORY = Path(".datasculpt/decisions")


def generate_decision_id() -> str:
    """Generate a unique decision ID using uuid4.

    Returns:
        A unique string identifier for the decision record.
    """
    return str(uuid4())


def get_timestamp() -> datetime:
    """Get the current UTC timestamp.

    Returns:
        The current datetime in UTC.
    """
    return datetime.utcnow()


def _serialize_enum(value: Any) -> Any:
    """Convert an enum to its value, or return the original value.

    Args:
        value: Any value that might be an enum.

    Returns:
        The enum's value if it's an enum, otherwise the original value.
    """
    if isinstance(value, Enum):
        return value.value
    return value


def _serialize_value(value: Any) -> Any:
    """Recursively serialize a value to JSON-compatible format.

    Args:
        value: Any value to serialize.

    Returns:
        A JSON-compatible representation of the value.
    """
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {_serialize_value(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def serialize_column_evidence(evidence: ColumnEvidence) -> dict[str, Any]:
    """Serialize a ColumnEvidence to a JSON-compatible dict.

    Args:
        evidence: The ColumnEvidence object to serialize.

    Returns:
        A dictionary that can be serialized to JSON.
    """
    data = asdict(evidence)
    return _serialize_value(data)


def serialize_hypothesis_score(score: HypothesisScore) -> dict[str, Any]:
    """Serialize a HypothesisScore to a JSON-compatible dict.

    Args:
        score: The HypothesisScore object to serialize.

    Returns:
        A dictionary that can be serialized to JSON.
    """
    data = asdict(score)
    return _serialize_value(data)


def serialize_hypotheses(hypotheses: list[HypothesisScore]) -> list[dict[str, Any]]:
    """Serialize a list of HypothesisScore objects to JSON-compatible format.

    Args:
        hypotheses: List of HypothesisScore objects to serialize.

    Returns:
        A list of dictionaries that can be serialized to JSON.
    """
    return [serialize_hypothesis_score(h) for h in hypotheses]


def serialize_grain_inference(grain: GrainInference) -> dict[str, Any]:
    """Serialize a GrainInference to a JSON-compatible dict.

    Args:
        grain: The GrainInference object to serialize.

    Returns:
        A dictionary that can be serialized to JSON.
    """
    data = asdict(grain)
    return _serialize_value(data)


def serialize_question(question: Question) -> dict[str, Any]:
    """Serialize a Question to a JSON-compatible dict.

    Args:
        question: The Question object to serialize.

    Returns:
        A dictionary that can be serialized to JSON.
    """
    data = asdict(question)
    return _serialize_value(data)


def serialize_decision_record(record: DecisionRecord) -> dict[str, Any]:
    """Serialize a DecisionRecord to a JSON-compatible dict.

    Args:
        record: The DecisionRecord object to serialize.

    Returns:
        A dictionary that can be serialized to JSON.
    """
    return {
        "decision_id": record.decision_id,
        "dataset_fingerprint": record.dataset_fingerprint,
        "timestamp": record.timestamp.isoformat(),
        "selected_hypothesis": record.selected_hypothesis.value,
        "hypotheses": serialize_hypotheses(record.hypotheses),
        "grain": serialize_grain_inference(record.grain),
        "column_evidence": {
            name: serialize_column_evidence(ev)
            for name, ev in record.column_evidence.items()
        },
        "questions": [serialize_question(q) for q in record.questions],
        "answers": _serialize_value(record.answers),
    }


def create_decision_record(
    dataset_fingerprint: str,
    selected_hypothesis: ShapeHypothesis,
    hypotheses: list[HypothesisScore],
    grain: GrainInference,
    column_evidence: dict[str, ColumnEvidence],
    questions: list[Question] | None = None,
    answers: dict[str, Any] | None = None,
) -> DecisionRecord:
    """Create a new DecisionRecord with generated ID and timestamp.

    Args:
        dataset_fingerprint: Unique identifier for the dataset being analyzed.
        selected_hypothesis: The chosen shape hypothesis.
        hypotheses: All evaluated hypothesis scores.
        grain: The inferred grain for the dataset.
        column_evidence: Evidence collected for each column.
        questions: Optional list of questions asked during inference.
        answers: Optional dictionary of answers to questions.

    Returns:
        A new DecisionRecord with unique ID and current timestamp.
    """
    return DecisionRecord(
        decision_id=generate_decision_id(),
        dataset_fingerprint=dataset_fingerprint,
        timestamp=get_timestamp(),
        selected_hypothesis=selected_hypothesis,
        hypotheses=hypotheses,
        grain=grain,
        column_evidence=column_evidence,
        questions=questions or [],
        answers=answers or {},
    )


def record_override(
    record: DecisionRecord,
    question_id: str,
    answer: Any,
    override_timestamp: datetime | None = None,
) -> DecisionRecord:
    """Record a user confirmation or override on a decision record.

    This creates a new DecisionRecord with the updated answer and timestamp.
    The original record is not modified.

    Args:
        record: The existing DecisionRecord to update.
        question_id: The ID of the question being answered.
        answer: The user's answer or override value.
        override_timestamp: Optional timestamp for the override. Defaults to now.

    Returns:
        A new DecisionRecord with the updated answer.
    """
    timestamp = override_timestamp or get_timestamp()

    # Create new answers dict with override metadata
    new_answers = dict(record.answers)
    new_answers[question_id] = {
        "value": answer,
        "timestamp": timestamp.isoformat(),
        "is_override": True,
    }

    return DecisionRecord(
        decision_id=record.decision_id,
        dataset_fingerprint=record.dataset_fingerprint,
        timestamp=record.timestamp,
        selected_hypothesis=record.selected_hypothesis,
        hypotheses=record.hypotheses,
        grain=record.grain,
        column_evidence=record.column_evidence,
        questions=record.questions,
        answers=new_answers,
    )


def export_to_json(record: DecisionRecord, path: str | Path) -> None:
    """Export a DecisionRecord to a JSON file.

    Args:
        record: The DecisionRecord to export.
        path: The file path to write the JSON to.
    """
    data = serialize_decision_record(record)
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_to_json_string(record: DecisionRecord) -> str:
    """Export a DecisionRecord to a JSON string.

    Args:
        record: The DecisionRecord to export.

    Returns:
        A JSON string representation of the record.
    """
    data = serialize_decision_record(record)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _deserialize_column_evidence(data: dict[str, Any]) -> ColumnEvidence:
    """Deserialize a dict to a ColumnEvidence object.

    Args:
        data: Dictionary containing column evidence data.

    Returns:
        A ColumnEvidence object.
    """
    from datasculpt.core.types import PrimitiveType, Role, StructuralType

    # Convert role_scores keys back to Role enums
    role_scores: dict[Role, float] = {}
    for role_str, score in data.get("role_scores", {}).items():
        role_scores[Role(role_str)] = score

    # Handle ParseResults - can be dict (legacy) or dataclass-serialized
    parse_results_data = data.get("parse_results", {})
    if isinstance(parse_results_data, dict):
        parse_results = ParseResults(
            date_parse_rate=parse_results_data.get("date_parse_rate", 0.0),
            has_time=parse_results_data.get("has_time", False),
            best_date_format=parse_results_data.get("best_date_format"),
            date_failure_examples=parse_results_data.get("date_failure_examples", []),
            json_array_rate=parse_results_data.get("json_array_rate", 0.0),
        )
    else:
        parse_results = ParseResults()

    # Handle ValueProfile
    value_profile_data = data.get("value_profile", {})
    if isinstance(value_profile_data, dict):
        value_profile = ValueProfile(
            min_value=value_profile_data.get("min_value"),
            max_value=value_profile_data.get("max_value"),
            mean=value_profile_data.get("mean"),
            integer_ratio=value_profile_data.get("integer_ratio", 0.0),
            non_negative_ratio=value_profile_data.get("non_negative_ratio", 0.0),
            bounded_0_1_ratio=value_profile_data.get("bounded_0_1_ratio", 0.0),
            bounded_0_100_ratio=value_profile_data.get("bounded_0_100_ratio", 0.0),
            low_cardinality=value_profile_data.get("low_cardinality", False),
            mostly_null=value_profile_data.get("mostly_null", False),
        )
    else:
        value_profile = ValueProfile()

    # Handle ArrayProfile (optional)
    array_profile_data = data.get("array_profile")
    array_profile = None
    if isinstance(array_profile_data, dict):
        array_profile = ArrayProfile(
            avg_length=array_profile_data.get("avg_length", 0.0),
            min_length=array_profile_data.get("min_length", 0),
            max_length=array_profile_data.get("max_length", 0),
            consistent_length=array_profile_data.get("consistent_length", False),
        )

    return ColumnEvidence(
        name=data["name"],
        primitive_type=PrimitiveType(data["primitive_type"]),
        structural_type=StructuralType(data["structural_type"]),
        null_rate=data.get("null_rate", 0.0),
        distinct_ratio=data.get("distinct_ratio", 0.0),
        unique_count=data.get("unique_count", 0),
        value_profile=value_profile,
        array_profile=array_profile,
        header_date_like=data.get("header_date_like", False),
        parse_results=parse_results,
        parse_results_dict=data.get("parse_results_dict", {}),
        role_scores=role_scores,
        external=data.get("external", {}),
        notes=data.get("notes", []),
    )


def _deserialize_hypothesis_score(data: dict[str, Any]) -> HypothesisScore:
    """Deserialize a dict to a HypothesisScore object.

    Args:
        data: Dictionary containing hypothesis score data.

    Returns:
        A HypothesisScore object.
    """
    return HypothesisScore(
        hypothesis=ShapeHypothesis(data["hypothesis"]),
        score=data["score"],
        reasons=data.get("reasons", []),
    )


def _deserialize_grain_inference(data: dict[str, Any]) -> GrainInference:
    """Deserialize a dict to a GrainInference object.

    Args:
        data: Dictionary containing grain inference data.

    Returns:
        A GrainInference object.
    """
    return GrainInference(
        key_columns=data["key_columns"],
        confidence=data["confidence"],
        uniqueness_ratio=data["uniqueness_ratio"],
        evidence=data.get("evidence", []),
    )


def _deserialize_question(data: dict[str, Any]) -> Question:
    """Deserialize a dict to a Question object.

    Args:
        data: Dictionary containing question data.

    Returns:
        A Question object.
    """
    from datasculpt.core.types import QuestionType

    return Question(
        id=data["id"],
        type=QuestionType(data["type"]),
        prompt=data["prompt"],
        choices=data.get("choices", []),
        default=data.get("default"),
        rationale=data.get("rationale"),
    )


def load_from_json(path: str | Path) -> DecisionRecord:
    """Load a DecisionRecord from a JSON file.

    Args:
        path: The file path to read the JSON from.

    Returns:
        A DecisionRecord object loaded from the file.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return _deserialize_decision_record(data)


def load_from_json_string(json_string: str) -> DecisionRecord:
    """Load a DecisionRecord from a JSON string.

    Args:
        json_string: A JSON string representation of a DecisionRecord.

    Returns:
        A DecisionRecord object.
    """
    data = json.loads(json_string)
    return _deserialize_decision_record(data)


def _deserialize_decision_record(data: dict[str, Any]) -> DecisionRecord:
    """Deserialize a dict to a DecisionRecord object.

    Args:
        data: Dictionary containing decision record data.

    Returns:
        A DecisionRecord object.
    """
    # Parse timestamp
    timestamp = datetime.fromisoformat(data["timestamp"])

    # Deserialize nested objects
    hypotheses = [_deserialize_hypothesis_score(h) for h in data["hypotheses"]]
    grain = _deserialize_grain_inference(data["grain"])
    column_evidence = {
        name: _deserialize_column_evidence(ev)
        for name, ev in data["column_evidence"].items()
    }
    questions = [_deserialize_question(q) for q in data.get("questions", [])]

    return DecisionRecord(
        decision_id=data["decision_id"],
        dataset_fingerprint=data["dataset_fingerprint"],
        timestamp=timestamp,
        selected_hypothesis=ShapeHypothesis(data["selected_hypothesis"]),
        hypotheses=hypotheses,
        grain=grain,
        column_evidence=column_evidence,
        questions=questions,
        answers=data.get("answers", {}),
    )


def list_decision_records(directory: Path) -> list[DecisionRecordSummary]:
    """List all decision records in a directory.

    Scans the directory for JSON files and extracts summary information
    from each valid decision record.

    Args:
        directory: Path to the directory containing decision record JSON files.

    Returns:
        A list of DecisionRecordSummary objects, sorted by timestamp (newest first).
    """
    summaries: list[DecisionRecordSummary] = []

    if not directory.exists():
        return summaries

    for file_path in directory.glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            summary = DecisionRecordSummary(
                decision_id=data["decision_id"],
                dataset_fingerprint=data["dataset_fingerprint"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                path=file_path,
                selected_hypothesis=data["selected_hypothesis"],
            )
            summaries.append(summary)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Skip files that aren't valid decision records
            logger.warning(f"Skipping invalid decision record file '{file_path}': {e}")
            continue

    # Sort by timestamp, newest first
    summaries.sort(key=lambda s: s.timestamp, reverse=True)
    return summaries


def load_decision_record(path: Path) -> DecisionRecord:
    """Load a decision record from a JSON file.

    Args:
        path: Path to the JSON file containing the decision record.

    Returns:
        A DecisionRecord object loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        KeyError: If required fields are missing from the JSON.
    """
    return load_from_json(path)


def find_decision_record(directory: Path, fingerprint: str) -> DecisionRecord | None:
    """Find a decision record by dataset fingerprint.

    Searches the directory for a decision record matching the given fingerprint.
    If multiple records exist for the same fingerprint, returns the most recent one.

    Args:
        directory: Path to the directory containing decision record JSON files.
        fingerprint: The dataset fingerprint to search for.

    Returns:
        The matching DecisionRecord if found, or None if no match exists.
    """
    if not directory.exists():
        return None

    matching_records: list[tuple[datetime, Path]] = []

    for file_path in directory.glob("*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("dataset_fingerprint") == fingerprint:
                timestamp = datetime.fromisoformat(data["timestamp"])
                matching_records.append((timestamp, file_path))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.debug("Could not parse decision record: %s", e)
            continue

    if not matching_records:
        return None

    # Return the most recent matching record
    matching_records.sort(key=lambda x: x[0], reverse=True)
    _, path = matching_records[0]
    return load_from_json(path)
