"""Proposal generation module for creating Invariant dataset registrations.

This module generates InvariantProposal objects from inference results,
including column specifications, warnings, and confirmation requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from datasculpt.core.types import (
    ColumnEvidence,
    ColumnSpec,
    DatasetKind,
    DecisionRecord,
    GrainInference,
    HypothesisScore,
    InvariantProposal,
    PrimitiveType,
    Role,
    ShapeHypothesis,
    StructuralType,
)

if TYPE_CHECKING:
    pass

# Try to import jsonschema for validation, fall back to manual validation
try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


# Shape hypothesis to DatasetKind mapping
SHAPE_TO_KIND: dict[ShapeHypothesis, DatasetKind] = {
    ShapeHypothesis.LONG_OBSERVATIONS: DatasetKind.OBSERVATIONS,
    ShapeHypothesis.LONG_INDICATORS: DatasetKind.INDICATORS_LONG,
    ShapeHypothesis.WIDE_TIME_COLUMNS: DatasetKind.TIMESERIES_WIDE,
    ShapeHypothesis.SERIES_COLUMN: DatasetKind.TIMESERIES_SERIES,
    ShapeHypothesis.WIDE_OBSERVATIONS: DatasetKind.OBSERVATIONS,  # Default fallback
}

# Thresholds for warnings and confirmations
LOW_CONFIDENCE_THRESHOLD = 0.7
HIGH_NULL_RATE_THRESHOLD = 0.5
AMBIGUOUS_HYPOTHESIS_GAP = 0.1


@dataclass
class ProposalConfig:
    """Configuration for proposal generation."""

    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD
    high_null_rate_threshold: float = HIGH_NULL_RATE_THRESHOLD
    ambiguous_hypothesis_gap: float = AMBIGUOUS_HYPOTHESIS_GAP


def map_shape_to_kind(shape: ShapeHypothesis) -> DatasetKind:
    """Map a ShapeHypothesis to a DatasetKind.

    Args:
        shape: The shape hypothesis to map.

    Returns:
        The corresponding DatasetKind.
    """
    return SHAPE_TO_KIND.get(shape, DatasetKind.OBSERVATIONS)


def get_primary_role(evidence: ColumnEvidence) -> Role:
    """Get the primary (highest-scoring) role for a column.

    Args:
        evidence: Column evidence with role scores.

    Returns:
        The role with the highest score, or METADATA as fallback.
    """
    if not evidence.role_scores:
        return Role.METADATA

    best_role = Role.METADATA
    best_score = 0.0

    for role, score in evidence.role_scores.items():
        if score > best_score:
            best_role = role
            best_score = score

    return best_role


def get_reference_system_hint(evidence: ColumnEvidence) -> str | None:
    """Infer reference system hint from column evidence.

    Reference systems include things like ISO country codes, FIPS codes,
    standard classification systems, etc.

    Args:
        evidence: Column evidence to analyze.

    Returns:
        Reference system hint string or None.
    """
    name_lower = evidence.name.lower()

    # Country/region identifiers
    if any(kw in name_lower for kw in ("country", "nation", "iso3", "iso_3", "iso2")):
        if "iso3" in name_lower or "iso_3" in name_lower:
            return "ISO 3166-1 alpha-3"
        if "iso2" in name_lower or "iso_2" in name_lower:
            return "ISO 3166-1 alpha-2"
        return "country_code"

    # US state identifiers
    if any(kw in name_lower for kw in ("state", "fips", "us_state")):
        if "fips" in name_lower:
            return "FIPS"
        return "us_state_code"

    # Industry classifications
    if any(kw in name_lower for kw in ("naics", "sic", "isic")):
        if "naics" in name_lower:
            return "NAICS"
        if "sic" in name_lower:
            return "SIC"
        if "isic" in name_lower:
            return "ISIC"

    # Currency
    if any(kw in name_lower for kw in ("currency", "curr", "ccy")):
        return "ISO 4217"

    return None


def get_concept_hint(evidence: ColumnEvidence) -> str | None:
    """Infer concept hint from column evidence.

    Concept hints describe what the column represents semantically.

    Args:
        evidence: Column evidence to analyze.

    Returns:
        Concept hint string or None.
    """
    name_lower = evidence.name.lower()

    # Common concepts
    concept_patterns = {
        "population": "population_count",
        "gdp": "gross_domestic_product",
        "revenue": "monetary_revenue",
        "income": "monetary_income",
        "sales": "monetary_sales",
        "temperature": "temperature_measurement",
        "latitude": "geographic_latitude",
        "longitude": "geographic_longitude",
        "age": "age_years",
        "date": "temporal_date",
        "time": "temporal_timestamp",
        "year": "temporal_year",
        "month": "temporal_month",
        "price": "monetary_price",
        "rate": "rate_value",
        "percentage": "percentage_value",
        "count": "count_value",
        "quantity": "quantity_value",
    }

    for pattern, concept in concept_patterns.items():
        if pattern in name_lower:
            return concept

    return None


def get_unit_hint(evidence: ColumnEvidence) -> str | None:
    """Infer unit hint from column evidence.

    Args:
        evidence: Column evidence to analyze.

    Returns:
        Unit hint string or None.
    """
    name_lower = evidence.name.lower()

    # Unit patterns
    unit_patterns = {
        "usd": "USD",
        "eur": "EUR",
        "gbp": "GBP",
        "dollars": "USD",
        "euros": "EUR",
        "pounds": "GBP",
        "percent": "%",
        "pct": "%",
        "kg": "kg",
        "kilogram": "kg",
        "km": "km",
        "kilometer": "km",
        "mile": "mi",
        "celsius": "C",
        "fahrenheit": "F",
        "kelvin": "K",
    }

    for pattern, unit in unit_patterns.items():
        if pattern in name_lower:
            return unit

    return None


def get_time_granularity(evidence: ColumnEvidence) -> str | None:
    """Infer time granularity from column evidence.

    Args:
        evidence: Column evidence to analyze.

    Returns:
        Time granularity string or None.
    """
    if evidence.primitive_type not in (PrimitiveType.DATE, PrimitiveType.DATETIME):
        # Also check role scores for TIME
        time_score = evidence.role_scores.get(Role.TIME, 0.0)
        if time_score < 0.3:
            return None

    name_lower = evidence.name.lower()

    # Granularity patterns
    if any(kw in name_lower for kw in ("year", "annual", "yearly")):
        return "year"
    if any(kw in name_lower for kw in ("quarter", "quarterly", "q1", "q2", "q3", "q4")):
        return "quarter"
    if any(kw in name_lower for kw in ("month", "monthly")):
        return "month"
    if any(kw in name_lower for kw in ("week", "weekly")):
        return "week"
    if any(kw in name_lower for kw in ("day", "daily", "date")):
        return "day"
    if any(kw in name_lower for kw in ("hour", "hourly")):
        return "hour"
    if any(kw in name_lower for kw in ("minute", "min")):
        return "minute"
    if any(kw in name_lower for kw in ("second", "sec", "timestamp")):
        return "second"

    # Default based on type
    if evidence.primitive_type == PrimitiveType.DATE:
        return "day"
    if evidence.primitive_type == PrimitiveType.DATETIME:
        return "second"

    return None


def create_column_spec(evidence: ColumnEvidence) -> ColumnSpec:
    """Create a ColumnSpec from column evidence.

    Args:
        evidence: Column evidence from profiling.

    Returns:
        ColumnSpec for the column.
    """
    role = get_primary_role(evidence)

    return ColumnSpec(
        name=evidence.name,
        role=role,
        primitive_type=evidence.primitive_type,
        structural_type=evidence.structural_type,
        reference_system_hint=get_reference_system_hint(evidence),
        concept_hint=get_concept_hint(evidence),
        unit_hint=get_unit_hint(evidence),
        time_granularity=get_time_granularity(evidence),
        notes=list(evidence.notes),
    )


def generate_warnings(
    column_evidence: dict[str, ColumnEvidence],
    shape_hypotheses: list[HypothesisScore],
    grain: GrainInference,
    config: ProposalConfig | None = None,
) -> list[str]:
    """Generate warnings based on inference results.

    Warnings are generated for:
    - Low confidence shape detection
    - Ambiguous shape (close scores between top hypotheses)
    - No stable grain found
    - Many columns with high null rates

    Args:
        column_evidence: Evidence for each column.
        shape_hypotheses: Ranked shape hypothesis scores.
        grain: Grain inference result.
        config: Configuration for thresholds.

    Returns:
        List of warning messages.
    """
    if config is None:
        config = ProposalConfig()

    warnings: list[str] = []

    # Check shape hypothesis confidence
    if shape_hypotheses:
        best_hypothesis = shape_hypotheses[0]

        # Low confidence warning
        if best_hypothesis.score < config.low_confidence_threshold:
            warnings.append(
                f"Low confidence in shape detection: {best_hypothesis.hypothesis.value} "
                f"(score: {best_hypothesis.score:.2f}, threshold: {config.low_confidence_threshold})"
            )

        # Ambiguous shape warning (close scores)
        if len(shape_hypotheses) >= 2:
            second_best = shape_hypotheses[1]
            gap = best_hypothesis.score - second_best.score
            if gap < config.ambiguous_hypothesis_gap:
                warnings.append(
                    f"Ambiguous shape: {best_hypothesis.hypothesis.value} ({best_hypothesis.score:.2f}) "
                    f"vs {second_best.hypothesis.value} ({second_best.score:.2f}), gap={gap:.2f}"
                )

    # Check grain stability
    if grain.confidence == 0.0:
        warnings.append(
            "No stable grain found: dataset may have duplicate rows or "
            "require all columns as grain"
        )
    elif grain.confidence < config.low_confidence_threshold:
        warnings.append(
            f"Low confidence in grain: {grain.key_columns} "
            f"(confidence: {grain.confidence:.2f}, uniqueness: {grain.uniqueness_ratio:.2%})"
        )

    # Check for high null rate columns
    high_null_columns = [
        name
        for name, evidence in column_evidence.items()
        if evidence.null_rate > config.high_null_rate_threshold
    ]
    if high_null_columns:
        warnings.append(
            f"High null rate columns (>{config.high_null_rate_threshold:.0%}): "
            f"{', '.join(high_null_columns)}"
        )

    # Check for columns with low role confidence
    low_confidence_columns = []
    for name, evidence in column_evidence.items():
        if evidence.role_scores:
            sorted_scores = sorted(evidence.role_scores.values(), reverse=True)
            if len(sorted_scores) >= 2:
                top_score = sorted_scores[0]
                second_score = sorted_scores[1]
                if top_score > 0 and (top_score - second_score) / top_score < 0.2:
                    low_confidence_columns.append(name)

    if low_confidence_columns:
        warnings.append(
            f"Ambiguous role assignment for columns: {', '.join(low_confidence_columns)}"
        )

    return warnings


def generate_confirmation_requirements(
    column_evidence: dict[str, ColumnEvidence],
    shape_hypotheses: list[HypothesisScore],
    grain: GrainInference,
    config: ProposalConfig | None = None,
) -> list[str]:
    """Generate list of items requiring user confirmation.

    Confirmation is required for:
    - Ambiguous shape detection
    - Overridden roles (when inference differs from naming conventions)
    - Uncertain grain

    Args:
        column_evidence: Evidence for each column.
        shape_hypotheses: Ranked shape hypothesis scores.
        grain: Grain inference result.
        config: Configuration for thresholds.

    Returns:
        List of items requiring user confirmation.
    """
    if config is None:
        config = ProposalConfig()

    confirmations: list[str] = []

    # Ambiguous shape requires confirmation
    if shape_hypotheses and len(shape_hypotheses) >= 2:
        best = shape_hypotheses[0]
        second = shape_hypotheses[1]
        gap = best.score - second.score
        if gap < config.ambiguous_hypothesis_gap:
            confirmations.append(
                f"Confirm dataset shape: Is this {best.hypothesis.value} or {second.hypothesis.value}?"
            )

    # Uncertain grain requires confirmation
    if grain.confidence < config.low_confidence_threshold:
        grain_cols = ", ".join(grain.key_columns) if grain.key_columns else "(none)"
        confirmations.append(
            f"Confirm grain columns: [{grain_cols}] (confidence: {grain.confidence:.2f})"
        )

    # Check for columns where role might need override
    for name, evidence in column_evidence.items():
        # Check if column name suggests a different role than assigned
        primary_role = get_primary_role(evidence)
        name_lower = name.lower()

        # If name contains 'id' but role is not KEY
        if ("_id" in name_lower or name_lower == "id") and primary_role != Role.KEY:
            confirmations.append(
                f"Confirm role for '{name}': assigned {primary_role.value}, but name suggests KEY"
            )

        # If name contains 'date' or 'time' but role is not TIME
        if (
            any(kw in name_lower for kw in ("date", "time", "timestamp"))
            and primary_role != Role.TIME
            and evidence.primitive_type
            not in (PrimitiveType.DATE, PrimitiveType.DATETIME)
        ):
            confirmations.append(
                f"Confirm role for '{name}': assigned {primary_role.value}, but name suggests TIME"
            )

    return confirmations


# JSON Schema for InvariantProposal validation
PROPOSAL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": [
        "dataset_name",
        "dataset_kind",
        "shape_hypothesis",
        "grain",
        "columns",
    ],
    "properties": {
        "dataset_name": {"type": "string", "minLength": 1},
        "dataset_kind": {
            "type": "string",
            "enum": [k.value for k in DatasetKind],
        },
        "shape_hypothesis": {
            "type": "string",
            "enum": [s.value for s in ShapeHypothesis],
        },
        "grain": {
            "type": "array",
            "items": {"type": "string"},
        },
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "role", "primitive_type", "structural_type"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "enum": [r.value for r in Role]},
                    "primitive_type": {
                        "type": "string",
                        "enum": [p.value for p in PrimitiveType],
                    },
                    "structural_type": {
                        "type": "string",
                        "enum": [s.value for s in StructuralType],
                    },
                    "reference_system_hint": {"type": ["string", "null"]},
                    "concept_hint": {"type": ["string", "null"]},
                    "unit_hint": {"type": ["string", "null"]},
                    "time_granularity": {"type": ["string", "null"]},
                    "notes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "required_user_confirmations": {"type": "array", "items": {"type": "string"}},
        "decision_record_id": {"type": "string"},
    },
}


def proposal_to_dict(proposal: InvariantProposal) -> dict[str, Any]:
    """Convert an InvariantProposal to a dictionary for validation.

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


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)


def validate_proposal_with_jsonschema(
    proposal_dict: dict[str, Any],
) -> ValidationResult:
    """Validate proposal using jsonschema library.

    Args:
        proposal_dict: Proposal as dictionary.

    Returns:
        ValidationResult with validity and any errors.
    """
    errors: list[str] = []

    try:
        jsonschema.validate(proposal_dict, PROPOSAL_SCHEMA)
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_proposal_manual(proposal_dict: dict[str, Any]) -> ValidationResult:
    """Manual validation of proposal structure without jsonschema.

    Args:
        proposal_dict: Proposal as dictionary.

    Returns:
        ValidationResult with validity and any errors.
    """
    errors: list[str] = []

    # Check required top-level fields
    required_fields = [
        "dataset_name",
        "dataset_kind",
        "shape_hypothesis",
        "grain",
        "columns",
    ]
    for field_name in required_fields:
        if field_name not in proposal_dict:
            errors.append(f"Missing required field: {field_name}")

    # Validate dataset_name
    if "dataset_name" in proposal_dict:
        if not isinstance(proposal_dict["dataset_name"], str):
            errors.append("dataset_name must be a string")
        elif len(proposal_dict["dataset_name"]) == 0:
            errors.append("dataset_name cannot be empty")

    # Validate dataset_kind
    if "dataset_kind" in proposal_dict:
        valid_kinds = {k.value for k in DatasetKind}
        if proposal_dict["dataset_kind"] not in valid_kinds:
            errors.append(
                f"Invalid dataset_kind: {proposal_dict['dataset_kind']}. "
                f"Must be one of: {valid_kinds}"
            )

    # Validate shape_hypothesis
    if "shape_hypothesis" in proposal_dict:
        valid_shapes = {s.value for s in ShapeHypothesis}
        if proposal_dict["shape_hypothesis"] not in valid_shapes:
            errors.append(
                f"Invalid shape_hypothesis: {proposal_dict['shape_hypothesis']}. "
                f"Must be one of: {valid_shapes}"
            )

    # Validate grain
    if "grain" in proposal_dict:
        if not isinstance(proposal_dict["grain"], list):
            errors.append("grain must be an array")
        else:
            for item in proposal_dict["grain"]:
                if not isinstance(item, str):
                    errors.append("grain items must be strings")
                    break

    # Validate columns
    if "columns" in proposal_dict:
        if not isinstance(proposal_dict["columns"], list):
            errors.append("columns must be an array")
        else:
            valid_roles = {r.value for r in Role}
            valid_primitives = {p.value for p in PrimitiveType}
            valid_structurals = {s.value for s in StructuralType}

            for i, col in enumerate(proposal_dict["columns"]):
                if not isinstance(col, dict):
                    errors.append(f"columns[{i}] must be an object")
                    continue

                # Check required column fields
                for field_name in ["name", "role", "primitive_type", "structural_type"]:
                    if field_name not in col:
                        errors.append(f"columns[{i}] missing required field: {field_name}")

                # Validate enum values
                if "role" in col and col["role"] not in valid_roles:
                    errors.append(
                        f"columns[{i}].role invalid: {col['role']}. "
                        f"Must be one of: {valid_roles}"
                    )
                if "primitive_type" in col and col["primitive_type"] not in valid_primitives:
                    errors.append(
                        f"columns[{i}].primitive_type invalid: {col['primitive_type']}. "
                        f"Must be one of: {valid_primitives}"
                    )
                if "structural_type" in col and col["structural_type"] not in valid_structurals:
                    errors.append(
                        f"columns[{i}].structural_type invalid: {col['structural_type']}. "
                        f"Must be one of: {valid_structurals}"
                    )

    # Validate optional array fields
    for field_name in ["warnings", "required_user_confirmations"]:
        if field_name in proposal_dict:
            if not isinstance(proposal_dict[field_name], list):
                errors.append(f"{field_name} must be an array")
            else:
                for item in proposal_dict[field_name]:
                    if not isinstance(item, str):
                        errors.append(f"{field_name} items must be strings")
                        break

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_proposal(proposal: InvariantProposal) -> ValidationResult:
    """Validate an InvariantProposal against the schema.

    Uses jsonschema if available, otherwise falls back to manual validation.

    Args:
        proposal: The proposal to validate.

    Returns:
        ValidationResult with validity and any errors.
    """
    proposal_dict = proposal_to_dict(proposal)

    if HAS_JSONSCHEMA:
        return validate_proposal_with_jsonschema(proposal_dict)
    return validate_proposal_manual(proposal_dict)


def generate_proposal(
    dataset_name: str,
    column_evidence: dict[str, ColumnEvidence],
    shape_hypotheses: list[HypothesisScore],
    grain: GrainInference,
    decision_record_id: str = "",
    config: ProposalConfig | None = None,
) -> InvariantProposal:
    """Generate an InvariantProposal from inference results.

    This is the main entry point for proposal generation. It:
    1. Maps shape hypothesis to DatasetKind
    2. Creates ColumnSpecs for all columns
    3. Generates warnings for low confidence, ambiguity, etc.
    4. Identifies items requiring user confirmation

    Args:
        dataset_name: Name for the dataset.
        column_evidence: Evidence for each column from profiling.
        shape_hypotheses: Ranked shape hypothesis scores.
        grain: Grain inference result.
        decision_record_id: ID of the associated decision record.
        config: Configuration for thresholds.

    Returns:
        InvariantProposal ready for user review.
    """
    if config is None:
        config = ProposalConfig()

    # Determine shape and kind
    if shape_hypotheses:
        selected_shape = shape_hypotheses[0].hypothesis
    else:
        selected_shape = ShapeHypothesis.LONG_OBSERVATIONS

    dataset_kind = map_shape_to_kind(selected_shape)

    # Create column specs
    columns = [
        create_column_spec(evidence)
        for evidence in column_evidence.values()
    ]

    # Generate warnings
    warnings = generate_warnings(column_evidence, shape_hypotheses, grain, config)

    # Generate confirmation requirements
    confirmations = generate_confirmation_requirements(
        column_evidence, shape_hypotheses, grain, config
    )

    return InvariantProposal(
        dataset_name=dataset_name,
        dataset_kind=dataset_kind,
        shape_hypothesis=selected_shape,
        grain=grain.key_columns,
        columns=columns,
        warnings=warnings,
        required_user_confirmations=confirmations,
        decision_record_id=decision_record_id,
    )


def generate_proposal_from_decision_record(
    decision_record: DecisionRecord,
    dataset_name: str,
    config: ProposalConfig | None = None,
) -> InvariantProposal:
    """Generate a proposal from a complete DecisionRecord.

    Convenience function that extracts all necessary information from
    a DecisionRecord.

    Args:
        decision_record: Complete decision record from inference.
        dataset_name: Name for the dataset.
        config: Configuration for thresholds.

    Returns:
        InvariantProposal ready for user review.
    """
    return generate_proposal(
        dataset_name=dataset_name,
        column_evidence=decision_record.column_evidence,
        shape_hypotheses=decision_record.hypotheses,
        grain=decision_record.grain,
        decision_record_id=decision_record.decision_id,
        config=config,
    )


# --- Proposal Diff Functionality ---


class ColumnChangeType(str, Enum):
    """Types of column changes between proposals."""

    ADDED = "added"
    REMOVED = "removed"
    ROLE_CHANGED = "role_changed"


@dataclass
class ColumnChange:
    """Represents a change to a column between two proposals."""

    column_name: str
    change_type: ColumnChangeType
    old_role: Role | None = None
    new_role: Role | None = None

    def __str__(self) -> str:
        """Return human-readable description of the change."""
        if self.change_type == ColumnChangeType.ADDED:
            role_str = f" (role: {self.new_role.value})" if self.new_role else ""
            return f"Added column '{self.column_name}'{role_str}"
        if self.change_type == ColumnChangeType.REMOVED:
            role_str = f" (was: {self.old_role.value})" if self.old_role else ""
            return f"Removed column '{self.column_name}'{role_str}"
        if self.change_type == ColumnChangeType.ROLE_CHANGED:
            old = self.old_role.value if self.old_role else "unknown"
            new = self.new_role.value if self.new_role else "unknown"
            return f"Column '{self.column_name}' role changed: {old} -> {new}"
        return f"Column '{self.column_name}' changed"


@dataclass
class ProposalDiff:
    """Difference between two InvariantProposals."""

    shape_changed: bool
    grain_changed: bool
    column_changes: list[ColumnChange]
    warning_changes: tuple[list[str], list[str]]  # (added_warnings, removed_warnings)
    summary: str

    @property
    def has_changes(self) -> bool:
        """Return True if there are any differences."""
        added_warnings, removed_warnings = self.warning_changes
        return (
            self.shape_changed
            or self.grain_changed
            or len(self.column_changes) > 0
            or len(added_warnings) > 0
            or len(removed_warnings) > 0
        )


def _build_diff_summary(
    old: InvariantProposal,
    new: InvariantProposal,
    shape_changed: bool,
    grain_changed: bool,
    column_changes: list[ColumnChange],
    added_warnings: list[str],
    removed_warnings: list[str],
) -> str:
    """Build a human-readable summary of proposal differences.

    Args:
        old: The original proposal.
        new: The updated proposal.
        shape_changed: Whether the shape changed.
        grain_changed: Whether the grain changed.
        column_changes: List of column changes.
        added_warnings: Warnings added in new proposal.
        removed_warnings: Warnings removed from old proposal.

    Returns:
        Human-readable diff summary.
    """
    lines: list[str] = []

    if not any([
        shape_changed,
        grain_changed,
        column_changes,
        added_warnings,
        removed_warnings,
    ]):
        return "No differences found between proposals."

    lines.append("Proposal Differences:")
    lines.append("")

    if shape_changed:
        lines.append(
            f"Shape: {old.shape_hypothesis.value} -> {new.shape_hypothesis.value}"
        )
        if old.dataset_kind != new.dataset_kind:
            lines.append(
                f"Kind: {old.dataset_kind.value} -> {new.dataset_kind.value}"
            )

    if grain_changed:
        old_grain = ", ".join(old.grain) if old.grain else "(none)"
        new_grain = ", ".join(new.grain) if new.grain else "(none)"
        lines.append(f"Grain: [{old_grain}] -> [{new_grain}]")

    if column_changes:
        lines.append("")
        lines.append("Column Changes:")
        for change in column_changes:
            lines.append(f"  - {change}")

    if added_warnings:
        lines.append("")
        lines.append("New Warnings:")
        for warning in added_warnings:
            lines.append(f"  + {warning}")

    if removed_warnings:
        lines.append("")
        lines.append("Resolved Warnings:")
        for warning in removed_warnings:
            lines.append(f"  - {warning}")

    return "\n".join(lines)


def diff_proposals(old: InvariantProposal, new: InvariantProposal) -> ProposalDiff:
    """Compare two InvariantProposals and return their differences.

    Args:
        old: The original/baseline proposal.
        new: The updated/new proposal.

    Returns:
        ProposalDiff containing all differences between the proposals.
    """
    # Check shape changes
    shape_changed = old.shape_hypothesis != new.shape_hypothesis

    # Check grain changes
    grain_changed = set(old.grain) != set(new.grain)

    # Build column maps for comparison
    old_columns: dict[str, ColumnSpec] = {col.name: col for col in old.columns}
    new_columns: dict[str, ColumnSpec] = {col.name: col for col in new.columns}

    old_column_names = set(old_columns.keys())
    new_column_names = set(new_columns.keys())

    # Find column changes
    column_changes: list[ColumnChange] = []

    # Added columns
    for name in sorted(new_column_names - old_column_names):
        column_changes.append(
            ColumnChange(
                column_name=name,
                change_type=ColumnChangeType.ADDED,
                new_role=new_columns[name].role,
            )
        )

    # Removed columns
    for name in sorted(old_column_names - new_column_names):
        column_changes.append(
            ColumnChange(
                column_name=name,
                change_type=ColumnChangeType.REMOVED,
                old_role=old_columns[name].role,
            )
        )

    # Role changes for columns present in both
    for name in sorted(old_column_names & new_column_names):
        old_role = old_columns[name].role
        new_role = new_columns[name].role
        if old_role != new_role:
            column_changes.append(
                ColumnChange(
                    column_name=name,
                    change_type=ColumnChangeType.ROLE_CHANGED,
                    old_role=old_role,
                    new_role=new_role,
                )
            )

    # Warning changes
    old_warnings_set = set(old.warnings)
    new_warnings_set = set(new.warnings)
    added_warnings = sorted(new_warnings_set - old_warnings_set)
    removed_warnings = sorted(old_warnings_set - new_warnings_set)

    # Build summary
    summary = _build_diff_summary(
        old=old,
        new=new,
        shape_changed=shape_changed,
        grain_changed=grain_changed,
        column_changes=column_changes,
        added_warnings=added_warnings,
        removed_warnings=removed_warnings,
    )

    return ProposalDiff(
        shape_changed=shape_changed,
        grain_changed=grain_changed,
        column_changes=column_changes,
        warning_changes=(added_warnings, removed_warnings),
        summary=summary,
    )
