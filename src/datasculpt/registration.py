"""Registration module for Invariant integration.

This module provides functions for registering datasets, variables,
and data products with Invariant. Currently contains stub implementations
that will be replaced with actual API calls when Invariant integration
is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasculpt.core.types import InvariantProposal


class RegistrationStatus(str, Enum):
    """Status of a registration operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    NOT_IMPLEMENTED = "not_implemented"
    CANCELLED = "cancelled"


@dataclass
class RegistrationResult:
    """Result of a catalog registration operation.

    Attributes:
        status: The status of the registration operation.
        dataset_id: The created dataset ID in Invariant (if successful).
        variable_ids: List of created variable IDs in Invariant (if successful).
        message: Human-readable message describing the result.
        errors: List of error messages if registration failed.
    """

    status: RegistrationStatus
    dataset_id: str | None = None
    variable_ids: list[str] = field(default_factory=list)
    message: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class DataProductResult:
    """Result of a DataProduct creation operation.

    Attributes:
        status: The status of the creation operation.
        data_product_id: The created DataProduct ID in Invariant (if successful).
        message: Human-readable message describing the result.
        errors: List of error messages if creation failed.
    """

    status: RegistrationStatus
    data_product_id: str | None = None
    message: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class SemanticViewResult:
    """Result of a semantic view registration operation.

    Attributes:
        status: The status of the registration operation.
        view_id: The created semantic view ID in Invariant (if successful).
        message: Human-readable message describing the result.
        errors: List of error messages if registration failed.
    """

    status: RegistrationStatus
    view_id: str | None = None
    message: str = ""
    errors: list[str] = field(default_factory=list)


def format_proposal_summary(proposal: InvariantProposal) -> str:
    """Format a proposal for display to the user.

    Args:
        proposal: The InvariantProposal to format.

    Returns:
        A human-readable string summary of the proposal.
    """
    lines = [
        f"Dataset: {proposal.dataset_name}",
        f"Kind: {proposal.dataset_kind.value}",
        f"Shape: {proposal.shape_hypothesis.value}",
        f"Grain: {', '.join(proposal.grain) if proposal.grain else '(none)'}",
        "",
        "Columns:",
    ]

    for col in proposal.columns:
        role_str = col.role.value
        type_str = col.primitive_type.value
        hints = []
        if col.reference_system_hint:
            hints.append(f"ref={col.reference_system_hint}")
        if col.concept_hint:
            hints.append(f"concept={col.concept_hint}")
        if col.unit_hint:
            hints.append(f"unit={col.unit_hint}")
        if col.time_granularity:
            hints.append(f"granularity={col.time_granularity}")

        hint_str = f" ({', '.join(hints)})" if hints else ""
        lines.append(f"  - {col.name}: {role_str} ({type_str}){hint_str}")

    if proposal.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in proposal.warnings:
            lines.append(f"  ! {warning}")

    if proposal.required_user_confirmations:
        lines.append("")
        lines.append("Items requiring confirmation:")
        for conf in proposal.required_user_confirmations:
            lines.append(f"  ? {conf}")

    return "\n".join(lines)


def confirm_registration(
    proposal: InvariantProposal,
    *,
    auto_confirm: bool = False,
) -> bool:
    """Display proposal and ask for user confirmation before registration.

    This function displays a formatted summary of the proposal and prompts
    the user to confirm before proceeding with registration.

    Args:
        proposal: The InvariantProposal to confirm.
        auto_confirm: If True, skip confirmation and return True.

    Returns:
        True if the user confirms, False otherwise.
    """
    if auto_confirm:
        return True

    summary = format_proposal_summary(proposal)
    print("\n" + "=" * 60)
    print("PROPOSED INVARIANT REGISTRATION")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    if proposal.required_user_confirmations:
        print("\nThis proposal has items requiring confirmation.")
        print("Please review the items marked with '?' above.")

    try:
        response = input("\nProceed with registration? [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return False


def register_catalog(proposal: InvariantProposal) -> RegistrationResult:
    """Register a Dataset and its Variables in Invariant.

    This function will create:
    1. A Dataset entity with the specified kind, grain, and metadata
    2. Variable entities for each column in the proposal

    The registration uses the proposal's decision_record_id to link
    back to the inference audit trail.

    Args:
        proposal: The InvariantProposal containing dataset and column specs.

    Returns:
        RegistrationResult with status and created IDs.

    Note:
        This is a stub implementation. When implemented, this function will:
        - Connect to the Invariant API
        - Create a Dataset with the specified kind and grain
        - Create Variable entities for each column
        - Handle rollback if any creation fails
        - Return the created IDs for reference
    """
    return RegistrationResult(
        status=RegistrationStatus.NOT_IMPLEMENTED,
        dataset_id=None,
        variable_ids=[],
        message="Catalog registration is not yet implemented. "
        "This stub will be replaced with Invariant API integration.",
        errors=[],
    )


def create_data_product(
    proposal: InvariantProposal,
    *,
    name: str | None = None,
    description: str | None = None,
) -> DataProductResult:
    """Create a DataProduct in Invariant for publishing.

    A DataProduct wraps a registered Dataset for external consumption,
    adding publishing metadata, access controls, and documentation.

    Args:
        proposal: The InvariantProposal for the underlying dataset.
        name: Optional override for the DataProduct name.
            Defaults to proposal.dataset_name.
        description: Optional description for the DataProduct.

    Returns:
        DataProductResult with status and created ID.

    Note:
        This is a stub implementation. When implemented, this function will:
        - Ensure the underlying Dataset is registered
        - Create a DataProduct entity wrapping the Dataset
        - Configure publishing settings and access controls
        - Generate documentation from the proposal metadata
    """
    return DataProductResult(
        status=RegistrationStatus.NOT_IMPLEMENTED,
        data_product_id=None,
        message="DataProduct creation is not yet implemented. "
        "This stub will be replaced with Invariant API integration.",
        errors=[],
    )


def register_semantic_view(
    proposal: InvariantProposal,
    *,
    view_name: str | None = None,
    base_dataset_id: str | None = None,
) -> SemanticViewResult:
    """Register a semantic dataset view in Invariant.

    A semantic view provides an alternative representation of a Dataset,
    such as a denormalized view, aggregation, or filtered subset.

    Args:
        proposal: The InvariantProposal defining the view structure.
        view_name: Optional name for the view. Defaults to proposal.dataset_name.
        base_dataset_id: ID of the base Dataset this view is derived from.
            If None, assumes the view is standalone.

    Returns:
        SemanticViewResult with status and created ID.

    Note:
        This is a stub implementation. When implemented, this function will:
        - Validate the view definition against the base dataset
        - Create a semantic view entity in Invariant
        - Link the view to its base dataset for lineage tracking
        - Register view-specific Variables if they differ from base
    """
    return SemanticViewResult(
        status=RegistrationStatus.NOT_IMPLEMENTED,
        view_id=None,
        message="Semantic view registration is not yet implemented. "
        "This stub will be replaced with Invariant API integration.",
        errors=[],
    )


@dataclass
class RollbackResult:
    """Result of a rollback operation.

    Attributes:
        status: The status of the rollback operation.
        rolled_back_entities: List of entity IDs that were rolled back.
        message: Human-readable message describing the result.
        errors: List of error messages if rollback failed.
    """

    status: RegistrationStatus
    rolled_back_entities: list[str] = field(default_factory=list)
    message: str = ""
    errors: list[str] = field(default_factory=list)


def rollback_registration(registration_id: str) -> bool:
    """Rollback a previous registration operation.

    This function undoes a registration by removing the created Dataset
    and Variable entities from Invariant.

    Args:
        registration_id: The ID of the registration to rollback. This can be
            a dataset_id from RegistrationResult, a data_product_id from
            DataProductResult, or a view_id from SemanticViewResult.

    Returns:
        True if rollback was successful or would be successful when implemented,
        False otherwise.

    Note:
        This is a stub implementation. When implemented, this function will:
        - Look up the registration by ID
        - Delete the Dataset entity and all associated Variable entities
        - Handle cascading deletes for dependent entities
        - Log the rollback operation for audit purposes
        - Return False if the registration doesn't exist or can't be rolled back
    """
    # Stub implementation - always returns True to indicate
    # the rollback would succeed when implemented
    return True


def get_rollback_result(registration_id: str) -> RollbackResult:
    """Get detailed result of a rollback operation.

    This function provides more detail than the boolean rollback_registration.

    Args:
        registration_id: The ID of the registration to rollback.

    Returns:
        RollbackResult with status and details.

    Note:
        This is a stub implementation. When implemented, this function will:
        - Perform the actual rollback via the Invariant API
        - Track which entities were successfully rolled back
        - Report any errors that occurred during rollback
    """
    return RollbackResult(
        status=RegistrationStatus.NOT_IMPLEMENTED,
        rolled_back_entities=[],
        message=f"Rollback for registration '{registration_id}' is not yet implemented. "
        "This stub will be replaced with Invariant API integration.",
        errors=[],
    )
