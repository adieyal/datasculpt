"""Frictionless Data adapter for schema inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasculpt.adapters.base import AdapterResult, BaseAdapter, safe_import
from datasculpt.core.types import ColumnEvidence, PrimitiveType

if TYPE_CHECKING:
    import pandas as pd

# Attempt to import frictionless
_frictionless = safe_import("frictionless")

AVAILABLE: bool = _frictionless is not None


class FrictionlessAdapter(BaseAdapter):
    """Adapter for Frictionless Data schema inference."""

    @property
    def name(self) -> str:
        return "frictionless"

    @property
    def available(self) -> bool:
        return AVAILABLE

    def profile(self, df: pd.DataFrame) -> AdapterResult:
        """Profile a DataFrame using Frictionless.

        Args:
            df: The DataFrame to profile.

        Returns:
            AdapterResult with inferred schema information.
        """
        if not AVAILABLE:
            return AdapterResult(
                warnings=["frictionless not installed; skipping adapter"]
            )

        return _profile_with_frictionless(df)


def _profile_with_frictionless(df: pd.DataFrame) -> AdapterResult:
    """Internal profiling implementation using frictionless."""
    from frictionless import Resource, Schema

    column_annotations: dict[str, dict[str, Any]] = {}
    dataset_annotations: dict[str, Any] = {}
    warnings: list[str] = []

    try:
        resource = Resource(df)
        resource.infer()
        schema: Schema = resource.schema

        for field in schema.fields:
            column_annotations[field.name] = {
                "frictionless_type": field.type,
                "frictionless_format": getattr(field, "format", None),
                "frictionless_constraints": getattr(field, "constraints", {}),
            }

        dataset_annotations["frictionless_schema"] = schema.to_dict()

    except Exception as e:
        warnings.append(f"frictionless profiling failed: {e}")

    return AdapterResult(
        column_annotations=column_annotations,
        dataset_annotations=dataset_annotations,
        warnings=warnings,
    )


def infer_schema(df: pd.DataFrame) -> dict[str, Any] | None:
    """Infer a Frictionless schema from a DataFrame.

    Args:
        df: The DataFrame to analyze.

    Returns:
        Schema dict if available, None otherwise.
    """
    if not AVAILABLE:
        return None

    from frictionless import Resource

    try:
        resource = Resource(df)
        resource.infer()
        return resource.schema.to_dict()
    except Exception:
        return None


# Mapping from frictionless types to PrimitiveType
FRICTIONLESS_TYPE_MAP: dict[str, PrimitiveType] = {
    "string": PrimitiveType.STRING,
    "integer": PrimitiveType.INTEGER,
    "number": PrimitiveType.NUMBER,
    "boolean": PrimitiveType.BOOLEAN,
    "date": PrimitiveType.DATE,
    "datetime": PrimitiveType.DATETIME,
    "time": PrimitiveType.DATETIME,
    "year": PrimitiveType.INTEGER,
    "yearmonth": PrimitiveType.STRING,
    "duration": PrimitiveType.STRING,
    "geopoint": PrimitiveType.STRING,
    "geojson": PrimitiveType.STRING,
    "array": PrimitiveType.STRING,
    "object": PrimitiveType.STRING,
    "any": PrimitiveType.UNKNOWN,
}


def map_frictionless_type(frictionless_type: str) -> PrimitiveType:
    """Map a frictionless field type to a PrimitiveType.

    Args:
        frictionless_type: The frictionless type string.

    Returns:
        Corresponding PrimitiveType.
    """
    return FRICTIONLESS_TYPE_MAP.get(frictionless_type, PrimitiveType.UNKNOWN)


def enrich_evidence_from_schema(
    evidence: ColumnEvidence,
    schema: dict[str, Any],
) -> ColumnEvidence:
    """Enrich ColumnEvidence with frictionless schema information.

    Args:
        evidence: The existing ColumnEvidence to enrich.
        schema: The frictionless schema dict.

    Returns:
        The enriched ColumnEvidence (modified in place).
    """
    if not schema or "fields" not in schema:
        return evidence

    for field_info in schema["fields"]:
        if field_info.get("name") == evidence.name:
            frictionless_type = field_info.get("type", "any")
            frictionless_format = field_info.get("format")
            constraints = field_info.get("constraints", {})

            evidence.external["frictionless"] = {
                "type": frictionless_type,
                "format": frictionless_format,
                "constraints": constraints,
                "inferred_primitive": map_frictionless_type(frictionless_type).value,
            }

            # Add note about frictionless inference
            if frictionless_type not in ("string", "any"):
                evidence.notes.append(
                    f"frictionless inferred type: {frictionless_type}"
                )

            break

    return evidence


def validate(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Validate a DataFrame using Frictionless.

    Args:
        df: The DataFrame to validate.

    Returns:
        List of validation errors, empty if valid or unavailable.
    """
    if not AVAILABLE:
        return []

    from frictionless import Resource

    try:
        resource = Resource(df)
        report = resource.validate()
        return [error.to_dict() for error in report.flatten(["rowNumber", "fieldName", "message"])]
    except Exception:
        return []


# Singleton adapter instance
_adapter = FrictionlessAdapter()


def get_adapter() -> FrictionlessAdapter:
    """Get the frictionless adapter instance."""
    return _adapter
