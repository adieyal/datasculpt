"""Datasculpt browser bundle for PyScript.

This is a bundled version of datasculpt core modules adapted for browser execution.
All modules have been flattened to avoid the 'core' subdirectory.
"""

from datasculpt.types import (
    ColumnEvidence,
    ColumnSpec,
    DatasetKind,
    DecisionRecord,
    GrainInference,
    HypothesisScore,
    InferenceConfig,
    InvariantProposal,
    PrimitiveType,
    Question,
    QuestionType,
    Role,
    ShapeHypothesis,
    StructuralType,
)

__version__ = "0.1.0"
__all__ = [
    "ColumnEvidence",
    "ColumnSpec",
    "DatasetKind",
    "DecisionRecord",
    "GrainInference",
    "HypothesisScore",
    "InferenceConfig",
    "InvariantProposal",
    "PrimitiveType",
    "Question",
    "QuestionType",
    "Role",
    "ShapeHypothesis",
    "StructuralType",
]
