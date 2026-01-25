"""Special column detection for Datasculpt (browser bundle).

Detects special columns like weights, denominators, suppression flags, etc.
"""


import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasculpt.types import ColumnEvidence


class SpecialColumnType(str, Enum):
    """Types of special columns in datasets."""

    WEIGHT = "weight"
    DENOMINATOR = "denominator"
    SUPPRESSION_FLAG = "suppression_flag"
    QUALITY_FLAG = "quality_flag"


class SpecialColumnStatus(str, Enum):
    """Status of special column detection."""

    AUTO_LOCKED = "auto_locked"
    NEEDS_CONFIRMATION = "needs_confirmation"
    USER_LOCKED = "user_locked"


@dataclass
class SpecialColumnCandidate:
    """A candidate special type for a column with scoring."""

    flag_type: SpecialColumnType
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def __lt__(self, other: SpecialColumnCandidate) -> bool:
        """Sort by confidence descending."""
        return self.confidence > other.confidence


@dataclass
class SpecialColumnResult:
    """Result of special column detection for a single column."""

    column_name: str
    candidates: list[SpecialColumnCandidate] = field(default_factory=list)
    selected: SpecialColumnType | None = None
    status: SpecialColumnStatus = SpecialColumnStatus.NEEDS_CONFIRMATION

    @property
    def requires_user_confirm(self) -> bool:
        """Whether user confirmation is needed."""
        return self.status == SpecialColumnStatus.NEEDS_CONFIRMATION


@dataclass
class SpecialColumnFlag:
    """Flag indicating a column has a special role."""

    column_name: str
    flag_type: SpecialColumnType
    confidence: float
    evidence: list[str]


# Patterns for detection
WEIGHT_PATTERNS = [
    re.compile(r"(?:^|_)weight(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)wgt(?:$|_)", re.IGNORECASE),
    re.compile(r"_wt$", re.IGNORECASE),
]

DENOMINATOR_PATTERNS = [
    re.compile(r"(?:^|_)denominator(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)denom(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)base(?:$|_)", re.IGNORECASE),
    re.compile(r"^n$", re.IGNORECASE),
]

SUPPRESSION_PATTERNS = [
    re.compile(r"(?:^|_)suppress", re.IGNORECASE),
    re.compile(r"(?:^|_)redact", re.IGNORECASE),
    re.compile(r"(?:^|_)masked(?:$|_)", re.IGNORECASE),
]

QUALITY_PATTERNS = [
    re.compile(r"(?:^|_)quality(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)reliability(?:$|_)", re.IGNORECASE),
    re.compile(r"(?:^|_)confidence(?:$|_)", re.IGNORECASE),
]


def _matches_patterns(name: str, patterns: list[re.Pattern]) -> list[str]:
    """Check if name matches patterns and return matched pattern strings."""
    matches = []
    for pattern in patterns:
        if pattern.search(name):
            matches.append(pattern.pattern)
    return matches


def score_weight_candidate(evidence: ColumnEvidence) -> SpecialColumnCandidate | None:
    """Score a column as a potential weight."""
    from datasculpt.types import PrimitiveType

    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    matches = _matches_patterns(name, WEIGHT_PATTERNS)
    if matches:
        score += 0.5
        evidence_list.extend([f"name:pattern:{p}" for p in matches])
    else:
        return None

    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        score += 0.2
        evidence_list.append("type:numeric")

    if evidence.value_profile.non_negative_ratio > 0.9:
        score += 0.15
        evidence_list.append("value:non_negative")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.WEIGHT,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


def score_denominator_candidate(evidence: ColumnEvidence) -> SpecialColumnCandidate | None:
    """Score a column as a potential denominator."""
    from datasculpt.types import PrimitiveType

    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    matches = _matches_patterns(name, DENOMINATOR_PATTERNS)
    if matches:
        score += 0.45
        evidence_list.extend([f"name:pattern:{p}" for p in matches])
    else:
        return None

    if evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        score += 0.2
        evidence_list.append("type:numeric")

    if evidence.value_profile.integer_ratio > 0.9:
        score += 0.15
        evidence_list.append("value:integer_like")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.DENOMINATOR,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


def score_suppression_candidate(evidence: ColumnEvidence) -> SpecialColumnCandidate | None:
    """Score a column as a potential suppression flag."""
    from datasculpt.types import PrimitiveType

    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    matches = _matches_patterns(name, SUPPRESSION_PATTERNS)
    if matches:
        score += 0.5
        evidence_list.extend([f"name:pattern:{p}" for p in matches])
    else:
        return None

    if evidence.primitive_type in (PrimitiveType.BOOLEAN, PrimitiveType.STRING):
        score += 0.15
        evidence_list.append("type:boolean_or_string")

    if evidence.value_profile.low_cardinality:
        score += 0.15
        evidence_list.append("value:low_cardinality")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.SUPPRESSION_FLAG,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


def score_quality_candidate(evidence: ColumnEvidence) -> SpecialColumnCandidate | None:
    """Score a column as a potential quality flag."""
    name = evidence.name
    evidence_list: list[str] = []
    score = 0.0

    matches = _matches_patterns(name, QUALITY_PATTERNS)
    if matches:
        score += 0.5
        evidence_list.extend([f"name:pattern:{p}" for p in matches])
    else:
        return None

    if evidence.value_profile.bounded_0_1_ratio > 0.9:
        score += 0.2
        evidence_list.append("value:probability_like")

    if evidence.value_profile.low_cardinality:
        score += 0.15
        evidence_list.append("value:low_cardinality")

    return SpecialColumnCandidate(
        flag_type=SpecialColumnType.QUALITY_FLAG,
        confidence=min(1.0, score),
        evidence=evidence_list,
    )


def detect_special_columns(
    evidence: dict[str, ColumnEvidence],
) -> list[SpecialColumnResult]:
    """Detect special columns with multi-candidate scoring.

    Args:
        evidence: Dictionary mapping column names to ColumnEvidence.

    Returns:
        List of SpecialColumnResult, one per column with candidates.
    """
    results: list[SpecialColumnResult] = []

    for col_name, col_evidence in evidence.items():
        candidates: list[SpecialColumnCandidate] = []

        weight_cand = score_weight_candidate(col_evidence)
        if weight_cand:
            candidates.append(weight_cand)

        denom_cand = score_denominator_candidate(col_evidence)
        if denom_cand:
            candidates.append(denom_cand)

        supp_cand = score_suppression_candidate(col_evidence)
        if supp_cand:
            candidates.append(supp_cand)

        qual_cand = score_quality_candidate(col_evidence)
        if qual_cand:
            candidates.append(qual_cand)

        if not candidates:
            continue

        candidates.sort()

        winner = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        margin = (
            winner.confidence - runner_up.confidence
            if runner_up
            else winner.confidence
        )

        if winner.confidence >= 0.75 and margin >= 0.15:
            status = SpecialColumnStatus.AUTO_LOCKED
            selected = winner.flag_type
        else:
            status = SpecialColumnStatus.NEEDS_CONFIRMATION
            selected = None

        result = SpecialColumnResult(
            column_name=col_name,
            candidates=candidates,
            selected=selected,
            status=status,
        )
        results.append(result)

    return results


def get_special_columns(
    evidence: dict[str, ColumnEvidence],
    flags: list[SpecialColumnFlag] | None = None,
) -> list[SpecialColumnFlag]:
    """Get all special columns from evidence and user flags.

    Args:
        evidence: Dictionary mapping column names to ColumnEvidence.
        flags: Optional list of user-provided SpecialColumnFlags.

    Returns:
        List of all detected and flagged special columns.
    """
    if flags is None:
        flags = []

    user_flagged = {f.column_name for f in flags}
    results = detect_special_columns(evidence)

    legacy_flags: list[SpecialColumnFlag] = list(flags)

    for result in results:
        if result.column_name in user_flagged:
            continue
        if result.candidates:
            top = result.candidates[0]
            legacy_flags.append(
                SpecialColumnFlag(
                    column_name=result.column_name,
                    flag_type=top.flag_type,
                    confidence=top.confidence,
                    evidence=top.evidence,
                )
            )

    legacy_flags.sort(key=lambda f: -f.confidence)
    return legacy_flags
