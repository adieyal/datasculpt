"""Microdata-specific analysis and survey metadata extraction.

This module provides utilities for detecting and analyzing survey/microdata
patterns, including:
- Survey type inference from question prefix patterns (DHS, LSMS, MICS, etc.)
- Section extraction from question column names
- Common prefix pattern detection
- Microdata profile extraction

These utilities support the microdata role scoring and shape detection
in the broader Datasculpt inference pipeline.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from datasculpt.core.roles import (
    CLUSTER_ID_PATTERNS,
    GEOGRAPHY_LEVEL_PATTERNS,
    RESPONDENT_ID_PATTERNS,
    SUBUNIT_ID_PATTERNS,
    SURVEY_QUESTION_PATTERNS,
    SURVEY_WEIGHT_PATTERNS,
    _matches_any_pattern,
)
from datasculpt.core.types import (
    ColumnEvidence,
    MicrodataLevel,
    MicrodataProfile,
    PrimitiveType,
    QuestionColumnProfile,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# Question column patterns for different survey types
# LSMS: s1aq1, s2bq3, s01aq01, etc.
LSMS_PATTERN = re.compile(r"^s(\d+)([a-z])?q(\d+)", re.IGNORECASE)

# DHS Household: hv001, hv002, hv103, etc.
DHS_HV_PATTERN = re.compile(r"^hv(\d+)", re.IGNORECASE)

# DHS Men's: mv001, mv002, etc.
DHS_MV_PATTERN = re.compile(r"^mv(\d+)", re.IGNORECASE)

# DHS Generic: v101, v102, etc.
DHS_V_PATTERN = re.compile(r"^v(\d+)", re.IGNORECASE)

# Other specify suffix
OTHER_SPECIFY_PATTERN = re.compile(r"_(os|other|oth|specify|spec|text|txt)$", re.IGNORECASE)

# Generic question patterns
GENERIC_Q_PATTERN = re.compile(r"^q(\d+)", re.IGNORECASE)

# Letter + number pattern (sh01, b2, etc.)
LETTER_NUM_PATTERN = re.compile(r"^([a-z]{1,2})(\d+)([a-z])?$", re.IGNORECASE)

# MICS prefixes
MICS_PREFIXES = ("hh", "hl", "wm", "mn", "cm", "br", "ub", "fs")


# ============ Survey Type Inference ============


def infer_survey_type(question_prefix: str | None) -> str | None:
    """Infer survey type from question prefix pattern.

    Analyzes the prefix pattern to determine the likely survey type:
    - DHS (Demographic and Health Survey) - hv*, mv*, v* patterns
    - LSMS (Living Standards Measurement Study) - s#q# patterns
    - MICS (Multiple Indicator Cluster Survey) - hh*, hl*, wm* patterns
    - Generic coded survey - Other recognizable patterns

    Args:
        question_prefix: The common prefix extracted from question columns.
                         Can be a single column name or pattern.

    Returns:
        One of:
        - "DHS": Demographic and Health Survey pattern
        - "LSMS": Living Standards Measurement Study pattern
        - "MICS": Multiple Indicator Cluster Survey pattern
        - None: Pattern not recognized

    Examples:
        >>> infer_survey_type("hv001")
        'DHS'
        >>> infer_survey_type("s1aq1")
        'LSMS'
        >>> infer_survey_type("v101")
        'DHS'
        >>> infer_survey_type("wm1")
        'MICS'
    """
    if not question_prefix:
        return None

    prefix_lower = question_prefix.lower()

    # DHS patterns
    if DHS_HV_PATTERN.match(prefix_lower):
        return "DHS"
    if DHS_MV_PATTERN.match(prefix_lower):
        return "DHS"
    if DHS_V_PATTERN.match(prefix_lower):
        return "DHS"

    # Check short prefixes for DHS
    if prefix_lower.startswith(("hv", "mv", "sh", "sm")):
        return "DHS"
    if prefix_lower == "v" or (prefix_lower.startswith("v") and len(prefix_lower) <= 2):
        return "DHS"

    # LSMS patterns
    if LSMS_PATTERN.match(prefix_lower):
        return "LSMS"
    if prefix_lower.startswith("s") and "q" in prefix_lower:
        return "LSMS"

    # MICS patterns
    for mics_prefix in MICS_PREFIXES:
        if prefix_lower.startswith(mics_prefix):
            return "MICS"

    return None


def infer_survey_type_from_columns(question_columns: list[str]) -> str | None:
    """Infer survey type from a list of question column names.

    Uses majority voting across all question columns.

    Args:
        question_columns: List of question column names.

    Returns:
        Most common survey type, or None if none detected.
    """
    if not question_columns:
        return None

    type_votes: Counter[str] = Counter()

    for col in question_columns:
        survey_type = infer_survey_type(col)
        if survey_type:
            type_votes[survey_type] += 1

    if not type_votes:
        return None

    return type_votes.most_common(1)[0][0]


# ============ Section Inference ============

# LSMS section mappings (approximate - varies by country)
LSMS_SECTION_MAP = {
    "1": "demographics",
    "2": "education",
    "3": "health",
    "4": "labor",
    "5": "housing",
    "6": "agriculture",
    "7": "consumption",
    "8": "income",
    "9": "assets",
    "10": "shocks",
    "11": "social_protection",
    "12": "migration",
}

# DHS section inference from prefix
DHS_SECTION_MAP = {
    "hv": "household",
    "mv": "men",
    "v": "women",
    "b": "birth_history",
    "m": "maternal_health",
    "h": "child_health",
    "sh": "hiv",
    "sm": "malaria",
}


def infer_section(question_columns: list[str]) -> str | None:
    """Infer survey section from question column patterns.

    For LSMS-style data, extracts section number and maps to section name.
    For DHS, infers section from variable prefix.

    Args:
        question_columns: List of question column names.

    Returns:
        Section name or section identifier, or None if not determinable.

    Examples:
        >>> infer_section(["s1aq1", "s1aq2", "s1aq3"])
        'demographics'
        >>> infer_section(["hv001", "hv002", "hv003"])
        'household'
        >>> infer_section(["s2bq1", "s2bq2"])
        'education'
    """
    if not question_columns:
        return None

    # Try LSMS section detection
    section_nums: Counter[str] = Counter()

    for col in question_columns:
        col_lower = col.lower()

        # Strip _os suffix if present
        if OTHER_SPECIFY_PATTERN.search(col_lower):
            col_lower = OTHER_SPECIFY_PATTERN.sub("", col_lower)

        lsms_match = LSMS_PATTERN.match(col_lower)
        if lsms_match:
            section_nums[lsms_match.group(1)] += 1

    if section_nums:
        most_common_num = section_nums.most_common(1)[0][0]
        return LSMS_SECTION_MAP.get(most_common_num, f"section_{most_common_num}")

    # Try DHS prefix detection
    dhs_sections: Counter[str] = Counter()
    for col in question_columns:
        col_lower = col.lower()
        for prefix, section in DHS_SECTION_MAP.items():
            if col_lower.startswith(prefix):
                dhs_sections[section] += 1
                break

    if dhs_sections:
        return dhs_sections.most_common(1)[0][0]

    return None


# ============ Question Prefix Extraction ============


def extract_question_prefix(question_columns: list[str]) -> str | None:
    """Extract common prefix pattern from question columns.

    Identifies the common prefix pattern across all question column names.

    Args:
        question_columns: List of question column names.

    Returns:
        The common prefix pattern, or None if columns don't share
        a recognizable pattern with >50% majority.

    Examples:
        >>> extract_question_prefix(["s1aq1", "s1aq2", "s1aq3"])
        's1a'
        >>> extract_question_prefix(["hv001", "hv002", "hv103"])
        'hv'
        >>> extract_question_prefix(["v101", "v102", "v103"])
        'v'
    """
    if not question_columns:
        return None

    prefix_counts: Counter[str] = Counter()

    for col in question_columns:
        col_lower = col.lower()

        # Strip _os suffix if present
        if OTHER_SPECIFY_PATTERN.search(col_lower):
            col_lower = OTHER_SPECIFY_PATTERN.sub("", col_lower)

        # LSMS: s1aq1 -> s1a
        lsms_match = LSMS_PATTERN.match(col_lower)
        if lsms_match:
            section_num = lsms_match.group(1)
            section_letter = lsms_match.group(2) or ""
            prefix_counts[f"s{section_num}{section_letter}"] += 1
            continue

        # DHS patterns
        if DHS_HV_PATTERN.match(col_lower):
            prefix_counts["hv"] += 1
            continue
        if DHS_MV_PATTERN.match(col_lower):
            prefix_counts["mv"] += 1
            continue
        if DHS_V_PATTERN.match(col_lower):
            prefix_counts["v"] += 1
            continue

        # Generic q pattern
        if GENERIC_Q_PATTERN.match(col_lower):
            prefix_counts["q"] += 1
            continue

        # Letter+number pattern
        letter_match = LETTER_NUM_PATTERN.match(col_lower)
        if letter_match:
            prefix_counts[letter_match.group(1)] += 1

    if not prefix_counts:
        return None

    # Check if there's a dominant prefix (>50% of columns)
    total = sum(prefix_counts.values())
    most_common_prefix, count = prefix_counts.most_common(1)[0]

    if count / total >= 0.5:
        return most_common_prefix

    return None


# ============ Question Column Parsing ============


def is_other_specify_column(column_name: str) -> bool:
    """Check if a column is an 'other specify' variant.

    'Other specify' columns capture free-text responses when a respondent
    selects 'other' in a multiple choice question.

    Args:
        column_name: The column name to check.

    Returns:
        True if the column appears to be an 'other specify' column.

    Examples:
        >>> is_other_specify_column("s1aq1_os")
        True
        >>> is_other_specify_column("s1aq1_other")
        True
        >>> is_other_specify_column("s1aq1")
        False
    """
    return bool(OTHER_SPECIFY_PATTERN.search(column_name.lower()))


def parse_question_column(column_name: str) -> dict[str, str | None]:
    """Parse a question column name into its components.

    Extracts section code, question number, and subquestion suffix.

    Args:
        column_name: The question column name.

    Returns:
        Dict with keys: 'section_code', 'question_number', 'subquestion'

    Examples:
        >>> parse_question_column('s1aq1')
        {'section_code': 's1a', 'question_number': 'q1', 'subquestion': None}
        >>> parse_question_column('s1aq1_os')
        {'section_code': 's1a', 'question_number': 'q1', 'subquestion': '_os'}
        >>> parse_question_column('hv001')
        {'section_code': 'hv', 'question_number': '001', 'subquestion': None}
    """
    result: dict[str, str | None] = {
        "section_code": None,
        "question_number": None,
        "subquestion": None,
    }

    col_lower = column_name.lower()

    # Check for other specify suffix
    os_match = OTHER_SPECIFY_PATTERN.search(col_lower)
    if os_match:
        result["subquestion"] = os_match.group(0)
        col_lower = OTHER_SPECIFY_PATTERN.sub("", col_lower)

    # LSMS: s1aq1 -> section=s1a, question=q1
    lsms_match = LSMS_PATTERN.match(col_lower)
    if lsms_match:
        section_num = lsms_match.group(1)
        section_letter = lsms_match.group(2) or ""
        result["section_code"] = f"s{section_num}{section_letter}"
        result["question_number"] = f"q{lsms_match.group(3)}"
        return result

    # DHS household: hv001 -> section=hv, question=001
    hv_match = DHS_HV_PATTERN.match(col_lower)
    if hv_match:
        result["section_code"] = "hv"
        result["question_number"] = hv_match.group(1)
        return result

    # DHS men's: mv001
    mv_match = DHS_MV_PATTERN.match(col_lower)
    if mv_match:
        result["section_code"] = "mv"
        result["question_number"] = mv_match.group(1)
        return result

    # DHS generic: v101
    v_match = DHS_V_PATTERN.match(col_lower)
    if v_match:
        result["section_code"] = "v"
        result["question_number"] = v_match.group(1)
        return result

    # Generic q pattern: q1
    q_match = GENERIC_Q_PATTERN.match(col_lower)
    if q_match:
        result["section_code"] = "q"
        result["question_number"] = q_match.group(1)
        return result

    return result


# ============ Column Detection Helpers ============


def detect_question_columns(columns: Sequence[ColumnEvidence]) -> list[ColumnEvidence]:
    """Detect columns that are survey question responses.

    Args:
        columns: Sequence of column evidence objects.

    Returns:
        List of columns matching question patterns.
    """
    return [
        col for col in columns
        if _matches_any_pattern(col.name, SURVEY_QUESTION_PATTERNS)
    ]


def detect_respondent_id_columns(columns: Sequence[ColumnEvidence]) -> list[ColumnEvidence]:
    """Detect primary respondent/unit ID columns.

    Args:
        columns: Sequence of column evidence objects.

    Returns:
        List of columns matching respondent ID patterns.
    """
    return [
        col for col in columns
        if _matches_any_pattern(col.name, RESPONDENT_ID_PATTERNS)
    ]


def detect_subunit_id_columns(columns: Sequence[ColumnEvidence]) -> list[ColumnEvidence]:
    """Detect secondary/subunit ID columns.

    Args:
        columns: Sequence of column evidence objects.

    Returns:
        List of columns matching subunit ID patterns.
    """
    return [
        col for col in columns
        if _matches_any_pattern(col.name, SUBUNIT_ID_PATTERNS)
    ]


def detect_cluster_columns(columns: Sequence[ColumnEvidence]) -> list[ColumnEvidence]:
    """Detect sampling cluster/EA ID columns.

    Args:
        columns: Sequence of column evidence objects.

    Returns:
        List of columns matching cluster ID patterns.
    """
    return [
        col for col in columns
        if _matches_any_pattern(col.name, CLUSTER_ID_PATTERNS)
    ]


def detect_weight_column(columns: Sequence[ColumnEvidence]) -> ColumnEvidence | None:
    """Detect survey weight column.

    Returns the first matching column (typically there's only one).

    Args:
        columns: Sequence of column evidence objects.

    Returns:
        Weight column evidence, or None if not found.
    """
    for col in columns:
        if _matches_any_pattern(col.name, SURVEY_WEIGHT_PATTERNS):
            return col
    return None


def detect_geography_columns(columns: Sequence[ColumnEvidence]) -> list[ColumnEvidence]:
    """Detect geography hierarchy columns.

    Args:
        columns: Sequence of column evidence objects.

    Returns:
        List of columns matching geography level patterns.
    """
    return [
        col for col in columns
        if _matches_any_pattern(col.name, GEOGRAPHY_LEVEL_PATTERNS)
    ]


# ============ Observation Level Inference ============

# Patterns suggesting individual-level data
INDIVIDUAL_PATTERNS = (
    re.compile(r"indiv", re.IGNORECASE),
    re.compile(r"person", re.IGNORECASE),
    re.compile(r"^age$", re.IGNORECASE),
    re.compile(r"^sex$", re.IGNORECASE),
    re.compile(r"^gender$", re.IGNORECASE),
    re.compile(r"member", re.IGNORECASE),
)

# Patterns suggesting household-level data
HOUSEHOLD_PATTERNS = (
    re.compile(r"^hh", re.IGNORECASE),
    re.compile(r"household", re.IGNORECASE),
    re.compile(r"dwelling", re.IGNORECASE),
)


def _count_pattern_matches(
    columns: list[ColumnEvidence],
    patterns: tuple[re.Pattern, ...],
) -> int:
    """Count how many columns match any of the given patterns."""
    return sum(
        1 for col in columns
        if any(p.search(col.name) for p in patterns)
    )


def infer_observation_level(
    primary_ids: list[ColumnEvidence],
    secondary_ids: list[ColumnEvidence],
    question_cols: list[ColumnEvidence],
    all_columns: list[ColumnEvidence],
) -> tuple[MicrodataLevel, float]:
    """Infer the observation level (household, individual, etc.).

    Uses ID structure and column naming patterns to determine what each row
    represents.

    Args:
        primary_ids: Detected primary ID columns.
        secondary_ids: Detected secondary ID columns.
        question_cols: Detected question columns.
        all_columns: All columns for additional signal.

    Returns:
        Tuple of (MicrodataLevel, confidence).
    """
    # Gather all relevant columns for pattern matching
    all_names = primary_ids + secondary_ids + question_cols + all_columns

    # Count signals for each level
    individual_signals = _count_pattern_matches(all_names, INDIVIDUAL_PATTERNS)
    household_signals = _count_pattern_matches(all_names, HOUSEHOLD_PATTERNS)

    # Strong signal: presence of secondary IDs suggests individual level
    if secondary_ids:
        return MicrodataLevel.INDIVIDUAL, 0.8

    # Compare signal strengths
    if individual_signals > household_signals and individual_signals > 0:
        confidence = min(0.5 + 0.1 * individual_signals, 0.9)
        return MicrodataLevel.INDIVIDUAL, confidence

    if household_signals > 0:
        confidence = min(0.5 + 0.1 * household_signals, 0.9)
        return MicrodataLevel.HOUSEHOLD, confidence

    # No clear signal
    if primary_ids:
        return MicrodataLevel.UNKNOWN, 0.4
    return MicrodataLevel.UNKNOWN, 0.3


# ============ Geography Hierarchy Ordering ============

# Ordering for common geography levels (broad to narrow)
GEOGRAPHY_LEVEL_ORDER = {
    "zone": 1,
    "region": 2,
    "state": 3,
    "province": 3,
    "county": 4,
    "district": 5,
    "lga": 5,  # Local Government Area (Nigeria)
    "sub_county": 6,
    "subcounty": 6,
    "municipality": 6,
    "ward": 7,
    "parish": 7,
    "sector": 8,
    "village": 9,
    "ea": 10,  # Enumeration Area
    "admin1": 2,
    "admin2": 4,
    "admin3": 6,
    "admin4": 8,
    "adm1": 2,
    "adm2": 4,
    "adm3": 6,
    "adm4": 8,
}


def order_geography_hierarchy(geo_columns: list[str]) -> list[str]:
    """Order geography columns from broad to narrow.

    Args:
        geo_columns: List of geography column names.

    Returns:
        List of column names ordered from broad to narrow administrative level.

    Examples:
        >>> order_geography_hierarchy(['lga', 'zone', 'state'])
        ['zone', 'state', 'lga']
    """
    def get_order(col_name: str) -> int:
        col_lower = col_name.lower()
        return GEOGRAPHY_LEVEL_ORDER.get(col_lower, 50)

    return sorted(geo_columns, key=get_order)


# ============ Profile Extraction ============


def extract_microdata_profile(
    columns: Sequence[ColumnEvidence],
    row_count: int | None = None,
) -> MicrodataProfile:
    """Extract detailed microdata profile from column evidence.

    Analyzes column patterns to build a comprehensive profile of the
    microdata structure including ID hierarchy, geography, and questions.

    Args:
        columns: Sequence of column evidence objects.
        row_count: Optional row count for additional inference.

    Returns:
        MicrodataProfile with detected structure.
    """
    columns_list = list(columns)

    # Detect ID columns
    primary_ids = detect_respondent_id_columns(columns)
    secondary_ids = detect_subunit_id_columns(columns)
    cluster_cols = detect_cluster_columns(columns)

    # Detect geography hierarchy
    geo_cols = detect_geography_columns(columns)
    geo_names = [c.name for c in geo_cols]
    ordered_geo = order_geography_hierarchy(geo_names)

    # Detect question columns and extract pattern
    question_cols = detect_question_columns(columns)
    question_names = [c.name for c in question_cols]
    question_prefix = extract_question_prefix(question_names)

    # Detect weight column
    weight_col = detect_weight_column(columns)

    # Infer observation level
    level, level_conf = infer_observation_level(
        primary_ids, secondary_ids, question_cols, columns_list
    )

    # Infer survey metadata
    survey_hint = infer_survey_type_from_columns(question_names)
    section_hint = infer_section(question_names)

    return MicrodataProfile(
        level=level,
        level_confidence=level_conf,
        primary_id_columns=[c.name for c in primary_ids],
        secondary_id_columns=[c.name for c in secondary_ids],
        cluster_columns=[c.name for c in cluster_cols],
        geography_hierarchy=ordered_geo,
        question_prefix_pattern=question_prefix,
        question_columns=question_names,
        weight_column=weight_col.name if weight_col else None,
        survey_type_hint=survey_hint,
        section_hint=section_hint,
    )


def create_question_profile(
    evidence: ColumnEvidence,
) -> QuestionColumnProfile:
    """Create a detailed profile for a question column.

    Args:
        evidence: Column evidence for the question.

    Returns:
        QuestionColumnProfile with parsed structure.
    """
    parsed = parse_question_column(evidence.name)

    # Determine response type from evidence
    if evidence.primitive_type == PrimitiveType.STRING:
        if evidence.distinct_ratio < 0.1 and evidence.unique_count <= 20:
            response_type = "categorical"
        else:
            response_type = "text"
    elif evidence.primitive_type in (PrimitiveType.INTEGER, PrimitiveType.NUMBER):
        if evidence.distinct_ratio < 0.05 and evidence.unique_count <= 10:
            response_type = "categorical"
        elif evidence.unique_count == 2:
            response_type = "binary"
        else:
            response_type = "numeric"
    else:
        response_type = "unknown"

    return QuestionColumnProfile(
        name=evidence.name,
        section_code=parsed["section_code"],
        question_number=parsed["question_number"],
        subquestion=parsed["subquestion"],
        response_type=response_type,
        distinct_values=evidence.unique_count,
        has_value_labels=False,  # Would need metadata from Stata/SPSS
        label_hint=None,  # Would need variable labels from source
    )
