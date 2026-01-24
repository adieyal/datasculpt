"""Query execution module for datasculpt.

Provides query interfaces for registered datasets using proposals
to interpret dimension and measure columns.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from datasculpt.core.types import InvariantProposal, Role


def _get_columns_by_role(proposal: InvariantProposal, role: Role) -> list[str]:
    """Get column names with a specific role from proposal.

    Args:
        proposal: The invariant proposal with column specs.
        role: The role to filter by.

    Returns:
        List of column names with the specified role.
    """
    return [col.name for col in proposal.columns if col.role == role]


def _get_dimension_columns(proposal: InvariantProposal) -> list[str]:
    """Get all dimension-like columns from proposal.

    Dimension columns include: DIMENSION, KEY, TIME, INDICATOR_NAME.

    Args:
        proposal: The invariant proposal with column specs.

    Returns:
        List of column names that serve as dimensions.
    """
    dimension_roles = {Role.DIMENSION, Role.KEY, Role.TIME, Role.INDICATOR_NAME}
    return [col.name for col in proposal.columns if col.role in dimension_roles]


def _get_measure_columns(proposal: InvariantProposal) -> list[str]:
    """Get all measure-like columns from proposal.

    Measure columns include: MEASURE, VALUE.

    Args:
        proposal: The invariant proposal with column specs.

    Returns:
        List of column names that serve as measures.
    """
    measure_roles = {Role.MEASURE, Role.VALUE}
    return [col.name for col in proposal.columns if col.role in measure_roles]


def apply_filters(
    df: pd.DataFrame,
    proposal: InvariantProposal,
    filters: dict[str, Any],
) -> pd.DataFrame:
    """Apply filters to a DataFrame based on dimension columns.

    Validates that filter keys correspond to dimension columns in the proposal.

    Args:
        df: Input DataFrame to filter.
        proposal: The invariant proposal with column specs.
        filters: Dictionary mapping column names to filter values.
            Values can be single values or lists for IN-style filtering.

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If a filter column is not a dimension column.
        KeyError: If a filter column does not exist in the DataFrame.
    """
    if not filters:
        return df

    # Get all dimension columns
    dimension_columns = set(_get_dimension_columns(proposal))

    # Validate filter columns
    for col_name in filters:
        if col_name not in df.columns:
            raise KeyError(f"Filter column '{col_name}' not found in DataFrame")
        if col_name not in dimension_columns:
            raise ValueError(
                f"Filter column '{col_name}' is not a dimension column. "
                f"Valid dimension columns: {sorted(dimension_columns)}"
            )

    # Apply filters
    mask = pd.Series(True, index=df.index)
    for col_name, filter_value in filters.items():
        if isinstance(filter_value, list):
            mask = mask & df[col_name].isin(filter_value)
        else:
            mask = mask & (df[col_name] == filter_value)

    return df[mask]


def query(
    df: pd.DataFrame,
    proposal: InvariantProposal,
    group_by: list[str] | None = None,
    measures: list[str] | None = None,
    filters: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Execute a query on a DataFrame using the proposal for interpretation.

    Args:
        df: Input DataFrame to query.
        proposal: The invariant proposal with column specs.
        group_by: Optional list of columns to group by. Must be dimension columns.
            If None, returns the filtered data without aggregation.
        measures: Optional list of measure columns to aggregate.
            If None, uses all measure columns from the proposal.
        filters: Optional dictionary of filters to apply before aggregation.

    Returns:
        Query result DataFrame with aggregated measures (sum by default).

    Raises:
        ValueError: If group_by columns are not dimension columns,
            or if measure columns are not measure columns.
        KeyError: If specified columns do not exist in the DataFrame.
    """
    # Apply filters first
    result_df = apply_filters(df, proposal, filters or {})

    # If no grouping requested, return filtered data
    if group_by is None:
        return result_df

    # Validate group_by columns
    dimension_columns = set(_get_dimension_columns(proposal))
    for col_name in group_by:
        if col_name not in df.columns:
            raise KeyError(f"Group by column '{col_name}' not found in DataFrame")
        if col_name not in dimension_columns:
            raise ValueError(
                f"Group by column '{col_name}' is not a dimension column. "
                f"Valid dimension columns: {sorted(dimension_columns)}"
            )

    # Determine measure columns
    if measures is None:
        measures = _get_measure_columns(proposal)
    else:
        # Validate specified measures
        measure_columns = set(_get_measure_columns(proposal))
        for col_name in measures:
            if col_name not in df.columns:
                raise KeyError(f"Measure column '{col_name}' not found in DataFrame")
            if col_name not in measure_columns:
                raise ValueError(
                    f"Measure column '{col_name}' is not a measure column. "
                    f"Valid measure columns: {sorted(measure_columns)}"
                )

    if not measures:
        raise ValueError(
            "No measure columns specified and none found in proposal. "
            "Cannot perform aggregation without measures."
        )

    # Perform aggregation
    return result_df.groupby(group_by, as_index=False)[measures].sum()


def explain_query(
    proposal: InvariantProposal,
    group_by: list[str] | None = None,
    measures: list[str] | None = None,
    filters: dict[str, Any] | None = None,
) -> str:
    """Generate a human-readable explanation of how a query will be interpreted.

    Args:
        proposal: The invariant proposal with column specs.
        group_by: Optional list of columns to group by.
        measures: Optional list of measure columns to aggregate.
        filters: Optional dictionary of filters.

    Returns:
        Human-readable explanation string.
    """
    lines: list[str] = []

    lines.append(f"Query Explanation for dataset: {proposal.dataset_name}")
    lines.append(f"Dataset kind: {proposal.dataset_kind.value}")
    lines.append("")

    # Column classification
    dimension_columns = _get_dimension_columns(proposal)
    measure_columns = _get_measure_columns(proposal)

    lines.append("Column Classification:")
    lines.append(f"  Dimension columns: {', '.join(dimension_columns) or '(none)'}")
    lines.append(f"  Measure columns: {', '.join(measure_columns) or '(none)'}")
    lines.append("")

    # Filters
    if filters:
        lines.append("Filters to apply:")
        for col_name, filter_value in filters.items():
            if isinstance(filter_value, list):
                lines.append(f"  {col_name} IN {filter_value}")
            else:
                lines.append(f"  {col_name} = {filter_value!r}")
        lines.append("")
    else:
        lines.append("No filters specified.")
        lines.append("")

    # Grouping
    if group_by:
        lines.append(f"Group by: {', '.join(group_by)}")
    else:
        lines.append("No grouping specified (will return filtered rows).")
    lines.append("")

    # Measures
    if group_by:
        effective_measures = measures if measures else measure_columns
        if effective_measures:
            lines.append(f"Measures to aggregate (sum): {', '.join(effective_measures)}")
        else:
            lines.append("WARNING: No measures available for aggregation.")
    lines.append("")

    # Result description
    lines.append("Query Result:")
    if group_by:
        if measures:
            lines.append(
                f"  Aggregated {', '.join(measures)} grouped by {', '.join(group_by)}"
            )
        else:
            lines.append(
                f"  Aggregated all measure columns grouped by {', '.join(group_by)}"
            )
    else:
        if filters:
            lines.append("  Filtered rows (no aggregation)")
        else:
            lines.append("  All rows (no filtering or aggregation)")

    return "\n".join(lines)


def validate_query(
    proposal: InvariantProposal,
    group_by: list[str] | None = None,
    measures: list[str] | None = None,
    filters: dict[str, Any] | None = None,
) -> list[str]:
    """Validate a query and return warnings about potential issues.

    Args:
        proposal: The invariant proposal with column specs.
        group_by: Optional list of columns to group by.
        measures: Optional list of measure columns to aggregate.
        filters: Optional dictionary of filters.

    Returns:
        List of warning messages (empty if no issues found).
    """
    warnings: list[str] = []

    # Get column classification
    dimension_columns = set(_get_dimension_columns(proposal))
    measure_columns = set(_get_measure_columns(proposal))
    all_column_names = {col.name for col in proposal.columns}

    # Check for ambiguous structure
    if not dimension_columns:
        warnings.append(
            "No dimension columns identified in proposal. "
            "Filtering and grouping may not be meaningful."
        )

    if not measure_columns:
        warnings.append(
            "No measure columns identified in proposal. "
            "Aggregation queries will fail."
        )

    # Check grain
    if not proposal.grain:
        warnings.append(
            "No grain (unique key) identified. "
            "Data may contain duplicates that affect query results."
        )

    # Validate group_by columns
    if group_by:
        for col_name in group_by:
            if col_name not in all_column_names:
                warnings.append(f"Group by column '{col_name}' not found in proposal.")
            elif col_name not in dimension_columns:
                warnings.append(
                    f"Group by column '{col_name}' is not a dimension. "
                    f"Expected one of: {sorted(dimension_columns)}"
                )

        # Check if grouping by grain columns
        grain_set = set(proposal.grain)
        group_by_set = set(group_by)
        if grain_set and not grain_set.intersection(group_by_set):
            warnings.append(
                "Group by columns do not include any grain columns. "
                "Results may aggregate across unique records."
            )

    # Validate measure columns
    if measures:
        for col_name in measures:
            if col_name not in all_column_names:
                warnings.append(f"Measure column '{col_name}' not found in proposal.")
            elif col_name not in measure_columns:
                warnings.append(
                    f"Measure column '{col_name}' is not identified as a measure. "
                    f"Expected one of: {sorted(measure_columns)}"
                )

    # Validate filter columns
    if filters:
        for col_name in filters:
            if col_name not in all_column_names:
                warnings.append(f"Filter column '{col_name}' not found in proposal.")
            elif col_name not in dimension_columns:
                warnings.append(
                    f"Filter column '{col_name}' is not a dimension. "
                    "Filtering on non-dimension columns may be unexpected."
                )

    # Check proposal warnings
    if proposal.warnings:
        warnings.append(
            f"Proposal has {len(proposal.warnings)} warning(s) that may affect queries: "
            f"{proposal.warnings[0][:50]}..."
            if len(proposal.warnings[0]) > 50
            else proposal.warnings[0]
        )

    # Check required confirmations
    if proposal.required_user_confirmations:
        warnings.append(
            "Proposal has unconfirmed items. "
            "Query results may change after confirmation."
        )

    return warnings
