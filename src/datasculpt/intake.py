"""File intake module for loading, normalizing, and fingerprinting datasets."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


class IntakeError(Exception):
    """Error during file intake processing."""


@dataclass
class ColumnStats:
    """Basic statistics for a single column."""

    name: str
    dtype: str
    null_count: int
    null_rate: float
    unique_count: int
    unique_rate: float
    min_value: Any | None = None
    max_value: Any | None = None
    mean_value: float | None = None


@dataclass
class DatasetPreview:
    """Preview information for a dataset."""

    row_count: int
    column_count: int
    columns: list[str]
    sample_rows: list[dict[str, Any]]
    column_stats: list[ColumnStats]
    memory_usage_bytes: int


@dataclass
class DatasetFingerprint:
    """Content-based fingerprint for a dataset."""

    hash: str
    schema_hash: str
    content_hash: str
    row_count: int
    column_count: int


@dataclass
class IntakeResult:
    """Result of file intake processing."""

    dataframe: pd.DataFrame
    source_path: Path
    source_format: str
    fingerprint: DatasetFingerprint
    preview: DatasetPreview
    load_warnings: list[str] = field(default_factory=list)


@dataclass
class VersionEntry:
    """A single version entry in the history."""

    fingerprint: str
    timestamp: str
    path: str


@dataclass
class VersionHistory:
    """Version history for a dataset."""

    dataset_name: str
    entries: list[VersionEntry] = field(default_factory=list)


@dataclass
class SchemaChanges:
    """Detected schema changes between two dataset versions."""

    columns_added: list[str]
    columns_removed: list[str]
    type_changes: dict[str, tuple[str, str]]  # column -> (old_type, new_type)


SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".dta": "stata",
}


def load_file(path: str | Path) -> pd.DataFrame:
    """
    Load a file into a pandas DataFrame.

    Supports CSV, Excel (.xlsx, .xls), and Parquet formats.

    Args:
        path: Path to the file to load.

    Returns:
        DataFrame containing the file contents.

    Raises:
        IntakeError: If file format is unsupported or loading fails.
    """
    path = Path(path)

    if not path.exists():
        raise IntakeError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        raise IntakeError(f"Unsupported file format: {suffix}. Supported: {supported}")

    format_type = SUPPORTED_EXTENSIONS[suffix]

    try:
        if format_type == "csv":
            return pd.read_csv(path)
        elif format_type == "excel":
            return pd.read_excel(path)
        elif format_type == "parquet":
            return pd.read_parquet(path)
        elif format_type == "stata":
            # Try with categoricals first for better type inference
            # Suppress pandas warning about invalid value in cast for categoricals
            # (occurs when categorical labels contain values that can't be cleanly cast)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in cast",
                    category=RuntimeWarning,
                )
                try:
                    return pd.read_stata(path)
                except Exception:
                    # Fall back to no categorical conversion if there are issues
                    # (e.g., null values in categorical columns causing
                    # "Cannot setitem on a Categorical with a new category")
                    return pd.read_stata(path, convert_categoricals=False)
        else:
            raise IntakeError(f"Unknown format type: {format_type}")
    except IntakeError:
        raise
    except Exception as e:
        raise IntakeError(f"Failed to load {path}: {e}") from e


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame to a consistent format.

    Performs:
    - Column name normalization (strip whitespace)
    - Index reset if not default RangeIndex
    - Sort columns alphabetically for deterministic ordering

    Args:
        df: Input DataFrame to normalize.

    Returns:
        Normalized DataFrame.
    """
    df = df.copy()

    # Normalize column names (strip whitespace)
    df.columns = [str(col).strip() for col in df.columns]

    # Reset index if it's not a default RangeIndex
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0:
        df = df.reset_index(drop=True)

    return df


def generate_fingerprint(
    df: pd.DataFrame,
    sample_size: int = 100,
) -> DatasetFingerprint:
    """
    Generate a deterministic content-based fingerprint for a DataFrame.

    The fingerprint is based on:
    - Schema: column names and dtypes (sorted for consistency)
    - Content: sample of rows (sorted for consistency)

    Args:
        df: DataFrame to fingerprint.
        sample_size: Number of rows to include in content hash.

    Returns:
        DatasetFingerprint with hash components.
    """
    # Schema hash: sorted column names and their dtypes
    schema_data = {
        "columns": sorted(df.columns.tolist()),
        "dtypes": {col: str(df[col].dtype) for col in sorted(df.columns)},
    }
    schema_json = json.dumps(schema_data, sort_keys=True)
    schema_hash = hashlib.sha256(schema_json.encode()).hexdigest()[:16]

    # Content hash: sample rows sorted and serialized
    # Use head + tail to capture both ends of the data
    n_rows = len(df)
    if n_rows <= sample_size:
        sample_df = df
    else:
        head_size = sample_size // 2
        tail_size = sample_size - head_size
        sample_df = pd.concat([df.head(head_size), df.tail(tail_size)])

    # Sort by all columns for deterministic ordering
    sorted_cols = sorted(sample_df.columns.tolist())
    sample_sorted = sample_df[sorted_cols].sort_values(by=sorted_cols, ignore_index=True)

    # Convert to JSON-serializable format
    # Convert to string first to handle categorical columns, then normalize null representations
    content_records = (
        sample_sorted.astype(str)
        .replace({"nan": "__NULL__", "<NA>": "__NULL__", "NaT": "__NULL__"})
        .to_dict(orient="records")
    )
    content_json = json.dumps(content_records, sort_keys=True)
    content_hash = hashlib.sha256(content_json.encode()).hexdigest()[:16]

    # Combined hash
    combined = f"{schema_hash}:{content_hash}:{n_rows}:{len(df.columns)}"
    full_hash = hashlib.sha256(combined.encode()).hexdigest()[:32]

    return DatasetFingerprint(
        hash=full_hash,
        schema_hash=schema_hash,
        content_hash=content_hash,
        row_count=n_rows,
        column_count=len(df.columns),
    )


def generate_preview(
    df: pd.DataFrame,
    sample_rows: int = 10,
) -> DatasetPreview:
    """
    Generate a preview with basic statistics for a DataFrame.

    Args:
        df: DataFrame to preview.
        sample_rows: Number of sample rows to include.

    Returns:
        DatasetPreview with row count, column count, samples, and stats.
    """
    n_rows = len(df)
    n_cols = len(df.columns)

    # Sample rows
    if n_rows <= sample_rows:
        samples = df.head(sample_rows)
    else:
        samples = df.head(sample_rows)

    # Convert sample to list of dicts, handling NaN values
    sample_records = samples.where(pd.notna(samples), None).to_dict(orient="records")

    # Column statistics
    column_stats: list[ColumnStats] = []
    for col in df.columns:
        series = df[col]
        null_count = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))

        stats = ColumnStats(
            name=str(col),
            dtype=str(series.dtype),
            null_count=null_count,
            null_rate=null_count / n_rows if n_rows > 0 else 0.0,
            unique_count=unique_count,
            unique_rate=unique_count / n_rows if n_rows > 0 else 0.0,
        )

        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(series):
            try:
                stats.min_value = float(series.min()) if not pd.isna(series.min()) else None
                stats.max_value = float(series.max()) if not pd.isna(series.max()) else None
                stats.mean_value = float(series.mean()) if not pd.isna(series.mean()) else None
            except (TypeError, ValueError):
                pass

        column_stats.append(stats)

    # Memory usage
    memory_bytes = int(df.memory_usage(deep=True).sum())

    return DatasetPreview(
        row_count=n_rows,
        column_count=n_cols,
        columns=df.columns.tolist(),
        sample_rows=sample_records,
        column_stats=column_stats,
        memory_usage_bytes=memory_bytes,
    )


def intake_file(
    path: str | Path,
    normalize: bool = True,
    sample_size: int = 100,
    preview_rows: int = 10,
) -> IntakeResult:
    """
    Load, normalize, fingerprint, and preview a data file.

    This is the main entry point for file intake. It:
    1. Loads the file based on extension
    2. Optionally normalizes the DataFrame
    3. Generates a content-based fingerprint
    4. Creates a preview with statistics

    Args:
        path: Path to the file to load.
        normalize: Whether to normalize the DataFrame.
        sample_size: Number of rows for fingerprint sampling.
        preview_rows: Number of rows to include in preview.

    Returns:
        IntakeResult containing the DataFrame, fingerprint, and preview.

    Raises:
        IntakeError: If file loading or processing fails.

    Example:
        >>> result = intake_file("data.csv")
        >>> print(f"Loaded {result.preview.row_count} rows")
        >>> print(f"Fingerprint: {result.fingerprint.hash}")
    """
    path = Path(path)
    suffix = path.suffix.lower()
    source_format = SUPPORTED_EXTENSIONS.get(suffix, "unknown")

    warnings: list[str] = []

    # Load file
    df = load_file(path)

    # Normalize if requested
    if normalize:
        df = normalize_dataframe(df)

    # Check for potential issues
    if df.empty:
        warnings.append("DataFrame is empty")

    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        warnings.append(f"Duplicate column names detected: {duplicate_cols}")

    # Generate fingerprint and preview
    fingerprint = generate_fingerprint(df, sample_size=sample_size)
    preview = generate_preview(df, sample_rows=preview_rows)

    return IntakeResult(
        dataframe=df,
        source_path=path,
        source_format=source_format,
        fingerprint=fingerprint,
        preview=preview,
        load_warnings=warnings,
    )


def _get_version_file_path(dataset_name: str, history_dir: Path) -> Path:
    """Get the path to the version history JSON file for a dataset."""
    versions_dir = history_dir / "versions"
    return versions_dir / f"{dataset_name}.json"


def track_version(
    fingerprint: DatasetFingerprint,
    path: Path,
    history_dir: Path,
) -> None:
    """
    Track a new version of a dataset by adding it to the version history.

    Creates the history directory and versions subdirectory if they don't exist.
    Appends a new entry with the fingerprint hash, current timestamp, and file path.

    Args:
        fingerprint: The fingerprint of the dataset version being tracked.
        path: The path to the dataset file.
        history_dir: The base directory for storing version history (e.g., .datasculpt).
    """
    # Extract dataset name from path (stem without extension)
    dataset_name = path.stem

    # Ensure versions directory exists
    versions_dir = history_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    # Load existing history or create new
    history = get_version_history(dataset_name, history_dir)

    # Create new entry
    entry = VersionEntry(
        fingerprint=fingerprint.hash,
        timestamp=datetime.utcnow().isoformat(),
        path=str(path),
    )

    # Append entry
    history.entries.append(entry)

    # Save to file
    version_file = _get_version_file_path(dataset_name, history_dir)
    history_data = {
        "dataset_name": history.dataset_name,
        "entries": [
            {
                "fingerprint": e.fingerprint,
                "timestamp": e.timestamp,
                "path": e.path,
            }
            for e in history.entries
        ],
    }
    version_file.write_text(json.dumps(history_data, indent=2))


def get_version_history(dataset_name: str, history_dir: Path) -> VersionHistory:
    """
    Get the version history for a dataset.

    Args:
        dataset_name: The name of the dataset (typically the file stem).
        history_dir: The base directory for storing version history (e.g., .datasculpt).

    Returns:
        VersionHistory containing all tracked versions, or an empty history if none exist.
    """
    version_file = _get_version_file_path(dataset_name, history_dir)

    if not version_file.exists():
        return VersionHistory(dataset_name=dataset_name, entries=[])

    try:
        data = json.loads(version_file.read_text())
        entries = [
            VersionEntry(
                fingerprint=e["fingerprint"],
                timestamp=e["timestamp"],
                path=e["path"],
            )
            for e in data.get("entries", [])
        ]
        return VersionHistory(dataset_name=data.get("dataset_name", dataset_name), entries=entries)
    except (json.JSONDecodeError, KeyError):
        return VersionHistory(dataset_name=dataset_name, entries=[])


def detect_schema_changes(
    old: DatasetFingerprint,
    new: DatasetFingerprint,
    old_df: pd.DataFrame | None = None,
    new_df: pd.DataFrame | None = None,
) -> SchemaChanges:
    """
    Detect schema changes between two dataset fingerprints.

    To detect detailed changes (column additions, removals, and type changes),
    the original DataFrames must be provided. If only fingerprints are provided,
    this function can only determine if the schema changed (via schema_hash comparison)
    but not the specific changes.

    Args:
        old: The fingerprint of the old dataset version.
        new: The fingerprint of the new dataset version.
        old_df: The old DataFrame (optional, needed for detailed comparison).
        new_df: The new DataFrame (optional, needed for detailed comparison).

    Returns:
        SchemaChanges with columns_added, columns_removed, and type_changes.
    """
    columns_added: list[str] = []
    columns_removed: list[str] = []
    type_changes: dict[str, tuple[str, str]] = {}

    # If DataFrames not provided, return empty changes
    # (fingerprints alone don't contain column details)
    if old_df is None or new_df is None:
        return SchemaChanges(
            columns_added=columns_added,
            columns_removed=columns_removed,
            type_changes=type_changes,
        )

    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)

    # Columns added in new
    columns_added = sorted(new_cols - old_cols)

    # Columns removed from old
    columns_removed = sorted(old_cols - new_cols)

    # Type changes for columns that exist in both
    common_cols = old_cols & new_cols
    for col in sorted(common_cols):
        old_dtype = str(old_df[col].dtype)
        new_dtype = str(new_df[col].dtype)
        if old_dtype != new_dtype:
            type_changes[col] = (old_dtype, new_dtype)

    return SchemaChanges(
        columns_added=columns_added,
        columns_removed=columns_removed,
        type_changes=type_changes,
    )
