"""Command-line interface for Datasculpt.

Provides commands for dataset inference and preview.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasculpt.core import (
    ColumnSpec,
    DatasetKind,
    InvariantProposal,
    Role,
    ShapeHypothesis,
    detect_shape,
    infer_grain,
)
from datasculpt.core.evidence import extract_dataframe_evidence
from datasculpt.core.roles import assign_roles, update_evidence_with_roles
from datasculpt.intake import (
    IntakeError,
    IntakeResult,
    intake_file,
)
from datasculpt.pipeline import apply_answers, infer
from datasculpt.proposal import ColumnChangeType, ProposalDiff, diff_proposals

# Exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
EXIT_FILE_ERROR = 2
EXIT_AMBIGUOUS = 3  # Used in non-interactive mode when questions would be generated


def _serialize_for_json(obj: Any) -> Any:
    """Convert dataclass instances and enums to JSON-serializable format."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize_for_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _run_inference(path: Path) -> tuple[IntakeResult, InvariantProposal]:
    """Run full inference pipeline on a file.

    Args:
        path: Path to the data file.

    Returns:
        Tuple of (IntakeResult, InvariantProposal).

    Raises:
        IntakeError: If file loading or processing fails.
    """
    # Load and fingerprint the file
    result = intake_file(path)
    df = result.dataframe

    # Extract column evidence
    column_evidence = extract_dataframe_evidence(df)

    # Detect indicator columns first
    has_indicator = False
    for evidence in column_evidence.values():
        from datasculpt.core.roles import score_indicator_name_role

        if score_indicator_name_role(evidence) >= 0.5:
            has_indicator = True
            break

    # Update evidence with role scores
    for _name, evidence in column_evidence.items():
        update_evidence_with_roles(evidence, has_indicator_column=has_indicator)

    # Detect shape
    shape_result = detect_shape(list(column_evidence.values()))

    # Infer grain (shape-aware)
    grain = infer_grain(df, column_evidence, detected_shape=shape_result.selected)

    # Assign roles
    role_assignments = assign_roles(list(column_evidence.values()))

    # Map shape to dataset kind
    shape_to_kind = {
        ShapeHypothesis.LONG_OBSERVATIONS: DatasetKind.OBSERVATIONS,
        ShapeHypothesis.LONG_INDICATORS: DatasetKind.INDICATORS_LONG,
        ShapeHypothesis.WIDE_OBSERVATIONS: DatasetKind.OBSERVATIONS,
        ShapeHypothesis.WIDE_TIME_COLUMNS: DatasetKind.TIMESERIES_WIDE,
        ShapeHypothesis.SERIES_COLUMN: DatasetKind.TIMESERIES_SERIES,
    }

    # Build column specs
    columns: list[ColumnSpec] = []
    for name, evidence in column_evidence.items():
        assignment = role_assignments.get(name)
        role = assignment.role if assignment else Role.METADATA

        columns.append(
            ColumnSpec(
                name=name,
                role=role,
                primitive_type=evidence.primitive_type,
                structural_type=evidence.structural_type,
                notes=evidence.notes.copy(),
            )
        )

    # Build warnings
    warnings: list[str] = result.load_warnings.copy()
    if shape_result.is_ambiguous:
        warnings.extend(shape_result.ambiguity_details)
    if grain.confidence < 0.5:
        warnings.append(f"Low grain confidence: {grain.confidence:.2f}")

    # Build proposal
    proposal = InvariantProposal(
        dataset_name=path.stem,
        dataset_kind=shape_to_kind.get(shape_result.selected, DatasetKind.OBSERVATIONS),
        shape_hypothesis=shape_result.selected,
        grain=grain.key_columns,
        columns=columns,
        warnings=warnings,
        decision_record_id=result.fingerprint.hash,
    )

    return result, proposal


def _format_confidence(value: float) -> str:
    """Format confidence value with color indicator."""
    percentage = value * 100
    if value >= 0.8:
        indicator = "[high]"
    elif value >= 0.5:
        indicator = "[medium]"
    else:
        indicator = "[low]"
    return f"{percentage:.1f}% {indicator}"


def _print_infer_summary(result: IntakeResult, proposal: InvariantProposal) -> None:
    """Print inference results in human-readable format."""
    print(f"File: {result.source_path}")
    print(f"Format: {result.source_format}")
    print(f"Fingerprint: {result.fingerprint.hash}")
    print()

    print("Dataset Summary")
    print("-" * 40)
    print(f"  Rows: {result.preview.row_count}")
    print(f"  Columns: {result.preview.column_count}")
    print()

    print("Inference Results")
    print("-" * 40)
    print(f"  Dataset Kind: {proposal.dataset_kind.value}")
    print(f"  Shape: {proposal.shape_hypothesis.value}")
    print(f"  Grain: {', '.join(proposal.grain) if proposal.grain else '(none detected)'}")
    print()

    print("Column Roles")
    print("-" * 40)
    for col in proposal.columns:
        notes_str = f" - {'; '.join(col.notes)}" if col.notes else ""
        print(f"  {col.name}: {col.role.value} ({col.primitive_type.value}){notes_str}")
    print()

    if proposal.warnings:
        print("Warnings")
        print("-" * 40)
        for warning in proposal.warnings:
            print(f"  ! {warning}")
        print()


def _print_preview_summary(result: IntakeResult) -> None:
    """Print dataset preview in human-readable format."""
    preview = result.preview

    print(f"File: {result.source_path}")
    print(f"Format: {result.source_format}")
    print()

    print("Dataset Summary")
    print("-" * 40)
    print(f"  Rows: {preview.row_count}")
    print(f"  Columns: {preview.column_count}")
    print(f"  Memory: {preview.memory_usage_bytes / 1024:.1f} KB")
    print()

    print("Columns")
    print("-" * 40)
    for stats in preview.column_stats:
        null_pct = stats.null_rate * 100
        unique_pct = stats.unique_rate * 100
        print(f"  {stats.name}")
        print(f"    Type: {stats.dtype}")
        print(f"    Nulls: {stats.null_count} ({null_pct:.1f}%)")
        print(f"    Unique: {stats.unique_count} ({unique_pct:.1f}%)")
        if stats.min_value is not None:
            print(f"    Range: {stats.min_value} to {stats.max_value}")
        if stats.mean_value is not None:
            print(f"    Mean: {stats.mean_value:.2f}")
    print()

    print("Sample Rows")
    print("-" * 40)
    if preview.sample_rows:
        # Print header
        columns = preview.columns
        col_widths = {col: max(len(col), 10) for col in columns}

        # Truncate column names for display
        header = " | ".join(col[:col_widths[col]].ljust(col_widths[col]) for col in columns)
        print(f"  {header}")
        print("  " + "-" * len(header))

        # Print rows (max 5 for readability)
        for row in preview.sample_rows[:5]:
            values = []
            for col in columns:
                val = row.get(col)
                val_str = str(val) if val is not None else ""
                val_str = val_str[:col_widths[col]]
                values.append(val_str.ljust(col_widths[col]))
            print(f"  {' | '.join(values)}")

        if len(preview.sample_rows) > 5:
            print(f"  ... ({len(preview.sample_rows) - 5} more rows)")
    else:
        print("  (no data)")
    print()

    if result.load_warnings:
        print("Warnings")
        print("-" * 40)
        for warning in result.load_warnings:
            print(f"  ! {warning}")
        print()


def _prompt_for_answers(questions: list) -> dict[str, Any]:
    """Prompt user for answers to inference questions.

    Args:
        questions: List of Question objects to prompt for.

    Returns:
        Dictionary mapping question IDs to user answers.
    """
    answers: dict[str, Any] = {}

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}")
        print("-" * 40)
        print(f"{question.prompt}")

        if question.rationale:
            print(f"  (Rationale: {question.rationale})")

        # Display choices
        if question.choices:
            print("\nOptions:")
            for j, choice in enumerate(question.choices, 1):
                label = choice.get("label", choice.get("value", f"Option {j}"))
                desc = choice.get("description", "")
                if desc:
                    print(f"  {j}. {label} - {desc}")
                else:
                    print(f"  {j}. {label}")

            # Show default if present
            default_idx = None
            if question.default:
                for j, choice in enumerate(question.choices, 1):
                    if choice.get("value") == question.default:
                        default_idx = j
                        break

            prompt_text = f"Enter choice (1-{len(question.choices)})"
            if default_idx:
                prompt_text += f" [default: {default_idx}]"
            prompt_text += ": "

            while True:
                user_input = input(prompt_text).strip()

                if not user_input and default_idx:
                    answers[question.id] = question.default
                    break

                try:
                    choice_num = int(user_input)
                    if 1 <= choice_num <= len(question.choices):
                        answers[question.id] = question.choices[choice_num - 1].get("value")
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(question.choices)}")
                except ValueError:
                    print("Please enter a valid number")
        else:
            # Free text input
            default_text = f" [default: {question.default}]" if question.default else ""
            user_input = input(f"Your answer{default_text}: ").strip()
            answers[question.id] = user_input if user_input else question.default

    return answers


def _format_ambiguities(questions: list) -> list[str]:
    """Format questions as ambiguity descriptions.

    Args:
        questions: List of Question objects.

    Returns:
        List of formatted ambiguity descriptions.
    """
    ambiguities: list[str] = []
    for question in questions:
        desc = question.prompt
        if question.rationale:
            desc += f" ({question.rationale})"
        ambiguities.append(desc)
    return ambiguities


def cmd_infer(args: argparse.Namespace) -> int:
    """Execute the infer command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    path = Path(args.file)

    # Check for mutually exclusive flags
    non_interactive = getattr(args, "non_interactive", False)
    interactive = getattr(args, "interactive", False)

    if non_interactive and interactive:
        print("Error: --non-interactive and --interactive are mutually exclusive", file=sys.stderr)
        return EXIT_ERROR

    try:
        # Use pipeline.infer for interactive/non-interactive modes
        if interactive or non_interactive:
            inference_result = infer(path, interactive=True)

            # Handle non-interactive mode
            if non_interactive and inference_result.pending_questions:
                ambiguities = _format_ambiguities(inference_result.pending_questions)
                print("Error: Ambiguities detected in non-interactive mode", file=sys.stderr)
                print("\nAmbiguities:", file=sys.stderr)
                for amb in ambiguities:
                    print(f"  - {amb}", file=sys.stderr)
                return EXIT_AMBIGUOUS

            # Handle interactive mode
            if interactive and inference_result.pending_questions:
                print("Inference generated questions that require clarification.\n")

                # Loop until no more questions
                while inference_result.pending_questions:
                    answers = _prompt_for_answers(inference_result.pending_questions)
                    inference_result = apply_answers(inference_result, answers)

                print("\nAll questions answered. Re-running inference with answers applied.\n")

            # Use the inference result
            result = intake_file(path)
            proposal = inference_result.proposal

            if args.json:
                output = {
                    "fingerprint": _serialize_for_json(result.fingerprint),
                    "proposal": _serialize_for_json(proposal),
                }
                print(json.dumps(output, indent=2))
            else:
                _print_infer_summary(result, proposal)
        else:
            # Default behavior (existing implementation)
            result, proposal = _run_inference(path)

            if args.json:
                output = {
                    "fingerprint": _serialize_for_json(result.fingerprint),
                    "proposal": _serialize_for_json(proposal),
                }
                print(json.dumps(output, indent=2))
            else:
                _print_infer_summary(result, proposal)

    except IntakeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR

    return EXIT_SUCCESS


def cmd_register(args: argparse.Namespace) -> int:
    """Execute the register command (stub).

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    path = Path(args.file)

    try:
        result, proposal = _run_inference(path)
    except IntakeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR

    # Show the proposal
    print("Invariant Registration Proposal")
    print("=" * 40)
    print(f"Dataset Name: {proposal.dataset_name}")
    print(f"Dataset Kind: {proposal.dataset_kind.value}")
    print(f"Shape: {proposal.shape_hypothesis.value}")
    print(f"Grain: {', '.join(proposal.grain) if proposal.grain else '(none detected)'}")
    print()

    print("Columns:")
    for col in proposal.columns:
        print(f"  {col.name}: {col.role.value} ({col.primitive_type.value})")
    print()

    if proposal.warnings:
        print("Warnings:")
        for warning in proposal.warnings:
            print(f"  ! {warning}")
        print()

    print("-" * 40)
    print("Registration to Invariant not yet implemented")
    print("-" * 40)

    return EXIT_SUCCESS


def _print_diff_summary(diff: ProposalDiff, file1: Path, file2: Path) -> None:
    """Print proposal diff in human-readable format."""
    print(f"Comparing: {file1.name} vs {file2.name}")
    print("=" * 50)
    print()

    if not diff.has_changes:
        print("No differences found between proposals.")
        return

    if diff.shape_changed:
        print("Shape Changes")
        print("-" * 40)
        # The summary already contains shape info, extract from there
        for line in diff.summary.split("\n"):
            if line.startswith("Shape:") or line.startswith("Kind:"):
                print(f"  {line}")
        print()

    if diff.grain_changed:
        print("Grain Changes")
        print("-" * 40)
        for line in diff.summary.split("\n"):
            if line.startswith("Grain:"):
                print(f"  {line}")
        print()

    if diff.column_changes:
        print("Column Changes")
        print("-" * 40)
        for change in diff.column_changes:
            if change.change_type == ColumnChangeType.ADDED:
                print(f"  + {change}")
            elif change.change_type == ColumnChangeType.REMOVED:
                print(f"  - {change}")
            else:
                print(f"  ~ {change}")
        print()

    added_warnings, removed_warnings = diff.warning_changes
    if added_warnings:
        print("New Warnings")
        print("-" * 40)
        for warning in added_warnings:
            print(f"  + {warning}")
        print()

    if removed_warnings:
        print("Resolved Warnings")
        print("-" * 40)
        for warning in removed_warnings:
            print(f"  - {warning}")
        print()


def cmd_diff(args: argparse.Namespace) -> int:
    """Execute the diff command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    path1 = Path(args.file1)
    path2 = Path(args.file2)

    try:
        # Run inference on both files
        _, proposal1 = _run_inference(path1)
        _, proposal2 = _run_inference(path2)

        # Compute diff
        diff = diff_proposals(proposal1, proposal2)

        if args.json:
            # Serialize ProposalDiff to JSON
            output = {
                "shape_changed": diff.shape_changed,
                "grain_changed": diff.grain_changed,
                "column_changes": [
                    {
                        "column_name": change.column_name,
                        "change_type": change.change_type.value,
                        "old_role": change.old_role.value if change.old_role else None,
                        "new_role": change.new_role.value if change.new_role else None,
                    }
                    for change in diff.column_changes
                ],
                "warning_changes": {
                    "added": diff.warning_changes[0],
                    "removed": diff.warning_changes[1],
                },
                "has_changes": diff.has_changes,
                "summary": diff.summary,
            }
            print(json.dumps(output, indent=2))
        else:
            _print_diff_summary(diff, path1, path2)

    except IntakeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR

    return EXIT_SUCCESS


def cmd_preview(args: argparse.Namespace) -> int:
    """Execute the preview command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code.
    """
    path = Path(args.file)

    try:
        result = intake_file(path)
    except IntakeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_FILE_ERROR
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_ERROR

    if args.json:
        output = {
            "source_path": str(result.source_path),
            "source_format": result.source_format,
            "fingerprint": _serialize_for_json(result.fingerprint),
            "preview": _serialize_for_json(result.preview),
            "warnings": result.load_warnings,
        }
        print(json.dumps(output, indent=2))
    else:
        _print_preview_summary(result)

    return EXIT_SUCCESS


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="datasculpt",
        description="Deterministic dataset shape and semantic inference",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
    )

    # infer command
    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference on a data file",
        description="Analyze a data file and generate an Invariant proposal",
    )
    infer_parser.add_argument(
        "file",
        help="Path to the data file (CSV, Excel, Parquet, or Stata)",
    )
    infer_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    infer_parser.add_argument(
        "--non-interactive",
        action="store_true",
        dest="non_interactive",
        help="Fail with exit code 3 if ambiguities are detected (for CI/automation)",
    )
    infer_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt user for answers to resolve ambiguities",
    )
    infer_parser.set_defaults(func=cmd_infer)

    # preview command
    preview_parser = subparsers.add_parser(
        "preview",
        help="Show dataset preview with statistics",
        description="Load a data file and display preview information",
    )
    preview_parser.add_argument(
        "file",
        help="Path to the data file (CSV, Excel, Parquet, or Stata)",
    )
    preview_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    preview_parser.set_defaults(func=cmd_preview)

    # register command (stub)
    register_parser = subparsers.add_parser(
        "register",
        help="Register dataset with Invariant (stub)",
        description="Generate and register dataset proposal with Invariant",
    )
    register_parser.add_argument(
        "file",
        help="Path to the data file (CSV, Excel, Parquet, or Stata)",
    )
    register_parser.set_defaults(func=cmd_register)

    # diff command
    diff_parser = subparsers.add_parser(
        "diff",
        help="Compare two dataset versions",
        description="Run inference on two files and display their differences",
    )
    diff_parser.add_argument(
        "file1",
        help="Path to the first data file (CSV, Excel, Parquet, or Stata)",
    )
    diff_parser.add_argument(
        "file2",
        help="Path to the second data file (CSV, Excel, Parquet, or Stata)",
    )
    diff_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    diff_parser.set_defaults(func=cmd_diff)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    # Suppress pandas warning about invalid value in cast for categorical columns
    # This occurs when converting categorical columns to strings (e.g., for fingerprinting)
    # and is a pandas internal issue that doesn't affect data quality
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in cast",
        category=RuntimeWarning,
    )

    parser = create_parser()
    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
