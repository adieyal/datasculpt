"""Main entry point for Datasculpt Inspector (PyScript).

This module initializes the application, sets up event handlers,
and orchestrates the user workflow.
"""


from typing import Any

# Global state
_current_result = None
_current_answers: dict[str, Any] = {}


async def on_file_selected(event) -> None:
    """Handle file selection from the file input.

    Args:
        event: JavaScript change event.
    """
    import asyncio

    global _current_result, _current_answers

    files = event.target.files
    if files.length == 0:
        return

    file = files.item(0)
    filename = file.name

    # Quick check for supported file type
    supported_extensions = (".csv", ".xlsx", ".xls")
    if not filename.lower().endswith(supported_extensions):
        from js import alert
        alert(f"Unsupported file type: {filename}")
        return

    # Import UI functions first (lightweight)
    from ui_renderer import (
        hide_processing,
        render_results,
        reset_ui,
        set_text,
        show_processing,
    )

    # Show processing IMMEDIATELY
    show_processing()
    set_text("processing-status", f"Loading {filename}...")

    # Yield to event loop so UI updates before heavy processing
    await asyncio.sleep(0)

    # Reset state
    _current_answers = {}

    try:
        # Now import heavy modules
        from file_handler import handle_file_upload
        from inference import get_result_summary, run_inference

        # Load file
        df, filename = await handle_file_upload(file)

        set_text("processing-status", f"Analyzing {len(df):,} rows...")
        await asyncio.sleep(0)  # Update UI again

        # Run inference
        _current_result = run_inference(df, filename, interactive=True)

        # Get summary for rendering
        summary = get_result_summary(_current_result)

        # Render results
        hide_processing()
        render_results(summary)

    except Exception as e:
        from js import alert, console
        console.error(f"Error processing file: {e}")
        alert(f"Error processing file: {str(e)}")
        reset_ui()


async def on_drag_over(event) -> None:
    """Handle drag over event on upload zone.

    Args:
        event: JavaScript dragover event.
    """
    event.preventDefault()
    from ui_renderer import add_class
    add_class("upload-section", "is-dragover")


async def on_drag_leave(event) -> None:
    """Handle drag leave event on upload zone.

    Args:
        event: JavaScript dragleave event.
    """
    event.preventDefault()
    from ui_renderer import remove_class
    remove_class("upload-section", "is-dragover")


async def on_drop(event) -> None:
    """Handle file drop event on upload zone.

    Args:
        event: JavaScript drop event.
    """
    event.preventDefault()
    from ui_renderer import remove_class
    remove_class("upload-section", "is-dragover")

    # Get dropped files
    files = event.dataTransfer.files
    if files.length > 0:
        # Create a synthetic event with the files
        class FileEvent:
            def __init__(self, files):
                self.target = type("Target", (), {"files": files})()

        await on_file_selected(FileEvent(files))


async def on_new_file_click(event) -> None:
    """Handle click on 'Analyze Another File' button.

    Args:
        event: JavaScript click event.
    """
    global _current_result, _current_answers
    _current_result = None
    _current_answers = {}

    from ui_renderer import reset_ui
    reset_ui()

    # Clear file input
    from js import document
    file_input = document.getElementById("file-input")
    file_input.value = ""


async def on_choice_click(event) -> None:
    """Handle click on a question choice button.

    Args:
        event: JavaScript click event.
    """
    from js import document

    global _current_answers

    target = event.target
    if not target.classList.contains("choice-button"):
        return

    question_id = target.dataset.question
    value = target.dataset.value

    # Try to parse value if it looks like JSON
    import json
    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed_value = value

    # Store answer
    _current_answers[question_id] = parsed_value

    # Update UI - deselect siblings, select this one
    parent = target.parentElement
    for btn in parent.children:
        btn.classList.remove("is-selected")
    target.classList.add("is-selected")


async def on_apply_answers_click(event) -> None:
    """Handle click on 'Apply Answers' button.

    Args:
        event: JavaScript click event.
    """
    global _current_result, _current_answers

    if _current_result is None:
        return

    from ui_renderer import (
        render_results,
        set_text,
        show_processing,
        hide_processing,
    )
    from inference import get_result_summary, rerun_with_answers

    show_processing()
    set_text("processing-status", "Applying answers...")

    try:
        # Re-run inference with answers
        _current_result = rerun_with_answers(_current_result, _current_answers)
        _current_answers = {}

        # Get summary for rendering
        summary = get_result_summary(_current_result)

        # Render results
        hide_processing()
        render_results(summary)

    except Exception as e:
        from js import alert, console
        console.error(f"Error applying answers: {e}")
        alert(f"Error: {str(e)}")
        hide_processing()


async def on_export_click(event) -> None:
    """Handle click on 'Download JSON' button.

    Args:
        event: JavaScript click event.
    """
    global _current_result

    if _current_result is None:
        return

    from export import download_json, generate_export_filename
    from inference import get_export_data

    # Get export data
    data = get_export_data(_current_result)

    # Generate filename
    filename = generate_export_filename(_current_result.proposal.dataset_name)

    # Download
    download_json(data, filename)


async def on_role_change(event) -> None:
    """Handle change on role dropdown.

    Args:
        event: JavaScript change event.
    """
    # Store the role override for re-inference
    global _current_answers

    target = event.target

    # Only handle role dropdowns (they have data-column attribute)
    if not hasattr(target, 'dataset') or not hasattr(target.dataset, 'column'):
        return

    column_name = target.dataset.column
    if not column_name:
        return

    new_role = target.value

    # Store as a special answer format
    _current_answers[f"role_{column_name}"] = new_role


def setup_event_handlers() -> None:
    """Set up all event handlers for the UI."""
    from js import document
    from pyodide.ffi import create_proxy

    # File input
    file_input = document.getElementById("file-input")
    file_input.addEventListener("change", create_proxy(on_file_selected))

    # Drag and drop
    upload_zone = document.getElementById("upload-section")
    upload_zone.addEventListener("dragover", create_proxy(on_drag_over))
    upload_zone.addEventListener("dragleave", create_proxy(on_drag_leave))
    upload_zone.addEventListener("drop", create_proxy(on_drop))

    # New file button
    new_file_btn = document.getElementById("new-file-btn")
    new_file_btn.addEventListener("click", create_proxy(on_new_file_click))

    # Apply answers button
    apply_btn = document.getElementById("apply-answers-btn")
    apply_btn.addEventListener("click", create_proxy(on_apply_answers_click))

    # Export button
    export_btn = document.getElementById("export-btn")
    export_btn.addEventListener("click", create_proxy(on_export_click))

    # Choice buttons (using event delegation)
    document.addEventListener("click", create_proxy(on_choice_click))

    # Role dropdowns (using event delegation)
    document.addEventListener("change", create_proxy(on_role_change))


def initialize() -> None:
    """Initialize the Datasculpt Inspector application."""
    from ui_renderer import hide_loading, update_loading_status

    update_loading_status("Setting up event handlers...")

    # Set up event handlers
    setup_event_handlers()

    update_loading_status("Ready!")

    # Hide loading overlay
    hide_loading()


# Entry point - called when PyScript loads
initialize()
