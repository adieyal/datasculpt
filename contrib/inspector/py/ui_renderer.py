"""UI rendering for Datasculpt Inspector.

This module provides functions to render inference results
to the HTML DOM using PyScript.
"""


from typing import Any


def get_element(element_id: str):
    """Get a DOM element by ID.

    Args:
        element_id: HTML element ID.

    Returns:
        DOM element.
    """
    from js import document
    return document.getElementById(element_id)


def set_text(element_id: str, text: str) -> None:
    """Set the text content of an element.

    Args:
        element_id: HTML element ID.
        text: Text to set.
    """
    element = get_element(element_id)
    if element:
        element.textContent = str(text)


def set_html(element_id: str, html: str) -> None:
    """Set the innerHTML of an element.

    Args:
        element_id: HTML element ID.
        html: HTML content to set.
    """
    element = get_element(element_id)
    if element:
        element.innerHTML = html


def show_element(element_id: str) -> None:
    """Show a hidden element.

    Args:
        element_id: HTML element ID.
    """
    element = get_element(element_id)
    if element:
        element.classList.remove("is-hidden")


def hide_element(element_id: str) -> None:
    """Hide an element.

    Args:
        element_id: HTML element ID.
    """
    element = get_element(element_id)
    if element:
        element.classList.add("is-hidden")


def add_class(element_id: str, class_name: str) -> None:
    """Add a CSS class to an element.

    Args:
        element_id: HTML element ID.
        class_name: CSS class to add.
    """
    element = get_element(element_id)
    if element:
        element.classList.add(class_name)


def remove_class(element_id: str, class_name: str) -> None:
    """Remove a CSS class from an element.

    Args:
        element_id: HTML element ID.
        class_name: CSS class to remove.
    """
    element = get_element(element_id)
    if element:
        element.classList.remove(class_name)


# Shape rendering

def render_shape_cards(shapes: list[dict[str, Any]]) -> str:
    """Render shape hypothesis cards.

    Args:
        shapes: List of shape hypothesis info.

    Returns:
        HTML string for shape cards.
    """
    cards_html = []

    for shape in shapes:
        winner_class = "is-winner" if shape["is_winner"] else ""
        score_pct = int(shape["score"] * 100)

        reasons_html = ""
        if shape["reasons"]:
            reasons_html = "<ul class='mt-2' style='font-size: 0.8rem; color: #7a7a7a;'>"
            for reason in shape["reasons"][:3]:
                reasons_html += f"<li>{_escape_html(reason)}</li>"
            reasons_html += "</ul>"

        card_html = f"""
        <div class="column is-one-third">
            <div class="shape-card {winner_class}">
                <p class="shape-name">{_escape_html(shape['label'])}</p>
                <p class="shape-score">{score_pct}%</p>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width: {score_pct}%;"></div>
                </div>
                {reasons_html}
            </div>
        </div>
        """
        cards_html.append(card_html)

    return "".join(cards_html)


# Grain rendering

def render_grain_info(grain: dict[str, Any]) -> str:
    """Render grain detection information.

    Args:
        grain: Grain inference info.

    Returns:
        HTML string for grain display.
    """
    columns_html = ""
    for col in grain["key_columns"]:
        columns_html += f'<span class="grain-column">{_escape_html(col)}</span>'

    confidence_pct = int(grain["confidence"] * 100)
    uniqueness_pct = f"{grain['uniqueness_ratio']:.1%}"

    confidence_class = "is-high" if confidence_pct >= 80 else "is-medium" if confidence_pct >= 50 else "is-low"

    evidence_html = ""
    if grain["evidence"]:
        evidence_html = "<div class='mt-3'><strong>Evidence:</strong><ul>"
        for ev in grain["evidence"][:5]:
            evidence_html += f"<li>{_escape_html(ev)}</li>"
        evidence_html += "</ul></div>"

    return f"""
    <div class="grain-columns">{columns_html}</div>
    <div class="confidence-meter">
        <span>Confidence:</span>
        <div class="confidence-bar">
            <div class="confidence-fill {confidence_class}" style="width: {confidence_pct}%;"></div>
        </div>
        <span class="confidence-value">{confidence_pct}%</span>
    </div>
    <p class="grain-uniqueness mt-2">Uniqueness: {uniqueness_pct}</p>
    {evidence_html}
    """


# Column rendering

def render_column_cards(columns: list[dict[str, Any]]) -> str:
    """Render column cards with role badges.

    Args:
        columns: List of column info.

    Returns:
        HTML string for column cards.
    """
    cards_html = []

    for col in columns:
        role_class = f"role-{col['role']}"
        null_pct = f"{col['null_rate']:.1%}"
        distinct_pct = f"{col['distinct_ratio']:.1%}"

        # Alternatives dropdown
        alternatives_html = ""
        if col["alternatives"]:
            options = "".join(
                f'<option value="{alt["role"]}">{alt["role"]} ({alt["score"]:.2f})</option>'
                for alt in col["alternatives"]
            )
            alternatives_html = f"""
            <div class="alternatives-dropdown">
                <select data-column="{_escape_html(col['name'])}">
                    <option value="{col['role']}" selected>{col['role']} (current)</option>
                    {options}
                </select>
            </div>
            """

        card_html = f"""
        <div class="column is-one-quarter">
            <div class="column-card">
                <p class="column-name">{_escape_html(col['name'])}</p>
                <span class="role-badge {role_class}">{col['role']}</span>
                <p class="column-type">{col['primitive_type']}</p>
                <p class="column-stats">
                    Null: {null_pct} | Distinct: {distinct_pct}
                </p>
                {alternatives_html}
            </div>
        </div>
        """
        cards_html.append(card_html)

    return "".join(cards_html)


# Warnings/diagnostics rendering

def render_diagnostics(warnings: list[str]) -> str:
    """Render diagnostic warnings.

    Args:
        warnings: List of warning messages.

    Returns:
        HTML string for diagnostics.
    """
    if not warnings:
        return '<div class="diagnostic-item is-success"><i class="fas fa-check"></i> No issues detected.</div>'

    items_html = []
    for warning in warnings:
        items_html.append(
            f'<div class="diagnostic-item is-warning"><i class="fas fa-exclamation-triangle"></i> {_escape_html(warning)}</div>'
        )

    return "".join(items_html)


# Questions rendering

def render_questions(questions: list[dict[str, Any]]) -> str:
    """Render question cards for interactive mode.

    Args:
        questions: List of question info.

    Returns:
        HTML string for question cards.
    """
    if not questions:
        return ""

    cards_html = []

    for q in questions:
        choices_html = ""
        for choice in q["choices"]:
            is_default = choice["value"] == q.get("default")
            selected_class = "is-selected" if is_default else ""
            choices_html += f"""
            <button class="choice-button {selected_class}"
                    data-question="{q['id']}"
                    data-value="{_escape_html(str(choice['value']))}">
                {_escape_html(choice.get('label', str(choice['value'])))}
            </button>
            """

        rationale_html = ""
        if q.get("rationale"):
            rationale_html = f'<p class="has-text-grey" style="font-size: 0.85rem; margin-top: 0.5rem;"><i class="fas fa-info-circle"></i> {_escape_html(q["rationale"])}</p>'

        card_html = f"""
        <div class="question-card" data-question-id="{q['id']}">
            <p class="question-text">{_escape_html(q['prompt'])}</p>
            <div class="question-choices">
                {choices_html}
            </div>
            {rationale_html}
        </div>
        """
        cards_html.append(card_html)

    return "".join(cards_html)


# Full results rendering

def render_results(summary: dict[str, Any]) -> None:
    """Render all inference results to the DOM.

    Args:
        summary: Inference result summary from get_result_summary().
    """
    # File info
    set_text("file-name", summary["dataset_name"])
    set_text("row-count", f"{summary['row_count']:,}")
    set_text("col-count", str(summary["column_count"]))

    # Shape hypotheses
    set_html("shape-hypotheses", render_shape_cards(summary["shapes"]))

    # Grain detection
    set_html("grain-result", render_grain_info(summary["grain"]))

    # Column cards
    set_html("column-cards", render_column_cards(summary["columns"]))

    # Diagnostics
    diagnostics_html = render_diagnostics(summary["warnings"])
    set_html("diagnostics-content", diagnostics_html)
    if summary["warnings"]:
        show_element("diagnostics-section")
    else:
        hide_element("diagnostics-section")

    # Questions
    if summary["questions"]:
        set_html("questions-container", render_questions(summary["questions"]))
        show_element("questions-section")
    else:
        hide_element("questions-section")

    # Show results section
    show_element("results-section")


def update_loading_status(message: str) -> None:
    """Update the loading status message.

    Args:
        message: Status message to display.
    """
    set_text("loading-status", message)


def show_loading() -> None:
    """Show the loading overlay."""
    add_class("loading-overlay", "is-active")


def hide_loading() -> None:
    """Hide the loading overlay."""
    remove_class("loading-overlay", "is-active")


def show_processing() -> None:
    """Show the processing indicator."""
    hide_element("upload-section")
    show_element("processing-section")


def hide_processing() -> None:
    """Hide the processing indicator."""
    hide_element("processing-section")


def reset_ui() -> None:
    """Reset the UI to initial state."""
    hide_element("results-section")
    hide_element("processing-section")
    hide_element("questions-section")
    hide_element("diagnostics-section")
    show_element("upload-section")


# Utility functions

def _escape_html(text: str) -> str:
    """Escape HTML special characters.

    Args:
        text: Text to escape.

    Returns:
        Escaped text.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
