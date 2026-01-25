"""Export functionality for browser.

This module provides functions to export inference results as JSON
files that can be downloaded by the user.
"""


import json
from typing import Any


def download_json(data: dict[str, Any], filename: str) -> None:
    """Trigger a JSON file download in the browser.

    Args:
        data: Dictionary to export as JSON.
        filename: Name for the downloaded file.
    """
    from js import Blob, URL, document

    # Convert to JSON string
    json_str = json.dumps(data, indent=2)

    # Create blob
    blob = Blob.new([json_str], {"type": "application/json"})

    # Create download link
    link = document.createElement("a")
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.style.display = "none"

    # Trigger download
    document.body.appendChild(link)
    link.click()

    # Cleanup
    document.body.removeChild(link)
    URL.revokeObjectURL(link.href)


def generate_export_filename(dataset_name: str) -> str:
    """Generate a filename for the export.

    Args:
        dataset_name: Name of the dataset.

    Returns:
        Filename with .json extension.
    """
    # Sanitize dataset name
    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_"
        for c in dataset_name
    )
    return f"{safe_name}_proposal.json"
