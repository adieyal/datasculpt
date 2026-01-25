"""File handling for browser file uploads.

This module provides functions to handle file uploads via the browser
File API and convert them to pandas DataFrames.
"""


from io import BytesIO

import pandas as pd


async def read_file_as_bytes(file) -> bytes:
    """Read a JavaScript File object as bytes.

    Args:
        file: JavaScript File object from file input.

    Returns:
        File contents as bytes.
    """
    from js import Uint8Array
    from pyodide.ffi import to_js

    array_buffer = await file.arrayBuffer()
    uint8_array = Uint8Array.new(array_buffer)
    return bytes(uint8_array)


def bytes_to_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Convert file bytes to a pandas DataFrame.

    Args:
        file_bytes: Raw file bytes.
        filename: Original filename (used to determine format).

    Returns:
        pandas DataFrame.

    Raises:
        ValueError: If file format is not supported.
    """
    buffer = BytesIO(file_bytes)
    filename_lower = filename.lower()

    if filename_lower.endswith(".csv"):
        return pd.read_csv(buffer)
    elif filename_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


async def handle_file_upload(file) -> tuple[pd.DataFrame, str]:
    """Handle a file upload from the browser.

    Args:
        file: JavaScript File object from file input or drag-and-drop.

    Returns:
        Tuple of (DataFrame, filename).
    """
    filename = file.name
    file_bytes = await read_file_as_bytes(file)
    df = bytes_to_dataframe(file_bytes, filename)
    return df, filename


def get_file_info(file) -> dict:
    """Get information about an uploaded file.

    Args:
        file: JavaScript File object.

    Returns:
        Dictionary with file info.
    """
    return {
        "name": file.name,
        "size": file.size,
        "type": file.type,
    }


def is_supported_file(filename: str) -> bool:
    """Check if a file type is supported.

    Args:
        filename: Name of the file.

    Returns:
        True if supported, False otherwise.
    """
    supported_extensions = (".csv", ".xlsx", ".xls")
    return filename.lower().endswith(supported_extensions)
