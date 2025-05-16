#!/usr/bin/env python3
"""
File utilities for Vietnamese Document Form Filler.

Handles file operations and validations.
"""

import logging
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory_path: Path to the directory

    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(file_path: str, file_type: str | None = None) -> tuple[bool, str]:
    """
    Validate that a file exists and optionally check its type.

    Args:
        file_path: Path to the file
        file_type: Optional extension to check (e.g., '.pdf', '.docx')

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        return False, f"File does not exist: {file_path}"

    # Check if it's a file (not a directory)
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"

    # Check file type if specified
    if file_type and path.suffix.lower() != file_type.lower():
        return False, f"File is not a {file_type} file: {file_path}"

    return True, ""


def list_files_by_extension(directory_path: str, extension: str) -> list[Path]:
    """
    List all files with a specific extension in a directory.

    Args:
        directory_path: Path to the directory
        extension: File extension to filter by (e.g., '.pdf', '.docx')

    Returns:
        List of Path objects for matching files
    """
    if not extension.startswith("."):
        extension = "." + extension

    path = Path(directory_path)
    if not path.exists() or not path.is_dir():
        logger.warning(f"Directory does not exist or is not a directory: {directory_path}")
        return []

    return sorted(path.glob(f"*{extension}"))
