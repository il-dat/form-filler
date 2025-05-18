#!/usr/bin/env python3
"""
Utils package for Vietnamese Document Form Filler.

Contains utility functions for the application.
"""

from form_filler.utils.file_utils import (
    ensure_directory_exists,
    list_files_by_extension,
    validate_file_exists,
)
from form_filler.utils.logging_utils import get_logger, setup_logging
from form_filler.utils.telemetry_blocker import block_telemetry, restore_original_functions

__all__ = [
    "block_telemetry",
    "ensure_directory_exists",
    "get_logger",
    "list_files_by_extension",
    "restore_original_functions",
    "setup_logging",
    "validate_file_exists",
]
