#!/usr/bin/env python3
"""
Logging utilities for Vietnamese Document Form Filler.

Handles logging configuration and formatting.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """

    Set up logging configuration.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to

    Returns:
        A configured logger instance
    """
    # Set up logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure handlers
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)

    # Apply configuration
    logging.basicConfig(level=numeric_level, format=log_format, handlers=handlers)

    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.debug(f"Logging initialized at level {log_level}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """

    Get a logger for a specific module/component.

    Args:
        name: The name of the module/component

    Returns:
        A properly configured logger
    """
    return logging.getLogger(name)
