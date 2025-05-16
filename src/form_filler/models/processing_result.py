#!/usr/bin/env python3
"""
Models for Vietnamese Document Form Filler.

Contains data classes for processing results.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ProcessingResult:
    """Data structure for processing results."""

    success: bool
    data: Any
    error: str | None = None
    metadata: dict[str, Any] | None = None
