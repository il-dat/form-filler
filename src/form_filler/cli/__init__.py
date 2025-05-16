#!/usr/bin/env python3
"""
CLI package for Vietnamese Document Form Filler.

Contains command-line interface implementation.
"""

from form_filler.cli.cli_commands import cli
from form_filler.cli.entry_point import main

__all__ = ["cli", "main"]
