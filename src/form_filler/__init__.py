#!/usr/bin/env python3
"""
Vietnamese Document Form Filler CLI.

A CrewAI-based multi-agent system for processing Vietnamese documents and filling English DOCX forms.
"""

import logging
import os
import sys

# Set early environment variables for LiteLLM to prevent network requests
# This needs to happen before any imports
os.environ["LITELLM_TELEMETRY"] = "false"
# Set an empty model cost map URL to prevent the request during import
os.environ["LITELLM_MODEL_COST_MAP_URL"] = ""

# Add debug logging for telemetry
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Disable tracking and telemetry for other libraries
os.environ["CREWAI_DO_NOT_TRACK"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACKING"] = "false"

# Import telemetry blocker and apply it for all commands
try:
    import sys

    from form_filler.utils.telemetry_blocker import block_telemetry

    # List of non-functional commands and flags
    non_functional_args = [
        "--help",
        "-h",
        "--version",
        "-v",
        "version",
        "check-ollama",
        "list-providers",
    ]

    # Check if this is a help command
    is_help_command = any(arg in sys.argv for arg in non_functional_args)

    # Always block telemetry, but let the blocker know if this is a help command
    # This ensures the LiteLLM patch returns an empty cost map for help commands
    block_telemetry(is_help_command=is_help_command)

    # Set litellm-specific environment variables to prevent network requests
    # This helps prevent the model_cost_map_url fetch when a user runs --help
    os.environ["LITELLM_TELEMETRY"] = "false"
except Exception as e:
    logging.warning(f"Failed to block telemetry: {e}")

# Import after setting environment variables
# ruff: noqa: E402
from form_filler.agents import (
    create_document_collector_agent,
    create_form_analyst_agent,
    create_form_filler_agent,
    create_translator_agent,
)
from form_filler.cli import cli, main
from form_filler.crew import DocumentProcessingCrew
from form_filler.models import ProcessingResult
from form_filler.tools import (
    DocumentExtractionTool,
    FormAnalysisTool,
    FormFillingTool,
    TranslationTool,
)
from form_filler.utils import get_logger, setup_logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Version information
try:
    from importlib.metadata import version

    __version__ = version("form-filler")
except ImportError:
    __version__ = "unknown"

__author__ = "Dat Nguyen"
__email__ = "datnguyen.it09@gmail.com"

# Main exports
__all__ = [
    "DocumentExtractionTool",
    "DocumentProcessingCrew",
    "FormAnalysisTool",
    "FormFillingTool",
    "ProcessingResult",
    "TranslationTool",
    "__author__",
    "__email__",
    "__version__",
    "cli",
    "create_document_collector_agent",
    "create_form_analyst_agent",
    "create_form_filler_agent",
    "create_translator_agent",
    "get_logger",
    "main",
    "setup_logging",
]
