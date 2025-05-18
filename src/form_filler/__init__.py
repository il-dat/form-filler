#!/usr/bin/env python3
"""
Vietnamese Document Form Filler CLI.

A CrewAI-based multi-agent system for processing Vietnamese documents and filling English DOCX forms.
"""

import logging
import os
import sys

# Set environment variables to disable telemetry and tracking
# This is a simpler approach than patching network libraries
os.environ["LITELLM_TELEMETRY"] = "false"
os.environ["LITELLM_MODEL_COST_MAP_URL"] = ""  # Prevent URL fetch at import time
os.environ["CREWAI_DO_NOT_TRACK"] = "true"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACKING"] = "false"

# Add debug logging for telemetry
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

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
