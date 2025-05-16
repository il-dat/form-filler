#!/usr/bin/env python3
"""
Vietnamese Document Form Filler CLI.

A CrewAI-based multi-agent system for processing Vietnamese documents and filling English DOCX forms.
"""

import logging

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
