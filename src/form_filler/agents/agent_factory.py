#!/usr/bin/env python3
"""
Agent Factory for Vietnamese Document Form Filler.

Creates CrewAI agents with appropriate roles and tools.
"""

from crewai import Agent

from form_filler.tools import (
    DocumentExtractionTool,
    FormAnalysisTool,
    FormFillingTool,
    TranslationTool,
)


def create_document_collector_agent(
    extraction_method: str = "traditional",
    vision_model: str = "llava:7b",
    openai_api_key: str = None,
    openai_model: str = "gpt-4-vision-preview",
) -> Agent:
    """Create the document collection agent.

    Args:
        extraction_method: Method to use for text extraction (traditional, ai, openai)
        vision_model: Vision model to use for AI text extraction
        openai_api_key: OpenAI API key for OpenAI extraction method
        openai_model: OpenAI model for OpenAI extraction method

    Returns:
        A CrewAI Agent configured for document collection
    """
    return Agent(
        role="Document Text Extractor",
        goal="Extract text content from Vietnamese documents (PDFs and images) with high accuracy",
        backstory="""You are a specialized document processing expert with advanced capabilities in text extraction.
        You can handle traditional OCR methods, cutting-edge AI vision models, and cloud-based AI services
        to extract text from various document formats. Your expertise includes processing Vietnamese documents
        with proper diacritics and special characters.""",
        tools=[
            DocumentExtractionTool(
                extraction_method=extraction_method,
                vision_model=vision_model,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
        ],
        verbose=True,
        allow_delegation=False,
    )


def create_translator_agent(model: str = "llama3.2:3b") -> Agent:
    """Create the translation agent.

    Args:
        model: Ollama model to use for translation

    Returns:
        A CrewAI Agent configured for text translation
    """
    return Agent(
        role="Vietnamese to English Translator",
        goal="Provide accurate and contextually appropriate translations from Vietnamese to English",
        backstory="""You are a professional translator with deep expertise in Vietnamese and English languages.
        You specialize in translating official documents, forms, and business communications while preserving
        the original meaning and maintaining formal language appropriate for document processing.""",
        tools=[TranslationTool(model=model)],
        verbose=True,
        allow_delegation=False,
    )


def create_form_analyst_agent() -> Agent:
    """Create the form analysis agent.

    Returns:
        A CrewAI Agent configured for form analysis
    """
    return Agent(
        role="Document Form Analyst",
        goal="Analyze DOCX forms and identify all fillable fields and their purposes",
        backstory="""You are an expert in document analysis and form processing. You can quickly identify
        form fields, understand their context and purpose, and determine the most appropriate content
        to fill each field based on available translated information.""",
        tools=[FormAnalysisTool()],
        verbose=True,
        allow_delegation=False,
    )


def create_form_filler_agent(model: str = "llama3.2:3b") -> Agent:
    """Create the form filling agent.

    Args:
        model: Ollama model to use for form filling

    Returns:
        A CrewAI Agent configured for form filling
    """
    return Agent(
        role="Form Completion Specialist",
        goal="Fill DOCX forms with translated content using intelligent field mapping",
        backstory="""You are a form completion specialist with the ability to understand document context
        and intelligently map translated content to appropriate form fields. You ensure accuracy and
        maintain proper formatting while filling forms.""",
        tools=[FormFillingTool(model=model)],
        verbose=True,
        allow_delegation=False,
    )
