#!/usr/bin/env python3
"""
Document Processing Crew for Vietnamese Document Form Filler.

Main CrewAI crew for processing documents and filling forms.
"""

import json
import logging

from crewai import Crew, Process, Task

from form_filler.agents import (
    create_document_collector_agent,
    create_form_analyst_agent,
    create_form_filler_agent,
    create_translator_agent,
)
from form_filler.models import ProcessingResult

# Setup logging
logger = logging.getLogger(__name__)


class DocumentProcessingCrew:
    """Main CrewAI crew for document processing."""

    def __init__(
        self,
        extraction_method: str = "traditional",
        provider_name: str = "ollama",
        text_model: str = "llama3.2:3b",
        vision_model: str = "llava:7b",
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        """Initialize the document processing crew.

        Args:
            extraction_method: Method to use for extraction (traditional, ai)
            provider_name: AI provider to use (ollama, openai, anthropic, etc.)
            text_model: Model name to use for text-based tasks
            vision_model: Model name to use for vision-based tasks
            api_key: API key for the AI provider (if needed)
            api_base: Base URL for the AI provider API (if needed)
        """
        # Create agents
        self.document_collector = create_document_collector_agent(
            extraction_method=extraction_method,
            provider_name=provider_name,
            model_name=vision_model,
            api_key=api_key,
            api_base=api_base,
        )

        self.translator = create_translator_agent(
            provider_name=provider_name,
            model_name=text_model,
            api_key=api_key,
            api_base=api_base,
        )

        self.form_analyst = create_form_analyst_agent()

        self.form_filler = create_form_filler_agent(
            provider_name=provider_name,
            model_name=text_model,
            api_key=api_key,
            api_base=api_base,
        )

        # Store configuration
        self.extraction_method = extraction_method
        self.provider_name = provider_name
        self.text_model = text_model
        self.vision_model = vision_model
        self.api_key = api_key
        self.api_base = api_base

    def process_document(
        self,
        source_path: str,
        form_path: str,
        output_path: str,
    ) -> ProcessingResult:
        """Process a document through the CrewAI pipeline.

        Args:
            source_path: Path to the source document
            form_path: Path to the form template
            output_path: Path to save the filled form

        Returns:
            ProcessingResult object containing success status and result data
        """
        try:
            # Define tasks
            # Determine extraction method description for task
            extraction_method_desc = "traditional OCR methods"
            if self.extraction_method == "ai":
                extraction_method_desc = "AI vision models"

            extraction_task = Task(
                description=f"""Extract all text content from the Vietnamese document at: {source_path}

                Requirements:
                - Extract all visible text including Vietnamese diacritics
                - Preserve formatting and structure where possible
                - Handle both text-based and image-based content
                - Use {extraction_method_desc}

                Return the complete extracted text.""",
                agent=self.document_collector,
                expected_output="Complete text content extracted from the Vietnamese document",
            )

            translation_task = Task(
                description="""Translate the extracted Vietnamese text to English.

                Requirements:
                - Maintain professional, formal language suitable for documents
                - Preserve all important information (names, dates, addresses, etc.)
                - Ensure accuracy and contextual appropriateness
                - Keep formatting structure where relevant

                Return the complete English translation.""",
                agent=self.translator,
                expected_output="Professional English translation of the Vietnamese text",
                context=[extraction_task],
            )

            form_analysis_task = Task(
                description=f"""Analyze the DOCX form structure at: {form_path}

                Requirements:
                - Identify all fillable fields and placeholders
                - Understand the purpose and context of each field
                - Provide detailed information about form structure

                Return structured information about the form fields.""",
                agent=self.form_analyst,
                expected_output="Detailed analysis of form structure and fillable fields",
            )

            form_filling_task = Task(
                description=f"""Fill the DOCX form with the translated content and save to: {output_path}

                Requirements:
                - Use the form analysis to understand field purposes
                - Intelligently map translated content to appropriate fields
                - Maintain original form formatting
                - Fill all relevant fields with appropriate content

                Return information about the filling process including number of fields filled.""",
                agent=self.form_filler,
                expected_output="Successfully filled form with detailed completion report",
                context=[translation_task, form_analysis_task],
            )

            # Create and execute crew
            crew = Crew(
                agents=[
                    self.document_collector,
                    self.translator,
                    self.form_analyst,
                    self.form_filler,
                ],
                tasks=[extraction_task, translation_task, form_analysis_task, form_filling_task],
                process=Process.sequential,
                verbose=True,
            )

            # Execute the crew
            logger.info("Starting CrewAI document processing pipeline")
            logger.info(f"Source: {source_path}, Form: {form_path}, Output: {output_path}")
            logger.info(f"Extraction method: {self.extraction_method}")
            logger.info(f"Provider: {self.provider_name}")
            logger.info(f"Text model: {self.text_model}")
            logger.info(f"Vision model: {self.vision_model}")

            result = crew.kickoff()

            # Parse the final result
            try:
                if isinstance(result, str):
                    final_result = json.loads(result)
                else:
                    final_result = {"output_path": output_path, "fields_filled": "Unknown"}

                return ProcessingResult(
                    success=True,
                    data=final_result,
                    metadata={
                        "extraction_method": self.extraction_method,
                        "provider_name": self.provider_name,
                        "text_model": self.text_model,
                        "vision_model": self.vision_model
                        if self.extraction_method == "ai"
                        else None,
                    },
                )
            except json.JSONDecodeError:
                # If result is not JSON, consider it successful anyway
                return ProcessingResult(
                    success=True,
                    data={"output_path": output_path, "result": str(result)},
                    metadata={
                        "extraction_method": self.extraction_method,
                        "provider_name": self.provider_name,
                        "text_model": self.text_model,
                        "vision_model": self.vision_model
                        if self.extraction_method == "ai"
                        else None,
                    },
                )

        except Exception as e:
            logger.error(f"CrewAI processing failed: {e}")
            return ProcessingResult(False, None, str(e))
