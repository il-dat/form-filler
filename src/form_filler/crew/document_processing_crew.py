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

    def __init__(  # noqa: D107
        self,
        text_model: str = "llama3.2:3b",
        extraction_method: str = "traditional",
        vision_model: str = "llava:7b",
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o",
    ):
        # Create agents
        self.document_collector = create_document_collector_agent(
            extraction_method=extraction_method,
            vision_model=vision_model,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
        )
        self.translator = create_translator_agent(text_model)
        self.form_analyst = create_form_analyst_agent()
        self.form_filler = create_form_filler_agent(text_model)

        # Store configuration
        self.extraction_method = extraction_method
        self.text_model = text_model
        self.vision_model = vision_model
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

    def process_document(
        self,
        source_path: str,
        form_path: str,
        output_path: str,
    ) -> ProcessingResult:
        """Process a document through the CrewAI pipeline."""
        try:
            # Define tasks
            # Determine extraction method description for task
            extraction_method_desc = "traditional OCR methods"
            if self.extraction_method == "ai":
                extraction_method_desc = "AI vision models"
            elif self.extraction_method == "openai":
                extraction_method_desc = "OpenAI Vision API"

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
                        "text_model": self.text_model,
                        "vision_model": (
                            self.vision_model if self.extraction_method == "ai" else None
                        ),
                        "openai_model": (
                            self.openai_model if self.extraction_method == "openai" else None
                        ),
                    },
                )
            except json.JSONDecodeError:
                # If result is not JSON, consider it successful anyway
                return ProcessingResult(
                    success=True,
                    data={"output_path": output_path, "result": str(result)},
                    metadata={
                        "extraction_method": self.extraction_method,
                        "text_model": self.text_model,
                        "vision_model": (
                            self.vision_model if self.extraction_method == "ai" else None
                        ),
                        "openai_model": (
                            self.openai_model if self.extraction_method == "openai" else None
                        ),
                    },
                )

        except Exception as e:
            logger.error(f"CrewAI processing failed: {e}")
            return ProcessingResult(False, None, str(e))
