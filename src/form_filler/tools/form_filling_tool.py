#!/usr/bin/env python3
"""
Form Filling Tool for Vietnamese Document Form Filler.

Handles filling DOCX forms with translated content.
"""

import json
import logging

from crewai.tools import BaseTool
from docx import Document
from langchain_community.chat_models import ChatOllama

from form_filler.tools.form_analysis_tool import FormAnalysisTool

# Setup logging
logger = logging.getLogger(__name__)


class FormFillingTool(BaseTool):
    """Tool for filling DOCX forms with translated content."""

    name: str = "form_filler"
    description: str = "Fill DOCX form fields with provided content"
    llm: any = None

    def __init__(self, model="llama3.2:3b", *args, **kwargs):
        """Initialize the object."""
        super().__init__(*args, **kwargs)
        self.llm = ChatOllama(model=model, base_url="http://localhost:11434")

    def _run(
        self, form_path: str, translated_text: str, output_path: str, field_mappings: str = None
    ) -> str:
        """Fill the form with mapped content."""
        try:
            doc = Document(form_path)

            # If no field mappings provided, generate them using AI
            if not field_mappings:
                field_mappings = self._generate_field_mappings(form_path, translated_text)

            # Parse field mappings
            try:
                mappings_data = json.loads(field_mappings)
                field_mappings_list = mappings_data.get("field_mappings", [])
            except json.JSONDecodeError:
                # Fallback: create simple mapping
                field_mappings_list = self._create_fallback_mappings(doc, translated_text)

            filled_count = 0

            # Fill paragraphs
            for paragraph in doc.paragraphs:
                for mapping in field_mappings_list:
                    field_text = mapping.get("field_text", "")
                    fill_content = mapping.get("fill_with", "")

                    if field_text in paragraph.text:
                        # Replace placeholder with content
                        new_text = paragraph.text.replace("____", fill_content)
                        new_text = new_text.replace(field_text, fill_content)
                        paragraph.clear()
                        paragraph.add_run(new_text)
                        filled_count += 1

            # Fill tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        for mapping in field_mappings_list:
                            field_text = mapping.get("field_text", "")
                            fill_content = mapping.get("fill_with", "")

                            if field_text in cell.text:
                                cell.text = cell.text.replace("____", fill_content)
                                cell.text = cell.text.replace(field_text, fill_content)
                                filled_count += 1

            # Save filled document
            doc.save(output_path)

            return json.dumps(
                {
                    "output_path": output_path,
                    "fields_filled": filled_count,
                    "total_mappings": len(field_mappings_list),
                }
            )

        except Exception as e:
            logger.error(f"Form filling failed: {e}")
            raise

    def _generate_field_mappings(self, form_path: str, content: str) -> str:
        """Generate field mappings using AI."""
        # Analyze form structure
        form_analyzer = FormAnalysisTool()
        form_fields = form_analyzer._run(form_path)

        system_prompt = """You are an expert at analyzing documents and mapping content to form fields.
        Given a form structure and translated content, determine how to fill each field appropriately.
        Return ONLY valid JSON in the specified format."""

        prompt = f"""Form fields found:
{form_fields}

Content to fill:
{content}

Please analyze and create a mapping of which content should fill which fields.
Consider the context and purpose of each field. Return only valid JSON in this format:
{{
    "field_mappings": [
        {{
            "field_text": "Original field text",
            "fill_with": "Content to fill this field",
            "confidence": 0.95
        }}
    ]
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.llm.invoke(messages)
            return response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            logger.error(f"AI field mapping failed: {e}")
            return self._create_fallback_json(form_fields, content)

    def _create_fallback_mappings(self, doc: Document, content: str) -> list[dict]:
        """Create simple fallback mappings when AI fails."""
        paragraphs_with_placeholders = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text and ("____" in text or "[" in text):
                paragraphs_with_placeholders.append(text)

        # Simple mapping: use first part of content for first field, etc.
        content_parts = content.split("\n")[: len(paragraphs_with_placeholders)]

        mappings = []
        for i, field_text in enumerate(paragraphs_with_placeholders):
            if i < len(content_parts):
                mappings.append(
                    {
                        "field_text": field_text,
                        "fill_with": content_parts[i][:100],  # Limit length
                        "confidence": 0.5,
                    }
                )

        return mappings

    def _create_fallback_json(self, form_fields: str, content: str) -> str:
        """Create fallback JSON when AI mapping fails."""
        try:
            fields = json.loads(form_fields)
            content_parts = content.split("\n")

            mappings = []
            for i, field in enumerate(fields[:3]):  # Limit to first 3 fields
                if i < len(content_parts):
                    mappings.append(
                        {
                            "field_text": field.get("text", ""),
                            "fill_with": content_parts[i][:100],
                            "confidence": 0.5,
                        }
                    )

            return json.dumps({"field_mappings": mappings})
        except Exception:
            return json.dumps({"field_mappings": []})
