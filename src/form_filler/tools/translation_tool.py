#!/usr/bin/env python3
"""
Translation Tool for Vietnamese Document Form Filler.

Handles translating Vietnamese to English using Ollama models.
"""

import logging
from typing import Any

from crewai.tools import BaseTool
from langchain_community.chat_models import ChatOllama

# Setup logging
logger = logging.getLogger(__name__)


class TranslationTool(BaseTool):
    """Tool for translating Vietnamese text to English."""

    name: str = "vietnamese_translator"
    description: str = "Translate Vietnamese text to English using Ollama LLM"
    llm: ChatOllama | None = None

    def __init__(self, model: str = "llama3.2:3b", *args: Any, **kwargs: Any) -> None:
        """Initialize the object."""
        super().__init__(*args, **kwargs)
        self.llm = ChatOllama(model=model, base_url="http://localhost:11434")

    def _run(self, vietnamese_text: str) -> str:
        """Translate Vietnamese text to English."""
        if vietnamese_text is None or not vietnamese_text.strip():
            raise Exception("Empty text provided for translation")

        system_prompt = """You are a professional translator specializing in Vietnamese to English translation.
        Translate the given Vietnamese text to clear, accurate English while preserving the original meaning and context.
        Focus on formal document language appropriate for forms and official papers.

        Return only the English translation without any additional commentary."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Please translate this Vietnamese text to English:\n\n{vietnamese_text}",
            },
        ]

        try:
            if not self.llm:
                raise ValueError("LLM model not initialized")

            response = self.llm.invoke(messages)
            result = response.content if hasattr(response, "content") else str(response)
            return str(result)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
