#!/usr/bin/env python3
"""
Translation Tool for Vietnamese Document Form Filler.

Handles translating Vietnamese to English using various AI providers.
"""

import logging
from typing import Any

from crewai.tools import BaseTool
from pydantic import Field, PrivateAttr

from form_filler.ai_providers import AIProviderFactory

# Setup logging
logger = logging.getLogger(__name__)


class TranslationTool(BaseTool):
    """Tool for translating Vietnamese text to English."""

    name: str = "vietnamese_translator"
    description: str = "Translate Vietnamese text to English using an AI model"

    # Provider configuration
    provider_name: str = Field(default="ollama")
    model_name: str = Field(default="llama3.2:3b")
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)

    # Private attribute for AI provider
    _ai_provider = PrivateAttr(default=None)  # Will hold AIProvider instance

    def __init__(
        self,
        provider_name: str = "ollama",
        model_name: str = "llama3.2:3b",
        api_key: str | None = None,
        api_base: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the translation tool.

        Args:
            provider_name: The name of the AI provider (ollama, openai, anthropic, etc.)
            model_name: The name of the model to use
            api_key: API key for the provider (if needed)
            api_base: Base URL for the API (if needed)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Support for legacy parameters
        if "model" in kwargs and model_name == "llama3.2:3b":
            model_name = kwargs.pop("model")

        # Set parameters
        kwargs["provider_name"] = provider_name
        kwargs["model_name"] = model_name
        kwargs["api_key"] = api_key
        kwargs["api_base"] = api_base

        super().__init__(*args, **kwargs)

        # Initialize AI provider
        try:
            self._ai_provider = AIProviderFactory.create_provider(
                provider_name=self.provider_name,
                model_name=self.model_name,
                api_key=self.api_key,
                api_base=self.api_base,
            )
        except Exception as e:
            logger.error(f"Failed to initialize AI provider: {e}")
            self._ai_provider = None

    def _run(self, vietnamese_text: str) -> str:
        """Translate Vietnamese text to English.

        Args:
            vietnamese_text: The Vietnamese text to translate

        Returns:
            The English translation of the Vietnamese text

        Raises:
            Exception: If the text is empty or translation fails
        """
        if vietnamese_text is None or not vietnamese_text.strip():
            raise Exception("Empty text provided for translation")

        # Check if AI provider is initialized
        if not self._ai_provider:
            raise ValueError("AI provider not initialized")

        system_prompt = """You are a professional translator specializing in Vietnamese to English translation.
        Translate the given Vietnamese text to clear, accurate English while preserving the original meaning and context.
        Focus on formal document language appropriate for forms and official papers.

        Return only the English translation without any additional commentary."""

        prompt = f"Please translate this Vietnamese text to English:\n\n{vietnamese_text}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            # Use the appropriate method based on provider capabilities
            result = self._ai_provider.chat_completion(
                messages=messages,
                max_tokens=1000,
                temperature=0.3,  # Lower temperature for more accurate translation
            )

            return str(result)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise
