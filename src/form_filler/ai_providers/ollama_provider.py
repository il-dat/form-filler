#!/usr/bin/env python3
"""
Ollama AI Provider implementation for form-filler.

Provides integration with Ollama API for local LLM inference.
"""

import base64
import logging
from typing import Any

from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaLLM

from form_filler.ai_providers.base_provider import AIProvider, register_provider

# Setup logging
logger = logging.getLogger(__name__)


@register_provider
class OllamaProvider(AIProvider):
    """AI Provider implementation for Ollama."""

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            model_name: The name of the Ollama model to use
            api_key: Not used for Ollama (included for API consistency)
            api_base: Base URL for Ollama API (default: http://localhost:11434)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model_name, api_key, api_base, **kwargs)

        # Set default API base if not provided
        if not self.api_base:
            self.api_base = "http://localhost:11434"

        # Determine if this is a vision model
        self._is_vision_model = any(
            x in model_name.lower() for x in ["llava", "vision", "bakllava"]
        )

        self._llm = None
        self._chat_model = None

    def initialize(self) -> None:
        """Initialize the Ollama client."""
        try:
            # Initialize both LLM (for completion/vision) and Chat model
            self._llm = OllamaLLM(
                model=self.model_name,
                base_url=self.api_base,
                **self.kwargs,
            )

            self._chat_model = ChatOllama(
                model=self.model_name,
                base_url=self.api_base,
                **self.kwargs,
            )

            logger.debug(f"Initialized Ollama provider with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise

    def text_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion using Ollama.

        Args:
            prompt: The prompt to generate a completion for
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated text completion
        """
        if not self._llm:
            self.initialize()

        # Format the prompt with system instructions if provided
        formatted_prompt = prompt
        if system_prompt:
            formatted_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            # Set parameters
            self._llm.num_predict = max_tokens
            self._llm.temperature = temperature

            # Generate completion
            response = self._llm.invoke(formatted_prompt)

            # Handle different response types
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return str(response.content)
            else:
                logger.warning(f"Unexpected response format: {type(response)}")
                return str(response)

        except Exception as e:
            logger.error(f"Ollama text completion failed: {e}")
            raise

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat completion using Ollama.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated chat completion text
        """
        if not self._chat_model:
            self.initialize()

        try:
            # Set parameters (if supported by the chat model)
            if hasattr(self._chat_model, "num_predict"):
                self._chat_model.num_predict = max_tokens
            if hasattr(self._chat_model, "temperature"):
                self._chat_model.temperature = temperature

            # Generate chat completion
            response = self._chat_model.invoke(messages)

            # Extract content from response
            if hasattr(response, "content"):
                return str(response.content)
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Ollama chat completion failed: {e}")
            raise

    def vision_completion(
        self,
        prompt: str,
        image_data: str | bytes,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion based on an image using Ollama.

        Args:
            prompt: The text prompt to accompany the image
            image_data: The image data (base64 string or bytes)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated text based on the image
        """
        if not self.supports_vision:
            raise ValueError(
                f"Ollama model '{self.model_name}' does not support vision. "
                "Please use a vision-capable model like 'llava:7b'."
            )

        if not self._llm:
            self.initialize()

        try:
            # Ensure image_data is a base64 string
            if isinstance(image_data, bytes):
                base64_image = base64.b64encode(image_data).decode("utf-8")
            else:
                # Assume it's already a base64 string
                base64_image = image_data

            # Format prompt with image for Ollama
            vision_prompt = f"""{prompt}

            <image>data:image/jpeg;base64,{base64_image}</image>"""

            # Set parameters
            self._llm.num_predict = max_tokens
            self._llm.temperature = temperature

            # Generate vision completion
            response = self._llm.invoke(vision_prompt)

            # Handle different response types
            if isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                return str(response.content)
            else:
                logger.warning(f"Unexpected response format: {type(response)}")
                return str(response)

        except Exception as e:
            logger.error(f"Ollama vision completion failed: {e}")
            raise

    @property
    def supports_vision(self) -> bool:
        """Check if this Ollama model supports vision capabilities.

        Returns:
            True if model is a vision model, False otherwise
        """
        return self._is_vision_model

    @property
    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.

        Returns:
            True as Ollama supports streaming
        """
        return True

    @property
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key.

        Returns:
            False as Ollama does not require API keys
        """
        return False
