#!/usr/bin/env python3
"""
OpenAI Provider implementation for form-filler.

Provides integration with OpenAI API for cloud-based LLM inference.
"""

import base64
import logging
from typing import Any

import requests

from form_filler.ai_providers.base_provider import AIProvider, register_provider

# Setup logging
logger = logging.getLogger(__name__)


@register_provider
class OpenAIProvider(AIProvider):
    """AI Provider implementation for OpenAI."""

    provider_name = "openai"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            model_name: The name of the OpenAI model to use
            api_key: OpenAI API key (required)
            api_base: Base URL for OpenAI API (default: https://api.openai.com/v1)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model_name, api_key, api_base, **kwargs)

        # Set default API base if not provided
        if not self.api_base:
            self.api_base = "https://api.openai.com/v1"

        # Determine if this is a vision model
        self._is_vision_model = "vision" in model_name.lower() or self.model_name in [
            "gpt-4-vision-preview",
            "gpt-4o",
        ]

        # Set default timeout
        self.timeout = kwargs.get("timeout", 30)

    def initialize(self) -> None:
        """Initialize the OpenAI client.

        Validates API key and connection.
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # For OpenAI, we don't need to create a persistent client,
        # but we can validate that the API key and base URL are working
        try:
            # Make a simple models request to check connectivity
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Create models URL from base URL
            models_url = f"{self.api_base}/models"

            # Send request
            response = requests.get(
                models_url,
                headers=headers,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200:
                raise ValueError(f"OpenAI API returned status code {response.status_code}")

            logger.debug(f"Successfully connected to OpenAI API using model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise

    def text_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion using OpenAI.

        Args:
            prompt: The prompt to generate a completion for
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated text completion
        """
        # For OpenAI, we use the chat completion API for text completions
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Use the chat completion method
        return self.chat_completion(messages, max_tokens, temperature)

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat completion using OpenAI.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated chat completion text
        """
        # Validate API key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            # Prepare payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add any additional parameters from kwargs
            for key, value in self.kwargs.items():
                if key not in payload and key != "timeout":
                    payload[key] = value

            # Create chat completions URL from base URL
            chat_url = f"{self.api_base}/chat/completions"

            # Send request
            response = requests.post(
                chat_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200:
                error_message = f"OpenAI API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_message = f"{error_message} - {error_data['error']['message']}"
                except Exception as err:
                    logger.debug(f"Could not parse error details: {err}")
                raise ValueError(error_message)

            # Parse response
            response_data = response.json()

            # Extract completion text
            completion_text = response_data["choices"][0]["message"]["content"]

            return completion_text

        except Exception as e:
            logger.error(f"OpenAI chat completion failed: {e}")
            raise

    def vision_completion(
        self,
        prompt: str,
        image_data: str | bytes,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion based on an image using OpenAI.

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
                f"OpenAI model '{self.model_name}' does not support vision. "
                "Please use a vision-capable model like 'gpt-4-vision-preview'."
            )

        # Validate API key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            # Ensure image_data is a base64 string
            if isinstance(image_data, bytes):
                base64_image = base64.b64encode(image_data).decode("utf-8")
            else:
                # Assume it's already a base64 string
                base64_image = image_data

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            # Prepare messages with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                },
            ]

            # Prepare payload
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            # Add any additional parameters from kwargs
            for key, value in self.kwargs.items():
                if key not in payload and key != "timeout":
                    payload[key] = value

            # Create chat completions URL from base URL
            chat_url = f"{self.api_base}/chat/completions"

            # Send request
            response = requests.post(
                chat_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200:
                error_message = f"OpenAI API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_message = f"{error_message} - {error_data['error']['message']}"
                except Exception as err:
                    logger.debug(f"Could not parse error details: {err}")
                raise ValueError(error_message)

            # Parse response
            response_data = response.json()

            # Extract completion text
            completion_text = response_data["choices"][0]["message"]["content"]

            return completion_text

        except Exception as e:
            logger.error(f"OpenAI vision completion failed: {e}")
            raise

    @property
    def supports_vision(self) -> bool:
        """Check if this OpenAI model supports vision capabilities.

        Returns:
            True if model is a vision model, False otherwise
        """
        return self._is_vision_model

    @property
    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.

        Returns:
            True as OpenAI supports streaming
        """
        return True

    @property
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key.

        Returns:
            True as OpenAI requires API keys
        """
        return True
