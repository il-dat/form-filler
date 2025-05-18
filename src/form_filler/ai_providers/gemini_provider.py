#!/usr/bin/env python3
"""
Google Gemini Provider implementation for form-filler.

Provides integration with Google's Gemini API for LLM inference.
"""

import base64
import logging
from typing import Any

import requests

from form_filler.ai_providers.base_provider import AIProvider, register_provider

# Setup logging
logger = logging.getLogger(__name__)


@register_provider
class GeminiProvider(AIProvider):
    """AI Provider implementation for Google Gemini."""

    provider_name = "gemini"

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Gemini provider.

        Args:
            model_name: The name of the Gemini model to use
            api_key: Google API key (required)
            api_base: Base URL for Gemini API (default: https://generativelanguage.googleapis.com/v1)
            **kwargs: Additional provider-specific parameters
        """
        super().__init__(model_name, api_key, api_base, **kwargs)

        # Set default API base if not provided
        if not self.api_base:
            self.api_base = "https://generativelanguage.googleapis.com/v1"

        # Determine if this is a vision model
        self._is_vision_model = "vision" in model_name.lower() or "pro" in model_name.lower()

        # Set default timeout
        self.timeout = kwargs.get("timeout", 30)

    def initialize(self) -> None:
        """Initialize the Gemini client.

        Validates API key and connection.
        """
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini")

        try:
            # Make a simple models request to check connectivity
            # For Gemini, we append the API key as a query parameter
            models_url = f"{self.api_base}/models?key={self.api_key}"

            # Send request
            response = requests.get(
                models_url,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200:
                error_message = f"Gemini API returned status code {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_message = f"{error_message} - {error_data['error']['message']}"
                except Exception as err:
                    logger.debug(f"Could not parse error details: {err}")
                raise ValueError(error_message)

            logger.debug(f"Successfully connected to Gemini API using model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Gemini provider: {e}")
            raise

    def text_completion(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a text completion using Gemini.

        Args:
            prompt: The prompt to generate a completion for
            system_prompt: Optional system instructions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated text completion
        """
        # Validate API key
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini")

        try:
            # Prepare payload
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }

            # Add system prompt if provided
            if system_prompt:
                payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

            # Add any additional parameters from kwargs
            for key, value in self.kwargs.items():
                # Handle generation config parameters
                if key not in ["timeout"] and key in [
                    "topP",
                    "topK",
                    "stopSequences",
                    "candidateCount",
                    "presencePenalty",
                    "frequencyPenalty",
                ]:
                    payload["generationConfig"][key] = value

            # Create generate content URL from base URL with API key as query parameter
            generate_url = (
                f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
            )

            # Send request
            response = requests.post(
                generate_url,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200:
                error_message = f"Gemini API error: {response.status_code}"
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
            # Gemini responses have a different structure
            completion_text = ""
            for candidate in response_data.get("candidates", []):
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        completion_text += part["text"]

            return completion_text

        except Exception as e:
            logger.error(f"Gemini text completion failed: {e}")
            raise

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a chat completion using Gemini.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            The generated chat completion text
        """
        # Validate API key
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini")

        try:
            # Convert from 'role'/'content' format to Gemini's format
            contents = []
            system_instruction = None

            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")

                # Handle system messages separately for Gemini
                if role == "system":
                    system_instruction = {"parts": [{"text": content}]}
                    continue

                # Map OpenAI roles to Gemini roles
                gemini_role = "user" if role == "user" else "model"

                # Create the content object
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

            # Prepare payload
            payload = {
                "contents": contents,
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }

            # Add system instructions if provided
            if system_instruction:
                payload["systemInstruction"] = system_instruction

            # Add any additional parameters from kwargs
            for key, value in self.kwargs.items():
                # Handle generation config parameters
                if key not in ["timeout"] and key in [
                    "topP",
                    "topK",
                    "stopSequences",
                    "candidateCount",
                    "presencePenalty",
                    "frequencyPenalty",
                ]:
                    payload["generationConfig"][key] = value

            # Create generate content URL from base URL with API key as query parameter
            generate_url = (
                f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
            )

            # Send request
            response = requests.post(
                generate_url,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200:
                error_message = f"Gemini API error: {response.status_code}"
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
            completion_text = ""
            for candidate in response_data.get("candidates", []):
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        completion_text += part["text"]

            return completion_text

        except Exception as e:
            logger.error(f"Gemini chat completion failed: {e}")
            raise

    def vision_completion(
        self,
        prompt: str,
        image_data: str | bytes,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion based on an image using Gemini.

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
                f"Gemini model '{self.model_name}' does not support vision. "
                "Please use a vision-capable model like 'gemini-1.5-pro'."
            )

        # Validate API key
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini")

        try:
            # Ensure image_data is a base64 string
            if isinstance(image_data, bytes):
                base64_image = base64.b64encode(image_data).decode("utf-8")
            else:
                # Assume it's already a base64 string
                base64_image = image_data

            # Prepare payload with multimodal content
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {"inlineData": {"mimeType": "image/jpeg", "data": base64_image}},
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }

            # Add any additional parameters from kwargs
            for key, value in self.kwargs.items():
                # Handle generation config parameters
                if key not in ["timeout"] and key in [
                    "topP",
                    "topK",
                    "stopSequences",
                    "candidateCount",
                    "presencePenalty",
                    "frequencyPenalty",
                ]:
                    payload["generationConfig"][key] = value

            # Create generate content URL from base URL with API key as query parameter
            generate_url = (
                f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
            )

            # Send request
            response = requests.post(
                generate_url,
                json=payload,
                timeout=self.timeout,
            )

            # Check response
            if response.status_code != 200:
                error_message = f"Gemini API error: {response.status_code}"
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
            completion_text = ""
            for candidate in response_data.get("candidates", []):
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    if "text" in part:
                        completion_text += part["text"]

            return completion_text

        except Exception as e:
            logger.error(f"Gemini vision completion failed: {e}")
            raise

    @property
    def supports_vision(self) -> bool:
        """Check if this Gemini model supports vision capabilities.

        Returns:
            True if model is a vision model, False otherwise
        """
        return self._is_vision_model

    @property
    def supports_streaming(self) -> bool:
        """Check if this provider supports streaming responses.

        Returns:
            True as Gemini supports streaming
        """
        return True

    @property
    def requires_api_key(self) -> bool:
        """Check if this provider requires an API key.

        Returns:
            True as Gemini requires API keys
        """
        return True
