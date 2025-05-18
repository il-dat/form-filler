#!/usr/bin/env python3
"""Document extraction tool for Vietnamese documents."""

import logging
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF for PDF
import pytesseract
from crewai.tools import BaseTool
from PIL import Image
from pydantic import Field, PrivateAttr

from form_filler.ai_providers import AIProviderFactory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DocumentExtractionTool(BaseTool):
    """Tool for extracting text from documents."""

    name: str = "document_extractor"
    description: str = "Extract text from PDF or image files using traditional or AI methods"
    extraction_method: str = Field(default="traditional")

    # Provider configuration
    provider_name: str = Field(default="ollama")
    model_name: str = Field(default="llava:7b")
    api_key: str | None = Field(default=None)
    api_base: str | None = Field(default=None)

    # Private attribute for AI provider
    _ai_provider = PrivateAttr(default=None)  # Will hold AIProvider instance

    def __init__(
        self,
        extraction_method: str = "traditional",
        provider_name: str = "ollama",
        model_name: str = "llava:7b",
        api_key: str | None = None,
        api_base: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the document extraction tool.

        Args:
            extraction_method: The method to use for extraction (traditional, ai)
            provider_name: The name of the AI provider (ollama, openai, anthropic, etc.)
            model_name: The name of the model to use
            api_key: API key for the provider (if needed)
            api_base: Base URL for the API (if needed)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Support for legacy parameters
        if "vision_model" in kwargs and model_name == "llava:7b":
            model_name = kwargs.pop("vision_model")
        if "openai_api_key" in kwargs and api_key is None:
            api_key = kwargs.pop("openai_api_key")
        if "openai_model" in kwargs and provider_name == "openai" and model_name == "llava:7b":
            model_name = kwargs.pop("openai_model")

        # Set parameters
        kwargs["extraction_method"] = extraction_method
        kwargs["provider_name"] = provider_name
        kwargs["model_name"] = model_name
        kwargs["api_key"] = api_key
        kwargs["api_base"] = api_base

        super().__init__(**kwargs)

        # Initialize AI provider if using AI extraction method
        if self.extraction_method != "traditional":
            try:
                self._ai_provider = AIProviderFactory.create_provider(
                    provider_name=self.provider_name,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    api_base=self.api_base,
                )

                # Ensure the model supports vision if we're using AI extraction
                if not self._ai_provider.supports_vision:
                    logger.warning(
                        f"Model {self.model_name} does not support vision. "
                        "Using traditional extraction instead."
                    )
                    self.extraction_method = "traditional"
                    self._ai_provider = None

            except Exception as e:
                logger.warning(
                    f"Failed to initialize AI provider: {e}. Using traditional extraction."
                )
                self.extraction_method = "traditional"
                self._ai_provider = None

    def _run(self, file_path: str) -> str:
        """Extract text from the given file."""
        try:
            # Convert string path to Path object
            path_obj = Path(file_path)

            if not path_obj.exists():
                raise Exception(f"File not found: {file_path}")

            if path_obj.suffix.lower() == ".pdf":
                if self.extraction_method == "traditional":
                    return self._extract_from_pdf_traditional(path_obj)
                else:
                    return self._extract_from_pdf_ai(path_obj)
            elif path_obj.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                if self.extraction_method == "traditional":
                    return self._extract_from_image_traditional(path_obj)
                else:
                    return self._extract_from_image_ai(path_obj)
            else:
                raise Exception(f"Unsupported file type: {path_obj.suffix}")

        except Exception as e:
            logger.error(f"Error in DocumentExtractionTool: {e}")
            raise

    def _extract_from_pdf_traditional(self, file_path: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def _extract_from_pdf_ai(self, file_path: Path) -> str:
        """Extract text from PDF using AI (convert to images and use vision model)."""
        try:
            # Check if AI provider is initialized
            if not self._ai_provider:
                logger.warning(
                    "AI provider not initialized, falling back to traditional extraction"
                )
                return self._extract_from_pdf_traditional(file_path)

            # Convert PDF pages to images and process with AI
            doc = fitz.open(file_path)
            images_text = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")

                # Save image temporarily
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
                    tmp_img.write(img_data)
                    tmp_img_path = tmp_img.name

                try:
                    # Use AI to extract text from image
                    page_text = self._extract_from_image_ai(Path(tmp_img_path))
                    images_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                finally:
                    # Clean up temporary image
                    Path(tmp_img_path).unlink()

            doc.close()
            return "\n".join(images_text)

        except Exception as e:
            logger.warning(f"AI PDF extraction failed, falling back to traditional: {e}")
            return self._extract_from_pdf_traditional(file_path)

    def _extract_from_image_traditional(self, file_path: Path) -> str:
        """Extract text from image using Tesseract OCR."""
        image = Image.open(file_path)
        result = pytesseract.image_to_string(image, lang="vie")
        return str(result)

    def _extract_from_image_ai(self, file_path: Path) -> str:
        """Extract text from image using AI vision model."""
        try:
            # Check if AI provider is initialized
            if not self._ai_provider:
                logger.warning("AI provider not initialized, falling back to OCR")
                return self._extract_from_image_traditional(file_path)

            # Read image data
            with Path(file_path).open("rb") as img_file:
                image_data = img_file.read()

            # Create prompt for text extraction
            prompt = """Please extract all text from this image. The image may contain Vietnamese text.
            Extract ALL visible text, preserving the original formatting and structure.
            Pay attention to Vietnamese diacritics and special characters.

            Return only the extracted text without any additional commentary."""

            # Get response from AI provider's vision model
            try:
                extracted_text = self._ai_provider.vision_completion(
                    prompt=prompt,
                    image_data=image_data,
                    max_tokens=1000,
                    temperature=0.3,  # Lower temperature for more accurate extraction
                )

                if not extracted_text:
                    raise ValueError("AI response returned empty text")

                return extracted_text

            except Exception as e:
                logger.warning(f"AI extraction error: {e}, falling back to OCR")
                return self._extract_from_image_traditional(file_path)

        except Exception as e:
            logger.warning(f"AI image extraction failed, falling back to OCR: {e}")
            return self._extract_from_image_traditional(file_path)
