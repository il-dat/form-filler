#!/usr/bin/env python3
"""Document extraction tool for Vietnamese documents."""

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF for PDF
import pytesseract
import requests
from crewai.tools import BaseTool
from langchain_ollama import OllamaLLM
from PIL import Image
from pydantic import Field, PrivateAttr

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
    vision_model: str = Field(default="llava:7b")
    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4-vision-preview")

    # Private attribute for ollama_llm
    _ollama_llm = PrivateAttr(default=None)  # Will hold OllamaLLM instance

    def __init__(
        self,
        extraction_method: str = "traditional",
        vision_model: str = "llava:7b",
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4-vision-preview",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the document extraction tool."""
        kwargs["extraction_method"] = extraction_method
        kwargs["vision_model"] = vision_model
        kwargs["openai_api_key"] = openai_api_key
        kwargs["openai_model"] = openai_model
        super().__init__(**kwargs)

        # Initialize Ollama LLM if using AI extraction method
        if self.extraction_method == "ai":
            self._ollama_llm = OllamaLLM(
                model=self.vision_model,
                base_url="http://localhost:11434",
            )

    def _run(self, file_path: str) -> str:
        """Extract text from the given file."""
        try:
            # Convert string path to Path object
            path_obj = Path(file_path)

            if not path_obj.exists():
                raise Exception(f"File not found: {file_path}")

            if path_obj.suffix.lower() == ".pdf":
                if self.extraction_method == "ai":
                    return self._extract_from_pdf_ai(path_obj)
                elif self.extraction_method == "openai":
                    return self._extract_from_pdf_openai(path_obj)
                else:
                    return self._extract_from_pdf_traditional(path_obj)
            elif path_obj.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                if self.extraction_method == "ai":
                    return self._extract_from_image_ai(path_obj)
                elif self.extraction_method == "openai":
                    return self._extract_from_image_openai(path_obj)
                else:
                    return self._extract_from_image_traditional(path_obj)
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
            # First check if ollama_llm is initialized
            if self._ollama_llm is None:
                logger.warning("AI model not initialized, falling back to OCR")
                return self._extract_from_image_traditional(file_path)

            # Convert image to base64 for AI processing
            with Path(file_path).open("rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Create prompt for text extraction
            prompt = f"""Please extract all text from this image. The image may contain Vietnamese text.
            Extract ALL visible text, preserving the original formatting and structure.
            Pay attention to Vietnamese diacritics and special characters.

            Return only the extracted text without any additional commentary.

            <image>data:image/png;base64,{base64_image}</image>

            Extracted text:"""

            # Get response from model
            response: Any = self._ollama_llm.invoke(prompt)

            # Initialize extracted_text
            extracted_text = ""

            # Try to get the text content based on the response type
            try:
                if response is None:
                    raise ValueError("AI response is None")
                elif isinstance(response, str):
                    extracted_text = response
                elif isinstance(response, dict) and "content" in response:
                    extracted_text = str(response["content"])
                elif hasattr(response, "content"):
                    extracted_text = str(response.content)
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")

                if not extracted_text:
                    raise ValueError("AI response returned empty text")

                return extracted_text

            except (ValueError, AttributeError, KeyError, TypeError) as e:
                logger.warning(f"AI extraction error: {e}, falling back to OCR")
                return self._extract_from_image_traditional(file_path)

        except Exception as e:
            logger.warning(f"AI image extraction failed, falling back to OCR: {e}")
            return self._extract_from_image_traditional(file_path)

    def _extract_from_image_openai(self, file_path: Path) -> str:
        """Extract text from image using OpenAI's vision model."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI extraction method")

        try:
            # Read image file and encode it
            with Path(file_path).open("rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            }

            payload = {
                "model": self.openai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract ALL text from this image. The image may contain Vietnamese text. Extract ALL visible text, preserving the original formatting and structure. Pay attention to Vietnamese diacritics and special characters. Return only the extracted text without any additional commentary.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
                "max_tokens": 1000,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,  # Add timeout for security
            )

            if response.status_code != 200:
                logger.error(f"OpenAI API error: {response.text}")
                raise Exception(f"OpenAI API error: {response.status_code}")

            result = response.json()
            extracted_text = result["choices"][0]["message"]["content"]
            return str(extracted_text)

        except Exception as e:
            logger.warning(f"OpenAI image extraction failed, falling back to OCR: {e}")
            return self._extract_from_image_traditional(file_path)

    def _extract_from_pdf_openai(self, file_path: Path) -> str:
        """Extract text from PDF using OpenAI (convert to images and use vision model)."""
        try:
            # Convert PDF pages to images and process with OpenAI
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
                    # Use OpenAI to extract text from image
                    page_text = self._extract_from_image_openai(Path(tmp_img_path))
                    images_text.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                finally:
                    # Clean up temporary image
                    Path(tmp_img_path).unlink()

            doc.close()
            return "\n".join(images_text)

        except Exception as e:
            logger.warning(f"OpenAI PDF extraction failed, falling back to traditional: {e}")
            return self._extract_from_pdf_traditional(file_path)
