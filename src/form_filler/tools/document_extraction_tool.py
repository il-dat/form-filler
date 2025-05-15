#!/usr/bin/env python3
"""
Document Extraction Tool for Vietnamese Document Form Filler.

Handles extracting text from documents using traditional OCR or AI methods.
"""

import base64
import logging
import tempfile
from pathlib import Path

import fitz  # PyMuPDF for PDF
import pytesseract
from crewai.tools import BaseTool
from langchain_ollama import OllamaLLM
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)


class DocumentExtractionTool(BaseTool):
    """Tool for extracting text from documents."""

    name: str = "document_extractor"
    description: str = "Extract text from PDF or image files using traditional or AI methods"
    extraction_method: str = "traditional"
    vision_model: str = "llava:7b"
    ollama_llm: OllamaLLM = None

    def __init__(self, extraction_method: str = "traditional", vision_model: str = "llava:7b"):
        """Initialize the document extraction tool."""
        super().__init__()
        self.extraction_method = extraction_method
        self.vision_model = vision_model
        self.ollama_llm = None
        if extraction_method == "ai":
            self.ollama_llm = OllamaLLM(model=vision_model, base_url="http://localhost:11434")

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
                else:
                    return self._extract_from_pdf_traditional(path_obj)
            elif path_obj.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                if self.extraction_method == "ai":
                    return self._extract_from_image_ai(path_obj)
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
                    images_text.append(f"\n--- Page {page_num+1} ---\n{page_text}")
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
        # Ensure we return a string from pytesseract
        result = pytesseract.image_to_string(image, lang="vie")
        if result is None:
            return ""
        return str(result)

    def _extract_from_image_ai(self, file_path: Path) -> str:
        """Extract text from image using AI vision model."""
        try:
            # Convert image to base64 for AI processing
            with open(file_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Create prompt for text extraction
            prompt = f"""Please extract all text from this image. The image may contain Vietnamese text.
            Extract ALL visible text, preserving the original formatting and structure.
            Pay attention to Vietnamese diacritics and special characters.

            Return only the extracted text without any additional commentary.

            <image>data:image/png;base64,{base64_image}</image>

            Extracted text:"""

            # Use vision model for extraction
            if self.ollama_llm is None:
                logger.warning("Ollama LLM not initialized, falling back to OCR")
                return self._extract_from_image_traditional(file_path)

            extracted_text = self.ollama_llm.invoke(prompt)

            if not extracted_text:
                logger.warning("AI image extraction failed, falling back to OCR")
                return self._extract_from_image_traditional(file_path)

            # Ensure we return a string
            if isinstance(extracted_text, str):
                return extracted_text
            else:
                # Handle potential non-string responses from different Ollama LLM versions
                logger.warning(f"Non-string response from Ollama: {type(extracted_text)}")
                if hasattr(extracted_text, '__str__'):
                    return str(extracted_text)
                return self._extract_from_image_traditional(file_path)

        except Exception as e:
            logger.warning(f"AI image extraction failed, falling back to OCR: {e}")
            return self._extract_from_image_traditional(file_path)
