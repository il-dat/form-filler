#!/usr/bin/env python3
"""
CLI Commands for Vietnamese Document Form Filler.

Implements Click commands for the application CLI.
"""

import logging
import sys
import time
from pathlib import Path

import aiohttp
import click

from form_filler.crew import DocumentProcessingCrew
from form_filler.tools import DocumentExtractionTool, TranslationTool

# Setup logging
logger = logging.getLogger(__name__)


def show_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    try:
        # Import version information inside the function to avoid circular imports
        from form_filler import __version__, __author__, __email__
        click.echo(f"Form-Filler version: {__version__}")
        click.echo(f"Author: {__author__}")
        click.echo(f"Email: {__email__}")
    except ImportError:
        click.echo("Form-Filler version: unknown")

    # Display Python version and system info
    import platform
    click.echo(f"Python version: {sys.version.split()[0]}")
    click.echo(f"System: {platform.system()} {platform.release()}")
    ctx.exit()

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--model", "-m", default="llama3.2:3b", help="Ollama model to use for text processing"
)
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    default="traditional",
    help="Text extraction method: traditional (PyMuPDF/Tesseract) or ai (vision models)",
)
@click.option("--vision-model", "-vm", default="llava:7b", help="Vision model for AI extraction")
@click.option('--version', is_flag=True, callback=show_version, expose_value=False,
              is_eager=True, help='Show the version and exit.')
@click.pass_context
def cli(ctx, verbose, model, extraction_method, vision_model):
    """Vietnamese to English Document Form Filler (CrewAI Edition).

    A CrewAI-based multi-agent system for processing Vietnamese documents (PDF/images)
    and filling English DOCX forms using local Ollama LLMs.

    Extraction Methods:
    - traditional: Use PyMuPDF for PDFs and Tesseract for images
    - ai: Use vision models for both PDFs and images (requires vision-capable models)
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["model"] = model
    ctx.obj["extraction_method"] = extraction_method
    ctx.obj["vision_model"] = vision_model

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("form", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--model", "-m", help="Override default Ollama model")
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    help="Override default extraction method",
)
@click.option("--vision-model", "-vm", help="Override default vision model")
@click.pass_context
def process(ctx, source, form, output, model, extraction_method, vision_model):
    """Process a Vietnamese document and fill an English DOCX form using CrewAI.

    SOURCE: Path to Vietnamese document (PDF or image)
    FORM: Path to English DOCX form template
    OUTPUT: Path where filled form will be saved

    Example with AI extraction:
    python document_processor.py -e ai -vm llava:7b process document.pdf form.docx output.docx
    """
    model = model or ctx.obj["model"]
    extraction_method = extraction_method or ctx.obj["extraction_method"]
    vision_model = vision_model or ctx.obj["vision_model"]

    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create CrewAI processor
    crew_processor = DocumentProcessingCrew(
        text_model=model, extraction_method=extraction_method, vision_model=vision_model
    )

    # Process the document
    start_time = time.time()
    result = crew_processor.process_document(source, form, output)
    processing_time = time.time() - start_time

    if result.success:
        click.echo(f"✅ Success! Document processed in {processing_time:.2f}s")
        click.echo(f"Filled form saved to: {output}")
        if result.metadata:
            click.echo(f"Extraction method: {result.metadata.get('extraction_method', 'N/A')}")
            click.echo(f"Text model: {result.metadata.get('text_model', 'N/A')}")
            if result.metadata.get("vision_model"):
                click.echo(f"Vision model: {result.metadata.get('vision_model')}")
        if result.data and isinstance(result.data, dict):
            fields_filled = result.data.get("fields_filled", "N/A")
            click.echo(f"Fields filled: {fields_filled}")
    else:
        click.echo(f"❌ Error: {result.error}")
        sys.exit(1)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    help="Override default extraction method",
)
@click.option("--vision-model", "-vm", help="Vision model for AI extraction")
@click.pass_context
def extract(ctx, file_path, extraction_method, vision_model):
    """Extract text from a Vietnamese document (for testing).

    Example with AI extraction:
    python document_processor.py -e ai extract document.pdf
    """
    extraction_method = extraction_method or ctx.obj["extraction_method"]
    vision_model = vision_model or ctx.obj["vision_model"]

    # Create extraction tool
    extractor = DocumentExtractionTool(
        extraction_method=extraction_method, vision_model=vision_model
    )

    try:
        click.echo(f"Extracting text using {extraction_method} method...")
        if extraction_method == "ai":
            click.echo(f"Vision model: {vision_model}")

        extracted_text = extractor._run(file_path)

        click.echo(f"Extracted text ({extraction_method} method):")
        click.echo("-" * 50)
        click.echo(extracted_text)
        click.echo("-" * 50)
        click.echo(f"Characters: {len(extracted_text)}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument("vietnamese_text")
@click.option("--model", "-m", help="Override default model")
@click.pass_context
def translate(ctx, vietnamese_text, model):
    """Translate Vietnamese text to English (for testing)."""
    model = model or ctx.obj["model"]

    translator = TranslationTool(model=model)

    try:
        click.echo(f"Translating with model: {model}")
        english_text = translator._run(vietnamese_text)

        click.echo("Translation:")
        click.echo("-" * 50)
        click.echo(english_text)
        click.echo("-" * 50)
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


async def check_ollama(host: str, port: int, check_vision: bool) -> None:
    """Check if Ollama is running and list available models."""
    url = f"http://{host}:{port}/api/tags"

    try:
        async with aiohttp.ClientSession() as session, session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                models = data.get("models", [])

                click.echo("✅ Ollama is running!")
                click.echo("\nAvailable models:")

                text_models = []
                vision_models = []

                for model in models:
                    name = model.get("name", "Unknown")
                    size = model.get("size", 0)
                    size_mb = size / (1024 * 1024) if size else 0

                    # Classify models
                    if (
                        "llava" in name.lower()
                        or "vision" in name.lower()
                        or "bakllava" in name.lower()
                    ):
                        vision_models.append(f"  - {name} ({size_mb:.1f} MB)")
                    else:
                        text_models.append(f"  - {name} ({size_mb:.1f} MB)")

                click.echo("\n📝 Text Models (for translation and form filling):")
                for model in text_models:
                    click.echo(model)

                if check_vision:
                    click.echo("\n👁️ Vision Models (for AI text extraction):")
                    if vision_models:
                        for model in vision_models:
                            click.echo(model)
                    else:
                        click.echo(
                            "  No vision models found. Install with: ollama pull llava:7b"
                        )
                        click.echo(
                            "  Vision models enable AI-powered text extraction from images and PDFs"
                        )
                        click.echo("  Supported models: llava:7b, llava:13b, bakllava")

                click.echo("\n🤖 CrewAI Integration Status: ✅ Ready")
                click.echo("Available extraction methods: traditional, ai")
            else:
                click.echo(f"❌ Ollama responded with status: {response.status}")
                sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error connecting to Ollama: {e}")
        click.echo(f"Make sure Ollama is running on {host}:{port}")
        sys.exit(1)


@cli.command()
def version():
    """Display the current version of the form-filler package."""
    # Use the same function that handles the --version flag
    show_version(click.get_current_context(), None, True)
