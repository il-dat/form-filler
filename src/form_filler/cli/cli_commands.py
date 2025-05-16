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
from rich.console import Console

from form_filler.crew import DocumentProcessingCrew
from form_filler.tools import DocumentExtractionTool, TranslationTool
from form_filler.utils.progress_utils import create_indeterminate_spinner

# Setup logging
logger = logging.getLogger(__name__)
console = Console()


def show_version(ctx, param, value):
    """Show current version of the package."""
    if not value or ctx.resilient_parsing:
        return
    try:
        # Import version information inside the function to avoid circular imports
        from form_filler import __author__, __email__, __version__

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
    type=click.Choice(["traditional", "ai", "openai"]),
    default="traditional",
    help="Text extraction method: traditional (PyMuPDF/Tesseract), ai (local vision models), or openai (OpenAI API)",
)
@click.option("--vision-model", "-vm", default="llava:7b", help="Vision model for AI extraction")
@click.option(
    "--openai-api-key", envvar="OPENAI_API_KEY", help="OpenAI API key for OpenAI extraction method"
)
@click.option(
    "--openai-model",
    default="gpt-4-vision-preview",
    help="OpenAI model for OpenAI extraction method",
)
@click.option(
    "--version",
    is_flag=True,
    callback=show_version,
    expose_value=False,
    is_eager=True,
    help="Show the version and exit.",
)
@click.pass_context
def cli(ctx, verbose, model, extraction_method, vision_model, openai_api_key, openai_model):
    """Vietnamese to English Document Form Filler (CrewAI Edition).

    A CrewAI-based multi-agent system for processing Vietnamese documents (PDF/images)
    and filling English DOCX forms using local Ollama LLMs.

    All commands display progress bars or spinners during execution to provide
    visual feedback for long-running operations.

    Extraction Methods:
    - traditional: Use PyMuPDF for PDFs and Tesseract for images
    - ai: Use local vision models for both PDFs and images (requires vision-capable models)
    - openai: Use OpenAI Vision API for both PDFs and images (requires API key)
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["model"] = model
    ctx.obj["extraction_method"] = extraction_method
    ctx.obj["vision_model"] = vision_model
    ctx.obj["openai_api_key"] = openai_api_key
    ctx.obj["openai_model"] = openai_model

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
    type=click.Choice(["traditional", "ai", "openai"]),
    help="Override default extraction method",
)
@click.option("--vision-model", "-vm", help="Override default vision model")
@click.option("--openai-api-key", help="OpenAI API key for OpenAI extraction method")
@click.option("--openai-model", help="OpenAI model for OpenAI extraction method")
@click.pass_context
def process(
    ctx, source, form, output, model, extraction_method, vision_model, openai_api_key, openai_model
):
    """Process a Vietnamese document and fill an English DOCX form using CrewAI.

    SOURCE: Path to Vietnamese document (PDF or image)
    FORM: Path to English DOCX form template
    OUTPUT: Path where filled form will be saved

    Examples:
    # Using local AI extraction with Ollama:
    python document_processor.py -e ai -vm llava:7b process document.pdf form.docx output.docx

    # Using OpenAI API:
    python document_processor.py -e openai --openai-api-key YOUR_API_KEY process document.pdf form.docx output.docx
    """
    model = model or ctx.obj["model"]
    extraction_method = extraction_method or ctx.obj["extraction_method"]
    vision_model = vision_model or ctx.obj["vision_model"]
    openai_api_key = openai_api_key or ctx.obj["openai_api_key"]
    openai_model = openai_model or ctx.obj["openai_model"]

    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create CrewAI processor
    crew_processor = DocumentProcessingCrew(
        text_model=model,
        extraction_method=extraction_method,
        vision_model=vision_model,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )

    # Process the document with progress bar
    with create_indeterminate_spinner("Processing document...") as progress:
        task_id = progress.add_task("Processing document...", total=None)

        start_time = time.time()
        result = crew_processor.process_document(source, form, output)
        processing_time = time.time() - start_time

        # Complete the progress
        progress.update(task_id, completed=True)

    if result.success:
        console.print(f"[green]✅ Success![/green] Document processed in {processing_time:.2f}s")
        console.print(f"Filled form saved to: [blue]{output}[/blue]")
        if result.metadata:
            console.print(
                f"Extraction method: [yellow]{result.metadata.get('extraction_method', 'N/A')}[/yellow]"
            )
            console.print(
                f"Text model: [yellow]{result.metadata.get('text_model', 'N/A')}[/yellow]"
            )
            if result.metadata.get("vision_model"):
                console.print(
                    f"Vision model: [yellow]{result.metadata.get('vision_model')}[/yellow]"
                )
            if result.metadata.get("openai_model") and extraction_method == "openai":
                console.print(
                    f"OpenAI model: [yellow]{result.metadata.get('openai_model')}[/yellow]"
                )
        if result.data and isinstance(result.data, dict):
            fields_filled = result.data.get("fields_filled", "N/A")
            console.print(f"Fields filled: [yellow]{fields_filled}[/yellow]")
    else:
        console.print(f"[red]❌ Error:[/red] {result.error}")
        sys.exit(1)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai", "openai"]),
    help="Override default extraction method",
)
@click.option("--vision-model", "-vm", help="Vision model for AI extraction")
@click.option("--openai-api-key", help="OpenAI API key for OpenAI extraction method")
@click.option("--openai-model", help="OpenAI model for OpenAI extraction method")
@click.pass_context
def extract(ctx, file_path, extraction_method, vision_model, openai_api_key, openai_model):
    """Extract text from a Vietnamese document (for testing).

    Examples:
    # Using local AI extraction with Ollama:
    python document_processor.py -e ai -vm llava:7b extract document.pdf

    # Using OpenAI API:
    python document_processor.py -e openai --openai-api-key YOUR_API_KEY extract document.pdf
    """
    extraction_method = extraction_method or ctx.obj["extraction_method"]
    vision_model = vision_model or ctx.obj["vision_model"]

    # Create extraction tool with the appropriate parameters
    openai_api_key = openai_api_key or ctx.obj["openai_api_key"]
    openai_model = openai_model or ctx.obj["openai_model"]

    # Create extraction tool
    extractor = DocumentExtractionTool(
        extraction_method=extraction_method,
        vision_model=vision_model,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )

    try:
        console.print(f"Extracting text using [yellow]{extraction_method}[/yellow] method...")
        if extraction_method == "ai":
            console.print(f"Vision model: [yellow]{vision_model}[/yellow]")

        # Add progress spinner during extraction
        with create_indeterminate_spinner("Extracting text...") as progress:
            task_id = progress.add_task("Extracting text...", total=None)

            start_time = time.time()
            extracted_text = extractor._run(file_path)
            processing_time = time.time() - start_time

            # Complete the progress
            progress.update(task_id, completed=True)

        console.print(f"\n[green]✅ Text extracted[/green] in {processing_time:.2f}s")
        console.print(f"Extracted text ([yellow]{extraction_method}[/yellow] method):")
        console.print("[blue]" + "-" * 50 + "[/blue]")
        console.print(extracted_text)
        console.print("[blue]" + "-" * 50 + "[/blue]")
        console.print(f"Characters: [yellow]{len(extracted_text)}[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {e}")
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
        console.print(f"Translating with model: [yellow]{model}[/yellow]")

        # Add progress spinner during translation
        with create_indeterminate_spinner("Translating text...") as progress:
            task_id = progress.add_task("Translating text...", total=None)

            start_time = time.time()
            english_text = translator._run(vietnamese_text)
            processing_time = time.time() - start_time

            # Complete the progress
            progress.update(task_id, completed=True)

        console.print(f"\n[green]✅ Text translated[/green] in {processing_time:.2f}s")
        console.print("Translation:")
        console.print("[blue]" + "-" * 50 + "[/blue]")
        console.print(english_text)
        console.print("[blue]" + "-" * 50 + "[/blue]")
    except Exception as e:
        console.print(f"[red]❌ Error:[/red] {e}")
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
                        click.echo("  No vision models found. Install with: ollama pull llava:7b")
                        click.echo(
                            "  Vision models enable AI-powered text extraction from images and PDFs"
                        )
                        click.echo("  Supported models: llava:7b, llava:13b, bakllava")

                click.echo("\n🤖 CrewAI Integration Status: ✅ Ready")
                click.echo("Available extraction methods: traditional, ai, openai")
                click.echo("\n💡 Note: OpenAI extraction requires an API key.")
                click.echo("Set it with --openai-api-key or OPENAI_API_KEY environment variable.")
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
