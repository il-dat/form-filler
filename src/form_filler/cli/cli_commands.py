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

from form_filler.ai_providers import AIProvider
from form_filler.crew import DocumentProcessingCrew
from form_filler.tools import DocumentExtractionTool, TranslationTool
from form_filler.utils.progress_utils import create_indeterminate_spinner

# Setup logging
logger = logging.getLogger(__name__)
console = Console()


def show_version(ctx: click.Context, param: click.Parameter, value: bool) -> None:
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
    "--provider",
    "-p",
    default="ollama",
    help="AI provider to use (ollama, openai, anthropic, deepseek, gemini)",
)
@click.option(
    "--text-model",
    "-m",
    default="llama3.2:3b",
    help="Model to use for text processing (translation, form filling)",
)
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    default="traditional",
    help="Text extraction method: traditional (PyMuPDF/Tesseract) or ai (vision models)",
)
@click.option(
    "--vision-model",
    "-vm",
    default="llava:7b",
    help="Vision model for AI extraction",
)
@click.option(
    "--api-key",
    envvar="AI_API_KEY",
    help="API key for the AI provider (if needed)",
)
@click.option(
    "--api-base",
    help="Base URL for the AI provider API (if needed)",
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
def cli(
    ctx: click.Context,
    verbose: bool,
    provider: str,
    text_model: str,
    extraction_method: str,
    vision_model: str,
    api_key: str | None,
    api_base: str | None,
) -> None:
    """Vietnamese to English Document Form Filler (CrewAI Edition).

    A CrewAI-based multi-agent system for processing Vietnamese documents (PDF/images)
    and filling English DOCX forms using various AI providers.

    All commands display progress bars or spinners during execution to provide
    visual feedback for long-running operations.

    Extraction Methods:
    - traditional: Use PyMuPDF for PDFs and Tesseract for images
    - ai: Use vision models for both PDFs and images (requires vision-capable models)

    Supported AI Providers:
    - ollama: Local Ollama models (default, no API key required)
    - openai: OpenAI API (requires API key)
    - anthropic: Anthropic Claude API (requires API key)
    - deepseek: DeepSeek API (requires API key)
    - gemini: Google Gemini API (requires API key)

    Note: For providers that require API keys, you can set the API_KEY environment
    variable instead of passing it as a command-line option.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["provider"] = provider
    ctx.obj["text_model"] = text_model
    ctx.obj["extraction_method"] = extraction_method
    ctx.obj["vision_model"] = vision_model
    ctx.obj["api_key"] = api_key
    ctx.obj["api_base"] = api_base

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("form", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--provider", "-p", help="Override default AI provider")
@click.option("--text-model", "-m", help="Override default text model")
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    help="Override default extraction method",
)
@click.option("--vision-model", "-vm", help="Override default vision model")
@click.option("--api-key", help="Override default API key")
@click.option("--api-base", help="Override default API base URL")
@click.pass_context
def process(
    ctx: click.Context,
    source: str | Path,
    form: str | Path,
    output: str | Path,
    provider: str | None,
    text_model: str | None,
    extraction_method: str | None,
    vision_model: str | None,
    api_key: str | None,
    api_base: str | None,
) -> None:
    """Process a Vietnamese document and fill an English DOCX form using CrewAI.

    SOURCE: Path to Vietnamese document (PDF or image)
    FORM: Path to English DOCX form template
    OUTPUT: Path where filled form will be saved

    Examples:
    # Using local AI extraction with Ollama:
    form-filler -e ai -vm llava:7b process document.pdf form.docx output.docx

    # Using OpenAI API:
    form-filler -p openai -m gpt-4 -e ai -vm gpt-4-vision-preview --api-key YOUR_API_KEY process document.pdf form.docx output.docx

    # Using Anthropic Claude:
    form-filler -p anthropic -m claude-3-sonnet-20240229 --api-key YOUR_API_KEY process document.pdf form.docx output.docx
    """
    provider = provider or ctx.obj["provider"]
    text_model = text_model or ctx.obj["text_model"]
    extraction_method = extraction_method or ctx.obj["extraction_method"]
    vision_model = vision_model or ctx.obj["vision_model"]
    api_key = api_key or ctx.obj["api_key"]
    api_base = api_base or ctx.obj["api_base"]

    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create CrewAI processor
    crew_processor = DocumentProcessingCrew(
        extraction_method=extraction_method,
        provider_name=provider,
        text_model=text_model,
        vision_model=vision_model,
        api_key=api_key,
        api_base=api_base,
    )

    # Process the document with progress bar
    with create_indeterminate_spinner("Processing document...") as progress:
        task_id = progress.add_task("Processing document...", total=None)

        start_time = time.time()
        # Convert Path to string if needed
        source_str = str(source) if hasattr(source, "__fspath__") else source
        form_str = str(form) if hasattr(form, "__fspath__") else form
        output_str = str(output_path)

        result = crew_processor.process_document(source_str, form_str, output_str)
        processing_time = time.time() - start_time

        # Complete the progress
        progress.update(task_id, completed=True)

    if result.success:
        console.print(f"[green]✅ Success![/green] Document processed in {processing_time:.2f}s")
        console.print(f"Filled form saved to: [blue]{output}[/blue]")
        if result.metadata:
            console.print(
                f"Provider: [yellow]{result.metadata.get('provider_name', 'N/A')}[/yellow]",
            )
            console.print(
                f"Extraction method: [yellow]{result.metadata.get('extraction_method', 'N/A')}[/yellow]",
            )
            console.print(
                f"Text model: [yellow]{result.metadata.get('text_model', 'N/A')}[/yellow]",
            )
            if result.metadata.get("vision_model"):
                console.print(
                    f"Vision model: [yellow]{result.metadata.get('vision_model')}[/yellow]",
                )
        if result.data and isinstance(result.data, dict):
            fields_filled = result.data.get("fields_filled", "N/A")
            console.print(f"Fields filled: [yellow]{fields_filled}[/yellow]")
    else:
        console.print(f"[red]❌ Error:[/red] {result.error}")
        sys.exit(1)


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--provider", "-p", help="Override default AI provider")
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    help="Override default extraction method",
)
@click.option("--vision-model", "-vm", help="Override default vision model")
@click.option("--api-key", help="Override default API key")
@click.option("--api-base", help="Override default API base URL")
@click.pass_context
def extract(
    ctx: click.Context,
    file_path: str | Path,
    provider: str | None,
    extraction_method: str | None,
    vision_model: str | None,
    api_key: str | None,
    api_base: str | None,
) -> None:
    """Extract text from a Vietnamese document (for testing).

    Examples:
    # Using local AI extraction with Ollama:
    form-filler -e ai -vm llava:7b extract document.pdf

    # Using OpenAI API for extraction:
    form-filler -p openai -e ai -vm gpt-4-vision-preview --api-key YOUR_API_KEY extract document.pdf
    """
    provider = provider or ctx.obj["provider"]
    extraction_method = extraction_method or ctx.obj["extraction_method"]
    vision_model = vision_model or ctx.obj["vision_model"]
    api_key = api_key or ctx.obj["api_key"]
    api_base = api_base or ctx.obj["api_base"]

    # Create extraction tool with the new API
    extractor = DocumentExtractionTool(
        extraction_method=extraction_method,
        provider_name=provider,
        model_name=vision_model,
        api_key=api_key,
        api_base=api_base,
    )

    try:
        console.print(f"Extracting text using [yellow]{extraction_method}[/yellow] method...")
        if extraction_method == "ai":
            console.print(f"Provider: [yellow]{provider}[/yellow]")
            console.print(f"Vision model: [yellow]{vision_model}[/yellow]")

        # Add progress spinner during extraction
        with create_indeterminate_spinner("Extracting text...") as progress:
            task_id = progress.add_task("Extracting text...", total=None)

            start_time = time.time()
            # Convert Path to string if needed
            file_path_str = str(file_path) if hasattr(file_path, "__fspath__") else file_path
            extracted_text = extractor._run(file_path_str)
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
@click.option("--provider", "-p", help="Override default AI provider")
@click.option("--model", "-m", help="Override default text model")
@click.option("--api-key", help="Override default API key")
@click.option("--api-base", help="Override default API base URL")
@click.pass_context
def translate(
    ctx: click.Context,
    vietnamese_text: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    api_base: str | None,
) -> None:
    """Translate Vietnamese text to English (for testing).

    Examples:
    # Using local Ollama:
    form-filler translate "Xin chào"

    # Using OpenAI:
    form-filler -p openai -m gpt-4 --api-key YOUR_API_KEY translate "Xin chào"
    """
    provider = provider or ctx.obj["provider"]
    model = model or ctx.obj["text_model"]
    api_key = api_key or ctx.obj["api_key"]
    api_base = api_base or ctx.obj["api_base"]

    translator = TranslationTool(
        provider_name=provider,
        model_name=model,
        api_key=api_key,
        api_base=api_base,
    )

    try:
        console.print(
            f"Translating with provider: [yellow]{provider}[/yellow], model: [yellow]{model}[/yellow]"
        )

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


@cli.command()
@click.option("--host", default="localhost", help="Ollama host")
@click.option("--port", default=11434, help="Ollama port")
@click.option("--check-vision", is_flag=True, help="Also check for vision models")
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
                            "  Vision models enable AI-powered text extraction from images and PDFs",
                        )
                        click.echo("  Supported models: llava:7b, llava:13b, bakllava")

                click.echo("\n🤖 CrewAI Integration Status: ✅ Ready")
                click.echo("Available providers: " + ", ".join(AIProvider.list_providers()))

            else:
                click.echo(f"❌ Ollama responded with status: {response.status}")
                sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error connecting to Ollama: {e}")
        click.echo(f"Make sure Ollama is running on {host}:{port}")
        sys.exit(1)


@cli.command()
def list_providers() -> None:
    """List all available AI providers."""
    providers = AIProvider.list_providers()
    click.echo("Available AI providers:")
    for provider in providers:
        click.echo(f"  - {provider}")


@cli.command()
def version() -> None:
    """Display the current version of the form-filler package."""

    # Create a dummy parameter for the show_version function
    class DummyParameter(click.Parameter):
        """Dummy parameter class for type compatibility."""

        def __init__(self) -> None:
            """Initialize with minimal required attributes."""
            self.name = "version"
            self.opts = ["--version"]

    # Use the same function that handles the --version flag
    show_version(click.get_current_context(), DummyParameter(), True)
