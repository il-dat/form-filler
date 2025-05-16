#!/usr/bin/env python3
"""
Batch Processing CLI for Vietnamese Document Form Filler.

Implements commands for batch processing multiple documents.
"""

import logging

import click

from form_filler.utils import ensure_directory_exists, list_files_by_extension

# Setup logging
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--model", "-m", default="llama3.2:3b", help="Ollama model to use for text processing"
)
@click.option("--crews", "-c", default=2, help="Number of concurrent CrewAI teams to use")
@click.option(
    "--extraction-method",
    "-e",
    type=click.Choice(["traditional", "ai"]),
    default="traditional",
    help="Text extraction method",
)
@click.pass_context
def batch_cli(ctx, verbose, model, crews, extraction_method):
    """Batch Processing for Vietnamese Document Form Filler (CrewAI Edition).

    Process multiple Vietnamese documents in parallel using CrewAI crews
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["model"] = model
    ctx.obj["crews"] = crews
    ctx.obj["extraction_method"] = extraction_method

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@batch_cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("form_template", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.pass_context
def process_directory(ctx, input_dir, form_template, output_dir):
    """Process all documents in a directory.

    Example:
    form-filler-cli batch process-directory ./input_docs/ ./form_template.docx ./output_docs/
    """
    # This is a placeholder implementation. The full implementation would:
    # 1. List all documents in input_dir
    # 2. Create concurrent CrewAI crews
    # 3. Distribute documents across crews
    # 4. Process in parallel
    # 5. Save results to output_dir

    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # List PDF and image files
    pdf_files = list_files_by_extension(input_dir, ".pdf")
    image_files = []
    for ext in [".jpg", ".jpeg", ".png", ".tiff"]:
        image_files.extend(list_files_by_extension(input_dir, ext))

    all_files = pdf_files + image_files

    click.echo(f"Found {len(all_files)} documents to process")
    click.echo(f"Using {ctx.obj['crews']} concurrent CrewAI crews")
    click.echo(f"Using extraction method: {ctx.obj['extraction_method']}")
    click.echo(f"Using model: {ctx.obj['model']}")
    click.echo("Batch processing not fully implemented in this placeholder module")


@batch_cli.command()
@click.pass_context
def crew_status(ctx):
    """Show status of processing crews."""
    # Placeholder for showing status of current processing crews
    click.echo("Crew Status")
    click.echo("=" * 40)
    click.echo("Batch processing status view is not implemented in this placeholder module")
    click.echo("This would show active crews, documents being processed, and completion status")


def main():
    """Main entry point for the batch CLI."""
    batch_cli()
