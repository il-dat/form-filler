#!/usr/bin/env python3
"""
Entry point for Vietnamese Document Form Filler CLI.

Contains the main CLI entry point and async command helpers.
"""

import asyncio
import sys

import click

from form_filler.cli.cli_commands import cli


# Create a sync version of the async check-ollama command
@cli.command()
@click.option("--host", default="localhost", help="Ollama host")
@click.option("--port", default=11434, help="Ollama port")
@click.option("--check-vision", is_flag=True, help="Also check for vision models")
def check_ollama(host: str, port: int, check_vision: bool) -> None:
    """Check if Ollama is running and list available models.

    Connects to the Ollama server to verify it's running and displays all
    available models, separating text and vision models for convenience.

    Examples:
    # Basic check for Ollama:
    form-filler check-ollama

    # Check for vision models too:
    form-filler check-ollama --check-vision

    # Connect to remote Ollama:
    form-filler check-ollama --host remote-server --port 11434
    """
    from form_filler.cli.cli_commands import check_ollama as async_check_ollama

    asyncio.run(async_check_ollama(host, port, check_vision))


def main() -> None:
    """Main entry point for the CLI."""
    try:
        cli(standalone_mode=False)
    except click.exceptions.Abort:
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# This prevents the runtime warning when executing as python -m
if __name__ == "__main__":
    main()
