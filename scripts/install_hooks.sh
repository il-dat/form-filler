#!/bin/bash
# Script to install pre-commit hooks in the current uv virtualenv

echo "Installing dev dependencies with uv..."
uv pip install -e ".[dev]"

echo "Installing pre-commit hooks..."
uv run pre-commit install --install-hooks

# Ensure gitleaks config exists (required for pre-commit hook)
if [ ! -f ".gitleaks.toml" ]; then
    echo "Setting up gitleaks config file..."
    ./scripts/setup_gitleaks_config.sh
fi

echo "Running pre-commit hooks on all files..."
uv run pre-commit run --all-files

echo "Done!"
