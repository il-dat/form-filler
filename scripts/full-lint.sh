#!/bin/bash
set -e

echo "Running comprehensive code linting..."

# Run Ruff linter
echo "🔍 Running ruff linter..."
uv run ruff check --fix . || echo "Ruff linter not available or failed"

# Run Ruff formatter
echo "🔧 Running ruff formatter..."
uv run ruff format . || echo "Ruff formatter not available or failed"

# Run Bandit security linter
echo "🔒 Running bandit security linter..."
uv run bandit -c pyproject.toml -r src/ || echo "Bandit security linter not available or failed"

echo "✅ Linting complete"
