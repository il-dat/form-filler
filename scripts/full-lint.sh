#!/bin/bash
set -e

echo "Running comprehensive code linting..."

# Run Ruff linter
echo "🔍 Running ruff linter..."
ruff check --fix .

# Run Ruff formatter
echo "🔧 Running ruff formatter..."
ruff format .

# Run Bandit security checks
echo "🔒 Running bandit security checks..."
bandit -c pyproject.toml -r src/

# Check for debug statements
echo "🐛 Checking for debug statements..."
grep -r "import pdb" --include="*.py" src/ || true
grep -r "import ipdb" --include="*.py" src/ || true
grep -r "breakpoint()" --include="*.py" src/ || true

# Check for trailing whitespace
echo "⬜ Checking for trailing whitespace..."
find . -type f -name "*.py" -exec grep -l " $" {} \; || true

echo "✅ Linting complete"
