#!/bin/bash
# Script to install pre-commit hooks in the current uv virtualenv

# Get the current Python executable path
PYTHON_EXEC=$(which python)

echo "Using Python at: $PYTHON_EXEC"

# Install pre-commit if not already installed
if ! $PYTHON_EXEC -m pip list | grep pre-commit >/dev/null; then
    echo "Installing pre-commit..."
    $PYTHON_EXEC -m pip install pre-commit
else
    echo "pre-commit already installed"
fi

# Install the hooks
echo "Installing pre-commit hooks..."
$PYTHON_EXEC -m pre_commit install

# Ensure gitleaks config exists (required for pre-commit hook)
if [ ! -f ".gitleaks.toml" ]; then
    echo "Setting up gitleaks config file..."
    ./scripts/setup_gitleaks_config.sh
fi

echo "Running pre-commit hooks on all files..."
$PYTHON_EXEC -m pre_commit run --all-files

echo "Done!"
