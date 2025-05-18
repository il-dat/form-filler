#!/bin/bash
# Script to create a custom gitleaks configuration within pyproject.toml

PYPROJECT_FILE="pyproject.toml"
FORCE_OVERWRITE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --force|-f)
      FORCE_OVERWRITE=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Check if pyproject.toml exists
if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "Error: $PYPROJECT_FILE not found"
    exit 1
fi

# Check if [tool.gitleaks] section already exists
if grep -q "\[tool\.gitleaks\]" "$PYPROJECT_FILE" && [ "$FORCE_OVERWRITE" = false ]; then
    echo "Gitleaks config section already exists in $PYPROJECT_FILE"
    echo "Edit this section to customize secret detection rules"
    echo "Use --force or -f to overwrite"
    exit 0
fi

# Creating a temporary file for the updated content
TMP_FILE=$(mktemp)

# If force overwrite is true, remove existing gitleaks section
if [ "$FORCE_OVERWRITE" = true ]; then
    # Remove existing [tool.gitleaks] section and its content
    sed '/\[tool\.gitleaks\]/,/^\[tool\./{ /^\[tool\./!d; }' "$PYPROJECT_FILE" > "$TMP_FILE"
    cp "$TMP_FILE" "$PYPROJECT_FILE"
fi

# Append gitleaks configuration to pyproject.toml
cat >> "$PYPROJECT_FILE" << 'EOF'

[tool.gitleaks]
title = "Form-Filler Gitleaks Configuration"

# This is in addition to the default ruleset provided by gitleaks

[[tool.gitleaks.rules]]
description = "Generic API Key"
regex = '''(?i)(['"]?)?([a-zA-Z0-9_-]+)_?(key|api|token|secret|passwd|password|auth)(['"]?)?[[:blank:]]*[:=>][[:blank:]]*(['"])?[a-zA-Z0-9_=+\/.]{16,45}(['"])?'''
tags = ["key", "API", "generic"]

[[tool.gitleaks.rules.allowlist]]
regexes = ['''example|test|fake|sample''']

[[tool.gitleaks.rules]]
description = "Generic Secret"
regex = '''(?i)['"]?[a-zA-Z0-9_-]*(secret|token|key|passwd|password|credential)s?['"]?[[:blank:]]*[:=>][[:blank:]]*['"][a-zA-Z0-9_=+\/.\-]{8,64}['"]'''
tags = ["key", "secret", "generic"]

[[tool.gitleaks.rules.allowlist]]
regexes = ['''example|test|fake|sample''']

[[tool.gitleaks.rules]]
description = "Database Connection String"
regex = '''(?i)(mongodb|postgresql|mysql|jdbc|redis|ldap):\/\/[^\s:]+:[^\s@]+@[^\s:]+:[0-9]+'''
tags = ["database", "connection"]

[[tool.gitleaks.rules]]
description = "Crypto Seed/Salt"
regex = '''(?i)['"]?seed['"]?[[:blank:]]*[:=>][[:blank:]]*['"][a-f0-9]{64}['"]'''
tags = ["crypto", "seed"]

[[tool.gitleaks.rules]]
description = "Environment Variable Assignment With Secret"
regex = '''(?i)(export|set)[[:blank:]]+[a-zA-Z0-9_]*(key|api|token|secret|passwd|password|auth)[a-zA-Z0-9_]*[[:blank:]]*=[[:blank:]]*['"].{8,64}['"]'''
tags = ["env", "secret"]

# Allow lists to prevent false positives
[tool.gitleaks.allowlist]
description = "Allow list to exclude common patterns"
paths = [
    '''(.*?)(test|example|sample|mock)(.*?)\.py''',
    '''.*pytest_cache.*''',
    '''.*\.git.*'''
]
EOF

echo "Added gitleaks custom configuration to $PYPROJECT_FILE"
echo "You can now edit this file to customize your secret detection rules"
echo "After editing, you may need to run 'pre-commit install' again to use the new config"

# If standalone .gitleaks.toml exists, suggest to remove it
if [ -f ".gitleaks.toml" ]; then
    echo ""
    echo "NOTE: A standalone .gitleaks.toml file was detected."
    echo "You may want to remove it to avoid configuration conflicts:"
    echo "  rm .gitleaks.toml"
fi

echo "Done!"
