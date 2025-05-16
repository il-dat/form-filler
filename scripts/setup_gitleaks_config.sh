#!/bin/bash
# Script to create a custom gitleaks configuration for specific secret patterns

CONFIG_FILE=".gitleaks.toml"
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

if [ -f "$CONFIG_FILE" ] && [ "$FORCE_OVERWRITE" = false ]; then
    echo "Gitleaks config file already exists at $CONFIG_FILE"
    echo "Edit this file to customize secret detection rules"
    echo "Use --force or -f to overwrite"
    exit 0
fi

cat > "$CONFIG_FILE" << 'EOF'
# Gitleaks configuration
# See https://github.com/zricethezav/gitleaks for documentation

title = "Form-Filler Gitleaks Configuration"

# This is in addition to the default ruleset provided by gitleaks

[[rules]]
description = "Generic API Key"
regex = '''(?i)(['"]?)?([a-zA-Z0-9_-]+)_?(key|api|token|secret|passwd|password|auth)(['"]?)?[[:blank:]]*[:=>][[:blank:]]*(['"])?[a-zA-Z0-9_=+\/.]{16,45}(['"])?'''
tags = ["key", "API", "generic"]
[rules.allowlist]
regexes = ['''example|test|fake|sample''']

[[rules]]
description = "Generic Secret"
regex = '''(?i)['"]?[a-zA-Z0-9_-]*(secret|token|key|passwd|password|credential)s?['"]?[[:blank:]]*[:=>][[:blank:]]*['"][a-zA-Z0-9_=+\/.\-]{8,64}['"]'''
tags = ["key", "secret", "generic"]
[rules.allowlist]
regexes = ['''example|test|fake|sample''']

[[rules]]
description = "Database Connection String"
regex = '''(?i)(mongodb|postgresql|mysql|jdbc|redis|ldap):\/\/[^\s:]+:[^\s@]+@[^\s:]+:[0-9]+'''
tags = ["database", "connection"]

[[rules]]
description = "Crypto Seed/Salt"
regex = '''(?i)['"]?seed['"]?[[:blank:]]*[:=>][[:blank:]]*['"][a-f0-9]{64}['"]'''
tags = ["crypto", "seed"]

[[rules]]
description = "Environment Variable Assignment With Secret"
regex = '''(?i)(export|set)[[:blank:]]+[a-zA-Z0-9_]*(key|api|token|secret|passwd|password|auth)[a-zA-Z0-9_]*[[:blank:]]*=[[:blank:]]*['"].{8,64}['"]'''
tags = ["env", "secret"]

# Allow lists to prevent false positives
[allowlist]
description = "Allow list to exclude common patterns"
paths = [
    '''(.*?)(test|example|sample|mock)(.*?)\.py''',
    '''.*pytest_cache.*''',
    '''.*\.git.*'''
]
EOF

echo "Created gitleaks custom configuration at $CONFIG_FILE"
echo "You can now edit this file to customize your secret detection rules"
echo "After editing, you may need to run 'pre-commit install' again to use the new config"

# Make file executable
chmod +x "$CONFIG_FILE"

echo "Done!"
