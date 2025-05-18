#!/bin/bash
# Run the full-lint script manually

set -e

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the full-lint script
"$DIR/full-lint.sh"
