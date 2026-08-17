#!/usr/bin/env bash

# Exit immediately if any command fails
set -euo pipefail

CONTEXT_LEN=${1:-8192}

# Build if probe_model is not compiled
if [ ! -f "./probe_model" ]; then
    echo "probe_model not found, building..."
    make probe_model
fi

echo "Running memory allocation probe for context length: $CONTEXT_LEN"
./probe_model "$CONTEXT_LEN"
