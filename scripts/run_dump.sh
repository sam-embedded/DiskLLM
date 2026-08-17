#!/usr/bin/env bash

# Exit immediately if any command fails
set -euo pipefail

# Determine GGUF file path
GGUF_FILE=""
if [ $# -gt 0 ]; then
    GGUF_FILE="$1"
elif [ -n "${MODEL_GGUF:-}" ]; then
    GGUF_FILE="$MODEL_GGUF"
else
    GGUF_FILE="/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf"
fi

echo "Running gguf_dump against $GGUF_FILE"
mkdir -p build
./gguf_dump "$GGUF_FILE" > build/gguf_dump.txt
echo "Dump saved to build/gguf_dump.txt"
