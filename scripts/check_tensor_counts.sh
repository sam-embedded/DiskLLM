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

echo "Using GGUF model file: $GGUF_FILE"

# Check if model file exists
if [ ! -f "$GGUF_FILE" ]; then
    echo "Error: GGUF file not found at $GGUF_FILE" >&2
    exit 1
fi

# Run gguf_dump and extract the tensor counts
DUMP_DIR="build"
DUMP_FILE="$DUMP_DIR/gguf_dump.txt"

# If dump file does not exist, run it now
if [ ! -f "$DUMP_FILE" ]; then
    echo "Dump file not found, running gguf_dump..."
    mkdir -p "$DUMP_DIR"
    ./gguf_dump "$GGUF_FILE" > "$DUMP_FILE"
fi

# Extract metadata tensor count and actual printed tensor count
EXPECTED_TENSORS=866

# Parse the GGUF Header tensor count
HEADER_TENSOR_COUNT=$(grep -m 1 "Tensor Count:" "$DUMP_FILE" | awk '{print $3}' || true)
# Parse the Summary total printed tensors
SUMMARY_TENSOR_COUNT=$(grep -m 1 "Total printed tensors:" "$DUMP_FILE" | awk '{print $4}' || true)

echo "Header Tensor Count: $HEADER_TENSOR_COUNT"
echo "Summary Tensor Count: $SUMMARY_TENSOR_COUNT"

if [ -z "$HEADER_TENSOR_COUNT" ] || [ -z "$SUMMARY_TENSOR_COUNT" ]; then
    echo "FAIL: Could not parse tensor counts from $DUMP_FILE" >&2
    exit 1
fi

if [ "$HEADER_TENSOR_COUNT" -ne "$EXPECTED_TENSORS" ] || [ "$SUMMARY_TENSOR_COUNT" -ne "$EXPECTED_TENSORS" ]; then
    echo "FAIL: Expected $EXPECTED_TENSORS tensors, but header has $HEADER_TENSOR_COUNT and summary has $SUMMARY_TENSOR_COUNT" >&2
    exit 1
fi

echo "SUCCESS: GGUF model file contains exactly $EXPECTED_TENSORS tensors!"
exit 0
