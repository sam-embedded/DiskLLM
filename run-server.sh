#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "          DiskLLM v3.0 Quickstart Server Launcher        "
echo "========================================================"

# 1. Build server if binary doesn't exist
if [ ! -f "diskllm-server" ]; then
    echo "[INFO] diskllm-server binary not found. Building with make..."
    make -j$(nproc)
fi

# 2. Search for GGUF model files
SEARCH_PATHS=(
    "$SCRIPT_DIR/models"
    "$HOME/models"
    "$SCRIPT_DIR"
)

FOUND_MODEL=""

for search_dir in "${SEARCH_PATHS[@]}"; do
    if [ -d "$search_dir" ]; then
        GGUF_FILE=$(find "$search_dir" -maxdepth 2 -type f -name "*.gguf" | head -n 1)
        if [ -n "$GGUF_FILE" ]; then
            FOUND_MODEL="$GGUF_FILE"
            break
        fi
    fi
done

if [ -z "$FOUND_MODEL" ]; then
    echo "[ERROR] No .gguf model file found in ./models, ~/models, or current directory."
    echo "Please place a GGUF model file in ./models or specify --model path manually:"
    echo "  ./diskllm-server --model /path/to/model.gguf --port 8080"
    exit 1
fi

PORT="${PORT:-8080}"
echo "[INFO] Found Model : $FOUND_MODEL"
echo "[INFO] Server Port : $PORT"
echo "========================================================"
echo "Starting DiskLLM OpenAI Compatible API Server..."
echo "  Connect Open WebUI to: http://localhost:$PORT/v1"
echo "========================================================"

exec ./diskllm-server --model "$FOUND_MODEL" --port "$PORT" --pin-weights
