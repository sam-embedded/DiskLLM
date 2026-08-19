#!/usr/bin/env bash
set -e

MODEL_PATH="${MODEL_PATH:-/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf}"
if [ ! -f "$MODEL_PATH" ]; then
    MODEL_PATH="models/model.gguf"
fi
SCRATCH_DIR="scratch"
mkdir -p "$SCRATCH_DIR"

echo "========================================================"
echo "          DiskLLM v1.0 Final Validation Suite           "
echo "========================================================"

STAGE_FAILURES=0

run_stage() {
    STAGE_NAME="$1"
    CMD="$2"
    echo -n "[TEST] $STAGE_NAME ... "
    if eval "$CMD" > /dev/null 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        STAGE_FAILURES=$((STAGE_FAILURES + 1))
    fi
}

# 1. BUILD STAGE
run_stage "BUILD (Make clean & build)" "make clean && make"

# 2. UNIT TESTS
run_stage "UNIT_TESTS (Dequant, Kernels, SSM, Attention)" "./test_dequant && ./test_kernels && ./test_ssm && ./test_attention"

# 3. TOKENIZER TESTS
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf}"
run_stage "TOKENIZER_TESTS (BPE & ChatML encoding)" "./test_tokenizer '$TOKENIZER_MODEL'"

# 4. GREEDY GENERATION
run_stage "GREEDY_GENERATION (Autoregressive decode)" "./diskllm --model '$MODEL_PATH' --prompt 'The capital of France is' --max-tokens 4 --greedy --threads 4 --quiet"

# 5. CHAT GENERATION
run_stage "CHAT_GENERATION (ChatML auto-formatting)" "./diskllm --model '$MODEL_PATH' --chat --prompt 'Hello' --max-tokens 2 --threads 4 --quiet"

# 6. STATE CACHE
STATE1="$SCRATCH_DIR/chk_turn1.state"
STATE2="$SCRATCH_DIR/chk_turn2.state"
run_stage "STATE_CACHE_SAVE (Turn 1)" "./diskllm --model '$MODEL_PATH' --prompt 'Paris is' --max-tokens 2 --save-state '$STATE1' --quiet"
run_stage "STATE_CACHE_INFO (--state-info inspection)" "./diskllm --state-info '$STATE1' | grep -q 'DKST'"
run_stage "STATE_CACHE_LOAD (Turn 2 state restore)" "./diskllm --model '$MODEL_PATH' --load-state '$STATE1' --max-tokens 2 --save-state '$STATE2' --quiet"

echo "========================================================"
if [ $STAGE_FAILURES -eq 0 ]; then
    echo "            FINAL RESULT: ALL STAGES PASSED            "
    echo "========================================================"
    exit 0
else
    echo "            FINAL RESULT: $STAGE_FAILURES STAGE(S) FAILED          "
    echo "========================================================"
    exit 1
fi
