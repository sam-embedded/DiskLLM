#!/usr/bin/env python3
"""
DiskLLM Validation Harness — Phase 14
Runs a suite of golden-output tests against the compiled diskllm binary.
Usage:
    python3 tests/run_golden_tests.py [--binary ./diskllm] [--model PATH] [--fast]
"""

import argparse
import subprocess
import sys
import os
import json
import time

MODEL_DEFAULT = "/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf"
BINARY_DEFAULT = "./diskllm"

# Each test case is:
#   name        - human label
#   args        - CLI args appended after --model and --greedy
#   expect_ids  - if set, first N generated token IDs must match
#   expect_text - if set, decoded generated text must contain this substring
#   expect_exit - expected exit code (default 0)
TESTS = [
    # --- Vocabulary lookup tests (fast, no inference) ---
    {
        "name": "lookup-id-17",
        "args": ["--lookup-id", "17"],
        "expect_stdout_contains": '"2"',
        "expect_exit": 0,
    },
    {
        "name": "lookup-id-out-of-range",
        "args": ["--lookup-id", "9999999"],
        "expect_exit": 1,
    },
    {
        "name": "lookup-ids-batch",
        "args": ["--lookup-ids", "15,16,17,18"],
        "expect_stdout_contains": '"0"',
        "expect_exit": 0,
    },
    {
        "name": "help-flag",
        "args": ["--help"],
        "expect_stdout_contains": "DiskLLM",
        "expect_exit": 0,
    },
    # --- Inference tests (slow, skipped with --fast) ---
    {
        "name": "prompt-capital-france-1tok",
        "slow": True,
        "args": [
            "--prompt-ids-file", "tests/prompts/real_prompt_01.ids",
            "--max-tokens", "1",
            "--greedy",
            "--decode-output",
        ],
        # After the bug-fix, first generated token for "The capital of France is"
        # should be Paris-related (not a digit).
        # We check that generated text is non-empty and doesn't start with a digit.
        "check_fn": lambda out: (
            "Generated text" in out and
            not out.split('Generated text')[1].strip().startswith(': "2')
        ),
        "expect_exit": 0,
    },
    {
        "name": "prompt-synthetic-ids-1tok",
        "slow": True,
        "args": [
            "--prompt-ids", "9826,382,1024",
            "--max-tokens", "1",
            "--greedy",
            "--logits-summary",
        ],
        "expect_stdout_contains": "[LOGITS-SUMMARY]",
        "expect_exit": 0,
    },
    {
        "name": "sampler-temperature",
        "slow": True,
        "args": [
            "--prompt-ids", "760,6511,314,9338,369",
            "--max-tokens", "4",
            "--temperature", "0.8",
            "--seed", "42",
        ],
        "expect_stdout_contains": "Token count",
        "expect_exit": 0,
    },
    {
        "name": "chat-capital-france",
        "slow": False, # Executed in both modes, using max-tokens 1 in fast mode and 16 in full mode
        "args": [
            "--prompt-ids-file", "tests/prompts/chat_capital_france.ids",
            "--greedy",
            "--decode-output",
            "--print-top-k", "8",
            "--logits-summary",
            "--stop-token", "248046",
        ],
        "expect_stdout_contains": "[GREEDY]",
        "expect_exit": 0,
    },
]


def run_test(binary, model, test, verbose=False):
    name = test["name"]
    args = [binary, "--model", model] + test.get("args", [])
    t0 = time.time()
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", f"Test '{name}' timed out"
    elapsed = time.time() - t0

    out = result.stdout + result.stderr
    expected_exit = test.get("expect_exit", 0)

    if result.returncode != expected_exit:
        return "FAIL", f"exit code {result.returncode} != {expected_exit}\nOutput:\n{out[:500]}"

    if "expect_stdout_contains" in test:
        needle = test["expect_stdout_contains"]
        if needle not in out:
            return "FAIL", f"stdout missing '{needle}'\nGot:\n{out[:500]}"

    if "check_fn" in test:
        try:
            ok = test["check_fn"](out)
        except Exception as e:
            return "FAIL", f"check_fn raised: {e}"
        if not ok:
            return "FAIL", f"check_fn returned False\nOutput:\n{out[:800]}"

    if verbose:
        print(f"  Output (first 400 chars):\n{out[:400]}")

    return "PASS", f"{elapsed:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=BINARY_DEFAULT)
    ap.add_argument("--model",  default=MODEL_DEFAULT)
    ap.add_argument("--fast",   action="store_true", help="Skip slow inference tests")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        print(f"ERROR: binary not found: {args.binary}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}")
        sys.exit(1)

    print(f"DiskLLM Validation Harness")
    print(f"Binary : {args.binary}")
    print(f"Model  : {args.model}")
    print(f"Mode   : {'fast (no inference)' if args.fast else 'full'}\n")

    passed = failed = skipped = 0

    for test in TESTS:
        name = test["name"]
        is_slow = test.get("slow", False)

        if is_slow and args.fast:
            print(f"  SKIP  {name}  (slow, --fast mode)")
            skipped += 1
            continue

        # Dynamic max-tokens for chat-capital-france
        test_copy = dict(test)
        if name == "chat-capital-france":
            max_tok = "1" if args.fast else "16"
            test_copy["args"] = ["--max-tokens", max_tok] + test.get("args", [])

        print(f"  RUN   {name} ... ", end="", flush=True)
        status, detail = run_test(args.binary, args.model, test_copy, verbose=args.verbose)
        print(f"{status}  ({detail})")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    total = passed + failed + skipped
    print(f"\nResults: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
