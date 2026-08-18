# DiskLLM Command-Line Reference

## Synopses

```bash
diskllm --model <MODEL_PATH.gguf> [options]
diskllm --state-info <STATE_FILE.state>
diskllm --version
diskllm --help
```

## Options

### Input Options
- `--model <PATH>`: Path to GGUF model file.
- `--prompt "text"`: UTF-8 input prompt.
- `--system "text"`: System prompt (used when `--chat` is enabled).
- `--chat`: Formats prompt into ChatML `<|im_start|>...<|im_end|>` tags.
- `--prompt-ids "id1,id2"`: Comma-separated list of prompt token IDs.
- `--prompt-ids-file <FILE>`: File containing token IDs.

### Generation & Sampling
- `--max-tokens N`: Maximum generated token count (default: 16).
- `--threads N, -t N`: Number of computing worker threads (default: 4).
- `--greedy`: Sets temperature to 0.0 for deterministic output.
- `--temp T`: Temperature sampling parameter.
- `--top-k K`: Top-K candidate filtering.
- `--top-p P`: Top-P nucleus sampling.
- `--min-p P`: Min-P sampling threshold.
- `--repeat-penalty R`: Repetition penalty factor (default: 1.0).
- `--stop-token ID`: Additional custom stop token ID.
- `--show-special-tokens`: Print special tokens during decoding.

### State Persistence
- `--save-state FILE`: Writes model state binary file after generation.
- `--load-state FILE`: Loads model state binary file (skips prefill disk reads).
- `--state-info FILE`: Inspects state header metadata and exits immediately.

### Diagnostic & I/O
- `--quiet`: Suppress informational logging to stderr; print ONLY generated text to stdout.
- `--io-mode <pread|mmap>`: Weight access mode (default: `pread`).
- `--warm-cache`: Pre-warms OS page cache before running inference.
- `--profile-decode`: Prints per-token decode timing breakdown.
- `--log-rss`: Prints process resident set memory size.
- `--version`: Prints `DiskLLM v1.0.0` and exits.
