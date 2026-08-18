# DiskLLM v1.0

> **Pure C11 Disk-Streaming LLM Engine for Hybrid SSM/Attention Models (Qwen2.5 / Qwen3.5 27B GGUF)**

DiskLLM is a high-performance, lightweight LLM inference engine written in **pure C11** with zero ML framework dependencies. It enables executing giant hybrid models—like **Qwen 27B Q4_K_M**—on consumer mobile devices (e.g., Android / Termux) with low system RAM (~1.6 GB RSS) by dynamically streaming model weights from disk storage.

---

## Key Features

- **Pure C11 Engine**: Built completely from scratch using standard C11 (`pthreads`, POSIX APIs). Zero Python or heavy framework runtime.
- **Disk-Streaming Weight Loader**: Double-buffered async prefetching keeps peak RAM consumption fixed at **~1.6 GB**, allowing execution of 16+ GB model files.
- **Hybrid SSM / Attention Architecture**: Full native support for Qwen2.5/Qwen3.5 hybrid architectures (64 layers: 48 Mamba/SSM layers + 16 Multi-Head Attention layers with QK-Norm & RoPE).
- **ARM NEON Acceleration**: Hand-optimized NEON vector kernels (`matvec` dot-products, dequantization for Q4_K, Q5_K, Q6_K, Q8_0).
- **Multithreaded Compute**: Parallel matrix-vector row processing scaling across CPU cores.
- **Native BPE Tokenizer**: Built-in pure C BPE tokenizer directly reading GGUF vocabulary and merge ranks with ChatML template support (`--chat`, `--system`).
- **Multi-Turn State Caching**: Save and restore compact binary model states (~165 MB) via `--save-state` and `--load-state` to skip prompt prefill disk reads (168x prefill speedup).

---

## Hardware Reality & Storage Bandwidth Boundedness

In a low-RAM environment where system memory is smaller than the model file size, model weight pages cannot remain cached in RAM across autoregressive decode steps. Consequently, weights must be read from flash storage on every generated token.

The theoretical single-token decode floor is bounded by storage bandwidth:

$$\text{Time per Token (s)} \ge \frac{\text{Model Weights Read per Step (GB)}}{\text{Storage Bandwidth (GB/s)}}$$

For a **15.5 GB** model on flash storage with **~130 MB/s** throughput:

$$\text{Time per Token} \ge \frac{1.55 \text{ GB}}{0.13 \text{ GB/s}} \approx 12-14 \text{ seconds/token}$$

---

## Build Instructions

### Prerequisites
- C11 compiler (`clang` or `gcc`)
- `make` or `cmake` (v3.10+)
- `pthreads` library (standard on Linux/Android Termux)

### Building with Make
```bash
make clean
make -j4
```

### Building with CMake
```bash
mkdir -p build && cd build
cmake ..
make -j4
```

---

## Quick Start & Usage Examples

### 1. Basic Text Completion
```bash
./diskllm \
    --model model.gguf \
    --prompt "The capital of France is" \
    --max-tokens 16 \
    --threads 4
```

### 2. Quiet Mode (Generated Text Only)
```bash
./diskllm \
    --model model.gguf \
    --chat \
    --system "You are a helpful AI assistant." \
    --prompt "What is quantum computing?" \
    --quiet
```

### 3. Multi-Turn State Caching (Skip Prefill Reads)
```bash
# Turn 1: Save state after initial prompt
./diskllm \
    --model model.gguf \
    --prompt "What is Paris?" \
    --save-state turn1.state

# Turn 2: Restore state and continue conversation (Prefill takes < 0.1s!)
./diskllm \
    --model model.gguf \
    --load-state turn1.state \
    --prompt "Tell me about its landmarks." \
    --save-state turn2.state
```

### 4. State File Metadata Inspection
```bash
./diskllm --state-info turn1.state
```

---

## System Architecture

DiskLLM is organized into modular components in `src/` and `include/`:

- **`tensor_catalog`**: GGUF v3 header and metadata parser.
- **`stream`**: Async double-buffered weight prefetcher using `pread` / `mmap`.
- **`tokenizer`**: Native GPT-2 style BPE encoder and UTF-8 decoder.
- **`matvec`**: ARM NEON dot-product matrix-vector multiply kernels.
- **`attention`**: Rotary Position Embedding (RoPE) & QK-Norm MHA layer.
- **`ssm`**: Selective State Space Model (Mamba) recurrence & 1D conv layer.
- **`sampler`**: Temperature, Top-K, Top-P, Min-P, and repetition penalty sampling.
- **`state`**: Serialization and loading of persistent KV/SSM state files.

---

## Documentation

Detailed technical documentation is available in `docs/`:
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Performance & Storage Profiling](docs/PERFORMANCE.md)
- [State Cache Specification](docs/STATE_CACHE.md)
- [CLI Reference](docs/CLI.md)
- [Release Report](docs/RELEASE_REPORT.md)

---

## License

MIT License. Free for open-source research and production use.
