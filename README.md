# DiskLLM v3.0

> **Modular, High-Performance Pure C11 Inference Engine & OpenAI-Compatible Server for Multi-Architecture LLMs (Qwen, Llama, Mistral, Gemma, Phi-3)**

DiskLLM is a ultra-lightweight, high-performance LLM inference library (`libdiskllm`), CLI tool (`diskllm-cli`), and OpenAI-compatible HTTP server (`diskllm-server`) written in **pure C11** with zero external ML framework dependencies.

It enables executing large language models—ranging from **1B to 27B+ parameters**—on consumer mobile devices (e.g. Android / Termux) and embedded hardware with **sub-1GB RAM footprint** by dynamically streaming model weights directly from disk storage.

---

## What's New in v3.0

- **Modular C Library (`libdiskllm`)**: Monolithic code refactored into a clean public API (`include/diskllm.h`), internal core engine (`src/core/`), and architecture registry (`include/arch/registry.h`).
- **OpenAI-Compatible HTTP Server (`diskllm-server`)**: Pure C POSIX HTTP socket server supporting `POST /v1/chat/completions` with real-time Server-Sent Events (SSE) streaming (`data: ...`, `data: [DONE]`).
- **Multi-Architecture Support**: Native C forward drivers for **Qwen** (hybrid SSM/Attention), **Llama 3.2**, **Mistral**, **Gemma**, and **Microsoft Phi-3** (fused QKV & packed SwiGLU).
- **RAM Weight Pinning (`--pin-weights`)**: Zero-I/O decode execution for small models by memory-mapping and locking weights in RAM (`mlock`).
- **Sub-1GB Disk Streaming**: Fixed ~600MB–1.6GB RAM footprint for running 27B models on constrained devices.

---

## Key Features

- **Pure C11 Engine**: Built completely from scratch using standard C11 (`pthreads`, POSIX sockets). Zero Python, PyTorch, or heavy framework runtimes.
- **OpenAI-API Compliance**: Drop-in compatible with **Open WebUI**, `curl`, LangChain, and standard OpenAI clients.
- **Disk-Streaming Weight Loader**: Double-buffered async prefetching (`pread` / `mmap` / `io_uring`) keeps peak RAM consumption fixed regardless of model size.
- **ARM NEON Acceleration**: Hand-optimized NEON vector kernels (`matvec` dot-products, dequantization for Q4_K, Q5_K, Q6_K, Q8_0).
- **Native Tokenizers**: Pure C tokenizers with direct GGUF metadata parsing, BPE/SentencePiece encoding, and ChatML/Llama/Phi-3 template auto-formatting.
- **Multi-Turn State Caching**: Save and restore compact binary model states (`.state`) to skip prompt prefill disk reads (100x+ prefill speedup).

---

## Hardware Reality & Storage Bandwidth Boundedness

In a low-RAM environment where system memory is smaller than the model file size, model weight pages cannot remain cached in RAM across autoregressive decode steps. Consequently, weights are streamed from flash storage on every generated token.

The theoretical single-token decode floor is bounded by storage bandwidth:

$$\text{Time per Token (s)} \ge \frac{\text{Model Weights Read per Step (GB)}}{\text{Storage Bandwidth (GB/s)}}$$

For small models (e.g. Llama 3.2 1B), `--pin-weights` locks weights into physical RAM for zero decode I/O ($330\text{ ms/tok}$).

---

## Build Instructions

### Prerequisites
- C11 compiler (`clang` or `gcc`)
- `make` or `cmake` (v3.10+)
- `pthreads` library (standard on Linux / Android Termux)

### Building with Make
```bash
make clean
make -j$(nproc)
```
This produces:
- `diskllm-cli` (and backwards-compatible `diskllm` binary)
- `diskllm-server`
- Unit test binaries (`test_dequant`, `test_kernels`, `test_attention`, `test_ssm`, `test_tokenizer`)

### Building with CMake
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

---

## Quick Start Guide

### 1. Run OpenAI API Server (`diskllm-server`)
```bash
# Auto-detect model and launch server on port 8080
./run-server.sh

# Or launch manually for a specific model
./diskllm-server --model ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8080 --pin-weights
```

### 2. Test Server with `curl` (SSE Streaming)
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "diskllm",
    "messages": [{"role": "user", "content": "What are the 3 main laws of robotics?"}],
    "stream": true
  }'
```

### 3. CLI Mode (`diskllm-cli`)
```bash
# Text Generation
./diskllm-cli --model ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --prompt "The capital of France is"

# Chat Mode
./diskllm-cli --model ~/models/Phi-3-mini-4k-instruct-Q4_K_M.gguf --chat --prompt "Hello!" --pin-weights
```

---

## How to Connect to Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) is a popular ChatGPT-style UI for local LLMs. You can connect Open WebUI to `diskllm-server` in seconds:

1. **Start `diskllm-server`**:
   ```bash
   ./diskllm-server --model ~/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8080 --pin-weights
   ```

2. **Configure Open WebUI Connection**:
   - Open WebUI Settings $\rightarrow$ **Admin Settings** $\rightarrow$ **Connections**.
   - Under **OpenAI API Connections**:
     - **API URL**: `http://localhost:8080/v1` (or `http://<your-device-ip>:8080/v1`)
     - **API Key**: `sk-diskllm` (any non-empty string)
   - Click **Save**.

3. **Start Chatting**:
   - Select `diskllm` from the model dropdown in Open WebUI.
   - Enjoy local streaming inference powered by DiskLLM!

---

## Multi-Architecture Support Matrix

| Architecture | GGUF Arch String | Models Tested | Key Features |
| :--- | :--- | :--- | :--- |
| **Qwen 2.5 / 3.5** | `qwen2`, `qwen35` | Qwen 27B, 3B, 0.5B | Hybrid Mamba/SSM + Attention, QK-Norm |
| **Llama 3 / 3.2** | `llama` | Llama-3.2-1B-Instruct | GQA, SwiGLU, RMSNorm |
| **Microsoft Phi-3** | `phi3` | Phi-3-mini-4k-instruct | Fused QKV tensor, packed SwiGLU |
| **Mistral** | `mistral` | Mistral-7B-v0.1 | Sliding Window Attention, GQA |
| **Gemma / Gemma 2**| `gemma`, `gemma2` | Gemma-2B-it | GeGLU activations, RMSNorm $+1$ |

---

## System Architecture

DiskLLM v3.0 is organized into modular layers in `include/` and `src/`:

- **`diskllm.h`**: Clean public C API header for `libdiskllm`.
- **`src/core/model.c`**: Model loading, GGUF metadata parsing, and RAM weight-pinning.
- **`src/core/context.c`**: KV cache allocation, layer block prefetching, and prefill/decode loops.
- **`src/core/arch/registry.c`**: Extensible architecture backend lookup table (`diskllm_arch_backend`).
- **`src/server/main.c`**: Pure C POSIX HTTP SSE server with `cJSON` request parsing.
- **`src/cli/main.c`**: Feature-complete CLI frontend with terminal streaming & execution profiling.

---

## License

MIT License. Free for open-source research, personal, and production use.
