# DiskLLM v3.0.0 — Modular Library, OpenAI Server & Multi-Arch Support

🚀 **DiskLLM v3.0.0 is officially released!**

This milestone release transforms DiskLLM from a single-file C prototype into a professional, modular C11 library (`libdiskllm`), CLI tool (`diskllm-cli`), and an OpenAI-compatible HTTP server (`diskllm-server`) capable of running local LLMs on mobile devices, SBCs, and embedded hardware with **sub-1GB RAM footprint**.

---

## 🌟 What's New in v3.0

### 1. Modular C Library Architecture (`libdiskllm`)
- **Public C API Header** (`include/diskllm.h`): Opaque data types (`diskllm_model`, `diskllm_context`, `diskllm_sampler`, `diskllm_tokenizer`) and simple C function calls for integration into third-party applications.
- **Extensible Architecture Registry** (`include/arch/registry.h`): Replaced monolithic conditional blocks with clean, decoupled architecture drivers (`diskllm_arch_backend`).

### 2. Multi-Architecture LLM Engine
DiskLLM now natively supports 5 major LLM architecture families:
- **Qwen 2.5 / 3.5**: Hybrid SSM/Mamba + Multi-Head Attention with QK-Norm.
- **Llama 3 / 3.2**: Grouped Query Attention (GQA), SwiGLU, RMSNorm.
- **Microsoft Phi-3**: Fused QKV projection tensor, packed SwiGLU FFN, and custom RoPE.
- **Mistral**: Sliding Window Attention, GQA.
- **Gemma / Gemma 2**: GeGLU activations and RMSNorm $+1$.

### 3. OpenAI-Compatible HTTP Server (`diskllm-server`)
- Pure C POSIX HTTP socket server (`socket`, `bind`, `listen`, `accept`).
- Endpoint `POST /v1/chat/completions` with **Server-Sent Events (SSE)** real-time streaming (`data: {"choices": [...]}`).
- Native integration with **Open WebUI**, `curl`, LangChain, and standard OpenAI SDKs.

### 4. RAM Weight Pinning (`--pin-weights`)
- Option to memory-map and lock weights into physical RAM (`mlock`), completely bypassing disk streaming during decode steps for small models (e.g. Llama-3.2-1B).
- Achieves **zero decode I/O** and **300+ ms/tok** generation speeds.

### 5. Quickstart Script (`run-server.sh`)
- Zero-config launcher script that automatically discovers GGUF models in `./models` or `~/models` and starts the OpenAI server on port `8080`.

---

## 🚀 Quickstart

```bash
# Clone and build
git clone https://github.com/sam-embedded/DiskLLM.git
cd DiskLLM
make -j$(nproc)

# Quickstart server launch
./run-server.sh
```

### Test OpenAI API with `curl`
```bash
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "diskllm",
    "messages": [{"role": "user", "content": "What are the 3 main laws of robotics?"}],
    "stream": true
  }'
```

---

## 📊 Summary of Journey

From a minimal experimental script testing disk streaming for Qwen 27B on Termux Android, DiskLLM has evolved into a complete open-source LLM inference ecosystem featuring ARM NEON kernels, double-buffered I/O, binary KV state persistence, multi-architecture backends, a clean C API, and a production-ready HTTP server.
