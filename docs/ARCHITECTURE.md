# DiskLLM Architecture Overview

DiskLLM is engineered from the ground up for low-memory environments where model weight tensors exceed total system RAM.

## System Topology & Pipeline

```
+-------------------------------------------------------------------+
|                            CLI Driver                             |
|                        (main/diskllm.c)                           |
+-------------------------------------------------------------------+
       |                                              |
       v                                              v
+------------------+                        +-------------------+
|  Native BPE      |                        |  State Caching    |
|  Tokenizer       |                        |  Engine           |
|  (src/tokenizer) |                        |  (src/state.c)    |
+------------------+                        +-------------------+
       |                                              |
       +-----------------------+----------------------+
                               |
                               v
               +----------------------------------+
               |   Double-Buffered Stream Context |
               |   (src/stream.c)                 |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               |  Hybrid Layer Forward Loop       |
               |  - 48 SSM / Mamba Layers         |
               |  - 16 Attention Layers           |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               |  Accelerated Kernels             |
               |  - ARM NEON Matvec               |
               |  - Quant Dequantization          |
               |  (src/matvec.c, src/dequant.c)   |
               +----------------------------------+
                               |
                               v
               +----------------------------------+
               |  Multi-Strategy Sampler          |
               |  (src/sampler.c)                 |
               +----------------------------------+
```

## Layer Execution & Streaming Strategy

1. **Layer Block Allocation**: Memory for weights is allocated as two ping-pong buffers (`buf_a`, `buf_b`) matching the size of a single layer block (~1.18 GB).
2. **Async Prefetching**: While worker threads compute layer $N$ in `buf_a`, a background `pthread` streams layer $N+1$ from disk into `buf_b`.
3. **SSM & Attention Hybrid**:
   - Layers $[0..47]$ execute SSM recurrent updates with 1D convolution state buffers.
   - Layers $[48..63]$ execute Multi-Head Attention with QK-Norm and partial RoPE embedding.
