# DiskLLM Performance & Benchmarking Report

## Storage Bandwidth & Scaling Analysis

Testing conducted on Qwen3.8-27B-Q4_K_M (15.5 GB GGUF model) under Termux / Android on ARMv8.2-A with 4 computing threads.

### Benchmark Matrix

| Execution Mode | Threads | I/O Mode | Prefill Time (5 tok) | Decode Speed | Peak RSS | Memory Pressure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pread Streaming** | 1 | `pread` | 68.90 s | 16.15 s/tok | 1,637 MB | Zero (Safe) |
| **Pread Streaming** | 4 | `pread` | 17.83 s | 13.63 s/tok | 1,628 MB | Zero (Safe) |
| **Mmap Mode** | 4 | `mmap` | 16.50 s | 14.10 s/tok | 5,074 MB | High Thrashing |
| **State Restored (Turn 2)**| 4 | `pread` | **0.10 s** | 13.30 s/tok | 1,597 MB | Zero (Safe) |

### Key Findings

1. **Multithread Acceleration**: Increasing matvec compute threads from 1 to 4 accelerated prompt prefill from 68.9s down to 17.8s (**3.86x prefill speedup**).
2. **Storage Bandwidth Bound**: Single-token decode is storage-bandwidth-bound (~130 MB/s read speed), consuming ~11.8s of I/O wait time per 13.6s token step.
3. **Pread vs Mmap Efficiency**: `pread` mode maintains a strict ~1.6 GB RSS memory cap, while `mmap` forces operating system page-fault thrashing under mobile DRAM constraints.
4. **State Caching Impact**: Restoring multi-turn state eliminates prompt prefill disk reads, enabling prompt resumption in **102.23 ms**.
