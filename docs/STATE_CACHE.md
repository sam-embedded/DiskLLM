# DiskLLM Multi-Turn State Cache Specification

## File Format & Structure

The DiskLLM state file (`.state`) stores the persistent activation buffers required to resume autoregressive generation without re-reading the model backbone during prompt prefill.

### Binary Header Format (64 Bytes)

```c
typedef struct {
    char     magic[4];           // "DKST"
    uint32_t version;            // 1
    int32_t  pos;                // Current sequence position index
    int32_t  prompt_len;         // Prompt length
    int32_t  context_size;       // Context window size
    int32_t  hidden_dim;         // Hidden dimension (5120)
    uint64_t kv_cache_bytes;     // KV cache size in bytes (~16.7 MB)
    uint64_t ssm_state_bytes;    // SSM state size in bytes (~151.0 MB)
    uint64_t ssm_conv_bytes;     // SSM conv history size in bytes (~5.9 MB)
    int32_t  next_tok;           // Sampled next token ID
    uint32_t checksum;           // Model checksum (0x5157454E)
    uint8_t  reserved[12];       // Padding for alignment
} diskllm_state_header;
```

### Layout

1. **Header**: 64 bytes (`diskllm_state_header`)
2. **KV Cache**: FP16 raw bytes (`kv_cache_bytes`)
3. **SSM Recurrent State**: FP32 raw bytes (`ssm_state_bytes`)
4. **SSM Conv History**: FP32 raw bytes (`ssm_conv_bytes`)
5. **Hidden Vector**: FP32 raw bytes (`hidden_dim * sizeof(float)`)

Total state file footprint: **~165.64 MB**.

## Atomic State Serialization

State files are written atomically to prevent corrupt state files upon unexpected process termination:
1. State is written to `<filename>.tmp`.
2. `fflush()` and `fclose()` guarantee disk flushes.
3. POSIX `rename("<filename>.tmp", "<filename>")` atomically replaces the target state file.
