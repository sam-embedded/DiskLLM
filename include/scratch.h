#ifndef SCRATCH_H
#define SCRATCH_H

#include <stdint.h>
#include <stddef.h>

// Temporary scratch space for forward pass activations and streaming
typedef struct {
    // 128 MiB total streaming buffers, split into two 64 MiB buffers
    uint8_t *stream_buffer;
    size_t stream_buffer_size;

    // Activations (floats)
    float *hidden_state;    // size 5120
    float *ffn_gate;        // size 17408
    float *ffn_up;          // size 17408
    float *attn_q;          // size 12288
    float *attn_kv;         // size 1024
    float *ssm_qkv;         // size 10240
    float *logits;          // size 248320
} scratch_buffers;

// Allocation and lifecycle functions
scratch_buffers *allocate_scratch_buffers(void);
void free_scratch_buffers(scratch_buffers *scratch);

#endif // SCRATCH_H
