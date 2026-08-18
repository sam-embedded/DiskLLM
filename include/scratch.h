#ifndef SCRATCH_H
#define SCRATCH_H

#include <stdint.h>
#include <stddef.h>
#include "model_config.h"

// Temporary scratch space for forward pass activations and streaming
typedef struct {
    uint8_t *stream_buffer;
    size_t stream_buffer_size;

    // Activations (floats)
    float *hidden_state;    // size hidden_dim
    float *ffn_gate;        // size ffn_dim
    float *ffn_up;          // size ffn_dim
    float *attn_q;          // size q_total_dim
    float *attn_kv;         // size kv_total_dim
    float *ssm_qkv;         // size ssm_conv_dim
    float *logits;          // size vocab_size
} scratch_buffers;

scratch_buffers *allocate_scratch_buffers(const qwen_model_config *cfg);
void free_scratch_buffers(scratch_buffers *scratch);

#endif // SCRATCH_H
