#ifndef ATTENTION_H
#define ATTENTION_H

#include "state.h"
#include "scratch.h"
#include "kernels.h"

// Weight pointers and types for a single attention layer
typedef struct {
    const float *attn_norm_w;          // shape [5120]
    const void *attn_q_w;              // shape [5120, 12288]
    const float *attn_q_norm_w;        // shape [256]
    const void *attn_k_w;              // shape [5120, 1024]
    const float *attn_k_norm_w;        // shape [256]
    const void *attn_v_w;              // shape [5120, 1024]
    const void *attn_output_w;         // shape [6144, 5120]
    
    int attn_q_w_type;
    int attn_k_w_type;
    int attn_v_w_type;
    int attn_output_w_type;
} attention_layer_weights;

// Forward pass for a single full attention layer
// - hidden_state: input/output activation vector of size 5120 (modified in-place)
// - pos: current sequence position index
// - layer_idx: absolute layer index of the block (0 to 63)
// - weights: pointers to the weights of the current layer
// - state: model persistent state (holding kv_cache)
// - scratch: temporary scratch buffers
// - freq_base: RoPE frequency base (e.g. 10000000.0)
// - rope_dim: RoPE dimension count (e.g. 64)
void attention_forward(
    float * restrict hidden_state,
    int pos,
    int layer_idx,
    const attention_layer_weights * restrict weights,
    model_state * restrict state,
    scratch_buffers * restrict scratch,
    double freq_base,
    int rope_dim
);

#endif // ATTENTION_H
