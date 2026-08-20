#ifndef ATTENTION_H
#define ATTENTION_H

#include "state.h"
#include "scratch.h"
#include "kernels.h"
#include "model_config.h"

// Weight pointers and types for a single attention layer
typedef struct {
    const float *attn_norm_w;
    const void *attn_q_w;
    const float *attn_q_b;
    const float *attn_q_norm_w;
    const void *attn_k_w;
    const float *attn_k_b;
    const float *attn_k_norm_w;
    const void *attn_v_w;
    const float *attn_v_b;
    const void *attn_output_w;
    const float *post_attn_norm_w; /* optional post-attention RMSNorm for Gemma */
    const void *attn_qkv_w;   /* fused QKV weight for Phi-3 style models */
    
    int attn_q_w_type;
    int attn_k_w_type;
    int attn_v_w_type;
    int attn_output_w_type;
    int attn_qkv_w_type;      /* type for fused QKV */

    const float *rope_freqs;  /* optional rope frequency multipliers (e.g. Gemma 4) */

    int q_total_dim;
    int k_total_dim;
    int v_total_dim;
    int attn_out_dim;
} attention_layer_weights;

void attention_forward(
    float * restrict hidden_state,
    int pos,
    int layer_idx,
    const attention_layer_weights * restrict weights,
    model_state * restrict state,
    scratch_buffers * restrict scratch,
    const qwen_model_config *cfg
);

#endif // ATTENTION_H
