#ifndef SSM_H
#define SSM_H

#include "state.h"
#include "scratch.h"
#include "kernels.h"

typedef struct {
    const float *attn_norm_w;          // [5120] (Input RMSNorm weight)
    const void *attn_qkv_w;            // [5120, 10240]
    const float *ssm_conv1d_w;         // [4, 10240]
    const float *ssm_a_w;              // [48]
    const void *ssm_alpha_w;           // [5120, 48]
    const void *ssm_beta_w;            // [5120, 48]
    const float *ssm_dt_bias;          // [48]
    const float *ssm_norm_w;           // [128]
    const void *ssm_out_w;             // [6144, 5120]
    const void *attn_gate_w;           // [5120, 6144]
    
    int attn_qkv_w_type;
    int ssm_alpha_w_type;
    int ssm_beta_w_type;
    int ssm_out_w_type;
    int attn_gate_w_type;
} ssm_layer_weights;

// Forward pass for a single SSM/DeltaNet layer
// - hidden_state: input/output activation vector of size 5120 (modified in-place)
// - pos: current sequence position index
// - layer_idx: absolute layer index of the block (0 to 63)
// - weights: pointers to the weights of the current layer
// - state: model persistent state (holding ssm_states and ssm_conv_histories)
// - scratch: temporary scratch buffers
void ssm_layer_forward(
    float * restrict hidden_state,
    int pos,
    int layer_idx,
    const ssm_layer_weights * restrict weights,
    model_state * restrict state,
    scratch_buffers * restrict scratch
);

#endif // SSM_H
