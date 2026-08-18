#ifndef SSM_H
#define SSM_H

#include "state.h"
#include "scratch.h"
#include "kernels.h"
#include "model_config.h"

typedef struct {
    const float *attn_norm_w;
    const void *attn_qkv_w;
    const float *ssm_conv1d_w;
    const float *ssm_a_w;
    const void *ssm_alpha_w;
    const void *ssm_beta_w;
    const float *ssm_dt_bias;
    const float *ssm_norm_w;
    const void *ssm_out_w;
    const void *attn_gate_w;
    
    int attn_qkv_w_type;
    int ssm_alpha_w_type;
    int ssm_beta_w_type;
    int ssm_out_w_type;
    int attn_gate_w_type;
} ssm_layer_weights;

void ssm_layer_forward(
    float * restrict hidden_state,
    int pos,
    int layer_idx,
    const ssm_layer_weights * restrict weights,
    model_state * restrict state,
    scratch_buffers * restrict scratch,
    const qwen_model_config *cfg
);

#endif // SSM_H
