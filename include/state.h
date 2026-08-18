#ifndef STATE_H
#define STATE_H

#include <stdint.h>
#include <stddef.h>
#include "model_config.h"

// Persistent model state
typedef struct {
    int context_length;
    
    // Mapping arrays
    int layer_to_attn_idx[1024];
    int layer_to_ssm_idx[1024];

    // KV Cache for Attention layers
    // Allocated as FP16 (uint16_t)
    uint16_t *kv_cache;
    size_t kv_cache_size; // in bytes

    // SSM Recurrent State for SSM layers
    float *ssm_states;
    size_t ssm_states_size; // in bytes

    // SSM Convolution History for SSM layers
    float *ssm_conv_histories;
    size_t ssm_conv_histories_size; // in bytes

    // Dimension sizes for state offsets
    int kv_dim;               // num_kv_heads * key_length
    int ssm_inner_size;       // e.g. 6144 or 2048
    int ssm_state_size;       // e.g. 128
    int ssm_conv_dim;         // ssm_inner_size + 2 * (ssm_group_count * ssm_state_size)
    int ssm_num_v_heads;      // ssm_time_step_rank (e.g. 48 or 16)
} model_state;

// Allocation and lifecycle functions
model_state *allocate_model_state(const qwen_model_config *cfg, int context_length);
void free_model_state(model_state *state);

#endif // STATE_H
