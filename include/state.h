#ifndef STATE_H
#define STATE_H

#include <stdint.h>
#include <stddef.h>
#include "model_config.h"

typedef enum {
    CACHE_TYPE_F16 = 0,
    CACHE_TYPE_Q8_0 = 1,
    CACHE_TYPE_Q4_0 = 2
} cache_type_t;

// Persistent model state
typedef struct {
    int context_length;
    cache_type_t cache_type;
    size_t kv_token_bytes; // bytes per token per layer for KV cache
    
    // Mapping arrays
    int layer_to_attn_idx[1024];
    int layer_to_ssm_idx[1024];

    // KV Cache for Attention layers (void pointer to support F16, Q8_0, Q4_0)
    void *kv_cache;
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
model_state *allocate_model_state_ex(const qwen_model_config *cfg, int context_length, cache_type_t cache_type);
model_state *allocate_model_state(const qwen_model_config *cfg, int context_length);
void free_model_state(model_state *state);

#endif // STATE_H
