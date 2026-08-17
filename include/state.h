#ifndef STATE_H
#define STATE_H

#include <stdint.h>
#include <stddef.h>

// Persistent model state
typedef struct {
    int context_length;
    
    // KV Cache for 16 Attention layers
    // Shape: [16, context_length, 2 (K and V), 1024]
    // Quantized to FP16 (uint16_t) to save memory
    uint16_t *kv_cache;
    size_t kv_cache_size; // in bytes

    // SSM Recurrent State for 48 SSM layers
    // Shape: [48, 6144, 128]
    float *ssm_states;
    size_t ssm_states_size; // in bytes

    // SSM Convolution History for 48 SSM layers
    // Shape: [48, 3, 6144]
    float *ssm_conv_histories;
    size_t ssm_conv_histories_size; // in bytes
} model_state;

// Allocation and lifecycle functions
model_state *allocate_model_state(int context_length);
void free_model_state(model_state *state);

#endif // STATE_H
