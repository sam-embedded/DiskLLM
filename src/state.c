#define _POSIX_C_SOURCE 200112L

#include "state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

model_state *allocate_model_state(int context_length) {
    if (context_length <= 0) {
        fprintf(stderr, "Error: Invalid context length %d\n", context_length);
        return NULL;
    }

    model_state *state = calloc(1, sizeof(model_state));
    if (!state) {
        fprintf(stderr, "Error: Failed to allocate model_state wrapper.\n");
        return NULL;
    }

    state->context_length = context_length;

    // 1. Allocate KV cache
    // Shape: [16, context_length, 2, 1024]
    // Allocated as uint16_t (FP16) to save memory
    state->kv_cache_size = 16ULL * (size_t)context_length * 2ULL * 1024ULL * sizeof(uint16_t);
    int ret = posix_memalign((void **)&state->kv_cache, 64, state->kv_cache_size);
    if (ret != 0 || !state->kv_cache) {
        fprintf(stderr, "Error: Failed to allocate KV cache of size %" PRIu64 " bytes.\n", (uint64_t)state->kv_cache_size);
        free(state);
        return NULL;
    }
    // Zero-initialize the allocated buffer
    memset(state->kv_cache, 0, state->kv_cache_size);

    // 2. Allocate SSM Recurrent States
    // Shape: [48, 6144, 128]
    state->ssm_states_size = 48ULL * 6144ULL * 128ULL * sizeof(float);
    ret = posix_memalign((void **)&state->ssm_states, 64, state->ssm_states_size);
    if (ret != 0 || !state->ssm_states) {
        fprintf(stderr, "Error: Failed to allocate SSM recurrent states of size %" PRIu64 " bytes.\n", (uint64_t)state->ssm_states_size);
        free(state->kv_cache);
        free(state);
        return NULL;
    }
    memset(state->ssm_states, 0, state->ssm_states_size);

    // 3. Allocate SSM Convolution Histories
    // Shape: [48, 3, 6144]
    // Note: conv kernel is 4, so history stores (4 - 1) = 3 states
    state->ssm_conv_histories_size = 48ULL * 3ULL * 6144ULL * sizeof(float);
    ret = posix_memalign((void **)&state->ssm_conv_histories, 64, state->ssm_conv_histories_size);
    if (ret != 0 || !state->ssm_conv_histories) {
        fprintf(stderr, "Error: Failed to allocate SSM conv history of size %" PRIu64 " bytes.\n", (uint64_t)state->ssm_conv_histories_size);
        free(state->ssm_states);
        free(state->kv_cache);
        free(state);
        return NULL;
    }
    memset(state->ssm_conv_histories, 0, state->ssm_conv_histories_size);

    return state;
}

void free_model_state(model_state *state) {
    if (!state) return;
    if (state->kv_cache) free(state->kv_cache);
    if (state->ssm_states) free(state->ssm_states);
    if (state->ssm_conv_histories) free(state->ssm_conv_histories);
    free(state);
}
