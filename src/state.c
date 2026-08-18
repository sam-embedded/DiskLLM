#define _POSIX_C_SOURCE 200112L

#include "state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

model_state *allocate_model_state_ex(const qwen_model_config *cfg, int context_length, cache_type_t cache_type) {
    if (context_length <= 0 || !cfg) {
        fprintf(stderr, "Error: Invalid context length %d or null config\n", context_length);
        return NULL;
    }

    model_state *state = calloc(1, sizeof(model_state));
    if (!state) {
        fprintf(stderr, "Error: Failed to allocate model_state wrapper.\n");
        return NULL;
    }

    state->context_length = context_length;
    state->cache_type = cache_type;
    state->kv_dim = cfg->num_kv_heads * cfg->key_length;
    state->ssm_inner_size = cfg->ssm_inner_size;
    state->ssm_state_size = cfg->ssm_state_size;
    state->ssm_conv_dim = cfg->ssm_inner_size + 2 * (cfg->ssm_group_count * cfg->ssm_state_size);
    state->ssm_num_v_heads = cfg->ssm_time_step_rank;

    int attn_count = 0;
    int ssm_count = 0;
    for (int b = 0; b < cfg->block_count; b++) {
        if (cfg->layer_types[b] == LAYER_TYPE_ATTENTION) {
            state->layer_to_attn_idx[b] = attn_count++;
            state->layer_to_ssm_idx[b] = -1;
        } else if (cfg->layer_types[b] == LAYER_TYPE_SSM) {
            state->layer_to_ssm_idx[b] = ssm_count++;
            state->layer_to_attn_idx[b] = -1;
        } else {
            state->layer_to_attn_idx[b] = -1;
            state->layer_to_ssm_idx[b] = -1;
        }
    }

    // 1. Calculate KV cache token size and allocate KV cache
    if (attn_count > 0 && state->kv_dim > 0) {
        if (cache_type == CACHE_TYPE_Q8_0) {
            size_t nb = ((size_t)state->kv_dim + 31) / 32;
            state->kv_token_bytes = nb * 34 * 2;
        } else if (cache_type == CACHE_TYPE_Q4_0) {
            size_t nb = ((size_t)state->kv_dim + 31) / 32;
            state->kv_token_bytes = nb * 18 * 2;
        } else {
            state->cache_type = CACHE_TYPE_F16;
            state->kv_token_bytes = (size_t)state->kv_dim * 2 * sizeof(uint16_t);
        }

        state->kv_cache_size = (size_t)attn_count * (size_t)context_length * state->kv_token_bytes;
        int ret = posix_memalign((void **)&state->kv_cache, 64, state->kv_cache_size);
        if (ret != 0 || !state->kv_cache) {
            fprintf(stderr, "Error: Failed to allocate KV cache of size %" PRIu64 " bytes.\n", (uint64_t)state->kv_cache_size);
            free(state);
            return NULL;
        }
        memset(state->kv_cache, 0, state->kv_cache_size);
    }

    // 2. Allocate SSM Recurrent States
    if (ssm_count > 0 && cfg->ssm_inner_size > 0 && cfg->ssm_state_size > 0) {
        state->ssm_states_size = (size_t)ssm_count * (size_t)cfg->ssm_time_step_rank * (size_t)cfg->ssm_state_size * (size_t)cfg->ssm_state_size * sizeof(float);
        int ret = posix_memalign((void **)&state->ssm_states, 64, state->ssm_states_size);
        if (ret != 0 || !state->ssm_states) {
            fprintf(stderr, "Error: Failed to allocate SSM recurrent states of size %" PRIu64 " bytes.\n", (uint64_t)state->ssm_states_size);
            if (state->kv_cache) free(state->kv_cache);
            free(state);
            return NULL;
        }
        memset(state->ssm_states, 0, state->ssm_states_size);

        // 3. Allocate SSM Convolution Histories
        int conv_hist_len = (cfg->ssm_conv_kernel > 1) ? (cfg->ssm_conv_kernel - 1) : 3;
        state->ssm_conv_histories_size = (size_t)ssm_count * (size_t)conv_hist_len * (size_t)state->ssm_conv_dim * sizeof(float);
        ret = posix_memalign((void **)&state->ssm_conv_histories, 64, state->ssm_conv_histories_size);
        if (ret != 0 || !state->ssm_conv_histories) {
            fprintf(stderr, "Error: Failed to allocate SSM conv history of size %" PRIu64 " bytes.\n", (uint64_t)state->ssm_conv_histories_size);
            if (state->ssm_states) free(state->ssm_states);
            if (state->kv_cache) free(state->kv_cache);
            free(state);
            return NULL;
        }
        memset(state->ssm_conv_histories, 0, state->ssm_conv_histories_size);
    }

    return state;
}

model_state *allocate_model_state(const qwen_model_config *cfg, int context_length) {
    return allocate_model_state_ex(cfg, context_length, CACHE_TYPE_F16);
}

void free_model_state(model_state *state) {
    if (!state) return;
    if (state->kv_cache) free(state->kv_cache);
    if (state->ssm_states) free(state->ssm_states);
    if (state->ssm_conv_histories) free(state->ssm_conv_histories);
    free(state);
}
