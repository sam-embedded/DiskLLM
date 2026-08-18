#define _POSIX_C_SOURCE 200112L

#include "scratch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

scratch_buffers *allocate_scratch_buffers(const qwen_model_config *cfg) {
    if (!cfg) {
        qwen_model_config dummy = {
            .hidden_dim = 5120, .ffn_dim = 17408,
            .num_attn_heads = 24, .key_length = 256,
            .num_kv_heads = 4, .ssm_inner_size = 6144,
            .ssm_group_count = 16, .ssm_state_size = 128,
            .vocab_size = 248320
        };
        return allocate_scratch_buffers(&dummy);
    }

    scratch_buffers *scratch = calloc(1, sizeof(scratch_buffers));
    if (!scratch) {
        fprintf(stderr, "Error: Failed to allocate scratch_buffers wrapper.\n");
        return NULL;
    }

    scratch->stream_buffer_size = 300ULL * 1024ULL * 1024ULL;
    int ret = posix_memalign((void **)&scratch->stream_buffer, 64, scratch->stream_buffer_size);
    if (ret != 0 || !scratch->stream_buffer) {
        fprintf(stderr, "Error: Failed to allocate stream buffer.\n");
        free(scratch);
        return NULL;
    }
    memset(scratch->stream_buffer, 0, scratch->stream_buffer_size);

    int hidden_dim = cfg->hidden_dim > 0 ? cfg->hidden_dim : 5120;
    int ffn_dim = cfg->ffn_dim > 0 ? cfg->ffn_dim : 17408;
    int vocab_size = cfg->vocab_size > 0 ? cfg->vocab_size : 248320;

    int q_total_dim = cfg->num_attn_heads * cfg->key_length * 2;
    if (q_total_dim < 12288) q_total_dim = 12288;

    int kv_total_dim = cfg->num_kv_heads * cfg->key_length;
    if (kv_total_dim < 1024) kv_total_dim = 1024;

    int ssm_conv_dim = cfg->ssm_inner_size + 2 * (cfg->ssm_group_count * cfg->ssm_state_size);
    if (ssm_conv_dim < 10240) ssm_conv_dim = 10240;

    #define ALLOC_ALIGNED_FLOAT(ptr, count) do { \
        ret = posix_memalign((void **)&scratch->ptr, 64, (size_t)(count) * sizeof(float)); \
        if (ret != 0 || !scratch->ptr) { \
            fprintf(stderr, "Error: Failed to allocate activation " #ptr " of size %d floats.\n", (int)(count)); \
            free_scratch_buffers(scratch); \
            return NULL; \
        } \
        memset(scratch->ptr, 0, (size_t)(count) * sizeof(float)); \
    } while(0)

    ALLOC_ALIGNED_FLOAT(hidden_state, hidden_dim);
    ALLOC_ALIGNED_FLOAT(ffn_gate, ffn_dim);
    ALLOC_ALIGNED_FLOAT(ffn_up, ffn_dim);
    ALLOC_ALIGNED_FLOAT(attn_q, q_total_dim);
    ALLOC_ALIGNED_FLOAT(attn_kv, kv_total_dim);
    ALLOC_ALIGNED_FLOAT(ssm_qkv, ssm_conv_dim);
    ALLOC_ALIGNED_FLOAT(logits, vocab_size);

    #undef ALLOC_ALIGNED_FLOAT

    return scratch;
}

void free_scratch_buffers(scratch_buffers *scratch) {
    if (!scratch) return;
    if (scratch->stream_buffer) free(scratch->stream_buffer);
    if (scratch->hidden_state) free(scratch->hidden_state);
    if (scratch->ffn_gate) free(scratch->ffn_gate);
    if (scratch->ffn_up) free(scratch->ffn_up);
    if (scratch->attn_q) free(scratch->attn_q);
    if (scratch->attn_kv) free(scratch->attn_kv);
    if (scratch->ssm_qkv) free(scratch->ssm_qkv);
    if (scratch->logits) free(scratch->logits);
    free(scratch);
}
