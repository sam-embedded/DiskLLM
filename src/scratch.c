#define _POSIX_C_SOURCE 200112L

#include "scratch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}

scratch_buffers *allocate_scratch_buffers(const qwen_model_config *cfg) {
    if (!cfg) {
        qwen_model_config dummy = {
            .hidden_dim = 5120, .ffn_dim = 17408,
            .num_attn_heads = 24, .key_length = 256,
            .num_kv_heads = 4, .ssm_inner_size = 6144,
            .ssm_group_count = 16, .ssm_state_size = 128,
            .ssm_time_step_rank = 48,
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
    int kv_total_dim = cfg->num_kv_heads * cfg->key_length;
    int attn_out_dim = cfg->num_attn_heads * cfg->key_length;

    int ssm_group_count = cfg->ssm_group_count > 0 ? cfg->ssm_group_count : 16;
    int ssm_state_size = cfg->ssm_state_size > 0 ? cfg->ssm_state_size : 128;
    int ssm_rank = cfg->ssm_time_step_rank > 0 ? cfg->ssm_time_step_rank : 48;
    int ssm_inner = cfg->ssm_inner_size > 0 ? cfg->ssm_inner_size : 6144;
    int ssm_conv_dim = ssm_inner + 2 * (ssm_group_count * ssm_state_size);

    // Compute maximum size required for each activation buffer across Attention and SSM layers
    int ffn_gate_alloc_dim = max_int(2 * ffn_dim, ssm_conv_dim);
    ffn_gate_alloc_dim = max_int(ffn_gate_alloc_dim, ssm_inner + ssm_rank * ssm_state_size);
    ffn_gate_alloc_dim = max_int(ffn_gate_alloc_dim, attn_out_dim);

    int ffn_up_alloc_dim = max_int(ffn_dim, ssm_conv_dim);
    ffn_up_alloc_dim = max_int(ffn_up_alloc_dim, 8192); // context length headroom

    int attn_q_alloc_dim = max_int(q_total_dim, ssm_inner);
    attn_q_alloc_dim = max_int(attn_q_alloc_dim, 8192);

    int attn_kv_alloc_dim = max_int(kv_total_dim, 2 * ssm_rank);
    attn_kv_alloc_dim = max_int(attn_kv_alloc_dim, 8192);

    int ssm_qkv_alloc_dim = max_int(ssm_conv_dim, ffn_dim);
    ssm_qkv_alloc_dim = max_int(ssm_qkv_alloc_dim, vocab_size);
    ssm_qkv_alloc_dim = max_int(ssm_qkv_alloc_dim, 8192);

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
    ALLOC_ALIGNED_FLOAT(ffn_gate, ffn_gate_alloc_dim);
    ALLOC_ALIGNED_FLOAT(ffn_up, ffn_up_alloc_dim);
    ALLOC_ALIGNED_FLOAT(attn_q, attn_q_alloc_dim);
    ALLOC_ALIGNED_FLOAT(attn_kv, attn_kv_alloc_dim);
    ALLOC_ALIGNED_FLOAT(ssm_qkv, ssm_qkv_alloc_dim);
    ALLOC_ALIGNED_FLOAT(logits, vocab_size);
    ALLOC_ALIGNED_FLOAT(ple_gate, 256);
    ALLOC_ALIGNED_FLOAT(ple_mult, 256);
    ALLOC_ALIGNED_FLOAT(ple_out, hidden_dim);
    ALLOC_ALIGNED_FLOAT(ple_proj_buf, 32768);
    ALLOC_ALIGNED_FLOAT(ple_token_buf, 32768);

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
    if (scratch->ple_gate) free(scratch->ple_gate);
    if (scratch->ple_mult) free(scratch->ple_mult);
    if (scratch->ple_out) free(scratch->ple_out);
    if (scratch->ple_proj_buf) free(scratch->ple_proj_buf);
    if (scratch->ple_token_buf) free(scratch->ple_token_buf);
    free(scratch);
}
