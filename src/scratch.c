#define _POSIX_C_SOURCE 200112L

#include "scratch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

scratch_buffers *allocate_scratch_buffers(void) {
    scratch_buffers *scratch = calloc(1, sizeof(scratch_buffers));
    if (!scratch) {
        fprintf(stderr, "Error: Failed to allocate scratch_buffers wrapper.\n");
        return NULL;
    }

    // Allocate 128 MiB for streaming
    scratch->stream_buffer_size = 128ULL * 1024ULL * 1024ULL;
    int ret = posix_memalign((void **)&scratch->stream_buffer, 64, scratch->stream_buffer_size);
    if (ret != 0 || !scratch->stream_buffer) {
        fprintf(stderr, "Error: Failed to allocate stream buffer of size 128 MiB.\n");
        free(scratch);
        return NULL;
    }
    memset(scratch->stream_buffer, 0, scratch->stream_buffer_size);

    // Helper macro to allocate and check aligned buffers
    #define ALLOC_ALIGNED_FLOAT(ptr, count) \
        ret = posix_memalign((void **)&scratch->ptr, 64, (count) * sizeof(float)); \
        if (ret != 0 || !scratch->ptr) { \
            fprintf(stderr, "Error: Failed to allocate activation " #ptr " of size " #count " floats.\n"); \
            free_scratch_buffers(scratch); \
            return NULL; \
        } \
        memset(scratch->ptr, 0, (count) * sizeof(float));

    // Allocate all activation buffers
    ALLOC_ALIGNED_FLOAT(hidden_state, 5120);
    ALLOC_ALIGNED_FLOAT(ffn_gate, 17408);
    ALLOC_ALIGNED_FLOAT(ffn_up, 17408);
    ALLOC_ALIGNED_FLOAT(attn_q, 12288);
    ALLOC_ALIGNED_FLOAT(attn_kv, 1024);
    ALLOC_ALIGNED_FLOAT(ssm_qkv, 10240);
    ALLOC_ALIGNED_FLOAT(logits, 248320);

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
