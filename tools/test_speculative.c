#include "speculative.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

void test_nextn_draft_step(void) {
    printf("Testing NextN Draft Step...\n");

    qwen_model_config cfg = {
        .hidden_dim = 128,
        .ffn_dim = 256,
        .num_attn_heads = 4,
        .key_length = 32,
        .vocab_size = 1000,
        .ssm_inner_size = 256,
        .ssm_group_count = 2,
        .ssm_state_size = 32,
        .ssm_time_step_rank = 16
    };

    scratch_buffers *scratch = allocate_scratch_buffers(&cfg);
    assert(scratch != NULL);

    int hidden_dim = cfg.hidden_dim;

    float *enorm_w = malloc(hidden_dim * sizeof(float));
    float *hnorm_w = malloc(hidden_dim * sizeof(float));
    float *snorm_w = malloc(hidden_dim * sizeof(float));
    float *eh_proj_w = malloc(2 * hidden_dim * hidden_dim * sizeof(float));

    for (int i = 0; i < hidden_dim; i++) {
        enorm_w[i] = 1.0f;
        hnorm_w[i] = 1.0f;
        snorm_w[i] = 1.0f;
    }
    for (int i = 0; i < 2 * hidden_dim * hidden_dim; i++) {
        eh_proj_w[i] = 0.001f * ((i % 11) - 5);
    }

    nextn_weights nweights = {
        .enorm_w = enorm_w,
        .hnorm_w = hnorm_w,
        .shared_head_norm_w = snorm_w,
        .eh_proj_w = eh_proj_w,
        .eh_proj_w_type = GGML_TYPE_F32
    };

    float *embed_t = malloc(hidden_dim * sizeof(float));
    float *hidden_t = malloc(hidden_dim * sizeof(float));
    float *out_hnext = malloc(hidden_dim * sizeof(float));

    for (int i = 0; i < hidden_dim; i++) {
        embed_t[i] = 0.1f * (i + 1);
        hidden_t[i] = 0.05f * (i + 1);
    }

    nextn_draft_step(embed_t, hidden_t, out_hnext, &nweights, scratch, &cfg);

    int has_nan = 0;
    for (int i = 0; i < hidden_dim; i++) {
        if (isnan(out_hnext[i]) || isinf(out_hnext[i])) {
            has_nan = 1;
            break;
        }
    }

    assert(!has_nan);
    printf("  NextN Draft Step executed cleanly without NaNs.\n");

    free(embed_t);
    free(hidden_t);
    free(out_hnext);
    free(enorm_w);
    free(hnorm_w);
    free(snorm_w);
    free(eh_proj_w);
    free_scratch_buffers(scratch);
}

int main(void) {
    printf("=== Starting NextN Speculative Engine Unit Tests ===\n\n");
    test_nextn_draft_step();
    printf("\n=== ALL SPECULATIVE TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
