#include "attention.h"
#include "state.h"
#include "scratch.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

// Helper to check if a float is NaN
static int is_nan(float x) {
    return x != x;
}

void test_attention_f32(void) {
    printf("Testing Attention Forward (F32 weights)...\n");

    int context_length = 64;
    int hidden_size = 5120;
    int q_size = 12288;
    int kv_size = 1024;
    int out_size = 6144;

    // Allocate state and scratch
    model_state *state = allocate_model_state(context_length);
    assert(state != NULL);
    scratch_buffers *scratch = allocate_scratch_buffers();
    assert(scratch != NULL);

    // Allocate memory for F32 weights
    float *attn_norm_w = malloc(hidden_size * sizeof(float));
    float *attn_q_w = malloc(hidden_size * q_size * sizeof(float));
    float *attn_q_norm_w = malloc(256 * sizeof(float));
    float *attn_k_w = malloc(hidden_size * kv_size * sizeof(float));
    float *attn_k_norm_w = malloc(256 * sizeof(float));
    float *attn_v_w = malloc(hidden_size * kv_size * sizeof(float));
    float *attn_output_w = malloc(out_size * hidden_size * sizeof(float));

    assert(attn_norm_w && attn_q_w && attn_q_norm_w && attn_k_w && attn_k_norm_w && attn_v_w && attn_output_w);

    // Initialize weights to simple values (e.g. constant scale / identity-like)
    for (int i = 0; i < hidden_size; i++) attn_norm_w[i] = 1.0f;
    for (int i = 0; i < 256; i++) {
        attn_q_norm_w[i] = 1.0f;
        attn_k_norm_w[i] = 1.0f;
    }
    
    // Fill projection matrices with small random values to avoid zeros
    for (int i = 0; i < hidden_size * q_size; i++) {
        attn_q_w[i] = 0.0001f * (i % 7 - 3);
    }
    for (int i = 0; i < hidden_size * kv_size; i++) {
        attn_k_w[i] = 0.0002f * (i % 5 - 2);
        attn_v_w[i] = 0.0003f * (i % 9 - 4);
    }
    for (int i = 0; i < out_size * hidden_size; i++) {
        attn_output_w[i] = 0.0001f * (i % 11 - 5);
    }

    attention_layer_weights weights = {
        .attn_norm_w = attn_norm_w,
        .attn_q_w = attn_q_w,
        .attn_q_norm_w = attn_q_norm_w,
        .attn_k_w = attn_k_w,
        .attn_k_norm_w = attn_k_norm_w,
        .attn_v_w = attn_v_w,
        .attn_output_w = attn_output_w,
        .attn_q_w_type = GGML_TYPE_F32,
        .attn_k_w_type = GGML_TYPE_F32,
        .attn_v_w_type = GGML_TYPE_F32,
        .attn_output_w_type = GGML_TYPE_F32,
    };

    // Initialize hidden state input
    float *hidden_state = malloc(hidden_size * sizeof(float));
    assert(hidden_state != NULL);
    for (int i = 0; i < hidden_size; i++) {
        hidden_state[i] = 0.1f * (i % 3 + 1); // 0.1, 0.2, 0.3
    }

    double freq_base = 10000000.0;
    int rope_dim = 64;
    int layer_idx = 3; // first attention layer (index % 4 == 3)

    // Call attention_forward at pos = 0
    printf("  Running forward pass at pos = 0...\n");
    attention_forward(hidden_state, 0, layer_idx, &weights, state, scratch, freq_base, rope_dim);

    // Verify output
    for (int i = 0; i < hidden_size; i++) {
        assert(!is_nan(hidden_state[i]));
    }
    printf("    pos = 0 completed successfully without NaNs.\n");

    // Call attention_forward at pos = 1
    // Update hidden state with some changes
    for (int i = 0; i < hidden_size; i++) {
        hidden_state[i] = 0.05f * (i % 5 + 1);
    }
    printf("  Running forward pass at pos = 1...\n");
    attention_forward(hidden_state, 1, layer_idx, &weights, state, scratch, freq_base, rope_dim);

    for (int i = 0; i < hidden_size; i++) {
        assert(!is_nan(hidden_state[i]));
    }
    printf("    pos = 1 completed successfully without NaNs.\n");

    // Clean up
    free(hidden_state);
    free(attn_norm_w);
    free(attn_q_w);
    free(attn_q_norm_w);
    free(attn_k_w);
    free(attn_k_norm_w);
    free(attn_v_w);
    free(attn_output_w);
    free_model_state(state);
    free_scratch_buffers(scratch);

    printf("  F32 Attention Forward PASSED.\n");
}

int main(void) {
    printf("=== Starting Attention Layer Unit Tests ===\n\n");
    test_attention_f32();
    printf("\n=== ALL ATTENTION TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
