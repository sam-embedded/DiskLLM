#include "ssm.h"
#include "state.h"
#include "scratch.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

// Helper to check if a float is NaN or Inf
static int is_invalid(float x) {
    return (x != x) || (x > 1e30f) || (x < -1e30f);
}

void test_ssm_layer_f32(void) {
    printf("Testing SSM Layer Forward (F32 weights)...\n");

    int context_length = 64;
    int hidden_size = 5120;
    int qkv_size = 10240;
    int ssm_a_size = 48;
    int alpha_beta_size = 48;
    int state_head_dim = 128;
    int out_size = 6144;

    // Allocate state and scratch
    model_state *state = allocate_model_state(context_length);
    assert(state != NULL);
    scratch_buffers *scratch = allocate_scratch_buffers();
    assert(scratch != NULL);

    // Allocate weights
    float *attn_norm_w = malloc(hidden_size * sizeof(float));
    float *attn_qkv_w = malloc(hidden_size * qkv_size * sizeof(float));
    float *ssm_conv1d_w = malloc(4 * qkv_size * sizeof(float));
    float *ssm_a_w = malloc(ssm_a_size * sizeof(float));
    float *ssm_alpha_w = malloc(hidden_size * alpha_beta_size * sizeof(float));
    float *ssm_beta_w = malloc(hidden_size * alpha_beta_size * sizeof(float));
    float *ssm_dt_bias = malloc(ssm_a_size * sizeof(float));
    float *ssm_norm_w = malloc(state_head_dim * sizeof(float));
    float *ssm_out_w = malloc(out_size * hidden_size * sizeof(float));
    float *attn_gate_w = malloc(hidden_size * out_size * sizeof(float));

    assert(attn_norm_w && attn_qkv_w && ssm_conv1d_w && ssm_a_w && ssm_alpha_w &&
           ssm_beta_w && ssm_dt_bias && ssm_norm_w && ssm_out_w && attn_gate_w);

    // Initialize weights to simple values
    for (int i = 0; i < hidden_size; i++) attn_norm_w[i] = 1.0f;
    for (int i = 0; i < state_head_dim; i++) ssm_norm_w[i] = 1.0f;

    // Small negative values for log(decay_gate) to keep decay scaling between 0 and 1
    for (int i = 0; i < ssm_a_size; i++) {
        ssm_a_w[i] = -0.5f;
        ssm_dt_bias[i] = 0.1f * (i % 3 - 1);
    }

    // Fill matrices with small random values to ensure non-zero activation flow
    for (int i = 0; i < hidden_size * qkv_size; i++) {
        attn_qkv_w[i] = 0.0001f * (i % 7 - 3);
    }
    for (int i = 0; i < 4 * qkv_size; i++) {
        ssm_conv1d_w[i] = 0.01f * (i % 5 - 2);
    }
    for (int i = 0; i < hidden_size * alpha_beta_size; i++) {
        ssm_alpha_w[i] = 0.0002f * (i % 3 - 1);
        ssm_beta_w[i] = 0.0003f * (i % 5 - 2);
    }
    for (int i = 0; i < out_size * hidden_size; i++) {
        ssm_out_w[i] = 0.0001f * (i % 11 - 5);
    }
    for (int i = 0; i < hidden_size * out_size; i++) {
        attn_gate_w[i] = 0.0001f * (i % 9 - 4);
    }

    ssm_layer_weights weights = {
        .attn_norm_w = attn_norm_w,
        .attn_qkv_w = attn_qkv_w,
        .ssm_conv1d_w = ssm_conv1d_w,
        .ssm_a_w = ssm_a_w,
        .ssm_alpha_w = ssm_alpha_w,
        .ssm_beta_w = ssm_beta_w,
        .ssm_dt_bias = ssm_dt_bias,
        .ssm_norm_w = ssm_norm_w,
        .ssm_out_w = ssm_out_w,
        .attn_gate_w = attn_gate_w,
        .attn_qkv_w_type = GGML_TYPE_F32,
        .ssm_alpha_w_type = GGML_TYPE_F32,
        .ssm_beta_w_type = GGML_TYPE_F32,
        .ssm_out_w_type = GGML_TYPE_F32,
        .attn_gate_w_type = GGML_TYPE_F32,
    };

    // Initialize hidden state
    float *hidden_state = malloc(hidden_size * sizeof(float));
    assert(hidden_state != NULL);
    for (int i = 0; i < hidden_size; i++) {
        hidden_state[i] = 0.1f * (i % 5 + 1);
    }

    // Run forward pass for multiple sequential positions
    int layer_idx = 0; // Layer 0 is an SSM layer

    printf("  Running forward pass at pos = 0...\n");
    ssm_layer_forward(hidden_state, 0, layer_idx, &weights, state, scratch);
    for (int i = 0; i < hidden_size; i++) {
        assert(!is_invalid(hidden_state[i]));
    }

    // Check that state buffers are updated and non-zero
    float sum_conv = 0.0f;
    for (size_t i = 0; i < state->ssm_conv_histories_size / sizeof(float); i++) {
        sum_conv += fabs(state->ssm_conv_histories[i]);
    }
    assert(sum_conv > 0.0f);
    printf("    pos = 0 completed successfully. Conv history is updated.\n");

    printf("  Running forward pass at pos = 1...\n");
    ssm_layer_forward(hidden_state, 1, layer_idx, &weights, state, scratch);
    for (int i = 0; i < hidden_size; i++) {
        assert(!is_invalid(hidden_state[i]));
    }

    float sum_ssm = 0.0f;
    for (size_t i = 0; i < state->ssm_states_size / sizeof(float); i++) {
        sum_ssm += fabs(state->ssm_states[i]);
    }
    assert(sum_ssm > 0.0f);
    printf("    pos = 1 completed successfully. Recurrent state is updated.\n");

    printf("  Running forward pass at pos = 2...\n");
    ssm_layer_forward(hidden_state, 2, layer_idx, &weights, state, scratch);
    for (int i = 0; i < hidden_size; i++) {
        assert(!is_invalid(hidden_state[i]));
    }
    printf("    pos = 2 completed successfully.\n");

    // Clean up
    free(hidden_state);
    free(attn_norm_w);
    free(attn_qkv_w);
    free(ssm_conv1d_w);
    free(ssm_a_w);
    free(ssm_alpha_w);
    free(ssm_beta_w);
    free(ssm_dt_bias);
    free(ssm_norm_w);
    free(ssm_out_w);
    free(attn_gate_w);
    free_model_state(state);
    free_scratch_buffers(scratch);

    printf("  SSM Layer Forward PASSED.\n");
}

int main(void) {
    printf("=== Starting SSM Layer Unit Tests ===\n\n");
    test_ssm_layer_f32();
    printf("\n=== ALL SSM TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
