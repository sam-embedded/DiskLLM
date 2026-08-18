#include "ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

static inline float sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline float ssm_silu(float x) {
    return x * sigmoid(x);
}

void ssm_layer_forward(
    float * restrict hidden_state,
    int pos,
    int layer_idx,
    const ssm_layer_weights * restrict weights,
    model_state * restrict state,
    scratch_buffers * restrict scratch,
    const qwen_model_config *cfg
) {
    (void)pos;
    
    int ssm_layer_idx = state->layer_to_ssm_idx[layer_idx];
    if (ssm_layer_idx < 0) ssm_layer_idx = 0;

    int hidden_dim     = cfg ? cfg->hidden_dim : 5120;
    int conv_channels  = state->ssm_conv_dim;
    int ssm_inner_size = state->ssm_inner_size;
    int ssm_state_size = state->ssm_state_size;
    int num_v_heads    = state->ssm_num_v_heads;
    int group_count    = cfg ? cfg->ssm_group_count : 16;
    int conv_kernel    = cfg ? cfg->ssm_conv_kernel : 4;
    int hist_len       = (conv_kernel > 1) ? (conv_kernel - 1) : 3;

    int qk_head_size   = ssm_state_size;
    int q_conv_size    = group_count * qk_head_size;
    int k_conv_size    = group_count * qk_head_size;

    /* 1. Input RMSNorm */
    rmsnorm(scratch->hidden_state, hidden_state, weights->attn_norm_w, hidden_dim, 1e-6f);

    /* 2. Input Projection (attn_qkv) */
    matvec(
        scratch->ffn_gate,
        weights->attn_qkv_w,
        scratch->hidden_state,
        hidden_dim,
        conv_channels,
        weights->attn_qkv_w_type,
        scratch->ssm_qkv
    );

    /* 3. Conv1D Convolution */
    float *history = state->ssm_conv_histories + (size_t)ssm_layer_idx * (size_t)hist_len * (size_t)conv_channels;

    for (int c = 0; c < conv_channels; c++) {
        float val = 0.0f;
        for (int k = 0; k < hist_len; k++) {
            val += history[k * conv_channels + c] * weights->ssm_conv1d_w[c * conv_kernel + k];
        }
        val += scratch->ffn_gate[c] * weights->ssm_conv1d_w[c * conv_kernel + (conv_kernel - 1)];
        scratch->ffn_up[c] = ssm_silu(val);
    }

    /* Update Conv1D history */
    for (int k = 0; k < hist_len - 1; k++) {
        memcpy(history + k * conv_channels, history + (k + 1) * conv_channels, conv_channels * sizeof(float));
    }
    memcpy(history + (hist_len - 1) * conv_channels, scratch->ffn_gate, conv_channels * sizeof(float));

    /* 4. Alpha & Beta Projections */
    matvec(
        scratch->attn_kv,
        weights->ssm_alpha_w,
        scratch->hidden_state,
        hidden_dim,
        num_v_heads,
        weights->ssm_alpha_w_type,
        scratch->ssm_qkv
    );

    matvec(
        scratch->attn_kv + num_v_heads,
        weights->ssm_beta_w,
        scratch->hidden_state,
        hidden_dim,
        num_v_heads,
        weights->ssm_beta_w_type,
        scratch->ssm_qkv
    );

    /* 5. Partition and L2-normalize Q_conv and K_conv */
    for (int h = 0; h < group_count; h++) {
        float *q_head = scratch->ffn_up + h * qk_head_size;
        float *k_head = scratch->ffn_up + q_conv_size + h * qk_head_size;

        double q_sum = 0.0;
        for (int i = 0; i < qk_head_size; i++) {
            q_sum += (double)q_head[i] * q_head[i];
        }
        float q_norm = (float)sqrt(q_sum + 1e-6);
        for (int i = 0; i < qk_head_size; i++) {
            q_head[i] /= q_norm;
        }

        double k_sum = 0.0;
        for (int i = 0; i < qk_head_size; i++) {
            k_sum += (double)k_head[i] * k_head[i];
        }
        float k_norm = (float)sqrt(k_sum + 1e-6);
        for (int i = 0; i < qk_head_size; i++) {
            k_head[i] /= k_norm;
        }
    }

    /* 6. Recurrent Gated Delta Rule state matrix update */
    size_t state_matrix_elements = (size_t)num_v_heads * (size_t)ssm_state_size * (size_t)ssm_state_size;
    float *ssm_state = state->ssm_states + (size_t)ssm_layer_idx * state_matrix_elements;
    float *attn_out_norm = scratch->ffn_gate + ssm_inner_size;
    float scale = 1.0f / sqrtf((float)ssm_state_size);
    int group_v_per_qk = num_v_heads / (group_count > 0 ? group_count : 1);

    for (int h = 0; h < num_v_heads; h++) {
        int h_qk = h / (group_v_per_qk > 0 ? group_v_per_qk : 1);
        const float *q_head = scratch->ffn_up + h_qk * qk_head_size;
        const float *k_head = scratch->ffn_up + q_conv_size + h_qk * qk_head_size;
        const float *v_head = scratch->ffn_up + q_conv_size + k_conv_size + h * ssm_state_size;
        float *S = ssm_state + (size_t)h * (size_t)ssm_state_size * (size_t)ssm_state_size;

        float biased_alpha = scratch->attn_kv[h] + weights->ssm_dt_bias[h];
        float gate_val = softplus(biased_alpha) * weights->ssm_a_w[h];
        float decay_scale = expf(gate_val);
        float beta_val = sigmoid(scratch->attn_kv[num_v_heads + h]);

        int matrix_len = ssm_state_size * ssm_state_size;
        for (int i = 0; i < matrix_len; i++) {
            S[i] *= decay_scale;
        }

        float delta[256];
        for (int j = 0; j < ssm_state_size; j++) {
            double sum = 0.0;
            for (int i = 0; i < ssm_state_size; i++) {
                sum += (double)S[j * ssm_state_size + i] * k_head[i];
            }
            delta[j] = (v_head[j] - (float)sum) * beta_val;
        }

        for (int j = 0; j < ssm_state_size; j++) {
            float d_val = delta[j];
            for (int i = 0; i < ssm_state_size; i++) {
                S[j * ssm_state_size + i] += k_head[i] * d_val;
            }
        }

        float *out_head = attn_out_norm + h * ssm_state_size;
        for (int j = 0; j < ssm_state_size; j++) {
            double sum = 0.0;
            for (int i = 0; i < ssm_state_size; i++) {
                sum += (double)S[j * ssm_state_size + i] * q_head[i];
            }
            out_head[j] = (float)sum * scale;
        }
    }

    /* 7. Gated Normalization */
    for (int h = 0; h < num_v_heads; h++) {
        float *out_head = attn_out_norm + h * ssm_state_size;
        double sum_sq = 0.0;
        for (int i = 0; i < ssm_state_size; i++) {
            sum_sq += (double)out_head[i] * out_head[i];
        }
        float rms = (float)sqrt(sum_sq / (double)ssm_state_size + 1e-6);
        for (int i = 0; i < ssm_state_size; i++) {
            out_head[i] = (out_head[i] / rms) * weights->ssm_norm_w[i];
        }
    }

    /* Project gating vector Z */
    matvec(
        scratch->attn_q,
        weights->attn_gate_w,
        scratch->hidden_state,
        hidden_dim,
        ssm_inner_size,
        weights->attn_gate_w_type,
        scratch->ssm_qkv
    );

    for (int i = 0; i < ssm_inner_size; i++) {
        attn_out_norm[i] *= ssm_silu(scratch->attn_q[i]);
    }

    /* 8. Output Projection */
    matvec(
        scratch->hidden_state,
        weights->ssm_out_w,
        attn_out_norm,
        ssm_inner_size,
        hidden_dim,
        weights->ssm_out_w_type,
        scratch->ssm_qkv
    );

    /* 9. Residual Addition */
    add_residual(hidden_state, hidden_state, scratch->hidden_state, hidden_dim);
}
