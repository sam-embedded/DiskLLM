#include "ssm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper softplus function
static inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

// Helper sigmoid function
static inline float sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

// Helper silu function
static inline float ssm_silu(float x) {
    return x * sigmoid(x);
}

void ssm_layer_forward(
    float * restrict hidden_state,
    int pos,
    int layer_idx,
    const ssm_layer_weights * restrict weights,
    model_state * restrict state,
    scratch_buffers * restrict scratch
) {
    (void)pos; // position parameter not strictly used in recurrent step but kept for api consistency
    
    // 1. Calculate ssm_layer_idx (SSM layer index within the 48 SSM layers)
    int attn_before = layer_idx / 4;
    int ssm_layer_idx = layer_idx - attn_before;

    // 2. Input RMSNorm: normalize hidden_state into scratch->hidden_state (size 5120)
    rmsnorm(scratch->hidden_state, hidden_state, weights->attn_norm_w, 5120, 1e-6f);

    // 3. Input Projection (attn_qkv): projects scratch->hidden_state (5120) to scratch->ffn_gate (10240)
    matvec(
        scratch->ffn_gate,
        weights->attn_qkv_w,
        scratch->hidden_state,
        5120,
        10240,
        weights->attn_qkv_w_type,
        scratch->ssm_qkv // dequantization buffer of size 10240
    );

    // 4. Conv1D Convolution:
    // conv_channels = 10240, kernel_size = 4, history size = 3.
    // The history is stored in state->ssm_conv_histories of shape [48, 3, 10240].
    float *history = state->ssm_conv_histories + ssm_layer_idx * 3 * 10240;
    float *hist_0 = history + 0 * 10240; // oldest
    float *hist_1 = history + 1 * 10240;
    float *hist_2 = history + 2 * 10240; // most recent

    // Compute convolution output + apply SiLU activation.
    // Output is stored in scratch->ffn_up (size 10240).
    // ssm_conv1d_w in GGUF: shape [4, 10240], contiguous layout is w[c * 4 + k] for channel c and kernel k
    for (int c = 0; c < 10240; c++) {
        float val = hist_0[c] * weights->ssm_conv1d_w[c * 4 + 0] +
                    hist_1[c] * weights->ssm_conv1d_w[c * 4 + 1] +
                    hist_2[c] * weights->ssm_conv1d_w[c * 4 + 2] +
                    scratch->ffn_gate[c] * weights->ssm_conv1d_w[c * 4 + 3];
        scratch->ffn_up[c] = ssm_silu(val);
    }

    // Update the Conv1D history buffer
    for (int c = 0; c < 10240; c++) {
        hist_0[c] = hist_1[c];
        hist_1[c] = hist_2[c];
        hist_2[c] = scratch->ffn_gate[c];
    }

    // 5. Alpha & Beta Projections:
    // Project scratch->hidden_state to alpha (size 48) and beta (size 48) in scratch->attn_kv
    matvec(
        scratch->attn_kv, // alpha starts at offset 0
        weights->ssm_alpha_w,
        scratch->hidden_state,
        5120,
        48,
        weights->ssm_alpha_w_type,
        scratch->ssm_qkv
    );

    matvec(
        scratch->attn_kv + 48, // beta starts at offset 48
        weights->ssm_beta_w,
        scratch->hidden_state,
        5120,
        48,
        weights->ssm_beta_w_type,
        scratch->ssm_qkv
    );

    // 6. Partition and L2-normalize Q_conv and K_conv:
    // Q_conv is first 2048 elements of scratch->ffn_up (16 heads of size 128)
    // K_conv is next 2048 elements of scratch->ffn_up (16 heads of size 128)
    // V_conv is remaining 6144 elements of scratch->ffn_up (48 heads of size 128)
    for (int h = 0; h < 16; h++) {
        float *q_head = scratch->ffn_up + h * 128;
        float *k_head = scratch->ffn_up + 2048 + h * 128;

        // Q head L2 norm
        double q_sum = 0.0;
        for (int i = 0; i < 128; i++) {
            q_sum += (double)q_head[i] * q_head[i];
        }
        float q_norm = (float)sqrt(q_sum + 1e-6);
        for (int i = 0; i < 128; i++) {
            q_head[i] /= q_norm;
        }

        // K head L2 norm
        double k_sum = 0.0;
        for (int i = 0; i < 128; i++) {
            k_sum += (double)k_head[i] * k_head[i];
        }
        float k_norm = (float)sqrt(k_sum + 1e-6);
        for (int i = 0; i < 128; i++) {
            k_head[i] /= k_norm;
        }
    }

    // 7. Recurrent Gated Delta Rule state matrix update:
    // Update the recurrent state matrices of size [128, 128] for the 48 heads in-place.
    // Store normalized attention outputs in scratch->ffn_gate + 6144 (size 6144).
    float *ssm_state = state->ssm_states + ssm_layer_idx * 6144 * 128;
    float *attn_out_norm = scratch->ffn_gate + 6144;
    float scale = 0.088388347f; // 1 / sqrt(128)

    for (int h = 0; h < 48; h++) {
        int h_qk = h / 3; // Group size = 3 (48 V heads / 16 QK heads)
        const float *q_head = scratch->ffn_up + h_qk * 128;
        const float *k_head = scratch->ffn_up + 2048 + h_qk * 128;
        const float *v_head = scratch->ffn_up + 4096 + h * 128;
        float *S = ssm_state + h * 128 * 128;

        // Compute decay factor: gate_val = softplus(alpha + dt_bias) * ssm_a
        // In GGUF, ssm_a is negative (-exp(A_log)), so gate_val is negative.
        // The decay multiplier for the state is exp(gate_val) (i.e. < 1).
        float biased_alpha = scratch->attn_kv[h] + weights->ssm_dt_bias[h];
        float gate_val = softplus(biased_alpha) * weights->ssm_a_w[h];
        float decay_scale = expf(gate_val);

        // Compute beta factor
        float beta_val = sigmoid(scratch->attn_kv[48 + h]);

        // 7.1 Decay state matrix in place
        for (int i = 0; i < 16384; i++) {
            S[i] *= decay_scale;
        }

        // 7.2 Compute delta vector: delta = (V - S @ K) * beta
        float delta[128];
        for (int j = 0; j < 128; j++) {
            double sum = 0.0;
            for (int i = 0; i < 128; i++) {
                sum += (double)S[j * 128 + i] * k_head[i];
            }
            delta[j] = (v_head[j] - (float)sum) * beta_val;
        }

        // 7.3 Add outer product to state: S += outer(delta, K)
        for (int j = 0; j < 128; j++) {
            float d_val = delta[j];
            for (int i = 0; i < 128; i++) {
                S[j * 128 + i] += k_head[i] * d_val;
            }
        }

        // 7.4 Compute attention output: out = S @ Q
        float *out_head = attn_out_norm + h * 128;
        for (int j = 0; j < 128; j++) {
            double sum = 0.0;
            for (int i = 0; i < 128; i++) {
                sum += (double)S[j * 128 + i] * q_head[i];
            }
            out_head[j] = (float)sum * scale;
        }
    }

    // 8. Gated Normalization: RMSNorm(out) * silu(Z)
    // 8.1 Apply ssm_norm weight per head of size 128
    for (int h = 0; h < 48; h++) {
        float *out_head = attn_out_norm + h * 128;
        double sum_sq = 0.0;
        for (int i = 0; i < 128; i++) {
            sum_sq += (double)out_head[i] * out_head[i];
        }
        float rms = (float)sqrt(sum_sq / 128.0 + 1e-6);
        for (int i = 0; i < 128; i++) {
            out_head[i] = (out_head[i] / rms) * weights->ssm_norm_w[i];
        }
    }

    // 8.2 Project hidden_state to gating vector Z (size 6144) using attn_gate_w
    // Z is stored in scratch->attn_q (size 6144)
    matvec(
        scratch->attn_q,
        weights->attn_gate_w,
        scratch->hidden_state,
        5120,
        6144,
        weights->attn_gate_w_type,
        scratch->ssm_qkv
    );

    // 8.3 Multiply by silu(Z)
    for (int i = 0; i < 6144; i++) {
        attn_out_norm[i] *= ssm_silu(scratch->attn_q[i]);
    }

    // 9. Output Projection: Project attn_out_norm (6144) to scratch->hidden_state (5120) using ssm_out_w
    matvec(
        scratch->hidden_state,
        weights->ssm_out_w,
        attn_out_norm,
        6144,
        5120,
        weights->ssm_out_w_type,
        scratch->ssm_qkv
    );

    // 10. Residual Addition: hidden_state += scratch->hidden_state
    add_residual(hidden_state, hidden_state, scratch->hidden_state, 5120);
}
