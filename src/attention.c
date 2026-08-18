#include "attention.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Portable float-to-FP16 converter
static inline uint16_t fp32_to_fp16(float val) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = val;
    uint32_t sign = (u.i >> 16) & 0x8000;
    int32_t exponent = ((u.i >> 23) & 0xff) - 127;
    uint32_t mantissa = u.i & 0x7fffff;

    if (exponent == -127) { // Zero or subnormal
        return sign;
    } else if (exponent > 15) { // Overflow, map to infinity
        return sign | 0x7c00;
    } else if (exponent < -14) { // Underflow, map to subnormal or zero
        int32_t shift = -14 - exponent;
        if (shift > 10) return sign;
        mantissa |= 0x800000; // Explicit leading bit
        mantissa >>= (13 + shift);
        return sign | mantissa;
    } else {
        return sign | ((exponent + 15) << 10) | (mantissa >> 13);
    }
}

// Portable FP16-to-float converter using bitwise manipulation
static inline float fp16_to_fp32(uint16_t h) {
    union {
        uint32_t u;
        float f;
    } conv;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7c00) >> 10;
    uint32_t mant = h & 0x03ff;

    if (exp == 0) {
        if (mant == 0) {
            conv.u = sign;
            return conv.f;
        }
        // Subnormal: normalize mantissa
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        // Infinity or NaN
        conv.u = sign | 0x7f800000 | (mant << 13);
        return conv.f;
    }

    conv.u = sign | (((exp - 15 + 127) & 0xff) << 23) | (mant << 13);
    return conv.f;
}

// Apply Rotary Position Embeddings (RoPE) to first rope_dim dimensions using half-split pairing
static void apply_rope(float * restrict vec, int pos, double freq_base, int rope_dim) {
    int half_dim = rope_dim / 2; // e.g. 32 for rope_dim = 64
    for (int i = 0; i < half_dim; i++) {
        double freq = 1.0 / pow(freq_base, (double)(2 * i) / rope_dim);
        double theta = (double)pos * freq;
        float cos_val = (float)cos(theta);
        float sin_val = (float)sin(theta);

        float x0 = vec[i];
        float x1 = vec[i + half_dim];

        vec[i]            = x0 * cos_val - x1 * sin_val;
        vec[i + half_dim] = x0 * sin_val + x1 * cos_val;
    }
}

void attention_forward(
    float * restrict hidden_state,
    int pos,
    int layer_idx,
    const attention_layer_weights * restrict weights,
    model_state * restrict state,
    scratch_buffers * restrict scratch,
    double freq_base,
    int rope_dim
) {
    // Hidden dimension size is 5120
    // Number of Q heads = 24, KV heads = 4. Head dimension = 256.
    // Q output size = 12288 (contains interleaved Query [256] + Gate [256] per head for 24 heads).
    // K and V projection outputs are 1024 each (4 heads * 256).

    // Step 1: Input RMSNorm
    rmsnorm(scratch->hidden_state, hidden_state, weights->attn_norm_w, 5120, 1e-6f);

    // Step 2: Projections
    // 2.1 Project Query + Gate (size 12288)
    matvec(
        scratch->attn_q,
        weights->attn_q_w,
        scratch->hidden_state,
        5120,
        12288,
        weights->attn_q_w_type,
        scratch->ssm_qkv
    );

    // 2.2 Project Key (size 1024)
    matvec(
        scratch->attn_kv,
        weights->attn_k_w,
        scratch->hidden_state,
        5120,
        1024,
        weights->attn_k_w_type,
        scratch->ssm_qkv
    );

    // 2.3 Apply QK-Norm to Q and K (BEFORE RoPE)
    // For each of the 24 Q heads, Q vector is at offset h_q * 512
    for (int h_q = 0; h_q < 24; h_q++) {
        float *q_head = scratch->attn_q + h_q * 512;
        rmsnorm(q_head, q_head, weights->attn_q_norm_w, 256, 1e-6f);
    }
    // For each of the 4 K heads, K vector is at offset h_kv * 256
    for (int h_kv = 0; h_kv < 4; h_kv++) {
        float *k_head = scratch->attn_kv + h_kv * 256;
        rmsnorm(k_head, k_head, weights->attn_k_norm_w, 256, 1e-6f);
    }

    // 2.4 Apply RoPE to Q and K (using rope_dim = 64)
    for (int h_q = 0; h_q < 24; h_q++) {
        float *q_head = scratch->attn_q + h_q * 512;
        apply_rope(q_head, pos, freq_base, rope_dim);
    }
    for (int h_kv = 0; h_kv < 4; h_kv++) {
        float *k_head = scratch->attn_kv + h_kv * 256;
        apply_rope(k_head, pos, freq_base, rope_dim);
    }

    // 2.5 Store Key in the KV Cache
    int cache_layer_idx = (layer_idx - 3) / 4;
    uint16_t *layer_cache = state->kv_cache + cache_layer_idx * state->context_length * 2 * 1024;
    uint16_t *cache_k = layer_cache + pos * 2048;
    for (int i = 0; i < 1024; i++) {
        cache_k[i] = fp32_to_fp16(scratch->attn_kv[i]);
    }

    // 2.6 Project Value (size 1024)
    matvec(
        scratch->attn_kv,
        weights->attn_v_w,
        scratch->hidden_state,
        5120,
        1024,
        weights->attn_v_w_type,
        scratch->ssm_qkv
    );

    // 2.7 Store Value in the KV Cache
    uint16_t *cache_v = layer_cache + pos * 2048 + 1024;
    for (int i = 0; i < 1024; i++) {
        cache_v[i] = fp32_to_fp16(scratch->attn_kv[i]);
    }

    // Step 3: Attention Computation (Grouped-Query Attention)
    float *attn_out = scratch->ffn_gate;
    float *scores = scratch->ffn_up;

    for (int h_q = 0; h_q < 24; h_q++) {
        int h_kv = h_q / 6; // Group size = 6 (24 Q heads / 4 KV heads)
        const float *q_head = scratch->attn_q + h_q * 512;

        // 3.1 Compute attention scores from position 0 up to pos (inclusive)
        for (int p = 0; p <= pos; p++) {
            const uint16_t *p_cache_k = state->kv_cache + cache_layer_idx * state->context_length * 2048 + p * 2048;
            const uint16_t *k_cache_head = p_cache_k + h_kv * 256;

            double sum = 0.0;
            for (int j = 0; j < 256; j++) {
                float k_val = fp16_to_fp32(k_cache_head[j]);
                sum += (double)q_head[j] * (double)k_val;
            }
            scores[p] = (float)sum * 0.0625f; // scale by 1 / sqrt(256) = 0.0625
        }

        // 3.2 Apply softmax over sequence positions
        float max_score = scores[0];
        for (int p = 1; p <= pos; p++) {
            if (scores[p] > max_score) {
                max_score = scores[p];
            }
        }
        double sum_exp = 0.0;
        for (int p = 0; p <= pos; p++) {
            scores[p] = expf(scores[p] - max_score);
            sum_exp += (double)scores[p];
        }
        float inv_sum = (float)(1.0 / sum_exp);
        for (int p = 0; p <= pos; p++) {
            scores[p] *= inv_sum;
        }

        // 3.3 Compute weighted sum of values
        float *out_head = attn_out + h_q * 256;
        memset(out_head, 0, 256 * sizeof(float));

        for (int p = 0; p <= pos; p++) {
            float prob = scores[p];
            const uint16_t *p_cache_v = state->kv_cache + cache_layer_idx * state->context_length * 2048 + p * 2048 + 1024;
            const uint16_t *v_cache_head = p_cache_v + h_kv * 256;

            for (int j = 0; j < 256; j++) {
                float v_val = fp16_to_fp32(v_cache_head[j]);
                out_head[j] += prob * v_val;
            }
        }

        // 3.4 Apply per-head Attention Gate (sigmoid activation)
        const float *gate_head = scratch->attn_q + h_q * 512 + 256;
        for (int j = 0; j < 256; j++) {
            float sig_gate = 1.0f / (1.0f + expf(-gate_head[j]));
            out_head[j] *= sig_gate;
        }
    }

    // Step 4: Output Projection
    // Project the gated attention outputs (size 6144) back to hidden_size (size 5120)
    matvec(
        scratch->hidden_state,
        weights->attn_output_w,
        attn_out,
        6144,
        5120,
        weights->attn_output_w_type,
        scratch->ssm_qkv
    );

    // Step 5: Residual Addition
    add_residual(hidden_state, hidden_state, scratch->hidden_state, 5120);
}
