#include "attention.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline uint16_t fp32_to_fp16(float val) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = val;
    uint32_t sign = (u.i >> 16) & 0x8000;
    int32_t exponent = ((u.i >> 23) & 0xff) - 127;
    uint32_t mantissa = u.i & 0x7fffff;

    if (exponent == -127) {
        return sign;
    } else if (exponent > 15) {
        return sign | 0x7c00;
    } else if (exponent < -14) {
        int32_t shift = -14 - exponent;
        if (shift > 10) return sign;
        mantissa |= 0x800000;
        mantissa >>= (13 + shift);
        return sign | mantissa;
    } else {
        return sign | ((exponent + 15) << 10) | (mantissa >> 13);
    }
}

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
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        conv.u = sign | 0x7f800000 | (mant << 13);
        return conv.f;
    }

    conv.u = sign | (((exp - 15 + 127) & 0xff) << 23) | (mant << 13);
    return conv.f;
}

static void apply_rope(float * restrict vec, int pos, double freq_base, int rope_dim, const qwen_model_config *cfg) {
    int half_dim = rope_dim / 2;
    double scaling_factor = (cfg && cfg->rope_scaling_factor > 1.0f) ? (double)cfg->rope_scaling_factor : 1.0;
    int orig_len = (cfg && cfg->rope_orig_context_len > 0) ? cfg->rope_orig_context_len : 4096;
    int scaling_type = cfg ? (int)cfg->rope_scaling_type : 0;

    for (int i = 0; i < half_dim; i++) {
        double freq = 1.0 / pow(freq_base, (double)(2 * i) / rope_dim);
        double theta = (double)pos * freq;

        if (scaling_type == ROPE_SCALING_LINEAR && scaling_factor > 1.0) {
            theta /= scaling_factor;
        } else if (scaling_type == ROPE_SCALING_YARN && scaling_factor > 1.0) {
            double wavelength = 2.0 * M_PI / freq;
            double b_fast = (cfg->rope_beta_fast > 0) ? (double)cfg->rope_beta_fast : 32.0;
            double b_slow = (cfg->rope_beta_slow > 0) ? (double)cfg->rope_beta_slow : 1.0;
            double gamma = (wavelength - b_slow * (double)orig_len) / ((b_fast - b_slow) * (double)orig_len);
            if (gamma < 0.0) gamma = 0.0;
            if (gamma > 1.0) gamma = 1.0;

            double s_i = (1.0 - gamma) + gamma * scaling_factor;
            theta /= s_i;
        }

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
    const qwen_model_config *cfg
) {
    int hidden_dim     = cfg ? cfg->hidden_dim : 5120;
    int num_attn_heads = cfg ? cfg->num_attn_heads : 24;
    int num_kv_heads   = cfg ? cfg->num_kv_heads : 4;
    int head_dim       = cfg ? cfg->key_length : 256;
    double freq_base   = cfg ? (double)cfg->rope_freq_base : 10000000.0;
    int rope_dim       = cfg ? cfg->rope_dim : 64;

    int q_dim_per_head = head_dim * 2;
    int q_total_dim    = num_attn_heads * q_dim_per_head;
    int k_total_dim    = num_kv_heads * head_dim;
    int v_total_dim    = num_kv_heads * head_dim;
    int attn_out_dim   = num_attn_heads * head_dim;
    int group_size     = num_attn_heads / (num_kv_heads > 0 ? num_kv_heads : 1);

    /* Step 1: Input RMSNorm */
    rmsnorm(scratch->hidden_state, hidden_state, weights->attn_norm_w, hidden_dim, 1e-6f);

    /* Step 2: Projections */
    /* 2.1 Project Query + Gate */
    matvec(
        scratch->attn_q,
        weights->attn_q_w,
        scratch->hidden_state,
        hidden_dim,
        q_total_dim,
        weights->attn_q_w_type,
        scratch->ssm_qkv
    );

    /* 2.2 Project Key */
    matvec(
        scratch->attn_kv,
        weights->attn_k_w,
        scratch->hidden_state,
        hidden_dim,
        k_total_dim,
        weights->attn_k_w_type,
        scratch->ssm_qkv
    );

    /* 2.3 Apply QK-Norm to Q and K (BEFORE RoPE) */
    if (weights->attn_q_norm_w) {
        for (int h_q = 0; h_q < num_attn_heads; h_q++) {
            float *q_head = scratch->attn_q + h_q * q_dim_per_head;
            rmsnorm(q_head, q_head, weights->attn_q_norm_w, head_dim, 1e-6f);
        }
    }
    if (weights->attn_k_norm_w) {
        for (int h_kv = 0; h_kv < num_kv_heads; h_kv++) {
            float *k_head = scratch->attn_kv + h_kv * head_dim;
            rmsnorm(k_head, k_head, weights->attn_k_norm_w, head_dim, 1e-6f);
        }
    }

    /* 2.4 Apply RoPE to Q and K */
    for (int h_q = 0; h_q < num_attn_heads; h_q++) {
        float *q_head = scratch->attn_q + h_q * q_dim_per_head;
        apply_rope(q_head, pos, freq_base, rope_dim, cfg);
    }
    for (int h_kv = 0; h_kv < num_kv_heads; h_kv++) {
        float *k_head = scratch->attn_kv + h_kv * head_dim;
        apply_rope(k_head, pos, freq_base, rope_dim, cfg);
    }

    /* 2.5 Store Key in the KV Cache */
    int cache_layer_idx = state->layer_to_attn_idx[layer_idx];
    if (cache_layer_idx < 0) cache_layer_idx = 0;
    uint16_t *layer_cache = state->kv_cache + (size_t)cache_layer_idx * state->context_length * 2 * state->kv_dim;
    uint16_t *cache_k = layer_cache + (size_t)pos * 2 * state->kv_dim;
    for (int i = 0; i < k_total_dim; i++) {
        cache_k[i] = fp32_to_fp16(scratch->attn_kv[i]);
    }

    /* 2.6 Project Value */
    matvec(
        scratch->attn_kv,
        weights->attn_v_w,
        scratch->hidden_state,
        hidden_dim,
        v_total_dim,
        weights->attn_v_w_type,
        scratch->ssm_qkv
    );

    /* 2.7 Store Value in the KV Cache */
    uint16_t *cache_v = cache_k + state->kv_dim;
    for (int i = 0; i < v_total_dim; i++) {
        cache_v[i] = fp32_to_fp16(scratch->attn_kv[i]);
    }

    /* Step 3: Attention Computation (Grouped-Query Attention) */
    float *attn_out = scratch->ffn_gate;
    float *scores = scratch->ffn_up;
    float scale = 1.0f / sqrtf((float)head_dim);
    if (cfg && cfg->rope_scaling_type == ROPE_SCALING_YARN && cfg->rope_scaling_factor > 1.0f) {
        float yarn_attn_scale = (cfg->rope_attn_factor > 0.0f) ? cfg->rope_attn_factor : (0.1f * logf(cfg->rope_scaling_factor) + 1.0f);
        scale *= yarn_attn_scale;
    }

    for (int h_q = 0; h_q < num_attn_heads; h_q++) {
        int h_kv = h_q / (group_size > 0 ? group_size : 1);
        const float *q_head = scratch->attn_q + h_q * q_dim_per_head;

        for (int p = 0; p <= pos; p++) {
            const uint16_t *p_cache_k = layer_cache + (size_t)p * 2 * state->kv_dim;
            const uint16_t *k_cache_head = p_cache_k + h_kv * head_dim;

            double sum = 0.0;
            for (int j = 0; j < head_dim; j++) {
                float k_val = fp16_to_fp32(k_cache_head[j]);
                sum += (double)q_head[j] * (double)k_val;
            }
            scores[p] = (float)sum * scale;
        }

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

        float *out_head = attn_out + h_q * head_dim;
        memset(out_head, 0, head_dim * sizeof(float));

        for (int p = 0; p <= pos; p++) {
            float prob = scores[p];
            const uint16_t *p_cache_v = layer_cache + (size_t)p * 2 * state->kv_dim + state->kv_dim;
            const uint16_t *v_cache_head = p_cache_v + h_kv * head_dim;

            for (int j = 0; j < head_dim; j++) {
                float v_val = fp16_to_fp32(v_cache_head[j]);
                out_head[j] += prob * v_val;
            }
        }

        const float *gate_head = q_head + head_dim;
        for (int j = 0; j < head_dim; j++) {
            float sig_gate = 1.0f / (1.0f + expf(-gate_head[j]));
            out_head[j] *= sig_gate;
        }
    }

    /* Step 4: Output Projection */
    matvec(
        scratch->hidden_state,
        weights->attn_output_w,
        attn_out,
        attn_out_dim,
        hidden_dim,
        weights->attn_output_w_type,
        scratch->ssm_qkv
    );

    /* Step 5: Residual Addition */
    add_residual(hidden_state, hidden_state, scratch->hidden_state, hidden_dim);
}
