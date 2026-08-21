#include "attention.h"
#include "kernels.h"
#include "dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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

static inline void quantize_q8_0(const float * restrict src, void * restrict dst, int k) {
    int nb = k / 32;
    block_q8_0 *b_dst = (block_q8_0 *)dst;
    for (int i = 0; i < nb; i++) {
        const float *x = src + i * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(x[j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d ? 1.0f / d : 0.0f;
        b_dst[i].d = fp32_to_fp16(d);
        for (int j = 0; j < 32; j++) {
            int val = (int)roundf(x[j] * id);
            if (val > 127) val = 127;
            if (val < -127) val = -127;
            b_dst[i].qs[j] = (int8_t)val;
        }
    }
}

static inline void quantize_q4_0(const float * restrict src, void * restrict dst, int k) {
    int nb = k / 32;
    block_q4_0 *b_dst = (block_q4_0 *)dst;
    for (int i = 0; i < nb; i++) {
        const float *x = src + i * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(x[j]);
            if (v > amax) amax = v;
        }
        float d = amax / 7.0f;
        float id = d ? 1.0f / d : 0.0f;
        b_dst[i].d = fp32_to_fp16(d);
        for (int j = 0; j < 16; j++) {
            int val0 = (int)roundf(x[j] * id) + 8;
            int val1 = (int)roundf(x[j + 16] * id) + 8;
            if (val0 < 0) val0 = 0;
            if (val0 > 15) val0 = 15;
            if (val1 < 0) val1 = 0;
            if (val1 > 15) val1 = 15;
            b_dst[i].qs[j] = (uint8_t)(val0 | (val1 << 4));
        }
    }
}

static inline void store_kv_vector(void * restrict dst, const float * restrict src, int k_dim, cache_type_t ctype) {
    if (ctype == CACHE_TYPE_Q8_0) {
        quantize_q8_0(src, dst, k_dim);
    } else if (ctype == CACHE_TYPE_Q4_0) {
        quantize_q4_0(src, dst, k_dim);
    } else {
        uint16_t *f16_dst = (uint16_t *)dst;
        for (int i = 0; i < k_dim; i++) {
            f16_dst[i] = fp32_to_fp16(src[i]);
        }
    }
}

static inline float dot_q8_0_f32(const block_q8_0 * restrict k_block, const float * restrict q_head, int head_dim) {
    int nb = head_dim / 32;
    double sum = 0.0;
    for (int b = 0; b < nb; b++) {
        float d = fp16_to_fp32(k_block[b].d);
        const int8_t *qs = k_block[b].qs;
        const float *q = q_head + b * 32;
        float bsum = 0.0f;
        for (int j = 0; j < 32; j++) {
            bsum += (float)qs[j] * q[j];
        }
        sum += (double)bsum * (double)d;
    }
    return (float)sum;
}

static inline float dot_q4_0_f32(const block_q4_0 * restrict k_block, const float * restrict q_head, int head_dim) {
    int nb = head_dim / 32;
    double sum = 0.0;
    for (int b = 0; b < nb; b++) {
        float d = fp16_to_fp32(k_block[b].d);
        const uint8_t *qs = k_block[b].qs;
        const float *q = q_head + b * 32;
        float bsum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uint8_t qv = qs[j];
            int v0 = (qv & 0x0F) - 8;
            int v1 = (qv >> 4) - 8;
            bsum += (float)v0 * q[j] + (float)v1 * q[j + 16];
        }
        sum += (double)bsum * (double)d;
    }
    return (float)sum;
}

static inline float compute_qk_score(const void * restrict k_head_ptr, const float * restrict q_head, int head_dim, cache_type_t ctype) {
    if (ctype == CACHE_TYPE_Q8_0) {
        return dot_q8_0_f32((const block_q8_0 *)k_head_ptr, q_head, head_dim);
    } else if (ctype == CACHE_TYPE_Q4_0) {
        return dot_q4_0_f32((const block_q4_0 *)k_head_ptr, q_head, head_dim);
    } else {
        const uint16_t *k_f16 = (const uint16_t *)k_head_ptr;
        double sum = 0.0;
        for (int j = 0; j < head_dim; j++) {
            float k_val = fp16_to_fp32(k_f16[j]);
            sum += (double)q_head[j] * (double)k_val;
        }
        return (float)sum;
    }
}

static inline void add_v_scaled(float * restrict out_head, const void * restrict v_head_ptr, float prob, int head_dim, cache_type_t ctype) {
    if (ctype == CACHE_TYPE_Q8_0) {
        const block_q8_0 *v_block = (const block_q8_0 *)v_head_ptr;
        int nb = head_dim / 32;
        for (int b = 0; b < nb; b++) {
            float scale = prob * fp16_to_fp32(v_block[b].d);
            float *out = out_head + b * 32;
            const int8_t *qs = v_block[b].qs;
            for (int j = 0; j < 32; j++) {
                out[j] += scale * (float)qs[j];
            }
        }
    } else if (ctype == CACHE_TYPE_Q4_0) {
        const block_q4_0 *v_block = (const block_q4_0 *)v_head_ptr;
        int nb = head_dim / 32;
        for (int b = 0; b < nb; b++) {
            float scale = prob * fp16_to_fp32(v_block[b].d);
            float *out = out_head + b * 32;
            const uint8_t *qs = v_block[b].qs;
            for (int j = 0; j < 16; j++) {
                uint8_t qv = qs[j];
                int v0 = (qv & 0x0F) - 8;
                int v1 = (qv >> 4) - 8;
                out[j]      += scale * (float)v0;
                out[j + 16] += scale * (float)v1;
            }
        }
    } else {
        const uint16_t *v_f16 = (const uint16_t *)v_head_ptr;
        for (int j = 0; j < head_dim; j++) {
            float v_val = fp16_to_fp32(v_f16[j]);
            out_head[j] += prob * v_val;
        }
    }
}

static void apply_rope(float * restrict vec, int pos, double freq_base, int rope_dim, const float *rope_freqs, const qwen_model_config *cfg) {
    int half_dim = rope_dim / 2;
    double scaling_factor = (cfg && cfg->rope_scaling_factor > 1.0f) ? (double)cfg->rope_scaling_factor : 1.0;
    int orig_len = (cfg && cfg->rope_orig_context_len > 0) ? cfg->rope_orig_context_len : 4096;
    int scaling_type = cfg ? (int)cfg->rope_scaling_type : 0;

    int is_neox = (cfg && (cfg->model_type == MODEL_TYPE_QWEN_HYBRID || cfg->model_type == MODEL_TYPE_QWEN_ATTENTION_ONLY || cfg->model_type == MODEL_TYPE_PHI3 || cfg->model_type == MODEL_TYPE_GEMMA || cfg->model_type == MODEL_TYPE_GEMMA2 || cfg->model_type == MODEL_TYPE_GEMMA3 || cfg->model_type == MODEL_TYPE_GEMMA4));

    for (int i = 0; i < half_dim; i++) {
        double freq = 1.0 / pow(freq_base, (double)(2 * i) / rope_dim);
        if (rope_freqs) {
            freq /= (double)rope_freqs[i];
        }
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

        if (is_neox) {
            float x0 = vec[i];
            float x1 = vec[i + half_dim];
            vec[i]            = x0 * cos_val - x1 * sin_val;
            vec[i + half_dim] = x0 * sin_val + x1 * cos_val;
        } else {
            float x0 = vec[2 * i];
            float x1 = vec[2 * i + 1];
            vec[2 * i]     = x0 * cos_val - x1 * sin_val;
            vec[2 * i + 1] = x0 * sin_val + x1 * cos_val;
        }
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
    int is_phi3 = (cfg && cfg->model_type == MODEL_TYPE_PHI3);
    int is_qwen_hybrid = (cfg && cfg->model_type == MODEL_TYPE_QWEN_HYBRID);

    int head_dim = cfg ? cfg->key_length : 256;
    if (weights && weights->q_total_dim > 0 && num_attn_heads > 0) {
        head_dim = weights->q_total_dim / (num_attn_heads * (is_qwen_hybrid ? 2 : 1));
    }
    double freq_base   = cfg ? (double)cfg->rope_freq_base : 10000000.0;
    if (cfg && cfg->model_type == MODEL_TYPE_GEMMA4) {
        if (head_dim <= 256) {
            freq_base = 10000.0;
        } else {
            freq_base = 1000000.0;
        }
    }
    int rope_dim       = cfg ? cfg->rope_dim : head_dim;
    if (rope_dim > head_dim) rope_dim = head_dim;

    int q_dim_per_head = is_qwen_hybrid ? (head_dim * 2) : head_dim;
    int q_total_dim    = (weights && weights->q_total_dim > 0) ? weights->q_total_dim : (num_attn_heads * q_dim_per_head);
    int k_total_dim    = (weights && weights->k_total_dim > 0) ? weights->k_total_dim : (num_kv_heads * head_dim);
    int v_total_dim    = (weights && weights->v_total_dim > 0) ? weights->v_total_dim : (num_kv_heads * head_dim);
    int attn_out_dim   = (weights && weights->attn_out_dim > 0) ? weights->attn_out_dim : (num_attn_heads * head_dim);
    int group_size     = num_attn_heads / (num_kv_heads > 0 ? num_kv_heads : 1);

    /* Step 1: Input RMSNorm */
    int add_one = 0;
    rmsnorm_ext(scratch->hidden_state, hidden_state, weights->attn_norm_w, hidden_dim, 1e-6f, add_one);

    /* Step 2: Projections */
    if (is_phi3 && weights && weights->attn_qkv_w) {
        /* Phi-3: single fused QKV matvec, then split */
        int fused_qkv_dim = q_total_dim + k_total_dim + v_total_dim;
        /* Use ssm_qkv scratch as temp buffer (must be large enough) */
        float *fused_buf = (float *)scratch->ssm_qkv;
        matvec(
            fused_buf,
            weights->attn_qkv_w,
            scratch->hidden_state,
            hidden_dim,
            fused_qkv_dim,
            weights->attn_qkv_w_type,
            scratch->ssm_qkv + fused_qkv_dim
        );
        /* Split: Q first, then K, then V */
        memcpy(scratch->attn_q,  fused_buf,                       q_total_dim * sizeof(float));
        memcpy(scratch->attn_kv, fused_buf + q_total_dim,         k_total_dim * sizeof(float));
        /* Store K cache first, then compute V */
        /* Step 2.4 Apply RoPE (neox style for Phi-3) */
        for (int h_q = 0; h_q < num_attn_heads; h_q++) {
            float *q_head = scratch->attn_q + h_q * q_dim_per_head;
            apply_rope(q_head, pos, freq_base, rope_dim, weights ? weights->rope_freqs : NULL, cfg);
        }
        for (int h_kv = 0; h_kv < num_kv_heads; h_kv++) {
            float *k_head = scratch->attn_kv + h_kv * head_dim;
            apply_rope(k_head, pos, freq_base, rope_dim, weights ? weights->rope_freqs : NULL, cfg);
        }

        int cache_layer_idx_f = state->layer_to_attn_idx[layer_idx];
        if (cache_layer_idx_f < 0) cache_layer_idx_f = 0;
        cache_type_t ctype_f = state->cache_type;
        size_t kv_token_bytes_f = state->kv_token_bytes;
        size_t vec_bytes_f = kv_token_bytes_f / 2;
        uint8_t *layer_cache_f = (uint8_t *)state->kv_cache + (size_t)cache_layer_idx_f * (size_t)state->context_length * kv_token_bytes_f;
        uint8_t *cache_k_f = layer_cache_f + (size_t)pos * kv_token_bytes_f;
        uint8_t *cache_v_f = cache_k_f + vec_bytes_f;
        store_kv_vector(cache_k_f, scratch->attn_kv, k_total_dim, ctype_f);
        /* Copy V from fused_buf */
        const float *v_src = fused_buf + q_total_dim + k_total_dim;
        store_kv_vector(cache_v_f, v_src, v_total_dim, ctype_f);

        /* Attention computation (GQA) */
        float *attn_out_f = scratch->ffn_gate;
        float *scores_f   = scratch->ffn_up;
        float scale_f = 1.0f / sqrtf((float)head_dim);
        float softcap_f = (cfg ? cfg->attn_logit_softcapping : 0.0f);
        size_t head_k_bytes_f = 0;
        if (ctype_f == CACHE_TYPE_Q8_0) {
            head_k_bytes_f = ((size_t)head_dim / 32) * 34;
        } else if (ctype_f == CACHE_TYPE_Q4_0) {
            head_k_bytes_f = ((size_t)head_dim / 32) * 18;
        } else {
            head_k_bytes_f = (size_t)head_dim * sizeof(uint16_t);
        }
        size_t head_v_bytes_f = head_k_bytes_f;

        for (int h_q = 0; h_q < num_attn_heads; h_q++) {
            int h_kv = h_q / (group_size > 0 ? group_size : 1);
            const float *q_head_f = scratch->attn_q + h_q * q_dim_per_head;
            for (int p = 0; p <= pos; p++) {
                const uint8_t *p_cache_k = layer_cache_f + (size_t)p * kv_token_bytes_f;
                const void *k_head_ptr = p_cache_k + (size_t)h_kv * head_k_bytes_f;
                float score = compute_qk_score(k_head_ptr, q_head_f, head_dim, ctype_f) * scale_f;
                if (softcap_f > 0.0f) score = softcap_f * tanhf(score / softcap_f);
                scores_f[p] = score;
            }
            float max_score_f = scores_f[0];
            for (int p = 1; p <= pos; p++) {
                if (scores_f[p] > max_score_f) max_score_f = scores_f[p];
            }
            double sum_exp_f = 0.0;
            for (int p = 0; p <= pos; p++) {
                scores_f[p] = expf(scores_f[p] - max_score_f);
                sum_exp_f += (double)scores_f[p];
            }
            float inv_sum_f = (float)(1.0 / sum_exp_f);
            for (int p = 0; p <= pos; p++) scores_f[p] *= inv_sum_f;

            float *out_head_f = attn_out_f + h_q * head_dim;
            memset(out_head_f, 0, head_dim * sizeof(float));
            for (int p = 0; p <= pos; p++) {
                const uint8_t *p_cache_v = layer_cache_f + (size_t)p * kv_token_bytes_f + vec_bytes_f;
                const void *v_head_ptr = p_cache_v + (size_t)h_kv * head_v_bytes_f;
                add_v_scaled(out_head_f, v_head_ptr, scores_f[p], head_dim, ctype_f);
            }
        }
        /* Output projection */
        matvec(
            scratch->hidden_state,
            weights->attn_output_w,
            attn_out_f,
            attn_out_dim,
            hidden_dim,
            weights->attn_output_w_type,
            scratch->ssm_qkv
        );
        add_residual(hidden_state, hidden_state, scratch->hidden_state, hidden_dim);
        return;
    }

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
    if (weights->attn_q_b) {
        for (int i = 0; i < q_total_dim; i++) scratch->attn_q[i] += weights->attn_q_b[i];
    }

    /* 2.1b Apply QK-Norm to Q (BEFORE RoPE) */
    if (weights->attn_q_norm_w) {
        for (int h_q = 0; h_q < num_attn_heads; h_q++) {
            float *q_head = scratch->attn_q + h_q * q_dim_per_head;
            rmsnorm_ext(q_head, q_head, weights->attn_q_norm_w, head_dim, 1e-6f, add_one);
        }
    }

    /* 2.1c Apply RoPE to Q */
    for (int h_q = 0; h_q < num_attn_heads; h_q++) {
        float *q_head = scratch->attn_q + h_q * q_dim_per_head;
        apply_rope(q_head, pos, freq_base, rope_dim, weights ? weights->rope_freqs : NULL, cfg);
    }

    int cache_layer_idx = state->layer_to_attn_idx[layer_idx];
    if (cache_layer_idx < 0) cache_layer_idx = 0;

    cache_type_t ctype = state->cache_type;
    size_t kv_token_bytes = state->kv_token_bytes;
    size_t vec_bytes = kv_token_bytes / 2;
    size_t head_k_bytes = 0;
    if (ctype == CACHE_TYPE_Q8_0) {
        head_k_bytes = ((size_t)head_dim / 32) * 34;
    } else if (ctype == CACHE_TYPE_Q4_0) {
        head_k_bytes = ((size_t)head_dim / 32) * 18;
    } else {
        head_k_bytes = (size_t)head_dim * sizeof(uint16_t);
    }
    size_t head_v_bytes = head_k_bytes;

    uint8_t *layer_cache = (uint8_t *)state->kv_cache + (size_t)cache_layer_idx * (size_t)state->context_length * kv_token_bytes;

    /* 2.2 Project Key & Value if present (non-shared KV layers) */
    if (weights->attn_k_w) {
        matvec(
            scratch->attn_kv,
            weights->attn_k_w,
            scratch->hidden_state,
            hidden_dim,
            k_total_dim,
            weights->attn_k_w_type,
            scratch->ssm_qkv
        );
        if (weights->attn_k_b) {
            for (int i = 0; i < k_total_dim; i++) scratch->attn_kv[i] += weights->attn_k_b[i];
        }

        /* 2.3 Apply QK-Norm to K (BEFORE RoPE) */
        if (weights->attn_k_norm_w) {
            for (int h_kv = 0; h_kv < num_kv_heads; h_kv++) {
                float *k_head = scratch->attn_kv + h_kv * head_dim;
                rmsnorm_ext(k_head, k_head, weights->attn_k_norm_w, head_dim, 1e-6f, add_one);
            }
        }

        /* 2.4 Apply RoPE to K */
        for (int h_kv = 0; h_kv < num_kv_heads; h_kv++) {
            float *k_head = scratch->attn_kv + h_kv * head_dim;
            apply_rope(k_head, pos, freq_base, rope_dim, weights ? weights->rope_freqs : NULL, cfg);
        }

        uint8_t *cache_k = layer_cache + (size_t)pos * kv_token_bytes;
        uint8_t *cache_v = cache_k + vec_bytes;

        /* 2.5 Store Key in the KV Cache */
        store_kv_vector(cache_k, scratch->attn_kv, k_total_dim, ctype);

        /* 2.6 Project Value */
        const void *v_w = weights->attn_v_w ? weights->attn_v_w : weights->attn_k_w;
        int v_w_type = weights->attn_v_w ? weights->attn_v_w_type : weights->attn_k_w_type;
        matvec(
            scratch->attn_kv,
            v_w,
            scratch->hidden_state,
            hidden_dim,
            v_total_dim,
            v_w_type,
            scratch->ssm_qkv
        );
        if (weights->attn_v_b) {
            for (int i = 0; i < v_total_dim; i++) scratch->attn_kv[i] += weights->attn_v_b[i];
        }

        if (cfg && cfg->model_type == MODEL_TYPE_GEMMA4) {
            for (int h_kv = 0; h_kv < num_kv_heads; h_kv++) {
                float *v_head = scratch->attn_kv + h_kv * head_dim;
                rmsnorm_ext(v_head, v_head, NULL, head_dim, 1e-6f, 0);
            }
        }

        /* 2.7 Store Value in the KV Cache */
        store_kv_vector(cache_v, scratch->attn_kv, v_total_dim, ctype);
    }

    /* Step 3: Attention Computation (Grouped-Query Attention) */
    float *attn_out = scratch->ffn_gate;
    float *scores = scratch->ffn_up;
    float scale = (cfg && cfg->model_type == MODEL_TYPE_GEMMA4) ? 1.0f : (1.0f / sqrtf((float)head_dim));
    if (cfg && cfg->rope_scaling_type == ROPE_SCALING_YARN && cfg->rope_scaling_factor > 1.0f) {
        float yarn_attn_scale = (cfg->rope_attn_factor > 0.0f) ? cfg->rope_attn_factor : (0.1f * logf(cfg->rope_scaling_factor) + 1.0f);
        scale *= yarn_attn_scale;
    }

    float softcap = (cfg ? cfg->attn_logit_softcapping : 0.0f);

    for (int h_q = 0; h_q < num_attn_heads; h_q++) {
        int h_kv = h_q / (group_size > 0 ? group_size : 1);
        const float *q_head = scratch->attn_q + h_q * q_dim_per_head;

        for (int p = 0; p <= pos; p++) {
            const uint8_t *p_cache_k = layer_cache + (size_t)p * kv_token_bytes;
            const void *k_head_ptr = p_cache_k + (size_t)h_kv * head_k_bytes;

            float score = compute_qk_score(k_head_ptr, q_head, head_dim, ctype) * scale;
            if (softcap > 0.0f) {
                score = softcap * tanhf(score / softcap);
            }
            scores[p] = score;
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
            const uint8_t *p_cache_v = layer_cache + (size_t)p * kv_token_bytes + vec_bytes;
            const void *v_head_ptr = p_cache_v + (size_t)h_kv * head_v_bytes;

            add_v_scaled(out_head, v_head_ptr, prob, head_dim, ctype);
        }

        if (is_qwen_hybrid) {
            const float *gate_head = q_head + head_dim;
            for (int j = 0; j < head_dim; j++) {
                float sig_gate = 1.0f / (1.0f + expf(-gate_head[j]));
                out_head[j] *= sig_gate;
            }
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

    if (weights->post_attn_norm_w) {
        rmsnorm_ext(scratch->hidden_state, scratch->hidden_state, weights->post_attn_norm_w, hidden_dim, 1e-6f, add_one);
    }

    /* Step 5: Residual Connection */
    add_residual(hidden_state, hidden_state, scratch->hidden_state, hidden_dim);
}
