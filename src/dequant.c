#include "dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Helper to convert float bits to float value
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

// Helper to convert float value to float bits
static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

// Portable C FP16 (uint16_t representation) to FP32 conversion
static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & 0x80000000;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0 << 23;
    const float exp_scale = fp32_from_bits(0x7800000);
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = 126 << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = 1 << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

// Helper to extract 6-bit scale and min values from packed scales array
static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; 
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Dequantize F32: Simple copy
void dequantize_f32(const void * restrict x, float * restrict y, int k) {
    const float *src = (const float *)x;
    for (int i = 0; i < k; i++) {
        y[i] = src[i];
    }
}

// Dequantize Q8_0
void dequantize_q8_0(const void * restrict x, float * restrict y, int k) {
    assert(k % QK8_0 == 0);
    const block_q8_0 *bx = (const block_q8_0 *)x;
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(bx[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            y[i*QK8_0 + j] = bx[i].qs[j] * d;
        }
    }
}

// Dequantize Q4_K
void dequantize_q4_K(const void * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const block_q4_K *bx = (const block_q4_K *)x;
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = bx[i].qs;
        const float d   = fp16_to_fp32(bx[i].d);
        const float min = fp16_to_fp32(bx[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, bx[i].scales, &sc, &m);
            const float d1 = d * sc; 
            const float m1 = min * m;
            
            get_scale_min_k4(is + 1, bx[i].scales, &sc, &m);
            const float d2 = d * sc; 
            const float m2 = min * m;
            
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2;
            }
            q += 32; 
            is += 2;
        }
    }
}

// Dequantize Q5_K
void dequantize_q5_K(const void * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const block_q5_K *bx = (const block_q5_K *)x;
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = bx[i].qs;
        const uint8_t * qh = bx[i].qh;
        const float d = fp16_to_fp32(bx[i].d);
        const float min = fp16_to_fp32(bx[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, bx[i].scales, &sc, &m);
            const float d1 = d * sc; 
            const float m1 = min * m;
            
            get_scale_min_k4(is + 1, bx[i].scales, &sc, &m);
            const float d2 = d * sc; 
            const float m2 = min * m;
            
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            }
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            }
            ql += 32; 
            is += 2;
            u1 <<= 2; 
            u2 <<= 2;
        }
    }
}

// Dequantize Q6_K
void dequantize_q6_K(const void * restrict x, float * restrict y, int k) {
    assert(k % QK_K == 0);
    const block_q6_K *bx = (const block_q6_K *)x;
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(bx[i].d);
        const uint8_t * ql = bx[i].ql;
        const uint8_t * qh = bx[i].qh;
        const int8_t  * sc = bx[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
