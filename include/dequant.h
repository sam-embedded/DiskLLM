#ifndef DEQUANT_H
#define DEQUANT_H

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define QK_K 256
#define K_SCALE_SIZE 12
#define QK8_0 32

// 4-bit quantization super-block
typedef struct {
    uint16_t d;    // super-block scale for quantized scales (FP16 representation)
    uint16_t dmin; // super-block scale for quantized mins (FP16 representation)
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4-bit quants
} block_q4_K;

// 5-bit quantization super-block
typedef struct {
    uint16_t d;    // super-block scale for quantized scales (FP16 representation)
    uint16_t dmin; // super-block scale for quantized mins (FP16 representation)
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;

// 6-bit quantization super-block
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    uint16_t d;              // super-block scale (FP16 representation)
} block_q6_K;

// 8-bit quantization block
typedef struct {
    uint16_t d;       // delta scale (FP16 representation)
    int8_t qs[QK8_0]; // 8-bit quants
} block_q8_0;

// 4-bit quantization block (standard 0-type)
typedef struct {
    uint16_t d;       // delta scale (FP16 representation)
    uint8_t qs[16];   // 4-bit quants (16 bytes = 32 values)
} block_q4_0;

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

// Dequantization function signatures
void dequantize_f32(const void * restrict x, float * restrict y, int k);
void dequantize_q8_0(const void * restrict x, float * restrict y, int k);
void dequantize_q4_K(const void * restrict x, float * restrict y, int k);
void dequantize_q5_K(const void * restrict x, float * restrict y, int k);
void dequantize_q6_K(const void * restrict x, float * restrict y, int k);

static inline void dequantize_row(float * restrict dest, const void * restrict src, int k, int type) {
    switch (type) {
        case 0:  dequantize_f32(src, dest, k); break;
        case 8:  dequantize_q8_0(src, dest, k); break;
        case 12: dequantize_q4_K(src, dest, k); break;
        case 13: dequantize_q5_K(src, dest, k); break;
        case 14: dequantize_q6_K(src, dest, k); break;
        default: dequantize_q4_K(src, dest, k); break;
    }
}

#endif // DEQUANT_H
