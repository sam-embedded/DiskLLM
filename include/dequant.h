#ifndef DEQUANT_H
#define DEQUANT_H

#include <stdint.h>
#include <stddef.h>

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

// Dequantization function signatures
void dequantize_f32(const void * restrict x, float * restrict y, int k);
void dequantize_q8_0(const void * restrict x, float * restrict y, int k);
void dequantize_q4_K(const void * restrict x, float * restrict y, int k);
void dequantize_q5_K(const void * restrict x, float * restrict y, int k);
void dequantize_q6_K(const void * restrict x, float * restrict y, int k);

#endif // DEQUANT_H
