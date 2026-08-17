#include "dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Helper to check if two floats are very close
static int approx_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

// Helper to convert float to half bits (IEEE 754 half-precision)
// Only needs to support exact integers or basic values for testing
static uint16_t float_to_half_bits(float f) {
    if (f == 0.0f) return 0;
    if (f == 1.0f) return 0x3C00;
    if (f == 2.0f) return 0x4000;
    if (f == 3.0f) return 0x4200;
    if (f == 4.0f) return 0x4400;
    if (f == 5.0f) return 0x4500;
    if (f == 6.0f) return 0x4600;
    if (f == 8.0f) return 0x4800;
    if (f == 10.0f) return 0x4900;
    // Fallback for simple testing
    fprintf(stderr, "float_to_half_bits: unsupported float %f\n", f);
    exit(1);
}

void test_f32(void) {
    printf("Testing F32 copy...\n");
    float x[10] = {1.5f, -2.0f, 3.14f, 0.0f, -0.001f, 42.0f, 100.1f, -99.9f, 0.5f, -0.5f};
    float y[10] = {0};

    dequantize_f32(x, y, 10);

    for (int i = 0; i < 10; i++) {
        assert(approx_equal(x[i], y[i]));
    }
    printf("  F32 copy PASSED.\n");
}

void test_q8_0(void) {
    printf("Testing Q8_0 dequantization...\n");
    block_q8_0 block;
    block.d = float_to_half_bits(2.0f); // Scale = 2.0
    for (int i = 0; i < QK8_0; i++) {
        block.qs[i] = i - 16; // values from -16 to 15
    }

    float y[QK8_0];
    dequantize_q8_0(&block, y, QK8_0);

    for (int i = 0; i < QK8_0; i++) {
        float expected = (i - 16) * 2.0f;
        assert(approx_equal(y[i], expected));
    }
    printf("  Q8_0 dequantization PASSED.\n");
}

void test_q4_K(void) {
    printf("Testing Q4_K dequantization...\n");
    block_q4_K block;
    memset(&block, 0, sizeof(block));

    block.d = float_to_half_bits(1.0f);    // super-block scale = 1.0
    block.dmin = float_to_half_bits(1.0f); // super-block min = 1.0

    // Set scale for block 0 to 5, min to 3
    block.scales[0] = 5; // scale = 5
    block.scales[4] = 3; // min = 3

    // Set scale for block 1 to 6, min to 4
    block.scales[1] = 6; // scale = 6
    block.scales[5] = 4; // min = 4

    // Set qs for block 0 (first 32 elements, lower nibble) to 7
    // Set qs for block 1 (elements 32-63, upper nibble) to 8
    for (int i = 0; i < 32; i++) {
        block.qs[i] = 7 | (8 << 4);
    }

    float y[QK_K];
    dequantize_q4_K(&block, y, QK_K);

    // Block 0: elements 0..31
    // scale = 5, min = 3, q = 7. Expected = 5 * 7 - 3 = 32
    for (int i = 0; i < 32; i++) {
        assert(approx_equal(y[i], 32.0f));
    }

    // Block 1: elements 32..63
    // scale = 6, min = 4, q = 8. Expected = 6 * 8 - 4 = 44
    for (int i = 32; i < 64; i++) {
        assert(approx_equal(y[i], 44.0f));
    }

    // Other blocks (2..7) have scale=0, min=0, q=0. Expected = 0
    for (int i = 64; i < QK_K; i++) {
        assert(approx_equal(y[i], 0.0f));
    }
    printf("  Q4_K dequantization PASSED.\n");
}

void test_q5_K(void) {
    printf("Testing Q5_K dequantization...\n");
    block_q5_K block;
    memset(&block, 0, sizeof(block));

    block.d = float_to_half_bits(1.0f);
    block.dmin = float_to_half_bits(1.0f);

    block.scales[0] = 5; // scale = 5
    block.scales[4] = 3; // min = 3

    // Block 0: qs lower nibble = 7, high bit = 1 (so q = 7 + 16 = 23)
    // Block 1: qs upper nibble = 8, high bit = 0 (so q = 8 + 0 = 8)
    for (int i = 0; i < 32; i++) {
        block.qs[i] = 7 | (8 << 4);
    }
    // Set high bit of block 0 elements to 1
    // qh[i] & u1 must be true. u1 starts at 1, so bit 0 of qh must be 1.
    for (int i = 0; i < 32; i++) {
        block.qh[i] = 1;
    }

    float y[QK_K];
    dequantize_q5_K(&block, y, QK_K);

    // Block 0 (elements 0..31): scale = 5, min = 3, q = 23. Expected = 5 * 23 - 3 = 112
    for (int i = 0; i < 32; i++) {
        assert(approx_equal(y[i], 112.0f));
    }

    // Block 1 (elements 32..63): scale = 0 (from scales[1] = 0), min = 0. Expected = 0
    for (int i = 32; i < 64; i++) {
        assert(approx_equal(y[i], 0.0f));
    }
    printf("  Q5_K dequantization PASSED.\n");
}

void test_q6_K(void) {
    printf("Testing Q6_K dequantization...\n");
    block_q6_K block;
    memset(&block, 0, sizeof(block));

    block.d = float_to_half_bits(1.0f);
    block.scales[0] = 10; // scale = 10 for block 0 (l=0..15) and block 1 (l=16..31)

    // Set ql lower nibble = 5
    // Set qh lower 2 bits = 2
    // Resulting 6-bit quant: 5 | (2 << 4) = 37. Subtract 32 = 5.
    for (int i = 0; i < 32; i++) {
        block.ql[i] = 5;
        block.qh[i] = 2;
    }

    float y[QK_K];
    dequantize_q6_K(&block, y, QK_K);

    // Block 0: elements 0..15. scale = 10, q = 5. Expected = 1.0 * 10 * 5 = 50
    for (int i = 0; i < 16; i++) {
        assert(approx_equal(y[i], 50.0f));
    }

    // Block 1: elements 16..31. scale = 0 (scales[1] = 0). Expected = 0
    for (int i = 16; i < 32; i++) {
        assert(approx_equal(y[i], 0.0f));
    }
    printf("  Q6_K dequantization PASSED.\n");
}

int main(void) {
    printf("=== Starting Dequantization Kernels Unit Tests ===\n\n");
    test_f32();
    test_q8_0();
    test_q4_K();
    test_q5_K();
    test_q6_K();
    printf("\n=== ALL DEQUANTIZATION TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
