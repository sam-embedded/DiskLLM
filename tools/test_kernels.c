#include "kernels.h"
#include "dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

// Helper to check if two floats are very close
static int approx_equal(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

// Portable float to half bits conversion (IEEE 754 half-precision)
// Needed to construct quantized blocks for testing
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
    
    // Simple conversion fallback for general testing values
    union { float f; uint32_t i; } u = {f};
    int s = (u.i >> 16) & 0x8000;
    int e = ((u.i >> 23) & 0xff) - 127 + 15;
    int m = u.i & 0x7fffff;
    if (e <= 0) {
        return s;
    } else if (e >= 31) {
        return s | 0x7c00;
    } else {
        return s | (e << 10) | (m >> 13);
    }
}

void test_rmsnorm(void) {
    printf("Testing RMSNorm...\n");
    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float w[5] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    float out[5] = {0};
    
    // ss = (1+4+9+16+25)/5 = 55/5 = 11
    // scale = 1 / sqrt(11 + 1e-6) = 1 / 3.3166249 = 0.3015113
    // out[i] = x[i] * scale * w[i]
    // out[0] = 1 * 0.3015113 * 0.5 = 0.1507556
    // out[1] = 2 * 0.3015113 * 1.0 = 0.6030226
    // out[4] = 5 * 0.3015113 * 2.5 = 3.7688918
    
    rmsnorm(out, x, w, 5, 1e-6f);
    
    assert(approx_equal(out[0], 0.1507556f));
    assert(approx_equal(out[1], 0.6030226f));
    assert(approx_equal(out[4], 3.7688918f));
    
    // Test in-place
    float x_inplace[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    rmsnorm(x_inplace, x_inplace, w, 5, 1e-6f);
    for (int i = 0; i < 5; i++) {
        assert(approx_equal(x_inplace[i], out[i]));
    }
    printf("  RMSNorm PASSED.\n");
}

void test_add_residual(void) {
    printf("Testing Add Residual...\n");
    float a[5] = {1.0f, -2.0f, 3.5f, 0.0f, 10.0f};
    float b[5] = {0.5f, 2.0f, -1.5f, 4.0f, -3.0f};
    float out[5] = {0};
    
    add_residual(out, a, b, 5);
    
    assert(approx_equal(out[0], 1.5f));
    assert(approx_equal(out[1], 0.0f));
    assert(approx_equal(out[2], 2.0f));
    assert(approx_equal(out[3], 4.0f));
    assert(approx_equal(out[4], 7.0f));
    
    // In-place testing
    float a_inplace[5] = {1.0f, -2.0f, 3.5f, 0.0f, 10.0f};
    add_residual(a_inplace, a_inplace, b, 5);
    for (int i = 0; i < 5; i++) {
        assert(approx_equal(a_inplace[i], out[i]));
    }
    printf("  Add Residual PASSED.\n");
}

void test_swiglu(void) {
    printf("Testing SiLU and SwiGLU...\n");
    float x[3] = {0.0f, 1.0f, -1.0f};
    float out_silu[3] = {0};
    
    // silu(0) = 0
    // silu(1) = 1 / (1 + exp(-1)) = 1 / (1 + 0.367879) = 0.7310586
    // silu(-1) = -1 / (1 + exp(1)) = -1 / (1 + 2.71828) = -0.2689414
    silu(out_silu, x, 3);
    assert(approx_equal(out_silu[0], 0.0f));
    assert(approx_equal(out_silu[1], 0.7310586f));
    assert(approx_equal(out_silu[2], -0.2689414f));
    
    float gate[3] = {0.0f, 1.0f, -1.0f};
    float up[3] = {2.0f, 3.0f, 4.0f};
    float out_swiglu[3] = {0};
    
    // swiglu(gate, up) = silu(gate) * up
    // out[0] = 0.0f * 2.0f = 0.0f
    // out[1] = 0.7310586f * 3.0f = 2.1931758f
    // out[2] = -0.2689414f * 4.0f = -1.0757656f
    swiglu(out_swiglu, gate, up, 3);
    assert(approx_equal(out_swiglu[0], 0.0f));
    assert(approx_equal(out_swiglu[1], 2.1931758f));
    assert(approx_equal(out_swiglu[2], -1.0757656f));
    
    printf("  SiLU and SwiGLU PASSED.\n");
}

void test_matvec_f32(void) {
    printf("Testing matvec F32...\n");
    // Matrix of shape [3, 2] -> in_features = 3, out_features = 2
    // Stored as flat array of size 6
    float w[6] = {
        1.0f, 2.0f, 3.0f, // Row 0
        4.0f, 5.0f, 6.0f  // Row 1
    };
    float x[3] = {0.5f, 1.0f, 2.0f};
    float out[2] = {0};
    
    // Row 0: 1*0.5 + 2*1 + 3*2 = 0.5 + 2 + 6 = 8.5
    // Row 1: 4*0.5 + 5*1 + 6*2 = 2 + 5 + 12 = 19
    matvec(out, w, x, 3, 2, GGML_TYPE_F32, NULL);
    
    assert(approx_equal(out[0], 8.5f));
    assert(approx_equal(out[1], 19.0f));
    printf("  matvec F32 PASSED.\n");
}

void test_matvec_q8_0(void) {
    printf("Testing matvec Q8_0 (with and without streaming buffer)...\n");
    
    int in_features = 64; // multiple of 32
    int out_features = 2;
    
    block_q8_0 weights[4]; // 2 rows, each has 2 blocks of 32 elements = 64 features total.
    memset(weights, 0, sizeof(weights));
    
    // Row 0, Block 0: scale = 2.0, qs = 1
    weights[0].d = float_to_half_bits(2.0f);
    for (int i = 0; i < 32; i++) weights[0].qs[i] = 1; // values = 2.0
    // Row 0, Block 1: scale = 3.0, qs = 2
    weights[1].d = float_to_half_bits(3.0f);
    for (int i = 0; i < 32; i++) weights[1].qs[i] = 2; // values = 6.0
    
    // Row 1, Block 0: scale = 1.0, qs = -1
    weights[2].d = float_to_half_bits(1.0f);
    for (int i = 0; i < 32; i++) weights[2].qs[i] = -1; // values = -1.0
    // Row 1, Block 1: scale = 4.0, qs = 3
    weights[3].d = float_to_half_bits(4.0f);
    for (int i = 0; i < 32; i++) weights[3].qs[i] = 3; // values = 12.0
    
    float x[64];
    for (int i = 0; i < 64; i++) {
        x[i] = 0.5f;
    }
    
    // Expected result:
    // Row 0: (32 * 2.0 * 0.5) + (32 * 6.0 * 0.5) = 32 + 96 = 128
    // Row 1: (32 * -1.0 * 0.5) + (32 * 12.0 * 0.5) = -16 + 192 = 176
    
    float out1[2] = {0};
    float out2[2] = {0};
    float dequant_buf[64];
    
    // Test with streaming/scratch buffer
    matvec(out1, weights, x, in_features, out_features, GGML_TYPE_Q8_0, dequant_buf);
    
    // Test without streaming/scratch buffer (block-by-block fallback)
    matvec(out2, weights, x, in_features, out_features, GGML_TYPE_Q8_0, NULL);
    
    assert(approx_equal(out1[0], 128.0f));
    assert(approx_equal(out1[1], 176.0f));
    assert(approx_equal(out2[0], 128.0f));
    assert(approx_equal(out2[1], 176.0f));
    
    printf("  matvec Q8_0 PASSED.\n");
}

void test_matvec_q4_K(void) {
    printf("Testing matvec Q4_K...\n");
    
    int in_features = 256; // 1 block of 256
    int out_features = 2;
    
    block_q4_K weights[2]; // 2 rows, each has 1 block of 256
    memset(weights, 0, sizeof(weights));
    
    // Row 0:
    weights[0].d = float_to_half_bits(1.0f);
    weights[0].dmin = float_to_half_bits(1.0f);
    // 8 blocks of 32 in the super-block. Let's set scale=5, min=3 for all.
    for (int is = 0; is < 4; is++) {
        weights[0].scales[is] = 5;
        weights[0].scales[is + 4] = 3;
    }
    for (int is = 8; is < 12; is++) {
        weights[0].scales[is] = 53; // 5 | (3 << 4)
    }
    // Set all qs to 7 (both nibbles) -> value = 5 * 7 - 3 = 32
    for (int i = 0; i < 128; i++) {
        weights[0].qs[i] = 7 | (7 << 4);
    }
    
    // Row 1:
    weights[1].d = float_to_half_bits(2.0f);
    weights[1].dmin = float_to_half_bits(1.0f);
    // Let's set scale=4 (d * scale = 2 * 4 = 8), min=2 (dmin * min = 1 * 2 = 2) for all.
    for (int is = 0; is < 4; is++) {
        weights[1].scales[is] = 4;
        weights[1].scales[is + 4] = 2;
    }
    for (int is = 8; is < 12; is++) {
        weights[1].scales[is] = 36; // 4 | (2 << 4)
    }
    // Set all qs to 5 (both nibbles) -> value = 8 * 5 - 2 = 38
    for (int i = 0; i < 128; i++) {
        weights[1].qs[i] = 5 | (5 << 4);
    }
    
    float x[256];
    for (int i = 0; i < 256; i++) {
        x[i] = 0.25f;
    }
    
    // Expected result:
    // Row 0: 256 * 32.0 * 0.25 = 2048.0
    // Row 1: 256 * 38.0 * 0.25 = 2432.0
    
    float out1[2] = {0};
    float out2[2] = {0};
    float dequant_buf[256];
    
    matvec(out1, weights, x, in_features, out_features, GGML_TYPE_Q4_K, dequant_buf);
    matvec(out2, weights, x, in_features, out_features, GGML_TYPE_Q4_K, NULL);
    
    assert(approx_equal(out1[0], 2048.0f));
    assert(approx_equal(out1[1], 2432.0f));
    assert(approx_equal(out2[0], 2048.0f));
    assert(approx_equal(out2[1], 2432.0f));
    
    printf("  matvec Q4_K PASSED.\n");
}

void test_matvec_q5_K(void) {
    printf("Testing matvec Q5_K...\n");
    
    int in_features = 256;
    int out_features = 2;
    
    block_q5_K weights[2];
    memset(weights, 0, sizeof(weights));
    
    // Row 0:
    weights[0].d = float_to_half_bits(1.0f);
    weights[0].dmin = float_to_half_bits(1.0f);
    for (int is = 0; is < 4; is++) {
        weights[0].scales[is] = 5;
        weights[0].scales[is + 4] = 3;
    }
    for (int is = 8; is < 12; is++) {
        weights[0].scales[is] = 53; // 5 | (3 << 4)
    }
    // qs=7, qh=1 (bit 0 set for first 32, bit 2 set for next 32, etc.)
    // Let's set qh to all 0x55 (binary 01010101) so alternate elements have high bit.
    // Actually, let's keep it simple: set qh = 0 -> value = 5 * 7 - 3 = 32
    for (int i = 0; i < 128; i++) {
        weights[0].qs[i] = 7 | (7 << 4);
    }
    memset(weights[0].qh, 0, sizeof(weights[0].qh));
    
    // Row 1:
    weights[1].d = float_to_half_bits(2.0f);
    weights[1].dmin = float_to_half_bits(1.0f);
    for (int is = 0; is < 4; is++) {
        weights[1].scales[is] = 4;
        weights[1].scales[is + 4] = 2;
    }
    for (int is = 8; is < 12; is++) {
        weights[1].scales[is] = 36; // 4 | (2 << 4)
    }
    // Set all qs to 5 (both nibbles) -> value = 8 * 5 - 2 = 38
    for (int i = 0; i < 128; i++) {
        weights[1].qs[i] = 5 | (5 << 4);
    }
    memset(weights[1].qh, 0, sizeof(weights[1].qh));
    
    float x[256];
    for (int i = 0; i < 256; i++) {
        x[i] = 0.125f;
    }
    
    // Expected result:
    // Row 0: 256 * 32.0 * 0.125 = 1024.0
    // Row 1: 256 * 38.0 * 0.125 = 1216.0
    
    float out1[2] = {0};
    float out2[2] = {0};
    float dequant_buf[256];
    
    matvec(out1, weights, x, in_features, out_features, GGML_TYPE_Q5_K, dequant_buf);
    matvec(out2, weights, x, in_features, out_features, GGML_TYPE_Q5_K, NULL);
    
    assert(approx_equal(out1[0], 1024.0f));
    assert(approx_equal(out1[1], 1216.0f));
    assert(approx_equal(out2[0], 1024.0f));
    assert(approx_equal(out2[1], 1216.0f));
    
    printf("  matvec Q5_K PASSED.\n");
}

void test_matvec_q6_K(void) {
    printf("Testing matvec Q6_K...\n");
    
    int in_features = 256;
    int out_features = 2;
    
    block_q6_K weights[2];
    memset(weights, 0, sizeof(weights));
    
    // Row 0:
    weights[0].d = float_to_half_bits(1.0f);
    // scale = 10, ql = 5, qh = 2 -> 6-bit quant: 5 | (2 << 4) = 37. Subtract 32 = 5.
    // Expected value = d * scale * q = 1.0 * 10 * 5 = 50.0
    for (int is = 0; is < 16; is++) {
        weights[0].scales[is] = 10;
    }
    for (int i = 0; i < 128; i++) {
        weights[0].ql[i] = 5 | (5 << 4);
    }
    for (int i = 0; i < 64; i++) {
        weights[0].qh[i] = 170;
    }
    
    // Row 1:
    weights[1].d = float_to_half_bits(2.0f);
    // scale = 5, ql = 5, qh = 2 -> 6-bit quant: 37 - 32 = 5.
    // Expected value = d * scale * q = 2.0 * 5 * 5 = 50.0
    for (int is = 0; is < 16; is++) {
        weights[1].scales[is] = 5;
    }
    for (int i = 0; i < 128; i++) {
        weights[1].ql[i] = 5 | (5 << 4);
    }
    for (int i = 0; i < 64; i++) {
        weights[1].qh[i] = 170;
    }
    
    float x[256];
    for (int i = 0; i < 256; i++) {
        x[i] = 0.5f;
    }
    
    // Expected result:
    // Row 0: 256 * 50.0 * 0.5 = 6400.0
    // Row 1: 256 * 50.0 * 0.5 = 6400.0
    
    float out1[2] = {0};
    float out2[2] = {0};
    float dequant_buf[256];
    
    matvec(out1, weights, x, in_features, out_features, GGML_TYPE_Q6_K, dequant_buf);
    matvec(out2, weights, x, in_features, out_features, GGML_TYPE_Q6_K, NULL);
    
    assert(approx_equal(out1[0], 6400.0f));
    assert(approx_equal(out1[1], 6400.0f));
    assert(approx_equal(out2[0], 6400.0f));
    assert(approx_equal(out2[1], 6400.0f));
    
    printf("  matvec Q6_K PASSED.\n");
}

int main(void) {
    printf("=== Starting Basic Math Kernels Unit Tests ===\n\n");
    test_rmsnorm();
    test_add_residual();
    test_swiglu();
    test_matvec_f32();
    test_matvec_q8_0();
    test_matvec_q4_K();
    test_matvec_q5_K();
    test_matvec_q6_K();
    printf("\n=== ALL KERNEL TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
