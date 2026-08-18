#ifndef KERNELS_H
#define KERNELS_H

#include <stdint.h>
#include <stddef.h>

// GGML tensor types, matching standard GGUF type IDs
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8   = 16,
    GGML_TYPE_I16  = 17,
    GGML_TYPE_I32  = 18,
} ggml_type;

// Configure threads for matvec matrix-vector operations
void matvec_set_num_threads(int num_threads);
int matvec_get_num_threads(void);

// Root Mean Square Normalization (RMSNorm)
void rmsnorm(float * restrict out, const float * restrict x, const float * restrict w, int n, float epsilon);

// Element-wise residual addition: out[i] = a[i] + b[i]
void add_residual(float * restrict out, const float * restrict a, const float * restrict b, int n);

// Element-wise SiLU (Swish) activation: out[i] = x[i] / (1.0f + expf(-x[i]))
void silu(float * restrict out, const float * restrict x, int n);

// SwiGLU activation gate and up projection combine: out[i] = SiLU(gate[i]) * up[i]
void swiglu(float * restrict out, const float * restrict gate, const float * restrict up, int n);

// Matrix-Vector Multiplication dispatcher (Multithreaded via pthreads)
void matvec(
    float * restrict out,
    const void * restrict w,
    const float * restrict x,
    int in_features,
    int out_features,
    int type,
    float * restrict dequant_buf
);

#endif // KERNELS_H
