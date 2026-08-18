#include "kernels.h"
#include <math.h>

void silu(float * restrict out, const float * restrict x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        out[i] = val / (1.0f + expf(-val));
    }
}

void swiglu(float * restrict out, const float * restrict gate, const float * restrict up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

void gelu(float * restrict out, const float * restrict x, int n) {
    const float kSqrtTwoOverPi = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float inner = kSqrtTwoOverPi * (val + 0.044715f * val * val * val);
        out[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

void geglu(float * restrict out, const float * restrict gate, const float * restrict up, int n) {
    const float kSqrtTwoOverPi = 0.7978845608028654f;
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float inner = kSqrtTwoOverPi * (g + 0.044715f * g * g * g);
        float gelu_g = 0.5f * g * (1.0f + tanhf(inner));
        out[i] = gelu_g * up[i];
    }
}
