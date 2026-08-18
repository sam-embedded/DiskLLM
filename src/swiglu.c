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
