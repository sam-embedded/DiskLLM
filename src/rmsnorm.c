#include "kernels.h"
#include <math.h>

void rmsnorm(float * restrict out, const float * restrict x, const float * restrict w, int n, float epsilon) {
    // Calculate sum of squares using double precision for numerical stability
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        ss += (double)x[i] * (double)x[i];
    }
    ss /= n;
    ss += (double)epsilon;
    
    float scale = (float)(1.0 / sqrt(ss));
    
    for (int i = 0; i < n; i++) {
        out[i] = x[i] * scale * w[i];
    }
}

void add_residual(float * restrict out, const float * restrict a, const float * restrict b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}
