#include "kernels.h"
#include "dequant.h"
#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// ─── ARM NEON SIMD ────────────────────────────────────────────────────────────
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

// ─── Threading configuration ──────────────────────────────────────────────────
static int g_num_threads = 4;

void matvec_set_num_threads(int num_threads) {
    if (num_threads >= 1 && num_threads <= 32) {
        g_num_threads = num_threads;
    }
}

int matvec_get_num_threads(void) {
    return g_num_threads;
}

// ─── Q4_K dot product ─────────────────────────────────────────────────────────

static inline double dot_q4_K(const block_q4_K * restrict bx, const float * restrict x) {
    const uint8_t *q   = bx->qs;
    const float    d   = fp16_to_fp32(bx->d);
    const float    min = fp16_to_fp32(bx->dmin);
    double sum = 0.0;
    int is = 0;
    uint8_t sc, m;

    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, bx->scales, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, bx->scales, &sc, &m);
        const float d2 = d * sc, m2 = min * m;

#if HAVE_NEON
        float32x4_t vd1 = vdupq_n_f32(d1), vm1 = vdupq_n_f32(m1);
        float32x4_t vd2 = vdupq_n_f32(d2), vm2 = vdupq_n_f32(m2);
        float32x4_t vsum = vdupq_n_f32(0.0f);

        for (int l = 0; l < 32; l += 8) {
            uint8x8_t q8  = vld1_u8(q + l);
            uint8x8_t lo4 = vand_u8(q8, vdup_n_u8(0x0F));
            uint16x8_t lo16 = vmovl_u8(lo4);
            float32x4_t qf_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16)));
            float32x4_t qf_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16)));
            float32x4_t xv_lo = vld1q_f32(x + l);
            float32x4_t xv_hi = vld1q_f32(x + l + 4);
            vsum = vmlaq_f32(vsum, xv_lo, vsubq_f32(vmulq_f32(vd1, qf_lo), vm1));
            vsum = vmlaq_f32(vsum, xv_hi, vsubq_f32(vmulq_f32(vd1, qf_hi), vm1));
        }
        x += 32;

        for (int l = 0; l < 32; l += 8) {
            uint8x8_t q8  = vld1_u8(q + l);
            uint8x8_t hi4 = vshr_n_u8(q8, 4);
            uint16x8_t hi16 = vmovl_u8(hi4);
            float32x4_t qf_lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16)));
            float32x4_t qf_hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16)));
            float32x4_t xv_lo = vld1q_f32(x + l);
            float32x4_t xv_hi = vld1q_f32(x + l + 4);
            vsum = vmlaq_f32(vsum, xv_lo, vsubq_f32(vmulq_f32(vd2, qf_lo), vm2));
            vsum = vmlaq_f32(vsum, xv_hi, vsubq_f32(vmulq_f32(vd2, qf_hi), vm2));
        }
        x += 32;

        sum += (double)vaddvq_f32(vsum);
#else
        for (int l = 0; l < 32; l++) sum += (double)x[l] * (d1 * (q[l] & 0xF) - m1);
        x += 32;
        for (int l = 0; l < 32; l++) sum += (double)x[l] * (d2 * (q[l] >> 4) - m2);
        x += 32;
#endif
        q  += 32;
        is += 2;
    }
    return sum;
}

// ─── Q5_K dot product ─────────────────────────────────────────────────────────

static inline double dot_q5_K(const block_q5_K * restrict bx, const float * restrict x) {
    const uint8_t *ql  = bx->qs;
    const uint8_t *qh  = bx->qh;
    const float    d   = fp16_to_fp32(bx->d);
    const float    min = fp16_to_fp32(bx->dmin);
    double sum = 0.0;
    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;

    for (int j = 0; j < QK_K; j += 64) {
        get_scale_min_k4(is + 0, bx->scales, &sc, &m);
        const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, bx->scales, &sc, &m);
        const float d2 = d * sc, m2 = min * m;

        for (int l = 0; l < 32; l++)
            sum += (double)x[l] * (d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1);
        x += 32;
        for (int l = 0; l < 32; l++)
            sum += (double)x[l] * (d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2);
        x += 32;
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    return sum;
}

// ─── Q6_K dot product ─────────────────────────────────────────────────────────

static inline double dot_q6_K(const block_q6_K * restrict bx, const float * restrict x) {
    const float     d  = fp16_to_fp32(bx->d);
    const uint8_t  *ql = bx->ql;
    const uint8_t  *qh = bx->qh;
    const int8_t   *sc = bx->scales;
    double sum = 0.0;

#if HAVE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);

    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l += 4) {
            int is = l / 16;
            int32_t q1a[4], q2a[4], q3a[4], q4a[4];
            for (int i = 0; i < 4; i++) {
                q1a[i] = (int8_t)((ql[l+i]    & 0xF) | (((qh[l+i] >> 0) & 3) << 4)) - 32;
                q2a[i] = (int8_t)((ql[l+i+32] & 0xF) | (((qh[l+i] >> 2) & 3) << 4)) - 32;
                q3a[i] = (int8_t)((ql[l+i]    >> 4)  | (((qh[l+i] >> 4) & 3) << 4)) - 32;
                q4a[i] = (int8_t)((ql[l+i+32] >> 4)  | (((qh[l+i] >> 6) & 3) << 4)) - 32;
            }

            float sc0 = d * sc[is + 0];
            float sc2 = d * sc[is + 2];
            float sc4 = d * sc[is + 4];
            float sc6 = d * sc[is + 6];

            float32x4_t qf1 = vcvtq_f32_s32(vld1q_s32(q1a));
            float32x4_t qf2 = vcvtq_f32_s32(vld1q_s32(q2a));
            float32x4_t qf3 = vcvtq_f32_s32(vld1q_s32(q3a));
            float32x4_t qf4 = vcvtq_f32_s32(vld1q_s32(q4a));

            float32x4_t xv0  = vld1q_f32(x + l);
            float32x4_t xv32 = vld1q_f32(x + l + 32);
            float32x4_t xv64 = vld1q_f32(x + l + 64);
            float32x4_t xv96 = vld1q_f32(x + l + 96);

            vsum = vmlaq_f32(vsum, xv0,  vmulq_f32(vdupq_n_f32(sc0), qf1));
            vsum = vmlaq_f32(vsum, xv32, vmulq_f32(vdupq_n_f32(sc2), qf2));
            vsum = vmlaq_f32(vsum, xv64, vmulq_f32(vdupq_n_f32(sc4), qf3));
            vsum = vmlaq_f32(vsum, xv96, vmulq_f32(vdupq_n_f32(sc6), qf4));
        }
        x  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
    sum = (double)vaddvq_f32(vsum);
#else
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l]    >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
            sum += (double)x[l]      * (d * sc[is + 0] * q1);
            sum += (double)x[l + 32] * (d * sc[is + 2] * q2);
            sum += (double)x[l + 64] * (d * sc[is + 4] * q3);
            sum += (double)x[l + 96] * (d * sc[is + 6] * q4);
        }
        x  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
#endif
    return sum;
}

// ─── Q8_0 dot product ─────────────────────────────────────────────────────────

static inline double dot_q8_0(const block_q8_0 * restrict bx, const float * restrict x) {
    const float d = fp16_to_fp32(bx->d);

#if HAVE_NEON
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (int j = 0; j < QK8_0; j += 8) {
        int8x8_t  q8   = vld1_s8(bx->qs + j);
        int16x8_t q16  = vmovl_s8(q8);
        float32x4_t qlo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
        float32x4_t qhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
        vsum = vmlaq_f32(vsum, vld1q_f32(x + j),     qlo);
        vsum = vmlaq_f32(vsum, vld1q_f32(x + j + 4), qhi);
    }
    return (double)(vaddvq_f32(vsum) * d);
#else
    double sum = 0.0;
    for (int j = 0; j < QK8_0; j++) sum += (double)x[j] * (bx->qs[j] * d);
    return sum;
#endif
}

// ─── Q4_0 dot product ─────────────────────────────────────────────────────────

static inline double dot_q4_0(const block_q4_0 * restrict bx, const float * restrict x) {
    const float d = fp16_to_fp32(bx->d);
    double sum = 0.0;
    for (int j = 0; j < 16; j++) {
        const uint8_t q = bx->qs[j];
        const float x0 = (float)((int8_t)(q & 0xF) - 8);
        const float x1 = (float)((int8_t)(q >> 4) - 8);
        sum += (double)x[j] * x0 + (double)x[j + 16] * x1;
    }
    return sum * (double)d;
}

// ─── Q5_0 dot product ─────────────────────────────────────────────────────────

static inline double dot_q5_0(const block_q5_0 * restrict bx, const float * restrict x) {
    const float d = fp16_to_fp32(bx->d);
    uint32_t qh;
    memcpy(&qh, bx->qh, sizeof(qh));
    double sum = 0.0;
    for (int j = 0; j < 16; j++) {
        const uint8_t q = bx->qs[j];
        const uint8_t h0 = (qh >> (j + 0)) & 1;
        const uint8_t h1 = (qh >> (j + 16)) & 1;
        const float x0 = (float)((int8_t)((q & 0xF) | (h0 << 4)) - 16);
        const float x1 = (float)((int8_t)((q >> 4)  | (h1 << 4)) - 16);
        sum += (double)x[j] * x0 + (double)x[j + 16] * x1;
    }
    return sum * (double)d;
}

// ─── Multithreaded Row Compute Kernel ─────────────────────────────────────────

static void compute_row_range(float * restrict out, const void * restrict w, const float * restrict x, int in_features, int r_start, int r_end, int type) {
    if (type == GGML_TYPE_F32) {
        const float *w_f32 = (const float *)w;
        for (int i = r_start; i < r_end; i++) {
            const float *row = w_f32 + i * in_features;
            double sum = 0.0;
#if HAVE_NEON
            float32x4_t vsum = vdupq_n_f32(0.0f);
            int j = 0;
            for (; j + 4 <= in_features; j += 4)
                vsum = vmlaq_f32(vsum, vld1q_f32(row + j), vld1q_f32(x + j));
            sum = (double)vaddvq_f32(vsum);
            for (; j < in_features; j++) sum += (double)row[j] * x[j];
#else
            for (int j = 0; j < in_features; j++) sum += (double)row[j] * (double)x[j];
#endif
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_Q4_0) {
        const block_q4_0 *bx = (const block_q4_0 *)w;
        int blocks_per_row = in_features / 32;
        for (int i = r_start; i < r_end; i++) {
            double sum = 0.0;
            const block_q4_0 *row_w = bx + i * blocks_per_row;
            const float *row_x = x;
            for (int j = 0; j < blocks_per_row; j++) {
                sum += dot_q4_0(&row_w[j], row_x);
                row_x += 32;
            }
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_Q5_0) {
        const block_q5_0 *bx = (const block_q5_0 *)w;
        int blocks_per_row = in_features / 32;
        for (int i = r_start; i < r_end; i++) {
            double sum = 0.0;
            const block_q5_0 *row_w = bx + i * blocks_per_row;
            const float *row_x = x;
            for (int j = 0; j < blocks_per_row; j++) {
                sum += dot_q5_0(&row_w[j], row_x);
                row_x += 32;
            }
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_Q4_K) {
        const block_q4_K *bx = (const block_q4_K *)w;
        int blocks_per_row = in_features / 256;
        for (int i = r_start; i < r_end; i++) {
            double sum = 0.0;
            const block_q4_K *row_w = bx + i * blocks_per_row;
            const float *row_x = x;
            for (int j = 0; j < blocks_per_row; j++) {
                sum += dot_q4_K(&row_w[j], row_x);
                row_x += 256;
            }
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_Q5_K) {
        const block_q5_K *bx = (const block_q5_K *)w;
        int blocks_per_row = in_features / 256;
        for (int i = r_start; i < r_end; i++) {
            double sum = 0.0;
            const block_q5_K *row_w = bx + i * blocks_per_row;
            const float *row_x = x;
            for (int j = 0; j < blocks_per_row; j++) {
                sum += dot_q5_K(&row_w[j], row_x);
                row_x += 256;
            }
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_Q6_K) {
        const block_q6_K *bx = (const block_q6_K *)w;
        int blocks_per_row = in_features / 256;
        for (int i = r_start; i < r_end; i++) {
            double sum = 0.0;
            const block_q6_K *row_w = bx + i * blocks_per_row;
            const float *row_x = x;
            for (int j = 0; j < blocks_per_row; j++) {
                sum += dot_q6_K(&row_w[j], row_x);
                row_x += 256;
            }
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_Q8_0) {
        const block_q8_0 *bx = (const block_q8_0 *)w;
        int blocks_per_row = in_features / 32;
        for (int i = r_start; i < r_end; i++) {
            double sum = 0.0;
            const block_q8_0 *row_w = bx + i * blocks_per_row;
            const float *row_x = x;
            for (int j = 0; j < blocks_per_row; j++) {
                sum += dot_q8_0(&row_w[j], row_x);
                row_x += 32;
            }
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_BF16) {
        const uint16_t *w_bf16 = (const uint16_t *)w;
        for (int i = r_start; i < r_end; i++) {
            const uint16_t *row = w_bf16 + i * in_features;
            double sum = 0.0;
            for (int j = 0; j < in_features; j++) {
                sum += (double)bf16_to_fp32(row[j]) * (double)x[j];
            }
            out[i] = (float)sum;
        }
    } else if (type == GGML_TYPE_F16) {
        const uint16_t *w_f16 = (const uint16_t *)w;
        for (int i = r_start; i < r_end; i++) {
            const uint16_t *row = w_f16 + i * in_features;
            double sum = 0.0;
            for (int j = 0; j < in_features; j++) {
                sum += (double)fp16_to_fp32(row[j]) * (double)x[j];
            }
            out[i] = (float)sum;
        }
    } else {
        fprintf(stderr, "Error: Unsupported tensor type %d in matvec\n", type);
        exit(1);
    }
}

// ─── Persistent Pthread Worker Pool ───────────────────────────────────────────

#define MAX_WORKERS 32

typedef struct {
    float *out;
    const void *w;
    const float *x;
    int in_features;
    int r_start;
    int r_end;
    int type;
} matvec_job;

typedef struct {
    pthread_t thread;
    int id;
    matvec_job job;
    pthread_mutex_t mutex;
    pthread_cond_t cond_start;
    pthread_cond_t cond_done;
    int work_available;
    int terminate;
} worker_thread;

static worker_thread g_workers[MAX_WORKERS];
static int g_pool_threads = 0;
static int g_pool_initialized = 0;
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

static void *worker_thread_fn(void *arg) {
    worker_thread *w = (worker_thread *)arg;
    while (1) {
        pthread_mutex_lock(&w->mutex);
        while (!w->work_available && !w->terminate) {
            pthread_cond_wait(&w->cond_start, &w->mutex);
        }
        if (w->terminate) {
            pthread_mutex_unlock(&w->mutex);
            break;
        }

        matvec_job job = w->job;
        pthread_mutex_unlock(&w->mutex);

        compute_row_range(job.out, job.w, job.x, job.in_features, job.r_start, job.r_end, job.type);

        pthread_mutex_lock(&w->mutex);
        w->work_available = 0;
        pthread_cond_signal(&w->cond_done);
        pthread_mutex_unlock(&w->mutex);
    }
    return NULL;
}

static void init_worker_pool(int num_threads) {
    pthread_mutex_lock(&g_pool_mutex);
    if (g_pool_initialized && g_pool_threads == num_threads) {
        pthread_mutex_unlock(&g_pool_mutex);
        return;
    }

    if (g_pool_initialized) {
        for (int i = 1; i < g_pool_threads; i++) {
            pthread_mutex_lock(&g_workers[i].mutex);
            g_workers[i].terminate = 1;
            pthread_cond_signal(&g_workers[i].cond_start);
            pthread_mutex_unlock(&g_workers[i].mutex);
            pthread_join(g_workers[i].thread, NULL);
            pthread_mutex_destroy(&g_workers[i].mutex);
            pthread_cond_destroy(&g_workers[i].cond_start);
            pthread_cond_destroy(&g_workers[i].cond_done);
        }
        g_pool_initialized = 0;
    }

    g_pool_threads = num_threads;
    for (int i = 1; i < num_threads; i++) {
        g_workers[i].id = i;
        g_workers[i].work_available = 0;
        g_workers[i].terminate = 0;
        pthread_mutex_init(&g_workers[i].mutex, NULL);
        pthread_cond_init(&g_workers[i].cond_start, NULL);
        pthread_cond_init(&g_workers[i].cond_done, NULL);
        pthread_create(&g_workers[i].thread, NULL, worker_thread_fn, &g_workers[i]);
    }
    g_pool_initialized = 1;
    pthread_mutex_unlock(&g_pool_mutex);
}

void matvec(
    float * restrict out,
    const void * restrict w,
    const float * restrict x,
    int in_features,
    int out_features,
    int type,
    float * restrict dequant_buf
) {
    (void)dequant_buf;

    if (g_vulkan_ctx) {
        if (vulkan_matvec(g_vulkan_ctx, out, w, x, in_features, out_features, type) == 0) {
            return;
        }
    }

    int num_threads = g_num_threads;

    if (num_threads <= 1 || out_features < 64) {
        compute_row_range(out, w, x, in_features, 0, out_features, type);
        return;
    }

    init_worker_pool(num_threads);

    int rows_per_thread = out_features / num_threads;

    for (int t = 1; t < num_threads; t++) {
        int r_start = t * rows_per_thread;
        int r_end = (t == num_threads - 1) ? out_features : (t + 1) * rows_per_thread;

        pthread_mutex_lock(&g_workers[t].mutex);
        g_workers[t].job = (matvec_job){
            .out = out,
            .w = w,
            .x = x,
            .in_features = in_features,
            .r_start = r_start,
            .r_end = r_end,
            .type = type
        };
        g_workers[t].work_available = 1;
        pthread_cond_signal(&g_workers[t].cond_start);
        pthread_mutex_unlock(&g_workers[t].mutex);
    }

    compute_row_range(out, w, x, in_features, 0, rows_per_thread, type);

    for (int t = 1; t < num_threads; t++) {
        pthread_mutex_lock(&g_workers[t].mutex);
        while (g_workers[t].work_available) {
            pthread_cond_wait(&g_workers[t].cond_done, &g_workers[t].mutex);
        }
        pthread_mutex_unlock(&g_workers[t].mutex);
    }
}
