#include "sampler.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ─── Greedy fast path ──────────────────────────────────────────────────────

int greedy_sample(const float *logits, int vocab_size) {
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void sampler_apply_repetition_penalty(float *logits, int vocab_size, const int *seen_tokens, int num_seen, float penalty) {
    if (penalty <= 1.0f || !seen_tokens || num_seen <= 0) return;
    for (int i = 0; i < num_seen; i++) {
        int tid = seen_tokens[i];
        if (tid >= 0 && tid < vocab_size) {
            if (logits[tid] < 0.0f) {
                logits[tid] *= penalty;
            } else {
                logits[tid] /= penalty;
            }
        }
    }
}

void sampler_apply_presence_penalty(float *logits, int vocab_size, const int *seen_tokens, int num_seen, float presence_penalty) {
    if (presence_penalty == 0.0f || !seen_tokens || num_seen <= 0) return;
    for (int i = 0; i < num_seen; i++) {
        int tid = seen_tokens[i];
        if (tid >= 0 && tid < vocab_size) {
            int first_seen = 1;
            for (int j = 0; j < i; j++) {
                if (seen_tokens[j] == tid) {
                    first_seen = 0;
                    break;
                }
            }
            if (first_seen) {
                logits[tid] -= presence_penalty;
            }
        }
    }
}

// ─── Sampler state ─────────────────────────────────────────────────────────

typedef struct {
    int   index;
    float value;
} logit_entry;

struct sampler {
    sampler_config cfg;
    uint64_t rng_state;          // xorshift64 state
    logit_entry *work_buf;       // scratch sort buffer [vocab_size]
    int          work_cap;
};

// xorshift64 PRNG
static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// Uniform float in [0, 1)
static inline float rand_f32(uint64_t *state) {
    return (float)(xorshift64(state) >> 11) * (1.0f / (float)(1ULL << 53));
}

sampler *sampler_create(const sampler_config *cfg) {
    sampler *s = calloc(1, sizeof(sampler));
    if (!s) return NULL;
    s->cfg = *cfg;
    s->rng_state = cfg->seed ? cfg->seed : 0xDEADBEEF12345678ULL;
    return s;
}

void sampler_free(sampler *s) {
    if (!s) return;
    if (s->work_buf) free(s->work_buf);
    free(s);
}

// Ensure work buffer is large enough
static int ensure_buf(sampler *s, int vocab_size) {
    if (s->work_cap < vocab_size) {
        logit_entry *p = realloc(s->work_buf, vocab_size * sizeof(logit_entry));
        if (!p) return -1;
        s->work_buf = p;
        s->work_cap = vocab_size;
    }
    return 0;
}

// Partial sort descending by value to find top_k entries (O(n*k) insertion)
static void partial_sort_top_k(logit_entry *buf, int n, int k) {
    // Ensure top-k entries sit in buf[0..k-1] sorted descending
    for (int i = 1; i < n && i < k; i++) {
        int j = i;
        logit_entry tmp = buf[i];
        while (j > 0 && buf[j-1].value < tmp.value) {
            buf[j] = buf[j-1];
            j--;
        }
        buf[j] = tmp;
    }
    // Now scan the rest and maintain sorted top-k window
    for (int i = k; i < n; i++) {
        if (buf[i].value > buf[k-1].value) {
            buf[k-1] = buf[i];
            // Re-insert into sorted position
            int j = k-1;
            logit_entry tmp = buf[j];
            while (j > 0 && buf[j-1].value < tmp.value) {
                buf[j] = buf[j-1];
                j--;
            }
            buf[j] = tmp;
        }
    }
}

int sampler_sample(sampler *s, float *logits, int vocab_size) {
    float temp = s->cfg.temperature;

    // Greedy shortcut
    if (temp == 0.0f) {
        return greedy_sample(logits, vocab_size);
    }

    // Fill work buffer
    if (ensure_buf(s, vocab_size) != 0) {
        return greedy_sample(logits, vocab_size);
    }
    for (int i = 0; i < vocab_size; i++) {
        s->work_buf[i].index = i;
        s->work_buf[i].value = logits[i];
    }

    // Apply temperature scaling
    float inv_temp = 1.0f / temp;
    float max_logit = s->work_buf[0].value;
    for (int i = 1; i < vocab_size; i++) {
        if (s->work_buf[i].value > max_logit) max_logit = s->work_buf[i].value;
    }
    // Numerically stable softmax with temperature
    for (int i = 0; i < vocab_size; i++) {
        s->work_buf[i].value = (s->work_buf[i].value - max_logit) * inv_temp;
    }

    // Top-K filtering
    int k = vocab_size;
    if (s->cfg.top_k > 0 && s->cfg.top_k < vocab_size) {
        k = s->cfg.top_k;
        partial_sort_top_k(s->work_buf, vocab_size, k);
        // Zero out entries beyond top-k by clamping n to k
    }

    // Convert to probabilities (softmax over selected k)
    float max_v = s->work_buf[0].value;
    for (int i = 1; i < k; i++) {
        if (s->work_buf[i].value > max_v) max_v = s->work_buf[i].value;
    }
    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        s->work_buf[i].value = expf(s->work_buf[i].value - max_v);
        sum += s->work_buf[i].value;
    }
    float inv_sum = (float)(1.0 / sum);
    for (int i = 0; i < k; i++) {
        s->work_buf[i].value *= inv_sum;
    }

    // Min-P filtering (remove tokens with prob < min_p * max_prob)
    if (s->cfg.min_p > 0.0f) {
        float min_prob = s->cfg.min_p; // work_buf[0] has highest prob after sort
        int nk = 0;
        for (int i = 0; i < k; i++) {
            if (s->work_buf[i].value >= min_prob) {
                s->work_buf[nk++] = s->work_buf[i];
            }
        }
        if (nk > 0) k = nk;
        // Renormalize
        double s2 = 0.0;
        for (int i = 0; i < k; i++) s2 += s->work_buf[i].value;
        float inv_s2 = (float)(1.0 / s2);
        for (int i = 0; i < k; i++) s->work_buf[i].value *= inv_s2;
    }

    // Top-P (nucleus) filtering: keep smallest prefix whose cumulative prob >= top_p
    if (s->cfg.top_p < 1.0f) {
        // work_buf[0..k-1] is sorted descending by probability
        double cumul = 0.0;
        int nk = k;
        for (int i = 0; i < k; i++) {
            cumul += s->work_buf[i].value;
            if ((float)cumul >= s->cfg.top_p) {
                nk = i + 1;
                break;
            }
        }
        k = nk;
        // Renormalize
        double s3 = 0.0;
        for (int i = 0; i < k; i++) s3 += s->work_buf[i].value;
        float inv_s3 = (float)(1.0 / s3);
        for (int i = 0; i < k; i++) s->work_buf[i].value *= inv_s3;
    }

    // Multinomial sampling
    float r = rand_f32(&s->rng_state);
    float cumul = 0.0f;
    for (int i = 0; i < k; i++) {
        cumul += s->work_buf[i].value;
        if (r < cumul) {
            return s->work_buf[i].index;
        }
    }
    return s->work_buf[k-1].index;
}
