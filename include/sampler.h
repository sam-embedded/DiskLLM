#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

// Opaque sampler state
typedef struct sampler sampler;

// Sampling configuration
typedef struct {
    float temperature;   // Sampling temperature (0.0 = greedy, >0 = stochastic)
    int   top_k;         // Top-K filtering (0 = disabled)
    float top_p;         // Top-P (nucleus) filtering (1.0 = disabled)
    float min_p;         // Min-P filtering (0.0 = disabled)
    uint64_t seed;       // RNG seed
} sampler_config;

// Create a sampler with given config
sampler *sampler_create(const sampler_config *cfg);

// Free a sampler
void sampler_free(sampler *s);

// Apply repetition penalty to logits for previously seen tokens
void sampler_apply_repetition_penalty(float *logits, int vocab_size, const int *seen_tokens, int num_seen, float penalty);

// Sample from logits. Returns token index.
int sampler_sample(sampler *s, float *logits, int vocab_size);

// Greedy sample (fast path, no state needed)
int greedy_sample(const float *logits, int vocab_size);

#endif // SAMPLER_H
