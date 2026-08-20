#ifndef DISKLLM_ARCH_REGISTRY_H
#define DISKLLM_ARCH_REGISTRY_H

#include <stdbool.h>

typedef struct diskllm_model diskllm_model;
typedef struct diskllm_context diskllm_context;

typedef struct diskllm_arch_backend {
    const char *name;
    bool (*init)(diskllm_model *model);
    bool (*prefill_layer)(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_states, int prompt_len);
    bool (*decode_layer)(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_single, int cur_pos);
} diskllm_arch_backend;

void diskllm_arch_registry_init(void);
const diskllm_arch_backend *diskllm_arch_find(int model_type);

/* Individual Architecture Drivers */
extern const diskllm_arch_backend diskllm_arch_qwen;
extern const diskllm_arch_backend diskllm_arch_qwen35;
extern const diskllm_arch_backend diskllm_arch_qwen2;
extern const diskllm_arch_backend diskllm_arch_llama;
extern const diskllm_arch_backend diskllm_arch_phi3;
extern const diskllm_arch_backend diskllm_arch_gemma;
extern const diskllm_arch_backend diskllm_arch_gemma2;
extern const diskllm_arch_backend diskllm_arch_gemma3;
extern const diskllm_arch_backend diskllm_arch_gemma4;
extern const diskllm_arch_backend diskllm_arch_mistral;

#endif /* DISKLLM_ARCH_REGISTRY_H */
