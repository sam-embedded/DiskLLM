#include "arch/registry.h"
#include "model_config.h"
#include <string.h>
#include <stdio.h>

static bool registry_initialized = false;

void diskllm_arch_registry_init(void) {
    if (registry_initialized) return;
    registry_initialized = true;
}

const diskllm_arch_backend *diskllm_arch_find(int model_type) {
    diskllm_arch_registry_init();

    switch (model_type) {
        case MODEL_TYPE_QWEN_HYBRID:
        case MODEL_TYPE_QWEN_ATTENTION_ONLY:
            return &diskllm_arch_qwen;
        case MODEL_TYPE_LLAMA:
            return &diskllm_arch_llama;
        case MODEL_TYPE_MISTRAL:
            return &diskllm_arch_mistral;
        case MODEL_TYPE_PHI3:
            return &diskllm_arch_phi3;
        case MODEL_TYPE_GEMMA:
            return &diskllm_arch_gemma;
        case MODEL_TYPE_GEMMA2:
            return &diskllm_arch_gemma2;
        case MODEL_TYPE_GEMMA3:
            return &diskllm_arch_gemma3;
        case MODEL_TYPE_GEMMA4:
            return &diskllm_arch_gemma4;
        default:
            return &diskllm_arch_qwen;
    }
}
