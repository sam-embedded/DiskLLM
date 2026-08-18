#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <stdint.h>
#include <stddef.h>
#include "layer_map.h"
#include "tensor_catalog.h"

typedef enum {
    MODEL_TYPE_QWEN_HYBRID = 0,
    MODEL_TYPE_QWEN_ATTENTION_ONLY = 1,
    MODEL_TYPE_UNSUPPORTED = 2
} model_type_enum;

typedef enum {
    ROPE_SCALING_NONE = 0,
    ROPE_SCALING_LINEAR = 1,
    ROPE_SCALING_YARN = 2
} rope_scaling_type_enum;

typedef struct qwen_model_config {
    char architecture[64];
    char model_name[64];
    model_type_enum model_type;

    int block_count;
    int hidden_dim;
    int ffn_dim;
    int vocab_size;

    int num_attn_heads;
    int num_kv_heads;
    int head_dim;
    int key_length;
    int value_length;

    float rope_freq_base;
    int rope_dim;

    rope_scaling_type_enum rope_scaling_type;
    float rope_scaling_factor;
    int rope_orig_context_len;
    float rope_ext_factor;
    float rope_attn_factor;
    float rope_beta_fast;
    float rope_beta_slow;
    int max_context_length;

    int ssm_conv_kernel;
    int ssm_state_size;
    int ssm_group_count;
    int ssm_time_step_rank;
    int ssm_inner_size;
    int full_attn_interval;

    int has_nextn;
    int is_tied_embedding;

    layer_type layer_types[1024];
    int layer_has_nextn[1024];

    int num_ssm_layers;
    int num_attn_layers;

    uint32_t eos_token_id;
    uint32_t bos_token_id;
} qwen_model_config;

qwen_model_config *load_qwen_model_config(const char *filepath, const tensor_catalog *cat, const char *arch_flag_str);
void free_qwen_model_config(qwen_model_config *cfg);

#endif // MODEL_CONFIG_H
