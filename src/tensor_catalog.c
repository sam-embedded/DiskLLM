#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif

#include "model_config.h"
#include "tensor_catalog.h"
#include "layer_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <inttypes.h>
#include <sys/types.h>
#include <unistd.h>

typedef enum {
    G_TYPE_UINT8   = 0,
    G_TYPE_INT8    = 1,
    G_TYPE_UINT16  = 2,
    G_TYPE_INT16   = 3,
    G_TYPE_UINT32  = 4,
    G_TYPE_INT32   = 5,
    G_TYPE_FLOAT32 = 6,
    G_TYPE_BOOL    = 7,
    G_TYPE_STRING  = 8,
    G_TYPE_ARRAY   = 9,
    G_TYPE_UINT64  = 10,
    G_TYPE_INT64   = 11,
    G_TYPE_FLOAT64 = 12,
} g_type;

static void skip_val(FILE *f, uint32_t val_type) {
    switch (val_type) {
        case G_TYPE_UINT8:
        case G_TYPE_INT8:
        case G_TYPE_BOOL:
            fseeko(f, 1, SEEK_CUR);
            break;
        case G_TYPE_UINT16:
        case G_TYPE_INT16:
            fseeko(f, 2, SEEK_CUR);
            break;
        case G_TYPE_UINT32:
        case G_TYPE_INT32:
        case G_TYPE_FLOAT32:
            fseeko(f, 4, SEEK_CUR);
            break;
        case G_TYPE_UINT64:
        case G_TYPE_INT64:
        case G_TYPE_FLOAT64:
            fseeko(f, 8, SEEK_CUR);
            break;
        case G_TYPE_STRING: {
            uint64_t len;
            if (fread(&len, 8, 1, f) != 1) return;
            fseeko(f, len, SEEK_CUR);
            break;
        }
        case G_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t len;
            if (fread(&elem_type, 4, 1, f) != 1) return;
            if (fread(&len, 8, 1, f) != 1) return;
            for (uint64_t i = 0; i < len; i++) {
                skip_val(f, elem_type);
            }
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown value type %u\n", val_type);
            exit(1);
    }
}

static uint64_t get_type_size(uint32_t type, uint64_t nelements) {
    switch (type) {
        case 0: return nelements * 4; // F32
        case 1: return nelements * 2; // F16
        case 2: return ((nelements + 31) / 32) * 18; // Q4_0
        case 3: return ((nelements + 31) / 32) * 20; // Q4_1
        case 6: return ((nelements + 31) / 32) * 22; // Q5_0
        case 7: return ((nelements + 31) / 32) * 24; // Q5_1
        case 8: return ((nelements + 31) / 32) * 34; // Q8_0
        case 9: return ((nelements + 31) / 32) * 36; // Q8_1
        case 12: return ((nelements + 255) / 256) * 144; // Q4_K
        case 13: return ((nelements + 255) / 256) * 176; // Q5_K
        case 14: return ((nelements + 255) / 256) * 210; // Q6_K
        case 15: return ((nelements + 255) / 256) * 288; // Q8_K
        case 16: return nelements * 1; // I8
        case 17: return nelements * 2; // I16
        case 18: return nelements * 4; // I32
        default: return 0;
    }
}

tensor_catalog *load_tensor_catalog(const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        perror("Error opening file in load_tensor_catalog");
        return NULL;
    }

    struct {
        char magic[4];
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    } header;

    if (fread(&header, 1, sizeof(header), f) != sizeof(header)) {
        fprintf(stderr, "Error: Failed to read GGUF header.\n");
        fclose(f);
        return NULL;
    }

    if (strncmp(header.magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Error: Invalid magic in GGUF file.\n");
        fclose(f);
        return NULL;
    }

    uint64_t alignment = 32;

    // Read metadata to resolve alignment
    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        char *key = malloc(key_len + 1);
        if (fread(key, 1, key_len, f) != key_len) {
            free(key);
            fclose(f);
            return NULL;
        }
        key[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) {
            free(key);
            fclose(f);
            return NULL;
        }

        if (strcmp(key, "general.alignment") == 0) {
            if (val_type == G_TYPE_UINT32) {
                uint32_t val;
                if (fread(&val, 4, 1, f) == 1) alignment = val;
            } else if (val_type == G_TYPE_UINT64) {
                uint64_t val;
                if (fread(&val, 8, 1, f) == 1) alignment = val;
            } else {
                skip_val(f, val_type);
            }
        } else {
            skip_val(f, val_type);
        }
        free(key);
    }

    // Allocate catalog structure
    tensor_catalog *catalog = malloc(sizeof(tensor_catalog));
    if (!catalog) {
        fclose(f);
        return NULL;
    }
    catalog->count = (int)header.tensor_count;
    catalog->tensors = malloc(catalog->count * sizeof(tensor_info));
    if (!catalog->tensors) {
        free(catalog);
        fclose(f);
        return NULL;
    }

    // Read tensor infos
    typedef struct {
        char *name;
        uint32_t n_dimensions;
        uint64_t dimensions[8];
        uint32_t type;
        uint64_t offset;
    } temp_tensor_info;

    temp_tensor_info *temp_tensors = malloc(header.tensor_count * sizeof(temp_tensor_info));
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        uint64_t name_len;
        if (fread(&name_len, 8, 1, f) != 1) {
            // Free and return
            fclose(f);
            return NULL;
        }
        temp_tensors[i].name = malloc(name_len + 1);
        if (fread(temp_tensors[i].name, 1, name_len, f) != name_len) {
            // Error
            fclose(f);
            return NULL;
        }
        temp_tensors[i].name[name_len] = '\0';

        if (fread(&temp_tensors[i].n_dimensions, 4, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        if (fread(temp_tensors[i].dimensions, 8, temp_tensors[i].n_dimensions, f) != temp_tensors[i].n_dimensions) {
            fclose(f);
            return NULL;
        }
        if (fread(&temp_tensors[i].type, 4, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        if (fread(&temp_tensors[i].offset, 8, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
    }

    off_t post_tensor_infos_offset = ftello(f);
    uint64_t tensor_data_block_offset = ((post_tensor_infos_offset + alignment - 1) / alignment) * alignment;

    // Build the final catalog
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        tensor_info *ti = &catalog->tensors[i];
        strncpy(ti->name, temp_tensors[i].name, sizeof(ti->name) - 1);
        ti->name[sizeof(ti->name) - 1] = '\0';
        ti->type = temp_tensors[i].type;
        ti->n_dims = temp_tensors[i].n_dimensions;
        
        uint64_t elements = 1;
        if (ti->n_dims == 0) {
            elements = 0;
        } else {
            for (uint32_t d = 0; d < ti->n_dims; d++) {
                ti->dims[d] = temp_tensors[i].dimensions[d];
                if (d < 4) {
                    elements *= ti->dims[d];
                }
            }
        }
        ti->absolute_offset = tensor_data_block_offset + temp_tensors[i].offset;
        ti->byte_size = get_type_size(ti->type, elements);

        free(temp_tensors[i].name);
    }

    free(temp_tensors);
    fclose(f);
    return catalog;
}

void free_tensor_catalog(tensor_catalog *catalog) {
    if (!catalog) return;
    if (catalog->tensors) free(catalog->tensors);
    free(catalog);
}

const tensor_info *find_tensor(const tensor_catalog *catalog, const char *name) {
    if (!catalog) return NULL;
    for (int i = 0; i < catalog->count; i++) {
        if (strcmp(catalog->tensors[i].name, name) == 0) {
            return &catalog->tensors[i];
        }
    }
    return NULL;
}

int get_metadata_uint32(const char *filepath, const char *key, uint32_t *out_val) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    struct {
        char magic[4];
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    } header;

    if (fread(&header, 1, sizeof(header), f) != sizeof(header)) {
        fclose(f);
        return -1;
    }

    if (strncmp(header.magic, "GGUF", 4) != 0) {
        fclose(f);
        return -1;
    }

    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) {
            fclose(f);
            return -1;
        }
        char *k = malloc(key_len + 1);
        if (fread(k, 1, key_len, f) != key_len) {
            free(k);
            fclose(f);
            return -1;
        }
        k[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) {
            free(k);
            fclose(f);
            return -1;
        }

        if (strcmp(k, key) == 0) {
            free(k);
            if (val_type == G_TYPE_UINT32) {
                uint32_t val;
                if (fread(&val, 4, 1, f) == 1) {
                    *out_val = val;
                    fclose(f);
                    return 0;
                }
            } else if (val_type == G_TYPE_INT32) {
                int32_t val;
                if (fread(&val, 4, 1, f) == 1) {
                    *out_val = (uint32_t)val;
                    fclose(f);
                    return 0;
                }
            }
            fclose(f);
            return -1;
        } else {
            skip_val(f, val_type);
        }
        free(k);
    }

    fclose(f);
    return -1;
}

int lookup_token_by_id(const char *filepath, uint32_t target_id, char *out_str, size_t max_len) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    struct {
        char magic[4];
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    } header;

    if (fread(&header, 1, sizeof(header), f) != sizeof(header)) {
        fclose(f);
        return -1;
    }

    if (strncmp(header.magic, "GGUF", 4) != 0) {
        fclose(f);
        return -1;
    }

    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) {
            fclose(f);
            return -1;
        }
        char *k = malloc(key_len + 1);
        if (fread(k, 1, key_len, f) != key_len) {
            free(k);
            fclose(f);
            return -1;
        }
        k[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) {
            free(k);
            fclose(f);
            return -1;
        }

        if (strcmp(k, "tokenizer.ggml.tokens") == 0) {
            free(k);
            if (val_type != 9) { // G_TYPE_ARRAY
                fclose(f);
                return -1;
            }
            uint32_t elem_type;
            uint64_t array_len;
            if (fread(&elem_type, 4, 1, f) != 1) {
                fclose(f);
                return -1;
            }
            if (fread(&array_len, 8, 1, f) != 1) {
                fclose(f);
                return -1;
            }
            if (elem_type != 8) { // G_TYPE_STRING
                fclose(f);
                return -1;
            }
            if (target_id >= array_len) {
                fclose(f);
                return -2; // out of range
            }
            // skip target_id strings
            for (uint32_t idx = 0; idx < target_id; idx++) {
                uint64_t str_len;
                if (fread(&str_len, 8, 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                fseeko(f, str_len, SEEK_CUR);
            }
            // read target_id string
            uint64_t target_str_len;
            if (fread(&target_str_len, 8, 1, f) != 1) {
                fclose(f);
                return -1;
            }
            size_t to_copy = (target_str_len < max_len - 1) ? target_str_len : max_len - 1;
            if (fread(out_str, 1, to_copy, f) != to_copy) {
                fclose(f);
                return -1;
            }
            out_str[to_copy] = '\0';
            fclose(f);
            return 0;
        } else {
            skip_val(f, val_type);
        }
        free(k);
    }

    fclose(f);
    return -1;
}

int search_token(const char *filepath, const char *query) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    struct {
        char magic[4];
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    } header;

    if (fread(&header, 1, sizeof(header), f) != sizeof(header)) {
        fclose(f);
        return -1;
    }

    if (strncmp(header.magic, "GGUF", 4) != 0) {
        fclose(f);
        return -1;
    }

    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) {
            fclose(f);
            return -1;
        }
        char *k = malloc(key_len + 1);
        if (fread(k, 1, key_len, f) != key_len) {
            free(k);
            fclose(f);
            return -1;
        }
        k[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) {
            free(k);
            fclose(f);
            return -1;
        }

        if (strcmp(k, "tokenizer.ggml.tokens") == 0) {
            free(k);
            if (val_type != 9) { // G_TYPE_ARRAY
                fclose(f);
                return -1;
            }
            uint32_t elem_type;
            uint64_t array_len;
            if (fread(&elem_type, 4, 1, f) != 1) {
                fclose(f);
                return -1;
            }
            if (fread(&array_len, 8, 1, f) != 1) {
                fclose(f);
                return -1;
            }
            if (elem_type != 8) { // G_TYPE_STRING
                fclose(f);
                return -1;
            }

            char str_buf[1024];
            for (uint32_t idx = 0; idx < array_len; idx++) {
                uint64_t str_len;
                if (fread(&str_len, 8, 1, f) != 1) {
                    fclose(f);
                    return -1;
                }
                size_t to_read = (str_len < sizeof(str_buf) - 1) ? str_len : sizeof(str_buf) - 1;
                if (fread(str_buf, 1, to_read, f) != to_read) {
                    fclose(f);
                    return -1;
                }
                str_buf[to_read] = '\0';
                if (str_len > to_read) {
                    fseeko(f, str_len - to_read, SEEK_CUR);
                }

                if (strcmp(str_buf, query) == 0) {
                    printf("Exact match - Token ID %u: \"%s\"\n", idx, str_buf);
                } else if (strstr(str_buf, query) != NULL) {
                    printf("Substring match - Token ID %u: \"%s\"\n", idx, str_buf);
                }
            }
            fclose(f);
            return 0;
        } else {
            skip_val(f, val_type);
        }
        free(k);
    }

    fclose(f);
    return -1;
}

int get_metadata_float(const char *filepath, const char *key, float *out_val) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    struct {
        char magic[4];
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    } header;

    if (fread(&header, 1, sizeof(header), f) != sizeof(header) || strncmp(header.magic, "GGUF", 4) != 0) {
        fclose(f);
        return -1;
    }

    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) { fclose(f); return -1; }
        char *k = malloc(key_len + 1);
        if (fread(k, 1, key_len, f) != key_len) { free(k); fclose(f); return -1; }
        k[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) { free(k); fclose(f); return -1; }

        if (strcmp(k, key) == 0) {
            free(k);
            if (val_type == G_TYPE_FLOAT32) {
                float val;
                if (fread(&val, 4, 1, f) == 1) { *out_val = val; fclose(f); return 0; }
            } else if (val_type == G_TYPE_FLOAT64) {
                double val;
                if (fread(&val, 8, 1, f) == 1) { *out_val = (float)val; fclose(f); return 0; }
            }
            fclose(f);
            return -1;
        } else {
            skip_val(f, val_type);
        }
        free(k);
    }

    fclose(f);
    return -1;
}

int get_metadata_string(const char *filepath, const char *key, char *out_str, size_t max_len) {
    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    struct {
        char magic[4];
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    } header;

    if (fread(&header, 1, sizeof(header), f) != sizeof(header) || strncmp(header.magic, "GGUF", 4) != 0) {
        fclose(f);
        return -1;
    }

    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) { fclose(f); return -1; }
        char *k = malloc(key_len + 1);
        if (fread(k, 1, key_len, f) != key_len) { free(k); fclose(f); return -1; }
        k[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) { free(k); fclose(f); return -1; }

        if (strcmp(k, key) == 0) {
            free(k);
            if (val_type == G_TYPE_STRING) {
                uint64_t str_len;
                if (fread(&str_len, 8, 1, f) != 1) { fclose(f); return -1; }
                size_t to_copy = (str_len < max_len - 1) ? str_len : max_len - 1;
                if (fread(out_str, 1, to_copy, f) != to_copy) { fclose(f); return -1; }
                out_str[to_copy] = '\0';
                fclose(f);
                return 0;
            }
            fclose(f);
            return -1;
        } else {
            skip_val(f, val_type);
        }
        free(k);
    }

    fclose(f);
    return -1;
}

qwen_model_config *load_qwen_model_config(const char *filepath, const tensor_catalog *cat, const char *arch_flag_str) {
    qwen_model_config *cfg = calloc(1, sizeof(qwen_model_config));
    if (!cfg) return NULL;

    /* 1. Read architecture name */
    if (get_metadata_string(filepath, "general.architecture", cfg->architecture, sizeof(cfg->architecture)) != 0) {
        strncpy(cfg->architecture, "qwen35", sizeof(cfg->architecture) - 1);
    }
    if (get_metadata_string(filepath, "general.name", cfg->model_name, sizeof(cfg->model_name)) != 0) {
        get_metadata_string(filepath, "general.basename", cfg->model_name, sizeof(cfg->model_name));
    }

    char key_buf[256];
    #define GET_U32(field, subkey, fallback_val) do { \
        snprintf(key_buf, sizeof(key_buf), "%s.%s", cfg->architecture, subkey); \
        uint32_t v; \
        if (get_metadata_uint32(filepath, key_buf, &v) == 0) { \
            field = (int)v; \
        } else { \
            snprintf(key_buf, sizeof(key_buf), "qwen35.%s", subkey); \
            if (get_metadata_uint32(filepath, key_buf, &v) == 0) { \
                field = (int)v; \
            } else { \
                snprintf(key_buf, sizeof(key_buf), "qwen2.%s", subkey); \
                if (get_metadata_uint32(filepath, key_buf, &v) == 0) { \
                    field = (int)v; \
                } else { \
                    field = fallback_val; \
                } \
            } \
        } \
    } while(0)

    #define GET_FLT(field, subkey, fallback_val) do { \
        snprintf(key_buf, sizeof(key_buf), "%s.%s", cfg->architecture, subkey); \
        float v; \
        if (get_metadata_float(filepath, key_buf, &v) == 0) { \
            field = v; \
        } else { \
            snprintf(key_buf, sizeof(key_buf), "qwen35.%s", subkey); \
            if (get_metadata_float(filepath, key_buf, &v) == 0) { \
                field = v; \
            } else { \
                snprintf(key_buf, sizeof(key_buf), "qwen2.%s", subkey); \
                if (get_metadata_float(filepath, key_buf, &v) == 0) { \
                    field = v; \
                } else { \
                    field = fallback_val; \
                } \
            } \
        } \
    } while(0)

    GET_U32(cfg->block_count, "block_count", 0);
    GET_U32(cfg->hidden_dim, "embedding_length", 5120);
    GET_U32(cfg->ffn_dim, "feed_forward_length", 17408);
    GET_U32(cfg->num_attn_heads, "attention.head_count", 24);
    GET_U32(cfg->num_kv_heads, "attention.head_count_kv", 4);
    GET_U32(cfg->key_length, "attention.key_length", 256);
    GET_U32(cfg->value_length, "attention.value_length", 256);
    cfg->head_dim = cfg->key_length;

    GET_FLT(cfg->rope_freq_base, "rope.freq_base", 10000000.0f);
    GET_U32(cfg->rope_dim, "rope.dimension_count", 64);

    GET_FLT(cfg->rope_scaling_factor, "rope.scaling.factor", 1.0f);
    GET_U32(cfg->rope_orig_context_len, "rope.scaling.original_context_length", 4096);
    GET_FLT(cfg->rope_ext_factor, "rope.scaling.ext_factor", 1.0f);
    GET_FLT(cfg->rope_attn_factor, "rope.scaling.attn_factor", 1.0f);
    GET_FLT(cfg->rope_beta_fast, "rope.scaling.beta_fast", 32.0f);
    GET_FLT(cfg->rope_beta_slow, "rope.scaling.beta_slow", 1.0f);
    GET_U32(cfg->max_context_length, "context_length", 262144);

    cfg->rope_scaling_type = ROPE_SCALING_NONE;
    char type_str[64] = {0};
    snprintf(key_buf, sizeof(key_buf), "%s.rope.scaling.type", cfg->architecture);
    if (get_metadata_string(filepath, key_buf, type_str, sizeof(type_str)) != 0) {
        snprintf(key_buf, sizeof(key_buf), "qwen35.rope.scaling.type");
        get_metadata_string(filepath, key_buf, type_str, sizeof(type_str));
    }
    if (type_str[0]) {
        if (!strcasecmp(type_str, "yarn")) cfg->rope_scaling_type = ROPE_SCALING_YARN;
        else if (!strcasecmp(type_str, "linear")) cfg->rope_scaling_type = ROPE_SCALING_LINEAR;
    }

    GET_U32(cfg->ssm_conv_kernel, "ssm.conv_kernel", 4);
    GET_U32(cfg->ssm_state_size, "ssm.state_size", 128);
    GET_U32(cfg->ssm_group_count, "ssm.group_count", 16);
    GET_U32(cfg->ssm_time_step_rank, "ssm.time_step_rank", 48);
    GET_U32(cfg->ssm_inner_size, "ssm.inner_size", 6144);
    GET_U32(cfg->full_attn_interval, "full_attention_interval", 4);

    #undef GET_U32
    #undef GET_FLT

    /* 2. Special Tokens */
    cfg->eos_token_id = 248046;
    cfg->bos_token_id = 248044;
    get_metadata_uint32(filepath, "tokenizer.ggml.eos_token_id", &cfg->eos_token_id);
    get_metadata_uint32(filepath, "tokenizer.ggml.bos_token_id", &cfg->bos_token_id);

    /* 3. Vocab Size from token_embd.weight shape */
    const tensor_info *ti_emb = find_tensor(cat, "token_embd.weight");
    if (ti_emb && ti_emb->n_dims >= 2) {
        cfg->vocab_size = (int)ti_emb->dims[1];
    } else {
        cfg->vocab_size = 248320;
    }

    /* 4. Tied embedding check */
    const tensor_info *ti_out = find_tensor(cat, "output.weight");
    if (!ti_out && ti_emb) {
        cfg->is_tied_embedding = 1;
    } else {
        cfg->is_tied_embedding = 0;
    }

    /* 5. Detect Block Count and Layer Types */
    int max_blk = -1;
    int has_ssm_tensors = 0;
    int has_attn_tensors = 0;

    for (int i = 0; i < cat->count; i++) {
        const char *name = cat->tensors[i].name;
        if (strncmp(name, "blk.", 4) == 0) {
            int b = atoi(name + 4);
            if (b >= 0 && b < 1024) {
                if (b > max_blk) max_blk = b;
            }
        }
    }

    int detected_blocks = max_blk + 1;
    if (cfg->block_count == 0 || detected_blocks < cfg->block_count) {
        cfg->block_count = detected_blocks;
    }

    if (cfg->block_count > 0) {
        char test_nm[256];
        snprintf(test_nm, sizeof(test_nm), "blk.%d.ffn_gate.weight", cfg->block_count - 1);
        if (!find_tensor(cat, test_nm) && cfg->block_count > 1) {
            snprintf(test_nm, sizeof(test_nm), "blk.%d.nextn.eh_proj.weight", cfg->block_count - 1);
            if (find_tensor(cat, test_nm)) {
                cfg->block_count--;
            }
        }
    }

    /* Classify layer types for each block */
    cfg->num_ssm_layers = 0;
    cfg->num_attn_layers = 0;
    cfg->has_nextn = 0;

    for (int b = 0; b < cfg->block_count; b++) {
        cfg->layer_types[b] = classify_layer_type(cat, b);
        if (cfg->layer_types[b] == LAYER_TYPE_SSM) {
            has_ssm_tensors = 1;
            cfg->num_ssm_layers++;
        } else if (cfg->layer_types[b] == LAYER_TYPE_ATTENTION) {
            has_attn_tensors = 1;
            cfg->num_attn_layers++;
        }

        char nm[256];
        snprintf(nm, sizeof(nm), "blk.%d.nextn.eh_proj.weight", b);
        if (find_tensor(cat, nm) != NULL) {
            cfg->layer_has_nextn[b] = 1;
            cfg->has_nextn = 1;
        } else {
            cfg->layer_has_nextn[b] = 0;
        }
    }
    char nextn_top[256];
    snprintf(nextn_top, sizeof(nextn_top), "blk.%d.nextn.eh_proj.weight", cfg->block_count);
    if (find_tensor(cat, nextn_top) != NULL) {
        cfg->has_nextn = 1;
    }

    /* 6. Model Type Auto Detection */
    if (arch_flag_str && !strcmp(arch_flag_str, "qwen-hybrid")) {
        cfg->model_type = MODEL_TYPE_QWEN_HYBRID;
    } else if (arch_flag_str && !strcmp(arch_flag_str, "qwen-attention")) {
        cfg->model_type = MODEL_TYPE_QWEN_ATTENTION_ONLY;
    } else {
        if (has_ssm_tensors) {
            cfg->model_type = MODEL_TYPE_QWEN_HYBRID;
        } else if (has_attn_tensors) {
            cfg->model_type = MODEL_TYPE_QWEN_ATTENTION_ONLY;
        } else {
            cfg->model_type = MODEL_TYPE_UNSUPPORTED;
        }
    }

    return cfg;
}

void free_qwen_model_config(qwen_model_config *cfg) {
    if (cfg) free(cfg);
}



