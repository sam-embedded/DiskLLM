#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif

#include "tensor_catalog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

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


