#define _FILE_OFFSET_BITS 64
#define _LARGEFILE_SOURCE
#define _LARGEFILE64_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <sys/types.h>

// GGUF value types definition
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type;

typedef enum {
    LAYER_TYPE_SSM = 0,
    LAYER_TYPE_ATTENTION = 1,
    LAYER_TYPE_NEXTN = 2,
    LAYER_TYPE_UNKNOWN = 3
} layer_type;

const char *layer_type_name(layer_type type) {
    switch (type) {
        case LAYER_TYPE_SSM: return "SSM";
        case LAYER_TYPE_ATTENTION: return "ATTENTION";
        case LAYER_TYPE_NEXTN: return "NEXTN";
        default: return "UNKNOWN";
    }
}

// Global tensors
const char *GLOBAL_TENSORS[] = {
    "token_embd.weight",
    "output_norm.weight",
    "output.weight"
};
#define GLOBAL_TENSORS_COUNT 3

// SSM Tensors
const char *SSM_TENSOR_SUFFIXES[] = {
    "attn_gate.weight",
    "attn_norm.weight",
    "attn_qkv.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight",
    "ssm_a",
    "ssm_alpha.weight",
    "ssm_beta.weight",
    "ssm_conv1d.weight",
    "ssm_dt.bias",
    "ssm_norm.weight",
    "ssm_out.weight"
};
#define SSM_TENSOR_SUFFIXES_COUNT 14

// Attention Tensors
const char *ATTN_TENSOR_SUFFIXES[] = {
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "post_attention_norm.weight"
};
#define ATTN_TENSOR_SUFFIXES_COUNT 11

// NextN Tensors
const char *NEXTN_TENSOR_SUFFIXES[] = {
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_norm.weight",
    "attn_output.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_v.weight",
    "ffn_down.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "nextn.eh_proj.weight",
    "nextn.enorm.weight",
    "nextn.hnorm.weight",
    "nextn.shared_head_norm.weight",
    "post_attention_norm.weight"
};
#define NEXTN_TENSOR_SUFFIXES_COUNT 15

typedef struct {
    char *name;
    uint32_t type;
    uint64_t offset;
    int assigned;
} parsed_tensor;

void skip_gguf_value(FILE *f, uint32_t val_type) {
    switch (val_type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            fseeko(f, 1, SEEK_CUR);
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            fseeko(f, 2, SEEK_CUR);
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            fseeko(f, 4, SEEK_CUR);
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            fseeko(f, 8, SEEK_CUR);
            break;
        case GGUF_TYPE_STRING: {
            uint64_t len;
            if (fread(&len, 8, 1, f) != 1) return;
            fseeko(f, len, SEEK_CUR);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t len;
            if (fread(&elem_type, 4, 1, f) != 1) return;
            if (fread(&len, 8, 1, f) != 1) return;
            for (uint64_t i = 0; i < len; i++) {
                skip_gguf_value(f, elem_type);
            }
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown value type %u in skip_gguf_value\n", val_type);
            exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <gguf_file>\n", argv[0]);
        return 1;
    }

    const char *filepath = argv[1];
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        perror("Error opening file");
        return 1;
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
        return 1;
    }

    if (strncmp(header.magic, "GGUF", 4) != 0) {
        fprintf(stderr, "Error: Invalid magic '%.4s' (expected 'GGUF').\n", header.magic);
        fclose(f);
        return 1;
    }

    // Skip metadata key-value pairs
    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read metadata key length.\n");
            fclose(f);
            return 1;
        }
        fseeko(f, key_len, SEEK_CUR);

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read metadata value type.\n");
            fclose(f);
            return 1;
        }
        skip_gguf_value(f, val_type);
    }

    // Read tensor infos
    parsed_tensor *tensors = malloc(header.tensor_count * sizeof(parsed_tensor));
    if (!tensors && header.tensor_count > 0) {
        fprintf(stderr, "Error: Failed to allocate memory for tensor infos.\n");
        fclose(f);
        return 1;
    }

    for (uint64_t i = 0; i < header.tensor_count; i++) {
        uint64_t name_len;
        if (fread(&name_len, 8, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " name length.\n", i);
            fclose(f);
            return 1;
        }
        tensors[i].name = malloc(name_len + 1);
        if (fread(tensors[i].name, 1, name_len, f) != name_len) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " name.\n", i);
            fclose(f);
            return 1;
        }
        tensors[i].name[name_len] = '\0';

        uint32_t n_dimensions;
        if (fread(&n_dimensions, 4, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " n_dimensions.\n", i);
            fclose(f);
            return 1;
        }
        fseeko(f, 8 * n_dimensions, SEEK_CUR);

        if (fread(&tensors[i].type, 4, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " type.\n", i);
            fclose(f);
            return 1;
        }

        if (fread(&tensors[i].offset, 8, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " offset.\n", i);
            fclose(f);
            return 1;
        }

        tensors[i].assigned = 0;
    }
    fclose(f);

    printf("Successfully loaded %" PRIu64 " tensors from %s\n", header.tensor_count, filepath);

    // 1. Assign global tensors
    int global_assigned = 0;
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        for (int g = 0; g < GLOBAL_TENSORS_COUNT; g++) {
            if (strcmp(tensors[i].name, GLOBAL_TENSORS[g]) == 0) {
                tensors[i].assigned = 1;
                global_assigned++;
                break;
            }
        }
    }
    printf("Assigned %d global tensors.\n", global_assigned);

    // 2. Classify block layers
    layer_type block_types[65];
    for (int b = 0; b < 65; b++) {
        block_types[b] = LAYER_TYPE_UNKNOWN;
    }

    for (int b = 0; b < 65; b++) {
        // Detect layer type for block b by scanning tensors
        int has_ssm_a = 0;
        int has_attn_q = 0;
        int has_nextn_eh = 0;

        for (uint64_t i = 0; i < header.tensor_count; i++) {
            int block_idx = -1;
            char suffix[256] = "";
            if (sscanf(tensors[i].name, "blk.%d.%255s", &block_idx, suffix) == 2) {
                if (block_idx == b) {
                    if (strcmp(suffix, "ssm_a") == 0) {
                        has_ssm_a = 1;
                    }
                    if (strcmp(suffix, "attn_q.weight") == 0) {
                        has_attn_q = 1;
                    }
                    if (strcmp(suffix, "nextn.eh_proj.weight") == 0) {
                        has_nextn_eh = 1;
                    }
                }
            }
        }

        if (b == 64) {
            if (has_nextn_eh) {
                block_types[b] = LAYER_TYPE_NEXTN;
            } else {
                fprintf(stderr, "Error: Block 64 does not contain NextN head tensors!\n");
                exit(1);
            }
        } else {
            if (has_ssm_a) {
                block_types[b] = LAYER_TYPE_SSM;
            } else if (has_attn_q) {
                block_types[b] = LAYER_TYPE_ATTENTION;
            } else {
                fprintf(stderr, "Error: Block %d could not be classified (no ssm_a or attn_q.weight found)!\n", b);
                exit(1);
            }
        }
    }

    // 3. Verify all expected tensors for each block are present and map them
    int ssm_count = 0;
    int attn_count = 0;
    int nextn_count = 0;

    for (int b = 0; b < 65; b++) {
        layer_type type = block_types[b];
        printf("Block %2d classified as: %s\n", b, layer_type_name(type));

        const char **expected_suffixes = NULL;
        int expected_count = 0;

        if (type == LAYER_TYPE_SSM) {
            expected_suffixes = SSM_TENSOR_SUFFIXES;
            expected_count = SSM_TENSOR_SUFFIXES_COUNT;
            ssm_count++;
        } else if (type == LAYER_TYPE_ATTENTION) {
            expected_suffixes = ATTN_TENSOR_SUFFIXES;
            expected_count = ATTN_TENSOR_SUFFIXES_COUNT;
            attn_count++;
        } else if (type == LAYER_TYPE_NEXTN) {
            expected_suffixes = NEXTN_TENSOR_SUFFIXES;
            expected_count = NEXTN_TENSOR_SUFFIXES_COUNT;
            nextn_count++;
        }

        for (int s = 0; s < expected_count; s++) {
            // Construct expected full tensor name
            char expected_name[256];
            snprintf(expected_name, sizeof(expected_name), "blk.%d.%s", b, expected_suffixes[s]);

            // Find it in the parsed tensors
            int found = 0;
            for (uint64_t i = 0; i < header.tensor_count; i++) {
                if (strcmp(tensors[i].name, expected_name) == 0) {
                    tensors[i].assigned = 1;
                    found = 1;
                    break;
                }
            }

            if (!found) {
                fprintf(stderr, "Error: Required tensor '%s' is missing in GGUF file!\n", expected_name);
                exit(1);
            }
        }
    }

    // 4. Verify unassigned tensors
    int unassigned_count = 0;
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        if (!tensors[i].assigned) {
            fprintf(stderr, "Error: Tensor '%s' in GGUF is unassigned to any layer!\n", tensors[i].name);
            unassigned_count++;
        }
    }

    if (unassigned_count > 0) {
        fprintf(stderr, "Error: Found %d unassigned tensors!\n", unassigned_count);
        exit(1);
    }

    // 5. Verify layer counts
    printf("\n=== Classification Summary ===\n");
    printf("SSM layers:       %d (Expected: 48)\n", ssm_count);
    printf("Attention layers: %d (Expected: 16)\n", attn_count);
    printf("NextN layers:     %d (Expected: 1)\n", nextn_count);
    printf("Total Tensors:    %" PRIu64 " (Expected: 866)\n", header.tensor_count);

    if (ssm_count != 48 || attn_count != 16 || nextn_count != 1) {
        fprintf(stderr, "Error: Layer count mismatch!\n");
        exit(1);
    }

    if (header.tensor_count != 866) {
        fprintf(stderr, "Error: Tensor count mismatch! Expected 866, got %" PRIu64 "\n", header.tensor_count);
        exit(1);
    }

    // Clean up
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        free(tensors[i].name);
    }
    free(tensors);

    printf("\nSUCCESS: All layer classifications verified and matched perfectly!\n");
    return 0;
}
