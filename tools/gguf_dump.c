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

// Forward declarations
void print_gguf_value(FILE *f, uint32_t val_type, uint64_t *alignment_out);
void skip_gguf_value(FILE *f, uint32_t val_type);

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

void print_gguf_value(FILE *f, uint32_t val_type, uint64_t *alignment_out) {
    switch (val_type) {
        case GGUF_TYPE_UINT8: {
            uint8_t v;
            if (fread(&v, 1, 1, f) != 1) return;
            printf("%u", v);
            break;
        }
        case GGUF_TYPE_INT8: {
            int8_t v;
            if (fread(&v, 1, 1, f) != 1) return;
            printf("%d", v);
            break;
        }
        case GGUF_TYPE_UINT16: {
            uint16_t v;
            if (fread(&v, 2, 1, f) != 1) return;
            printf("%u", v);
            break;
        }
        case GGUF_TYPE_INT16: {
            int16_t v;
            if (fread(&v, 2, 1, f) != 1) return;
            printf("%d", v);
            break;
        }
        case GGUF_TYPE_UINT32: {
            uint32_t v;
            if (fread(&v, 4, 1, f) != 1) return;
            printf("%u", v);
            if (alignment_out) {
                *alignment_out = v;
            }
            break;
        }
        case GGUF_TYPE_INT32: {
            int32_t v;
            if (fread(&v, 4, 1, f) != 1) return;
            printf("%d", v);
            break;
        }
        case GGUF_TYPE_FLOAT32: {
            float v;
            if (fread(&v, 4, 1, f) != 1) return;
            printf("%g", v);
            break;
        }
        case GGUF_TYPE_BOOL: {
            uint8_t v;
            if (fread(&v, 1, 1, f) != 1) return;
            printf("%s", v ? "true" : "false");
            break;
        }
        case GGUF_TYPE_STRING: {
            uint64_t len;
            if (fread(&len, 8, 1, f) != 1) return;
            char *s = malloc(len + 1);
            if (!s) {
                fprintf(stderr, "Error: Out of memory reading string of size %" PRIu64 "\n", len);
                exit(1);
            }
            if (fread(s, 1, len, f) != len) {
                free(s);
                return;
            }
            s[len] = '\0';
            printf("\"%s\"", s);
            free(s);
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t len;
            if (fread(&elem_type, 4, 1, f) != 1) return;
            if (fread(&len, 8, 1, f) != 1) return;
            printf("[");
            for (uint64_t i = 0; i < len; i++) {
                if (i > 0) printf(", ");
                if (i < 5) {
                    print_gguf_value(f, elem_type, NULL);
                } else {
                    skip_gguf_value(f, elem_type);
                    if (i == 5) {
                        printf("... (+%" PRIu64 " more)", len - 5);
                    }
                }
            }
            printf("]");
            break;
        }
        case GGUF_TYPE_UINT64: {
            uint64_t v;
            if (fread(&v, 8, 1, f) != 1) return;
            printf("%" PRIu64, v);
            if (alignment_out) {
                *alignment_out = v;
            }
            break;
        }
        case GGUF_TYPE_INT64: {
            int64_t v;
            if (fread(&v, 8, 1, f) != 1) return;
            printf("%" PRId64, v);
            break;
        }
        case GGUF_TYPE_FLOAT64: {
            double v;
            if (fread(&v, 8, 1, f) != 1) return;
            printf("%g", v);
            break;
        }
        default:
            fprintf(stderr, "Error: Unknown value type %u\n", val_type);
            exit(1);
    }
}

const char *ggml_type_name(uint32_t type) {
    switch (type) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 6: return "Q5_0";
        case 7: return "Q5_1";
        case 8: return "Q8_0";
        case 9: return "Q8_1";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "I8";
        case 17: return "I16";
        case 18: return "I32";
        default: return "UNKNOWN";
    }
}

uint64_t ggml_type_size(uint32_t type, uint64_t nelements) {
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

    printf("GGUF Magic: %.4s\n", header.magic);
    printf("GGUF Version: %u\n", header.version);
    printf("Tensor Count: %" PRIu64 "\n", header.tensor_count);
    printf("Metadata KV Count: %" PRIu64 "\n", header.metadata_kv_count);

    uint64_t alignment = 32; // Default GGUF alignment

    printf("\n=== Metadata Key-Values ===\n");
    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        uint64_t key_len;
        if (fread(&key_len, 8, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read metadata key length.\n");
            fclose(f);
            return 1;
        }
        char *key = malloc(key_len + 1);
        if (!key) {
            fprintf(stderr, "Error: Out of memory allocating key of size %" PRIu64 "\n", key_len);
            fclose(f);
            return 1;
        }
        if (fread(key, 1, key_len, f) != key_len) {
            fprintf(stderr, "Error: Failed to read metadata key.\n");
            free(key);
            fclose(f);
            return 1;
        }
        key[key_len] = '\0';

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read metadata value type for key '%s'.\n", key);
            free(key);
            fclose(f);
            return 1;
        }

        printf("%s (type %u): ", key, val_type);
        uint64_t parsed_alignment = 0;
        if (strcmp(key, "general.alignment") == 0) {
            print_gguf_value(f, val_type, &parsed_alignment);
            if (parsed_alignment != 0) {
                alignment = parsed_alignment;
            }
        } else {
            print_gguf_value(f, val_type, NULL);
        }
        printf("\n");
        free(key);
    }

    printf("\nResolved Alignment: %" PRIu64 " bytes\n", alignment);

    // Now read tensor infos
    printf("\n=== Tensor Infos ===\n");
    printf("%-5s %-60s %-20s %-10s %-18s %-18s\n", "Index", "Name", "Dimensions", "Type", "Byte Size", "Absolute Offset");
    printf("---------------------------------------------------------------------------------------------------------------------------------------\n");

    typedef struct {
        char *name;
        uint32_t n_dimensions;
        uint64_t dimensions[8];
        uint32_t type;
        uint64_t offset;
    } temp_tensor_info;

    temp_tensor_info *tensors = malloc(header.tensor_count * sizeof(temp_tensor_info));
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
        if (!tensors[i].name) {
            fprintf(stderr, "Error: Out of memory allocating name for tensor %" PRIu64 "\n", i);
            fclose(f);
            return 1;
        }
        if (fread(tensors[i].name, 1, name_len, f) != name_len) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " name.\n", i);
            fclose(f);
            return 1;
        }
        tensors[i].name[name_len] = '\0';

        if (fread(&tensors[i].n_dimensions, 4, 1, f) != 1) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " n_dimensions.\n", i);
            fclose(f);
            return 1;
        }

        if (tensors[i].n_dimensions > 8) {
            fprintf(stderr, "Error: Tensor %s has %u dimensions, which exceeds the max of 8.\n", tensors[i].name, tensors[i].n_dimensions);
            fclose(f);
            return 1;
        }

        if (fread(tensors[i].dimensions, 8, tensors[i].n_dimensions, f) != tensors[i].n_dimensions) {
            fprintf(stderr, "Error: Failed to read tensor %" PRIu64 " dimensions.\n", i);
            fclose(f);
            return 1;
        }

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
    }

    off_t post_tensor_infos_offset = ftello(f);
    printf("File offset post-tensor-infos: %lld\n", (long long)post_tensor_infos_offset);

    // Calculate start of the tensor data block
    uint64_t tensor_data_block_offset = ((post_tensor_infos_offset + alignment - 1) / alignment) * alignment;
    printf("Tensor data block starts at aligned offset: %" PRIu64 "\n\n", tensor_data_block_offset);

    for (uint64_t i = 0; i < header.tensor_count; i++) {
        // Compute total elements
        uint64_t elements = 1;
        if (tensors[i].n_dimensions == 0) {
            elements = 0;
        } else {
            for (uint32_t d = 0; d < tensors[i].n_dimensions; d++) {
                elements *= tensors[i].dimensions[d];
            }
        }

        uint64_t byte_size = ggml_type_size(tensors[i].type, elements);
        uint64_t absolute_offset = tensor_data_block_offset + tensors[i].offset;

        // Print dimensions list
        char dims_str[64] = "";
        if (tensors[i].n_dimensions == 0) {
            strcpy(dims_str, "[]");
        } else {
            int pos = 0;
            pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, "[");
            for (uint32_t d = 0; d < tensors[i].n_dimensions; d++) {
                pos += snprintf(dims_str + pos, sizeof(dims_str) - pos, "%" PRIu64 "%s", 
                                tensors[i].dimensions[d], (d == tensors[i].n_dimensions - 1) ? "" : ", ");
            }
            snprintf(dims_str + pos, sizeof(dims_str) - pos, "]");
        }

        printf("%-5" PRIu64 " %-60s %-20s %-10s %-18" PRIu64 " %-18" PRIu64 "\n",
               i,
               tensors[i].name,
               dims_str,
               ggml_type_name(tensors[i].type),
               byte_size,
               absolute_offset);
    }

    // Clean up
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        free(tensors[i].name);
    }
    free(tensors);
    fclose(f);

    printf("\n=== Summary ===\n");
    printf("Total printed tensors: %" PRIu64 "\n", header.tensor_count);
    return 0;
}
