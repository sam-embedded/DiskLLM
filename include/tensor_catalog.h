#ifndef TENSOR_CATALOG_H
#define TENSOR_CATALOG_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    char name[128];
    uint32_t type;
    uint32_t n_dims;
    uint64_t dims[4];
    uint64_t absolute_offset;
    uint64_t byte_size;
} tensor_info;

typedef struct {
    tensor_info *tensors;
    int count;
} tensor_catalog;

tensor_catalog *load_tensor_catalog(const char *filepath);
void free_tensor_catalog(tensor_catalog *catalog);
const tensor_info *find_tensor(const tensor_catalog *catalog, const char *name);

int lookup_token_by_id(const char *filepath, uint32_t target_id, char *out_str, size_t max_len);
int get_metadata_uint32(const char *filepath, const char *key, uint32_t *out_val);
int search_token(const char *filepath, const char *query);

#endif // TENSOR_CATALOG_H
