#ifndef LAYER_MAP_H
#define LAYER_MAP_H

#include "tensor_catalog.h"

typedef enum {
    LAYER_TYPE_SSM = 0,
    LAYER_TYPE_ATTENTION = 1,
    LAYER_TYPE_UNKNOWN = 2
} layer_type;

layer_type classify_layer_type(const tensor_catalog *cat, int block_idx);

#endif // LAYER_MAP_H
