#ifndef LAYER_MAP_H
#define LAYER_MAP_H

typedef enum {
    LAYER_TYPE_SSM = 0,
    LAYER_TYPE_ATTENTION = 1
} layer_type;

layer_type get_layer_type(int layer_idx);

#endif // LAYER_MAP_H
