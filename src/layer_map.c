#include "layer_map.h"

layer_type get_layer_type(int layer_idx) {
    // Attention layers are index 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
    // SSM layers are all other 48 layers.
    if ((layer_idx - 3) % 4 == 0) {
        return LAYER_TYPE_ATTENTION;
    }
    return LAYER_TYPE_SSM;
}
