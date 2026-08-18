#include "tensor_catalog.h"
#include "layer_map.h"
#include <stdio.h>
#include <string.h>

layer_type classify_layer_type(const tensor_catalog *cat, int block_idx) {
    char nm[256];

    /* 1. Check for SSM specific tensor: blk.N.ssm_a */
    snprintf(nm, sizeof(nm), "blk.%d.ssm_a", block_idx);
    if (find_tensor(cat, nm) != NULL) {
        return LAYER_TYPE_SSM;
    }

    /* 2. Check for separate Q projection: blk.N.attn_q.weight */
    snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", block_idx);
    if (find_tensor(cat, nm) != NULL) {
        return LAYER_TYPE_ATTENTION;
    }

    /* 3. Check for fused attn_qkv.weight: determine if SSM or Attention */
    snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", block_idx);
    if (find_tensor(cat, nm) != NULL) {
        snprintf(nm, sizeof(nm), "blk.%d.ssm_conv1d.weight", block_idx);
        if (find_tensor(cat, nm) != NULL) {
            return LAYER_TYPE_SSM;
        }
        snprintf(nm, sizeof(nm), "blk.%d.ssm_out.weight", block_idx);
        if (find_tensor(cat, nm) != NULL) {
            return LAYER_TYPE_SSM;
        }
        return LAYER_TYPE_ATTENTION;
    }

    /* 4. Check for standard attn_norm or ffn_norm */
    snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight", block_idx);
    if (find_tensor(cat, nm) != NULL) {
        return LAYER_TYPE_ATTENTION;
    }

    return LAYER_TYPE_UNKNOWN;
}
