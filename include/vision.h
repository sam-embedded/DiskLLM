#ifndef DISKLLM_VISION_H
#define DISKLLM_VISION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "tensor_catalog.h"
#include "model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int image_size;          /* e.g., 224, 336, 448 */
    int patch_size;          /* e.g., 14, 16 */
    int hidden_dim;          /* e.g., 768, 1024, 1152 */
    int num_heads;           /* e.g., 12, 16 */
    int num_layers;          /* e.g., 12, 24, 32 */
    int projection_dim;      /* e.g., 1024, 2048, 2560 (LLM hidden dim) */
    int ffn_dim;             /* e.g., 3072, 4304 */
    float eps;
    float image_mean[3];
    float image_std[3];
    char projector_type[64];
} diskllm_vision_config;

typedef struct {
    char mmproj_path[512];
    tensor_catalog *catalog;
    diskllm_vision_config cfg;
    uint8_t *mmap_base;
    size_t mmap_size;
    int fd;
} diskllm_vision_model;

/* Load image from file to RGB float buffer normalized to [0, 1] or with mean/std */
float *diskllm_image_load_rgb(const char *filepath, int target_w, int target_h, int *out_w, int *out_h, const float mean[3], const float std[3]);
void diskllm_image_free(float *rgb_data);

/* Load Vision Projector model from GGUF */
diskllm_vision_model *diskllm_vision_load(const char *mmproj_path);
void diskllm_vision_free(diskllm_vision_model *vmodel);

/* Run Vision Transformer on image to produce visual patch embeddings for the LLM.
 * Returns a dynamically allocated buffer of size [num_patches * projection_dim] floats.
 * out_num_patches receives the patch count (e.g. 256).
 */
float *diskllm_vision_encode(diskllm_vision_model *vmodel, const float *image_rgb, int img_w, int img_h, int *out_num_patches, int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* DISKLLM_VISION_H */
