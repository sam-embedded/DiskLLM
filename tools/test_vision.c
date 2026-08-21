#include "vision.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *mmproj_path = (argc > 1) ? argv[1] : "/home/sam/models/gemma-4-E4B-it-mmproj.gguf";
    const char *image_path = (argc > 2) ? argv[2] : "test.jpeg";

    printf("=== Testing DiskLLM Vision Pipeline ===\n");
    printf("Loading Vision Model: %s\n", mmproj_path);
    diskllm_vision_model *vm = diskllm_vision_load(mmproj_path);
    if (!vm) {
        printf("Failed to load vision model\n");
        return 1;
    }

    printf("Loading Image: %s\n", image_path);
    int img_w = 0, img_h = 0;
    float *rgb = diskllm_image_load_rgb(image_path, vm->cfg.image_size, vm->cfg.image_size, &img_w, &img_h, vm->cfg.image_mean, vm->cfg.image_std);
    if (!rgb) {
        printf("Failed to load image\n");
        diskllm_vision_free(vm);
        return 1;
    }
    printf("Image decoded and normalized: %dx%d (channels=3)\n", img_w, img_h);

    int n_patches = 0;
    printf("Running ViT Forward Pass...\n");
    float *embeddings = diskllm_vision_encode(vm, rgb, img_w, img_h, &n_patches, 4);
    if (!embeddings) {
        printf("Failed to compute vision embeddings\n");
        diskllm_image_free(rgb);
        diskllm_vision_free(vm);
        return 1;
    }

    printf("[SUCCESS] Generated %d visual patch embeddings (dimension: %d)\n", n_patches, vm->cfg.projection_dim);
    printf("First embedding preview [0..4]: %f, %f, %f, %f, %f\n",
           embeddings[0], embeddings[1], embeddings[2], embeddings[3], embeddings[4]);

    free(embeddings);
    diskllm_image_free(rgb);
    diskllm_vision_free(vm);
    return 0;
}
