#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family_index;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline_q8_0;
    VkPipeline pipeline_q4_k;
    VkPipeline pipeline_f32;

    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;

    // Host-visible mapped buffers
    VkBuffer buf_x;
    VkDeviceMemory mem_x;
    float *map_x;
    size_t cap_x_bytes;

    VkBuffer buf_w;
    VkDeviceMemory mem_w;
    void *map_w;
    size_t cap_w_bytes;

    VkBuffer buf_y;
    VkDeviceMemory mem_y;
    float *map_y;
    size_t cap_y_bytes;
} vulkan_context;

// Global Vulkan context pointer (NULL if GPU mode disabled or init failed)
extern vulkan_context *g_vulkan_ctx;

vulkan_context *vulkan_backend_init(void);
void vulkan_backend_free(vulkan_context *ctx);

int vulkan_matvec(
    vulkan_context *ctx,
    float *out,
    const void *w,
    const float *x,
    int in_features,
    int out_features,
    int type
);

#endif // VULKAN_BACKEND_H
