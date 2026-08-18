#include "vulkan_backend.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

vulkan_context *g_vulkan_ctx = NULL;

typedef struct {
    uint32_t K;
    uint32_t N;
} matvec_push_constants;

static uint32_t find_memory_type(VkPhysicalDevice phy, uint32_t type_filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phy, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return 0xFFFFFFFF;
}

static int create_buffer(
    VkDevice device,
    VkPhysicalDevice phy,
    size_t size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags props,
    VkBuffer *buf,
    VkDeviceMemory *mem
) {
    VkBufferCreateInfo buf_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    if (vkCreateBuffer(device, &buf_info, NULL, buf) != VK_SUCCESS) return -1;

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, *buf, &mem_reqs);

    uint32_t mem_type = find_memory_type(phy, mem_reqs.memoryTypeBits, props);
    if (mem_type == 0xFFFFFFFF) return -1;

    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = mem_type
    };
    if (vkAllocateMemory(device, &alloc_info, NULL, mem) != VK_SUCCESS) return -1;

    vkBindBufferMemory(device, *buf, *mem, 0);
    return 0;
}

static VkShaderModule create_shader_module(VkDevice device, const char *filepath) {
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "[VULKAN] Error: Shader file %s not found\n", filepath);
        return VK_NULL_HANDLE;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || (size % 4 != 0)) {
        fclose(f);
        return VK_NULL_HANDLE;
    }

    uint32_t *code = malloc(size);
    if (!code) { fclose(f); return VK_NULL_HANDLE; }
    if (fread(code, 1, size, f) != (size_t)size) {
        free(code);
        fclose(f);
        return VK_NULL_HANDLE;
    }
    fclose(f);

    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code
    };
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &create_info, NULL, &module) != VK_SUCCESS) {
        fprintf(stderr, "[VULKAN] Error: Failed to create shader module for %s\n", filepath);
    }
    free(code);
    return module;
}

vulkan_context *vulkan_backend_init(void) {
    vulkan_context *ctx = calloc(1, sizeof(vulkan_context));
    if (!ctx) return NULL;

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "DiskLLM-Vulkan",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "DiskLLM",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_1
    };

    VkInstanceCreateInfo inst_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };

    if (vkCreateInstance(&inst_info, NULL, &ctx->instance) != VK_SUCCESS) {
        fprintf(stderr, "[VULKAN] Failed to create Vulkan instance.\n");
        free(ctx);
        return NULL;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);
    if (device_count == 0) {
        fprintf(stderr, "[VULKAN] No Vulkan physical devices found.\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }

    VkPhysicalDevice devices[16];
    if (device_count > 16) device_count = 16;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);

    int selected_idx = 0;
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            selected_idx = i;
            break;
        }
    }
    ctx->physical_device = devices[selected_idx];

    VkPhysicalDeviceProperties sel_props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &sel_props);
    printf("[VULKAN] Selected Device: %s (API v%u.%u.%u)\n",
           sel_props.deviceName,
           VK_VERSION_MAJOR(sel_props.apiVersion),
           VK_VERSION_MINOR(sel_props.apiVersion),
           VK_VERSION_PATCH(sel_props.apiVersion));

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties queue_families[16];
    if (queue_family_count > 16) queue_family_count = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, queue_families);

    uint32_t compute_queue_idx = 0xFFFFFFFF;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_idx = i;
            break;
        }
    }
    if (compute_queue_idx == 0xFFFFFFFF) {
        fprintf(stderr, "[VULKAN] No compute queue family found.\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }
    ctx->compute_queue_family_index = compute_queue_idx;

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = compute_queue_idx,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };

    VkDeviceCreateInfo dev_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info
    };

    if (vkCreateDevice(ctx->physical_device, &dev_info, NULL, &ctx->device) != VK_SUCCESS) {
        fprintf(stderr, "[VULKAN] Failed to create Vulkan logical device.\n");
        vkDestroyInstance(ctx->instance, NULL);
        free(ctx);
        return NULL;
    }
    vkGetDeviceQueue(ctx->device, compute_queue_idx, 0, &ctx->compute_queue);

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = compute_queue_idx
    };
    vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool);

    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    vkAllocateCommandBuffers(ctx->device, &cmd_alloc, &ctx->command_buffer);

    // Descriptor Set Layout: 3 Storage Buffers
    VkDescriptorSetLayoutBinding bindings[3] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT }
    };
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings
    };
    vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->descriptor_set_layout);

    // Push Constants Layout: (K, N) = 8 bytes
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(matvec_push_constants)
    };

    VkPipelineLayoutCreateInfo pipe_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx->descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range
    };
    vkCreatePipelineLayout(ctx->device, &pipe_layout_info, NULL, &ctx->pipeline_layout);

    // Create Shaders & Pipelines
    VkShaderModule sm_q8_0 = create_shader_module(ctx->device, "shaders/matvec_q8_0.spv");
    VkShaderModule sm_q4_k = create_shader_module(ctx->device, "shaders/matvec_q4_k.spv");
    VkShaderModule sm_f32  = create_shader_module(ctx->device, "shaders/matvec_f32.spv");

    if (sm_q8_0 != VK_NULL_HANDLE) {
        VkComputePipelineCreateInfo pipe_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm_q8_0,
                .pName = "main"
            },
            .layout = ctx->pipeline_layout
        };
        vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &ctx->pipeline_q8_0);
        vkDestroyShaderModule(ctx->device, sm_q8_0, NULL);
    }

    if (sm_q4_k != VK_NULL_HANDLE) {
        VkComputePipelineCreateInfo pipe_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm_q4_k,
                .pName = "main"
            },
            .layout = ctx->pipeline_layout
        };
        vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &ctx->pipeline_q4_k);
        vkDestroyShaderModule(ctx->device, sm_q4_k, NULL);
    }

    if (sm_f32 != VK_NULL_HANDLE) {
        VkComputePipelineCreateInfo pipe_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm_f32,
                .pName = "main"
            },
            .layout = ctx->pipeline_layout
        };
        vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipe_info, NULL, &ctx->pipeline_f32);
        vkDestroyShaderModule(ctx->device, sm_f32, NULL);
    }

    // Allocate Descriptor Pool and Descriptor Set
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3
    };
    VkDescriptorPoolCreateInfo pool_create_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size
    };
    vkCreateDescriptorPool(ctx->device, &pool_create_info, NULL, &ctx->descriptor_pool);

    VkDescriptorSetAllocateInfo desc_alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx->descriptor_set_layout
    };
    vkAllocateDescriptorSets(ctx->device, &desc_alloc_info, &ctx->descriptor_set);

    // Initial Buffer Allocations (Host-Visible / Host-Coherent)
    ctx->cap_x_bytes = 128 * 1024;        // 128 KB for input vector
    ctx->cap_y_bytes = 16 * 1024 * 1024;  // 16 MB for output vector
    ctx->cap_w_bytes = 300 * 1024 * 1024; // 300 MB for weights matrix

    VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    create_buffer(ctx->device, ctx->physical_device, ctx->cap_x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, &ctx->buf_x, &ctx->mem_x);
    create_buffer(ctx->device, ctx->physical_device, ctx->cap_w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, &ctx->buf_w, &ctx->mem_w);
    create_buffer(ctx->device, ctx->physical_device, ctx->cap_y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, &ctx->buf_y, &ctx->mem_y);

    vkMapMemory(ctx->device, ctx->mem_x, 0, ctx->cap_x_bytes, 0, (void **)&ctx->map_x);
    vkMapMemory(ctx->device, ctx->mem_w, 0, ctx->cap_w_bytes, 0, &ctx->map_w);
    vkMapMemory(ctx->device, ctx->mem_y, 0, ctx->cap_y_bytes, 0, (void **)&ctx->map_y);

    // Update Descriptor Sets with Buffer Info
    VkDescriptorBufferInfo buf_infos[3] = {
        { .buffer = ctx->buf_x, .offset = 0, .range = VK_WHOLE_SIZE },
        { .buffer = ctx->buf_w, .offset = 0, .range = VK_WHOLE_SIZE },
        { .buffer = ctx->buf_y, .offset = 0, .range = VK_WHOLE_SIZE }
    };
    VkWriteDescriptorSet writes[3] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ctx->descriptor_set, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[0] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ctx->descriptor_set, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[1] },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ctx->descriptor_set, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &buf_infos[2] }
    };
    vkUpdateDescriptorSets(ctx->device, 3, writes, 0, NULL);

    printf("[VULKAN] Engine successfully initialized.\n");
    return ctx;
}

void vulkan_backend_free(vulkan_context *ctx) {
    if (!ctx) return;
    if (ctx->device) {
        vkDeviceWaitIdle(ctx->device);
        if (ctx->map_x) vkUnmapMemory(ctx->device, ctx->mem_x);
        if (ctx->map_w) vkUnmapMemory(ctx->device, ctx->mem_w);
        if (ctx->map_y) vkUnmapMemory(ctx->device, ctx->mem_y);

        if (ctx->buf_x) vkDestroyBuffer(ctx->device, ctx->buf_x, NULL);
        if (ctx->mem_x) vkFreeMemory(ctx->device, ctx->mem_x, NULL);
        if (ctx->buf_w) vkDestroyBuffer(ctx->device, ctx->buf_w, NULL);
        if (ctx->mem_w) vkFreeMemory(ctx->device, ctx->mem_w, NULL);
        if (ctx->buf_y) vkDestroyBuffer(ctx->device, ctx->buf_y, NULL);
        if (ctx->mem_y) vkFreeMemory(ctx->device, ctx->mem_y, NULL);

        if (ctx->pipeline_q8_0) vkDestroyPipeline(ctx->device, ctx->pipeline_q8_0, NULL);
        if (ctx->pipeline_q4_k) vkDestroyPipeline(ctx->device, ctx->pipeline_q4_k, NULL);
        if (ctx->pipeline_f32)  vkDestroyPipeline(ctx->device, ctx->pipeline_f32, NULL);
        if (ctx->pipeline_layout) vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
        if (ctx->descriptor_set_layout) vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptor_set_layout, NULL);
        if (ctx->descriptor_pool) vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
        if (ctx->command_pool) vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);

        vkDestroyDevice(ctx->device, NULL);
    }
    if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
}

int vulkan_matvec(
    vulkan_context *ctx,
    float *out,
    const void *w,
    const float *x,
    int in_features,
    int out_features,
    int type
) {
    if (!ctx) return -1;

    VkPipeline target_pipeline = VK_NULL_HANDLE;
    size_t w_bytes = 0;

    if (type == GGML_TYPE_Q8_0) {
        target_pipeline = ctx->pipeline_q8_0;
        w_bytes = ((size_t)in_features / 32) * 34 * (size_t)out_features;
    } else if (type == GGML_TYPE_Q4_K) {
        target_pipeline = ctx->pipeline_q4_k;
        w_bytes = ((size_t)in_features / 256) * 144 * (size_t)out_features;
    } else if (type == GGML_TYPE_F32) {
        target_pipeline = ctx->pipeline_f32;
        w_bytes = (size_t)in_features * (size_t)out_features * sizeof(float);
    }

    if (target_pipeline == VK_NULL_HANDLE) {
        return -1; // Fallback to CPU for unsupported kernel types
    }

    size_t x_bytes = (size_t)in_features * sizeof(float);
    size_t y_bytes = (size_t)out_features * sizeof(float);

    if (x_bytes > ctx->cap_x_bytes || w_bytes > ctx->cap_w_bytes || y_bytes > ctx->cap_y_bytes) {
        return -1; // Fallback to CPU if memory bounds exceeded
    }

    // 1. Copy data to mapped Vulkan SSBOs
    memcpy(ctx->map_x, x, x_bytes);
    memcpy(ctx->map_w, w, w_bytes);

    // 2. Record Command Buffer
    vkResetCommandBuffer(ctx->command_buffer, 0);
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(ctx->command_buffer, &begin_info);

    vkCmdBindPipeline(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, target_pipeline);
    vkCmdBindDescriptorSets(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layout, 0, 1, &ctx->descriptor_set, 0, NULL);

    matvec_push_constants pc = {
        .K = (uint32_t)in_features,
        .N = (uint32_t)out_features
    };
    vkCmdPushConstants(ctx->command_buffer, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t group_count_x = ((uint32_t)out_features + 63) / 64;
    vkCmdDispatch(ctx->command_buffer, group_count_x, 1, 1);

    vkEndCommandBuffer(ctx->command_buffer);

    // 3. Submit Command Buffer to Compute Queue
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &ctx->command_buffer
    };
    vkQueueSubmit(ctx->compute_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->compute_queue);

    // 4. Copy Output Y back to CPU buffer
    memcpy(out, ctx->map_y, y_bytes);
    return 0;
}
