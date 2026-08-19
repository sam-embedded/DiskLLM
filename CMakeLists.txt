cmake_minimum_required(VERSION 3.10)
project(DiskLLM C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE)

include_directories(include include/arch)

find_package(Threads REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3 -ffast-math")

# ARM NEON detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv8")
    add_compile_options(-march=armv8.2-a+dotprod)
endif()

set(SRCS_CORE
    src/attention.c
    src/ssm.c
    src/dequant.c
    src/rmsnorm.c
    src/swiglu.c
    src/matvec.c
    src/state.c
    src/scratch.c
    src/tensor_catalog.c
    src/layer_map.c
    src/sampler.c
    src/stream.c
    src/tokenizer.c
    src/speculative.c
    src/vulkan_backend.c
    src/core/model.c
    src/core/context.c
    src/core/tokenizer_api.c
    src/core/sampler_api.c
    src/core/arch/registry.c
    src/core/arch/qwen.c
    src/core/arch/llama.c
    src/core/arch/phi3.c
    src/core/arch/gemma.c
    src/core/arch/mistral.c
)

# Core Library
add_library(diskllm_core STATIC ${SRCS_CORE})
target_link_libraries(diskllm_core PRIVATE m Threads::Threads)

# CLI Executable
add_executable(diskllm-cli src/cli/main.c)
target_link_libraries(diskllm-cli PRIVATE diskllm_core m Threads::Threads)

# Main alias executable for backwards compatibility
add_executable(diskllm src/cli/main.c)
target_link_libraries(diskllm PRIVATE diskllm_core m Threads::Threads)

# OpenAI HTTP Server Executable
add_executable(diskllm-server src/server/main.c src/server/cJSON.c)
target_link_libraries(diskllm-server PRIVATE diskllm_core m Threads::Threads)

# Tools & Benchmarks
add_executable(gguf_dump tools/gguf_dump.c)
target_link_libraries(gguf_dump PRIVATE m Threads::Threads)

add_executable(tensor_map tools/tensor_map.c)
target_link_libraries(tensor_map PRIVATE m Threads::Threads)

add_executable(probe_model tools/probe_model.c src/state.c src/scratch.c)
target_link_libraries(probe_model PRIVATE m Threads::Threads)

add_executable(bench_stream tools/bench_stream.c src/state.c src/scratch.c src/stream.c)
target_link_libraries(bench_stream PRIVATE m Threads::Threads)

# Tests
add_executable(test_dequant tools/test_dequant.c src/dequant.c)
target_link_libraries(test_dequant PRIVATE m Threads::Threads)

add_executable(test_kernels tools/test_kernels.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c)
target_link_libraries(test_kernels PRIVATE m Threads::Threads)

add_executable(test_attention tools/test_attention.c src/attention.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/state.c src/scratch.c)
target_link_libraries(test_attention PRIVATE m Threads::Threads)

add_executable(test_ssm tools/test_ssm.c src/ssm.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/state.c src/scratch.c)
target_link_libraries(test_ssm PRIVATE m Threads::Threads)

add_executable(test_tokenizer tools/test_tokenizer.c src/tokenizer.c)
target_link_libraries(test_tokenizer PRIVATE m Threads::Threads)

# Install rule
install(TARGETS diskllm-cli diskllm-server diskllm DESTINATION bin)
