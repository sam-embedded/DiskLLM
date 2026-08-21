INCLUDES = -Iinclude -Iinclude/arch
DEFINES  = -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE

UNAME_M := $(shell uname -m)
ARCH_FLAGS =
ifneq (,$(filter aarch64 arm64 armv8%,$(UNAME_M)))
    ARCH_FLAGS = -march=armv8.2-a+dotprod
endif

ALL_CFLAGS = -std=c11 -Wall -Wextra -O3 -Wno-restrict -Wno-sign-compare $(DEFINES) $(INCLUDES) $(ARCH_FLAGS) -ffast-math $(CFLAGS)

# Link flags
LFLAGS = -lm -lpthread -lvulkan

SRCS_CORE = src/attention.c src/ssm.c src/dequant.c src/rmsnorm.c \
            src/swiglu.c src/matvec.c src/state.c src/scratch.c \
            src/tensor_catalog.c src/layer_map.c src/sampler.c src/stream.c \
            src/tokenizer.c src/speculative.c src/vulkan_backend.c \
            src/core/model.c src/core/context.c src/core/tokenizer_api.c src/core/sampler_api.c \
            src/core/vision.c \
            src/core/arch/registry.c src/core/arch/qwen35.c src/core/arch/qwen2.c src/core/arch/llama.c \
            src/core/arch/phi3.c src/core/arch/gemma.c src/core/arch/gemma2.c src/core/arch/gemma3.c src/core/arch/gemma4.c src/core/arch/mistral.c

all: shaders gguf_dump tensor_map probe_model bench_stream test_dequant test_kernels test_attention test_ssm test_tokenizer diskllm-cli diskllm-server diskllm

shaders: shaders/matvec_q8_0.spv shaders/matvec_q4_k.spv shaders/matvec_f32.spv

shaders/%.spv: shaders/%.comp
	glslc $< -o $@

gguf_dump: tools/gguf_dump.c
	$(CC) $(ALL_CFLAGS) $< -o $@ $(LFLAGS)

tensor_map: tools/tensor_map.c
	$(CC) $(ALL_CFLAGS) $< -o $@ $(LFLAGS)

probe_model: tools/probe_model.c src/state.c src/scratch.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

bench_stream: tools/bench_stream.c src/state.c src/scratch.c src/stream.c src/tensor_catalog.c src/layer_map.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_dequant: tools/test_dequant.c src/dequant.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_kernels: tools/test_kernels.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/vulkan_backend.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_attention: tools/test_attention.c src/attention.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/state.c src/scratch.c src/vulkan_backend.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_ssm: tools/test_ssm.c src/ssm.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/state.c src/scratch.c src/vulkan_backend.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_tokenizer: tools/test_tokenizer.c src/tokenizer.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_vision: tools/test_vision.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

diskllm-cli: src/cli/main.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

diskllm-server: src/server/main.c src/server/cJSON.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

diskllm: diskllm-cli
	cp diskllm-cli diskllm

clean:
	rm -f gguf_dump tensor_map probe_model bench_stream \
	      test_dequant test_kernels test_attention test_ssm test_tokenizer test_speculative \
	      test_vision test_fast_vision test_web_image test_qwen35_vision test_tok_ids test_direct_prompt \
	      diskllm diskllm-cli diskllm-server

test_tok_ids: tools/test_tok_ids.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_fast_vision: tools/test_fast_vision.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_direct_prompt: tools/test_direct_prompt.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_web_image: tools/test_web_image.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_qwen35_vision: tools/test_qwen35_vision.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)
