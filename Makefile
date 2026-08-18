INCLUDES = -Iinclude
DEFINES  = -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE

UNAME_M := $(shell uname -m)
ARCH_FLAGS =
ifneq (,$(filter aarch64 arm64 armv8%,$(UNAME_M)))
    ARCH_FLAGS = -march=armv8.2-a+dotprod
endif

ALL_CFLAGS = -std=c11 -Wall -Wextra -O3 $(DEFINES) $(INCLUDES) $(ARCH_FLAGS) -ffast-math $(CFLAGS)

# Link flags
LFLAGS = -lm -lpthread

SRCS_CORE = src/attention.c src/ssm.c src/dequant.c src/rmsnorm.c \
            src/swiglu.c src/matvec.c src/state.c src/scratch.c \
            src/tensor_catalog.c src/layer_map.c src/sampler.c src/stream.c \
            src/tokenizer.c src/speculative.c

all: gguf_dump tensor_map probe_model bench_stream test_dequant test_kernels test_attention test_ssm test_tokenizer diskllm

gguf_dump: tools/gguf_dump.c
	$(CC) $(ALL_CFLAGS) $< -o $@ $(LFLAGS)

tensor_map: tools/tensor_map.c
	$(CC) $(ALL_CFLAGS) $< -o $@ $(LFLAGS)

probe_model: tools/probe_model.c src/state.c src/scratch.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

bench_stream: tools/bench_stream.c src/state.c src/scratch.c src/stream.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_dequant: tools/test_dequant.c src/dequant.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_kernels: tools/test_kernels.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_attention: tools/test_attention.c src/attention.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/state.c src/scratch.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_ssm: tools/test_ssm.c src/ssm.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/state.c src/scratch.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_tokenizer: tools/test_tokenizer.c src/tokenizer.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

test_speculative: tools/test_speculative.c src/speculative.c src/tensor_catalog.c src/layer_map.c src/dequant.c src/rmsnorm.c src/swiglu.c src/matvec.c src/scratch.c
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

diskllm: main/diskllm.c $(SRCS_CORE)
	$(CC) $(ALL_CFLAGS) $^ -o $@ $(LFLAGS)

clean:
	rm -f gguf_dump tensor_map probe_model bench_stream \
	      test_dequant test_kernels test_attention test_ssm test_tokenizer test_speculative diskllm

