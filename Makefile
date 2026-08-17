CC = cc
CFLAGS = -std=c11 -Wall -Wextra -O2 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -Iinclude

all: gguf_dump tensor_map probe_model bench_stream test_dequant

gguf_dump: tools/gguf_dump.c
	$(CC) $(CFLAGS) $< -o $@

tensor_map: tools/tensor_map.c
	$(CC) $(CFLAGS) $< -o $@

probe_model: tools/probe_model.c src/state.c src/scratch.c
	$(CC) $(CFLAGS) $^ -o $@

bench_stream: tools/bench_stream.c src/state.c src/scratch.c src/stream.c
	$(CC) $(CFLAGS) $^ -o $@

test_dequant: tools/test_dequant.c src/dequant.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

clean:
	rm -f gguf_dump tensor_map probe_model bench_stream test_dequant
