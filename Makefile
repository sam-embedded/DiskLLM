CC = cc
CFLAGS = -std=c11 -Wall -Wextra -O2 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE

all: gguf_dump

gguf_dump: tools/gguf_dump.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f gguf_dump
