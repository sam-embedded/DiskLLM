#define _GNU_SOURCE
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif

#include "stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/stat.h>

// Exact read loop using pread
static ssize_t exact_pread(int fd, void *buf, size_t count, off_t offset) {
    size_t bytes_read = 0;
    uint8_t *ptr = (uint8_t *)buf;
    while (bytes_read < count) {
        ssize_t ret = pread(fd, ptr + bytes_read, count - bytes_read, offset + bytes_read);
        if (ret < 0) {
            return -1; // read error
        }
        if (ret == 0) {
            break; // EOF
        }
        bytes_read += ret;
    }
    return bytes_read;
}

stream_context *init_stream_context(const char *filepath, uint8_t *scratch_stream_buffer, size_t scratch_stream_buffer_size) {
    if (!filepath || !scratch_stream_buffer) {
        fprintf(stderr, "Error: Invalid arguments to init_stream_context.\n");
        return NULL;
    }

    // GGUF requires 128 MiB buffer size to split into two 64 MiB buffers
    size_t expected_size = 128ULL * 1024ULL * 1024ULL;
    if (scratch_stream_buffer_size < expected_size) {
        fprintf(stderr, "Error: Scratch stream buffer size (%zu) is less than expected 128 MiB.\n", scratch_stream_buffer_size);
        return NULL;
    }

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) {
        perror("Error: Failed to open model file for streaming");
        return NULL;
    }

    // Try to advise the OS about sequential access (optional but helpful for streaming)
#ifdef POSIX_FADV_SEQUENTIAL
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

    stream_context *ctx = calloc(1, sizeof(stream_context));
    if (!ctx) {
        fprintf(stderr, "Error: Failed to allocate stream_context wrapper.\n");
        close(fd);
        return NULL;
    }

    ctx->fd = fd;
    ctx->chunk_size = expected_size / 2; // 64 MiB
    ctx->buffer_a = scratch_stream_buffer;
    ctx->buffer_b = scratch_stream_buffer + ctx->chunk_size;
    ctx->total_bytes_read = 0;

    return ctx;
}

void close_stream_context(stream_context *ctx) {
    if (!ctx) return;
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    free(ctx);
}

int stream_tensor_chunks(stream_context *ctx, uint64_t file_offset, uint64_t byte_size, stream_chunk_callback callback, void *user_data) {
    if (!ctx || !callback) {
        fprintf(stderr, "Error: Invalid parameters for stream_tensor_chunks.\n");
        return -1;
    }

    uint64_t bytes_streamed = 0;
    int buffer_toggle = 0; // 0 for buffer_a, 1 for buffer_b

    while (bytes_streamed < byte_size) {
        uint64_t bytes_to_read = byte_size - bytes_streamed;
        if (bytes_to_read > ctx->chunk_size) {
            bytes_to_read = ctx->chunk_size;
        }

        uint8_t *active_buffer = (buffer_toggle == 0) ? ctx->buffer_a : ctx->buffer_b;

        // Perform exact pread
        ssize_t read_bytes = exact_pread(ctx->fd, active_buffer, bytes_to_read, (off_t)(file_offset + bytes_streamed));
        if (read_bytes < 0) {
            fprintf(stderr, "Error: Disk read failed at offset %" PRIu64 "\n", file_offset + bytes_streamed);
            return -1;
        }
        if ((size_t)read_bytes < bytes_to_read) {
            fprintf(stderr, "Warning: Short read at offset %" PRIu64 " (requested %" PRIu64 ", got %zd)\n",
                    file_offset + bytes_streamed, bytes_to_read, read_bytes);
        }

        ctx->total_bytes_read += read_bytes;

        // Invoke the callback for the processed chunk
        callback(active_buffer, read_bytes, file_offset + bytes_streamed, user_data);

        bytes_streamed += read_bytes;
        
        if (read_bytes == 0) {
            break; // reached EOF
        }

        // Toggle buffer for the next chunk
        buffer_toggle = 1 - buffer_toggle;
    }

    return 0;
}
