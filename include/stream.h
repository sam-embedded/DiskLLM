#ifndef STREAM_H
#define STREAM_H

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

// Stream context struct
typedef struct {
    int fd;                   // file descriptor of the GGUF file
    uint8_t *buffer_a;        // first 64 MiB buffer
    uint8_t *buffer_b;        // second 64 MiB buffer
    size_t chunk_size;        // size of each buffer (64 MiB)
    uint64_t total_bytes_read;// stats: total bytes read from disk
} stream_context;

// Callback signature for chunked reads
// - buffer: pointer to the buffer containing the chunk data
// - size: size of the chunk in bytes
// - offset: file offset of the chunk
// - user_data: user-defined context pointer passed through the streaming call
typedef void (*stream_chunk_callback)(const uint8_t *buffer, size_t size, uint64_t offset, void *user_data);

// Lifecycle functions
stream_context *init_stream_context(const char *filepath, uint8_t *scratch_stream_buffer, size_t scratch_stream_buffer_size);
void close_stream_context(stream_context *ctx);

// Stream a tensor's data in chunks
// - ctx: streaming context
// - file_offset: absolute file offset of the tensor data block
// - byte_size: size of the tensor data in bytes
// - callback: callback function called for each loaded chunk
// - user_data: custom pointer passed to the callback
int stream_tensor_chunks(stream_context *ctx, uint64_t file_offset, uint64_t byte_size, stream_chunk_callback callback, void *user_data);

#endif // STREAM_H
