#ifndef STREAM_H
#define STREAM_H

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

typedef enum {
    IO_MODE_PREAD = 0,
    IO_MODE_MMAP,
    IO_MODE_DIRECT,
    IO_MODE_IOURING
} io_mode_t;

// Stream context struct
typedef struct {
    int fd;                    // file descriptor of the GGUF file
    io_mode_t mode;            // active I/O mode
    int is_direct;             // 1 if O_DIRECT is active
    int ring_fd;               // io_uring ring file descriptor (-1 if unused)

    // io_uring ring pointers and offsets
    void *sq_mmap;
    void *cq_mmap;
    size_t sq_mmap_size;
    size_t cq_mmap_size;

    void *sqes_ptr;
    void *cqes_ptr;

    uint32_t *sq_khead;
    uint32_t *sq_ktail;
    uint32_t *sq_kring_mask;
    uint32_t *sq_karray;

    uint32_t *cq_khead;
    uint32_t *cq_ktail;
    uint32_t *cq_kring_mask;

    uint8_t *buffer_a;         // first 64 MiB buffer (4KB aligned)
    uint8_t *buffer_b;         // second 64 MiB buffer (4KB aligned)
    size_t chunk_size;         // size of each buffer (64 MiB)
    uint64_t total_bytes_read; // stats: total bytes read from disk
} stream_context;

// Callback signature for chunked reads
typedef void (*stream_chunk_callback)(const uint8_t *buffer, size_t size, uint64_t offset, void *user_data);

// Lifecycle functions
stream_context *init_stream_context_ex(const char *filepath, uint8_t *scratch_stream_buffer, size_t scratch_stream_buffer_size, io_mode_t mode);
stream_context *init_stream_context(const char *filepath, uint8_t *scratch_stream_buffer, size_t scratch_stream_buffer_size);
void close_stream_context(stream_context *ctx);

// Read exact bytes using the active I/O mode (pread, O_DIRECT, or io_uring)
ssize_t stream_read_exact(stream_context *ctx, void *buf, size_t count, off_t offset);

// Stream a tensor's data in chunks
int stream_tensor_chunks(stream_context *ctx, uint64_t file_offset, uint64_t byte_size, stream_chunk_callback callback, void *user_data);

#endif // STREAM_H
