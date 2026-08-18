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
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/io_uring.h>

static inline int sys_io_uring_setup(unsigned entries, struct io_uring_params *p) {
    return (int)syscall(__NR_io_uring_setup, entries, p);
}

static inline int sys_io_uring_enter(int ring_fd, unsigned to_submit, unsigned min_complete, unsigned flags, const void *sig, size_t sigsz) {
    return (int)syscall(__NR_io_uring_enter, ring_fd, to_submit, min_complete, flags, sig, sigsz);
}

static ssize_t exact_pread_standard(int fd, void *buf, size_t count, off_t offset) {
    size_t bytes_read = 0;
    uint8_t *ptr = (uint8_t *)buf;
    while (bytes_read < count) {
        ssize_t ret = pread(fd, ptr + bytes_read, count - bytes_read, offset + bytes_read);
        if (ret < 0) return -1;
        if (ret == 0) break;
        bytes_read += ret;
    }
    return (ssize_t)bytes_read;
}

static ssize_t exact_pread_direct(int fd, void *buf, size_t count, off_t offset) {
    size_t block_size = 4096;
    off_t aligned_offset = (offset / block_size) * block_size;
    size_t delta = (size_t)(offset - aligned_offset);
    size_t aligned_count = ((delta + count + block_size - 1) / block_size) * block_size;

    uint8_t *aligned_buf = NULL;
    if (posix_memalign((void **)&aligned_buf, block_size, aligned_count) != 0 || !aligned_buf) {
        return exact_pread_standard(fd, buf, count, offset);
    }

    ssize_t r = pread(fd, aligned_buf, aligned_count, aligned_offset);
    if (r >= (ssize_t)(delta + count)) {
        memcpy(buf, aligned_buf + delta, count);
        free(aligned_buf);
        return (ssize_t)count;
    }

    free(aligned_buf);
    return exact_pread_standard(fd, buf, count, offset);
}

static int init_io_uring_ring(stream_context *ctx) {
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));

    int ring_fd = sys_io_uring_setup(32, &p);
    if (ring_fd < 0) {
        return -1;
    }

    ctx->ring_fd = ring_fd;
    ctx->sq_mmap_size = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    ctx->cq_mmap_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);

    void *sq_ptr = mmap(NULL, ctx->sq_mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
    if (sq_ptr == MAP_FAILED) {
        close(ring_fd);
        ctx->ring_fd = -1;
        return -1;
    }
    ctx->sq_mmap = sq_ptr;

    void *sqes_ptr = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES);
    if (sqes_ptr == MAP_FAILED) {
        munmap(sq_ptr, ctx->sq_mmap_size);
        close(ring_fd);
        ctx->ring_fd = -1;
        return -1;
    }
    ctx->sqes_ptr = sqes_ptr;

    void *cq_ptr = mmap(NULL, ctx->cq_mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
    if (cq_ptr == MAP_FAILED) {
        munmap(sqes_ptr, p.sq_entries * sizeof(struct io_uring_sqe));
        munmap(sq_ptr, ctx->sq_mmap_size);
        close(ring_fd);
        ctx->ring_fd = -1;
        return -1;
    }
    ctx->cq_mmap = cq_ptr;
    ctx->cqes_ptr = (void *)((uint8_t *)cq_ptr + p.cq_off.cqes);

    ctx->sq_khead = (uint32_t *)((uint8_t *)sq_ptr + p.sq_off.head);
    ctx->sq_ktail = (uint32_t *)((uint8_t *)sq_ptr + p.sq_off.tail);
    ctx->sq_kring_mask = (uint32_t *)((uint8_t *)sq_ptr + p.sq_off.ring_mask);
    ctx->sq_karray = (uint32_t *)((uint8_t *)sq_ptr + p.sq_off.array);

    ctx->cq_khead = (uint32_t *)((uint8_t *)cq_ptr + p.cq_off.head);
    ctx->cq_ktail = (uint32_t *)((uint8_t *)cq_ptr + p.cq_off.tail);
    ctx->cq_kring_mask = (uint32_t *)((uint8_t *)cq_ptr + p.cq_off.ring_mask);

    return 0;
}

stream_context *init_stream_context_ex(const char *filepath, uint8_t *scratch_stream_buffer, size_t scratch_stream_buffer_size, io_mode_t mode) {
    if (!filepath || !scratch_stream_buffer) {
        fprintf(stderr, "Error: Invalid arguments to init_stream_context_ex.\n");
        return NULL;
    }

    size_t expected_size = 128ULL * 1024ULL * 1024ULL;
    if (scratch_stream_buffer_size < expected_size) {
        fprintf(stderr, "Error: Scratch stream buffer size (%zu) is less than expected 128 MiB.\n", scratch_stream_buffer_size);
        return NULL;
    }

    int flags = O_RDONLY;
    int is_direct = (mode == IO_MODE_DIRECT || mode == IO_MODE_IOURING);
    if (is_direct) {
        flags |= O_DIRECT;
    }

    int fd = open(filepath, flags);
    if (fd < 0 && is_direct) {
        // Fallback to standard open without O_DIRECT
        flags &= ~O_DIRECT;
        fd = open(filepath, flags);
        is_direct = 0;
    }

    if (fd < 0) {
        perror("Error: Failed to open model file for streaming");
        return NULL;
    }

#ifdef POSIX_FADV_SEQUENTIAL
    if (!is_direct) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
#endif

    stream_context *ctx = calloc(1, sizeof(stream_context));
    if (!ctx) {
        close(fd);
        return NULL;
    }

    ctx->fd = fd;
    ctx->mode = mode;
    ctx->is_direct = is_direct;
    ctx->ring_fd = -1;
    ctx->chunk_size = expected_size / 2;
    ctx->buffer_a = scratch_stream_buffer;
    ctx->buffer_b = scratch_stream_buffer + ctx->chunk_size;
    ctx->total_bytes_read = 0;

    if (mode == IO_MODE_IOURING) {
        if (init_io_uring_ring(ctx) != 0) {
            fprintf(stderr, "[WARNING] io_uring setup failed. Falling back to %s mode.\n", is_direct ? "O_DIRECT" : "pread");
            ctx->mode = is_direct ? IO_MODE_DIRECT : IO_MODE_PREAD;
        } else {
            fprintf(stderr, "[INFO] io_uring backend initialized successfully.\n");
        }
    } else if (mode == IO_MODE_DIRECT && is_direct) {
        fprintf(stderr, "[INFO] O_DIRECT backend initialized successfully.\n");
    }

    return ctx;
}

stream_context *init_stream_context(const char *filepath, uint8_t *scratch_stream_buffer, size_t scratch_stream_buffer_size) {
    return init_stream_context_ex(filepath, scratch_stream_buffer, scratch_stream_buffer_size, IO_MODE_PREAD);
}

void close_stream_context(stream_context *ctx) {
    if (!ctx) return;
    if (ctx->ring_fd >= 0) {
        if (ctx->sqes_ptr) munmap(ctx->sqes_ptr, 32 * sizeof(struct io_uring_sqe));
        if (ctx->sq_mmap) munmap(ctx->sq_mmap, ctx->sq_mmap_size);
        if (ctx->cq_mmap) munmap(ctx->cq_mmap, ctx->cq_mmap_size);
        close(ctx->ring_fd);
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    free(ctx);
}

ssize_t stream_read_exact(stream_context *ctx, void *buf, size_t count, off_t offset) {
    if (!ctx || ctx->fd < 0) return -1;

    if (ctx->mode == IO_MODE_IOURING && ctx->ring_fd >= 0) {
        unsigned tail = *ctx->sq_ktail;
        unsigned index = tail & *ctx->sq_kring_mask;
        struct io_uring_sqe *sqe = &((struct io_uring_sqe *)ctx->sqes_ptr)[index];

        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = ctx->fd;
        sqe->off = (uint64_t)offset;
        sqe->addr = (uint64_t)buf;
        sqe->len = (uint32_t)count;
        sqe->user_data = 1;

        ctx->sq_karray[index] = index;
        *ctx->sq_ktail = tail + 1;

        int res = sys_io_uring_enter(ctx->ring_fd, 1, 1, IORING_ENTER_GETEVENTS, NULL, 0);
        if (res >= 0) {
            unsigned head = *ctx->cq_khead;
            struct io_uring_cqe *cqe = &((struct io_uring_cqe *)ctx->cqes_ptr)[head & *ctx->cq_kring_mask];
            int ret = cqe->res;
            *ctx->cq_khead = head + 1;
            if (ret >= 0) return (ssize_t)ret;
        }

        // Fallback if SQE entry failed
        if (ctx->is_direct) return exact_pread_direct(ctx->fd, buf, count, offset);
        return exact_pread_standard(ctx->fd, buf, count, offset);
    }

    if (ctx->is_direct) {
        return exact_pread_direct(ctx->fd, buf, count, offset);
    }

    return exact_pread_standard(ctx->fd, buf, count, offset);
}

int stream_tensor_chunks(stream_context *ctx, uint64_t file_offset, uint64_t byte_size, stream_chunk_callback callback, void *user_data) {
    if (!ctx || !callback) {
        fprintf(stderr, "Error: Invalid parameters for stream_tensor_chunks.\n");
        return -1;
    }

    uint64_t bytes_streamed = 0;
    int buffer_toggle = 0;

    while (bytes_streamed < byte_size) {
        uint64_t bytes_to_read = byte_size - bytes_streamed;
        if (bytes_to_read > ctx->chunk_size) {
            bytes_to_read = ctx->chunk_size;
        }

        uint8_t *active_buffer = (buffer_toggle == 0) ? ctx->buffer_a : ctx->buffer_b;

        ssize_t read_bytes = stream_read_exact(ctx, active_buffer, bytes_to_read, (off_t)(file_offset + bytes_streamed));
        if (read_bytes < 0) {
            fprintf(stderr, "Error: Disk read failed at offset %" PRIu64 "\n", file_offset + bytes_streamed);
            return -1;
        }

        ctx->total_bytes_read += read_bytes;
        callback(active_buffer, read_bytes, file_offset + bytes_streamed, user_data);
        bytes_streamed += read_bytes;

        if (read_bytes == 0) break;
        buffer_toggle = 1 - buffer_toggle;
    }

    return 0;
}
