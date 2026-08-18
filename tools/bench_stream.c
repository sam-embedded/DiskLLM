#define _POSIX_C_SOURCE 200112L

#include "state.h"
#include "scratch.h"
#include "stream.h"
#include "tensor_catalog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void bench_callback(const uint8_t *buffer, size_t size, uint64_t offset, void *user_data) {
    (void)buffer;
    (void)offset;
    uint64_t *total_bytes = (uint64_t *)user_data;
    *total_bytes += size;
}

int main(int argc, char **argv) {
    const char *model_path = "/home/sam/models/Qwen3.5-0.8B-Q4_K_M.gguf";
    io_mode_t io_mode = IO_MODE_PREAD;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(argv[i], "--io-mode") && i + 1 < argc) {
            const char *m = argv[++i];
            if (!strcmp(m, "direct")) io_mode = IO_MODE_DIRECT;
            else if (!strcmp(m, "iouring") || !strcmp(m, "io_uring")) io_mode = IO_MODE_IOURING;
            else if (!strcmp(m, "mmap")) io_mode = IO_MODE_MMAP;
            else io_mode = IO_MODE_PREAD;
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        }
    }

    printf("=== DiskLLM I/O Streaming Engine Benchmark ===\n");
    printf("Model Path: %s\n", model_path);
    printf("I/O Mode  : %s\n\n", (io_mode == IO_MODE_IOURING ? "io_uring" : (io_mode == IO_MODE_DIRECT ? "O_DIRECT" : "pread")));

    tensor_catalog *cat = load_tensor_catalog(model_path);
    if (!cat) {
        fprintf(stderr, "FAIL: Could not load GGUF catalog for %s\n", model_path);
        return 1;
    }

    const tensor_info *ti = cat->tensors;
    if (!ti) {
        fprintf(stderr, "FAIL: Empty catalog in %s\n", model_path);
        free_tensor_catalog(cat);
        return 1;
    }

    // Benchmark on first large tensor in catalog
    uint64_t test_offset = ti->absolute_offset;
    uint64_t test_size = ti->byte_size;

    printf("Benchmarking streaming read of tensor '%s' (%.2f MB)...\n",
           ti->name, (double)test_size / (1024.0 * 1024.0));

    scratch_buffers *scratch = allocate_scratch_buffers(NULL);
    if (!scratch) {
        fprintf(stderr, "FAIL: Scratch buffers allocation failed.\n");
        free_tensor_catalog(cat);
        return 1;
    }

    stream_context *ctx = init_stream_context_ex(model_path, scratch->stream_buffer, scratch->stream_buffer_size, io_mode);
    if (!ctx) {
        fprintf(stderr, "FAIL: Streaming context initialization failed.\n");
        free_scratch_buffers(scratch);
        free_tensor_catalog(cat);
        return 1;
    }

    uint64_t bytes_received = 0;
    double start_time = get_time_sec();
    int ret = stream_tensor_chunks(ctx, test_offset, test_size, bench_callback, &bytes_received);
    double end_time = get_time_sec();

    if (ret != 0) {
        fprintf(stderr, "FAIL: Error streaming tensor data.\n");
        close_stream_context(ctx);
        free_scratch_buffers(scratch);
        free_tensor_catalog(cat);
        return 1;
    }

    double elapsed = end_time - start_time;
    double throughput = (bytes_received / (1024.0 * 1024.0)) / (elapsed > 0 ? elapsed : 0.000001);

    printf("  Bytes Read : %" PRIu64 "\n", bytes_received);
    printf("  Time Taken : %.4f seconds\n", elapsed);
    printf("  Throughput : %.2f MB/s\n", throughput);

    close_stream_context(ctx);
    free_scratch_buffers(scratch);
    free_tensor_catalog(cat);

    printf("\nSUCCESS: Disk I/O benchmark completed successfully!\n");
    return 0;
}
