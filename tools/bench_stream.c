#define _POSIX_C_SOURCE 200112L

#include "state.h"
#include "scratch.h"
#include "stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

// Helper to get monotonic time in seconds
double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Callback function to count bytes read
void bench_callback(const uint8_t *buffer, size_t size, uint64_t offset, void *user_data) {
    // Suppress compiler warnings
    (void)buffer;
    (void)offset;
    uint64_t *total_bytes = (uint64_t *)user_data;
    *total_bytes += size;
}

int main(int argc, char **argv) {
    const char *model_path = "/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf";
    if (argc > 1) {
        model_path = argv[1];
    }

    printf("=== Streaming Engine Benchmark ===\n");
    printf("Model Path: %s\n\n", model_path);

    // 1. Allocate scratch buffers (for the 128 MiB stream buffer)
    printf("Allocating scratch buffers (128 MiB)...\n");
    scratch_buffers *scratch = allocate_scratch_buffers(NULL);
    if (!scratch) {
        fprintf(stderr, "FAIL: Scratch buffers allocation failed.\n");
        return 1;
    }

    // 2. Initialize streaming context
    printf("Initializing stream context...\n");
    stream_context *ctx = init_stream_context(model_path, scratch->stream_buffer, scratch->stream_buffer_size);
    if (!ctx) {
        fprintf(stderr, "FAIL: Streaming context initialization failed.\n");
        free_scratch_buffers(scratch);
        return 1;
    }

    // Benchmark 1: Stream output.weight
    // Offset: 10996704, Size: 1042944000 bytes
    uint64_t output_offset = 10996704;
    uint64_t output_size = 1042944000;
    uint64_t output_bytes_received = 0;

    printf("\nBenchmarking: Streaming output.weight (%.2f MB)...\n", output_size / (1024.0 * 1024.0));
    double start_time = get_time_sec();
    int ret = stream_tensor_chunks(ctx, output_offset, output_size, bench_callback, &output_bytes_received);
    double end_time = get_time_sec();

    if (ret != 0) {
        fprintf(stderr, "FAIL: Error streaming output.weight.\n");
        close_stream_context(ctx);
        free_scratch_buffers(scratch);
        return 1;
    }

    double elapsed = end_time - start_time;
    double throughput = (output_bytes_received / (1024.0 * 1024.0)) / elapsed;
    printf("  Bytes expected: %" PRIu64 "\n", output_size);
    printf("  Bytes read:     %" PRIu64 "\n", output_bytes_received);
    printf("  Time taken:     %.4f seconds\n", elapsed);
    printf("  Throughput:     %.2f MB/s\n", throughput);

    if (output_bytes_received != output_size) {
        fprintf(stderr, "FAIL: Bytes read mismatch for output.weight.\n");
        close_stream_context(ctx);
        free_scratch_buffers(scratch);
        return 1;
    }

    // Benchmark 2: Stream blk.0.ffn_down.weight
    // Offset: 1829845984, Size: 73113600 bytes
    uint64_t ffn_offset = 1829845984;
    uint64_t ffn_size = 73113600;
    uint64_t ffn_bytes_received = 0;

    printf("\nBenchmarking: Streaming blk.0.ffn_down.weight (%.2f MB)...\n", ffn_size / (1024.0 * 1024.0));
    start_time = get_time_sec();
    ret = stream_tensor_chunks(ctx, ffn_offset, ffn_size, bench_callback, &ffn_bytes_received);
    end_time = get_time_sec();

    if (ret != 0) {
        fprintf(stderr, "FAIL: Error streaming blk.0.ffn_down.weight.\n");
        close_stream_context(ctx);
        free_scratch_buffers(scratch);
        return 1;
    }

    elapsed = end_time - start_time;
    throughput = (ffn_bytes_received / (1024.0 * 1024.0)) / elapsed;
    printf("  Bytes expected: %" PRIu64 "\n", ffn_size);
    printf("  Bytes read:     %" PRIu64 "\n", ffn_bytes_received);
    printf("  Time taken:     %.4f seconds\n", elapsed);
    printf("  Throughput:     %.2f MB/s\n", throughput);

    if (ffn_bytes_received != ffn_size) {
        fprintf(stderr, "FAIL: Bytes read mismatch for blk.0.ffn_down.weight.\n");
        close_stream_context(ctx);
        free_scratch_buffers(scratch);
        return 1;
    }

    // Verify stats
    printf("\n=== Summary ===\n");
    printf("Total registered bytes read: %" PRIu64 "\n", ctx->total_bytes_read);
    uint64_t expected_total = output_size + ffn_size;
    if (ctx->total_bytes_read != expected_total) {
        fprintf(stderr, "FAIL: Total bytes read statistics mismatch.\n");
        close_stream_context(ctx);
        free_scratch_buffers(scratch);
        return 1;
    }

    // Clean up
    close_stream_context(ctx);
    free_scratch_buffers(scratch);

    printf("SUCCESS: Disk streaming engine verified successfully!\n");
    return 0;
}
