#define _POSIX_C_SOURCE 200112L

#include "state.h"
#include "scratch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to parse VmRSS from /proc/self/status
long get_vm_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long rss = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %ld", &rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

int main(int argc, char **argv) {
    int context_length = 8192;
    if (argc > 1) {
        context_length = atoi(argv[1]);
        if (context_length <= 0) {
            fprintf(stderr, "Error: Invalid context length '%s'. Using default 8192.\n", argv[1]);
            context_length = 8192;
        }
    }

    printf("=== Model State & Scratch Allocation Probe ===\n");
    printf("Target Context Length: %d tokens\n\n", context_length);

    // Initial RSS measurement
    long rss_initial = get_vm_rss_kb();
    printf("Initial RSS: %.2f MB (%ld KB)\n", rss_initial / 1024.0, rss_initial);

    // 1. Allocate Model State
    printf("\nAllocating Model State...\n");
    model_state *state = allocate_model_state(context_length);
    if (!state) {
        fprintf(stderr, "FAIL: Model state allocation failed.\n");
        return 1;
    }
    printf("  KV cache size:            %.2f MB (%zu bytes)\n", state->kv_cache_size / (1024.0 * 1024.0), state->kv_cache_size);
    printf("  SSM recurrent state size:  %.2f MB (%zu bytes)\n", state->ssm_states_size / (1024.0 * 1024.0), state->ssm_states_size);
    printf("  SSM conv history size:     %.2f MB (%zu bytes)\n", state->ssm_conv_histories_size / (1024.0 * 1024.0), state->ssm_conv_histories_size);

    // 2. Allocate Scratch Buffers
    printf("\nAllocating Scratch Buffers...\n");
    scratch_buffers *scratch = allocate_scratch_buffers();
    if (!scratch) {
        fprintf(stderr, "FAIL: Scratch buffers allocation failed.\n");
        free_model_state(state);
        return 1;
    }
    printf("  Streaming buffer size:    %.2f MB (%zu bytes)\n", scratch->stream_buffer_size / (1024.0 * 1024.0), scratch->stream_buffer_size);

    // Post-Allocation RSS measurement
    long rss_allocated = get_vm_rss_kb();
    printf("\nPost-Allocation RSS: %.2f MB (%ld KB)\n", rss_allocated / 1024.0, rss_allocated);
    printf("RSS Net Increase:    %.2f MB\n", (rss_allocated - rss_initial) / 1024.0);

    // 3. Free everything
    printf("\nFreeing state and scratch memory...\n");
    free_scratch_buffers(scratch);
    free_model_state(state);

    // Post-Cleanup RSS measurement
    long rss_final = get_vm_rss_kb();
    printf("Post-Cleanup RSS:    %.2f MB (%ld KB)\n", rss_final / 1024.0, rss_final);
    printf("Remaining Overhead:  %.2f MB\n", (rss_final - rss_initial) / 1024.0);

    if (rss_final - rss_initial > 50 * 1024) { // Allow some slack for libc allocations
        printf("\nWARNING: Possible memory leak detected (residual RSS is high).\n");
    } else {
        printf("\nSUCCESS: Memory allocation and deallocation verified successfully without significant leaks!\n");
    }

    return 0;
}
