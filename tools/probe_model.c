#define _POSIX_C_SOURCE 200112L

#include "state.h"
#include "scratch.h"
#include "model_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    qwen_model_config cfg = {
        .block_count = 64,
        .hidden_dim = 5120,
        .ffn_dim = 17408,
        .num_attn_heads = 24,
        .num_kv_heads = 4,
        .key_length = 256,
        .value_length = 256,
        .ssm_conv_kernel = 4,
        .ssm_state_size = 128,
        .ssm_group_count = 16,
        .ssm_time_step_rank = 48,
        .ssm_inner_size = 6144,
        .vocab_size = 248320
    };
    for (int b = 0; b < 64; b++) {
        cfg.layer_types[b] = (b - 3) % 4 == 0 ? LAYER_TYPE_ATTENTION : LAYER_TYPE_SSM;
    }

    printf("=== Model State & Scratch Allocation Probe ===\n");
    printf("Target Context Length: %d tokens\n\n", context_length);

    long rss_initial = get_vm_rss_kb();
    printf("Initial RSS: %.2f MB (%ld KB)\n", rss_initial / 1024.0, rss_initial);

    printf("\nAllocating Model State...\n");
    model_state *state = allocate_model_state(&cfg, context_length);
    if (!state) {
        fprintf(stderr, "FAIL: Model state allocation failed.\n");
        return 1;
    }
    printf("  KV cache size:            %.2f MB (%zu bytes)\n", state->kv_cache_size / (1024.0 * 1024.0), state->kv_cache_size);
    printf("  SSM recurrent state size:  %.2f MB (%zu bytes)\n", state->ssm_states_size / (1024.0 * 1024.0), state->ssm_states_size);
    printf("  SSM conv history size:     %.2f MB (%zu bytes)\n", state->ssm_conv_histories_size / (1024.0 * 1024.0), state->ssm_conv_histories_size);

    printf("\nAllocating Scratch Buffers...\n");
    scratch_buffers *scratch = allocate_scratch_buffers(&cfg);
    if (!scratch) {
        fprintf(stderr, "FAIL: Scratch buffers allocation failed.\n");
        free_model_state(state);
        return 1;
    }
    printf("  Streaming buffer size:    %.2f MB (%zu bytes)\n", scratch->stream_buffer_size / (1024.0 * 1024.0), scratch->stream_buffer_size);

    long rss_allocated = get_vm_rss_kb();
    printf("\nPost-Allocation RSS: %.2f MB (%ld KB)\n", rss_allocated / 1024.0, rss_allocated);
    printf("RSS Net Increase:    %.2f MB\n", (rss_allocated - rss_initial) / 1024.0);

    printf("\nFreeing state and scratch memory...\n");
    free_scratch_buffers(scratch);
    free_model_state(state);

    long rss_final = get_vm_rss_kb();
    printf("Post-Cleanup RSS:    %.2f MB (%ld KB)\n", rss_final / 1024.0, rss_final);
    printf("Remaining Overhead:  %.2f MB\n", (rss_final - rss_initial) / 1024.0);

    if (rss_final - rss_initial > 50 * 1024) {
        printf("\nWARNING: Possible memory leak detected (residual RSS is high).\n");
    } else {
        printf("\nSUCCESS: Memory allocation and deallocation verified successfully without significant leaks!\n");
    }

    return 0;
}
