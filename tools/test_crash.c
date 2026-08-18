#define _POSIX_C_SOURCE 200112L
#define _FILE_OFFSET_BITS 64
#include "state.h"
#include "scratch.h"
#include "tensor_catalog.h"
#include "stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    const char *model_path = "/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf";
    
    fprintf(stderr, "[1] loading catalog\n");
    tensor_catalog *catalog = load_tensor_catalog(model_path);
    fprintf(stderr, "[2] catalog: count=%d\n", catalog ? catalog->count : -1);
    if (!catalog) return 1;
    
    fprintf(stderr, "[3] allocating state\n");
    model_state *state = allocate_model_state(256);
    fprintf(stderr, "[4] state=%p\n", (void*)state);
    if (!state) return 1;
    
    fprintf(stderr, "[5] allocating scratch\n");
    scratch_buffers *scratch = allocate_scratch_buffers();
    fprintf(stderr, "[6] scratch=%p stream_buf=%p\n", (void*)scratch,
            scratch ? (void*)scratch->stream_buffer : NULL);
    if (!scratch) return 1;
    
    fprintf(stderr, "[7] allocating buf_a 300MB\n");
    uint8_t *buf_a = malloc(300ULL * 1024 * 1024);
    fprintf(stderr, "[8] buf_a=%p\n", (void*)buf_a);
    if (!buf_a) return 1;
    
    fprintf(stderr, "[9] allocating buf_b 300MB\n");
    uint8_t *buf_b = malloc(300ULL * 1024 * 1024);
    fprintf(stderr, "[10] buf_b=%p\n", (void*)buf_b);
    if (!buf_b) return 1;
    
    fprintf(stderr, "[11] init stream context\n");
    stream_context *sctx = init_stream_context(model_path, scratch->stream_buffer,
                                                scratch->stream_buffer_size);
    fprintf(stderr, "[12] sctx=%p fd=%d\n", (void*)sctx, sctx ? sctx->fd : -1);
    if (!sctx) return 1;
    
    fprintf(stderr, "[13] finding output_norm\n");
    const tensor_info *ti = find_tensor(catalog, "output_norm.weight");
    fprintf(stderr, "[14] ti=%p byte_size=%llu\n", (void*)ti,
            ti ? (unsigned long long)ti->byte_size : 0ULL);
    if (!ti) return 1;
    
    fprintf(stderr, "[15] malloc output_norm %llu\n", (unsigned long long)ti->byte_size);
    float *output_norm = malloc(ti->byte_size);
    fprintf(stderr, "[16] buf=%p\n", (void*)output_norm);
    
    fprintf(stderr, "[17] done OK\n");
    return 0;
}
