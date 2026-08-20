#ifndef DISKLLM_INTERNAL_H
#define DISKLLM_INTERNAL_H

#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <time.h>
#include "diskllm.h"
#include "model_config.h"
#include "tensor_catalog.h"
#include "layer_map.h"
#include "state.h"
#include "scratch.h"
#include "stream.h"
#include "tokenizer.h"
#include "sampler.h"
#include "kernels.h"
#include "dequant.h"
#include "attention.h"
#include "ssm.h"
#include "arch/registry.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct diskllm_model {
    char model_path[512];
    qwen_model_config *cfg;
    tensor_catalog *catalog;
    tokenizer *tok;
    int fd;
    stream_context *sctx;
    
    /* Memory Mapping / RAM Pinning */
    uint8_t *g_mmap_full;
    size_t   g_mmap_full_size;
    const uint8_t *mmap_output_weight;
    
    /* Core Tensor Infos */
    const tensor_info *ti_emb;
    const tensor_info *ti_outw;
    size_t embed_row_bytes;
    size_t logit_row_bytes;
    float *output_norm;

    /* Gemma 4 PLE (Per-Layer Embeddings) */
    const void *per_layer_token_embd;
    int per_layer_token_embd_type;
    size_t per_layer_token_embd_row_bytes;
    const void *per_layer_model_proj;
    int per_layer_model_proj_type;
    const float *per_layer_proj_norm;
    int n_embd_per_layer;
    const float *rope_freqs;
    
    diskllm_model_params params;
    const diskllm_arch_backend *arch_backend;
};

struct diskllm_context {
    diskllm_model *model;
    model_state   *state;
    scratch_buffers *scratch;
    diskllm_context_params params;
    
    uint8_t *layer_buf_a;
    uint8_t *layer_buf_b;
    float   *hidden_single;
    
    int cur_pos;
    int prompt_len;
    int gen_count;
    
    uint64_t bytes_read;
    uint64_t decode_start_bytes_read;
    
    double t_prefill_start;
    double t_prefill_end;
    double t_gen_start;
    double t_gen_end;
    
    long peak_rss;
    
    int  *prompt_tokens;
    int  *gen_tokens;
    int   next_tok;
    bool  debug_hidden_norm;

    /* Gemma 4 PLE per-token cache: size [block_count * n_embd_per_layer] */
    float *ple_cache;
    float *ple_prompt_cache;
};

struct diskllm_sampler {
    sampler *smp;
    diskllm_sampler_params params;
};

struct diskllm_tokenizer {
    tokenizer *tok;
};

/* Helper Functions */
double diskllm_get_time_ms(void);
long   diskllm_read_rss_mb(void);
uint64_t diskllm_get_available_memory_bytes(void);

#endif /* DISKLLM_INTERNAL_H */
