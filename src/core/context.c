#include "diskllm_internal.h"
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>

typedef struct {
    const float *post_attn_norm_w;
    const void  *ffn_gate_w; int ffn_gate_w_type;
    const void  *ffn_up_w;   int ffn_up_w_type;
    const void  *ffn_down_w; int ffn_down_w_type;
    layer_type   l_type;
    union {
        attention_layer_weights attn;
        ssm_layer_weights       ssm;
    } u;
} layer_block_weights_internal;

typedef struct {
    int                     fd;
    const tensor_catalog   *cat;
    const qwen_model_config *cfg;
    int                     layer_idx;
    uint8_t                *buf;
    layer_block_weights_internal *blk;
    uint64_t                bytes_read;
    int                     status;
} prefetch_args;

static ssize_t exact_pread(int fd, void *buf, size_t count, off_t offset) {
    size_t done = 0;
    uint8_t *ptr = (uint8_t *)buf;
    while (done < count) {
        ssize_t r = pread(fd, ptr + done, count - done, offset + done);
        if (r < 0) return -1;
        if (r == 0) break;
        done += r;
    }
    return done;
}

static int load_tensor_to_buf(int fd, const tensor_info *ti, void *dest, uint64_t *bytes_counter) {
    if (!ti) return -1;
    ssize_t r = exact_pread(fd, dest, ti->byte_size, ti->absolute_offset);
    if (r < (ssize_t)ti->byte_size) {
        fprintf(stderr, "pread failed for tensor %s\n", ti->name);
        return -1;
    }
    if (bytes_counter) *bytes_counter += ti->byte_size;
    return 0;
}

static int load_layer_block_weights(int fd, const tensor_catalog *cat, const qwen_model_config *cfg, int li,
                                     uint8_t *buf, layer_block_weights_internal *blk, uint64_t *ctr) {
    char nm[256];
    uint8_t *p = buf;
    blk->l_type = cfg->layer_types[li];

#define LOAD(field, name_fmt, type_field) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (!_ti) { fprintf(stderr, "[ERROR] Missing layer tensor: %s\n", nm); return -1; } \
    if (load_tensor_to_buf(fd, _ti, p, ctr) != 0) return -1; \
    field = (const void *)p; type_field = _ti->type; p += _ti->byte_size; \
} while(0)
#define LOAD_NORM(field, name_fmt) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (!_ti) { fprintf(stderr, "[ERROR] Missing layer tensor: %s\n", nm); return -1; } \
    if (load_tensor_to_buf(fd, _ti, p, ctr) != 0) return -1; \
    field = (const float *)p; p += _ti->byte_size; \
} while(0)
#define LOAD_NORM_OPTIONAL(field, name_fmt) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (_ti) { \
        if (load_tensor_to_buf(fd, _ti, p, ctr) != 0) return -1; \
        field = (const float *)p; p += _ti->byte_size; \
    } else { \
        field = NULL; \
    } \
} while(0)

    if (blk->l_type == LAYER_TYPE_ATTENTION) {
        LOAD_NORM(blk->u.attn.attn_norm_w,               "blk.%d.attn_norm.weight");
        if (cfg->has_fused_qkv) {
            LOAD(blk->u.attn.attn_qkv_w, "blk.%d.attn_qkv.weight", blk->u.attn.attn_qkv_w_type);
            blk->u.attn.attn_q_w = NULL;
            blk->u.attn.attn_k_w = NULL;
            blk->u.attn.attn_v_w = NULL;
            snprintf(nm, sizeof(nm), "blk.%d.attn_qkv.weight", li);
            const tensor_info *ti_qkv = find_tensor(cat, nm);
            if (ti_qkv && ti_qkv->n_dims >= 2) {
                int head_dim  = cfg->key_length;
                int n_heads   = cfg->num_attn_heads;
                int n_kv      = cfg->num_kv_heads;
                blk->u.attn.q_total_dim = n_heads * head_dim;
                blk->u.attn.k_total_dim = n_kv    * head_dim;
                blk->u.attn.v_total_dim = n_kv    * head_dim;
            }
        } else {
            LOAD(blk->u.attn.attn_q_w,                       "blk.%d.attn_q.weight",   blk->u.attn.attn_q_w_type);
            LOAD_NORM_OPTIONAL(blk->u.attn.attn_q_norm_w,   "blk.%d.attn_q_norm.weight");
            LOAD(blk->u.attn.attn_k_w,                       "blk.%d.attn_k.weight",   blk->u.attn.attn_k_w_type);
            LOAD_NORM_OPTIONAL(blk->u.attn.attn_k_norm_w,   "blk.%d.attn_k_norm.weight");
            LOAD(blk->u.attn.attn_v_w,                       "blk.%d.attn_v.weight",   blk->u.attn.attn_v_w_type);
            blk->u.attn.attn_qkv_w = NULL;
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", li);
            const tensor_info *ti_q = find_tensor(cat, nm);
            blk->u.attn.q_total_dim = (ti_q && ti_q->n_dims >= 2) ? (int)ti_q->dims[1] : 0;
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", li);
            const tensor_info *ti_k = find_tensor(cat, nm);
            blk->u.attn.k_total_dim = (ti_k && ti_k->n_dims >= 2) ? (int)ti_k->dims[1] : 0;
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", li);
            const tensor_info *ti_v = find_tensor(cat, nm);
            blk->u.attn.v_total_dim = (ti_v && ti_v->n_dims >= 2) ? (int)ti_v->dims[1] : 0;
        }
        LOAD(blk->u.attn.attn_output_w,                  "blk.%d.attn_output.weight", blk->u.attn.attn_output_w_type);
        snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", li);
        const tensor_info *ti_o = find_tensor(cat, nm);
        blk->u.attn.attn_out_dim = (ti_o && ti_o->n_dims >= 2) ? (int)ti_o->dims[0] : 0;
    } else {
        LOAD_NORM(blk->u.ssm.attn_norm_w,     "blk.%d.attn_norm.weight");
        LOAD(blk->u.ssm.attn_qkv_w,           "blk.%d.attn_qkv.weight", blk->u.ssm.attn_qkv_w_type);
        LOAD_NORM(blk->u.ssm.ssm_conv1d_w,    "blk.%d.ssm_conv1d.weight");
        LOAD_NORM(blk->u.ssm.ssm_a_w,         "blk.%d.ssm_a");
        LOAD(blk->u.ssm.ssm_alpha_w,          "blk.%d.ssm_alpha.weight", blk->u.ssm.ssm_alpha_w_type);
        LOAD(blk->u.ssm.ssm_beta_w,           "blk.%d.ssm_beta.weight",  blk->u.ssm.ssm_beta_w_type);
        LOAD_NORM(blk->u.ssm.ssm_dt_bias,     "blk.%d.ssm_dt.bias");
        LOAD_NORM(blk->u.ssm.ssm_norm_w,      "blk.%d.ssm_norm.weight");
        LOAD(blk->u.ssm.ssm_out_w,            "blk.%d.ssm_out.weight",   blk->u.ssm.ssm_out_w_type);
        LOAD(blk->u.ssm.attn_gate_w,          "blk.%d.attn_gate.weight", blk->u.ssm.attn_gate_w_type);
    }
    snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", li);
    const tensor_info *_ti_post = find_tensor(cat, nm);
    if (!_ti_post) {
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", li);
        _ti_post = find_tensor(cat, nm);
    }
    if (!_ti_post) { fprintf(stderr, "[ERROR] Missing layer norm tensor: blk.%d.(post_attention_norm|ffn_norm).weight\n", li); return -1; }
    if (load_tensor_to_buf(fd, _ti_post, p, ctr) != 0) return -1;
    blk->post_attn_norm_w = (const float *)p; p += _ti_post->byte_size;
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", li);
    if (find_tensor(cat, nm)) {
        LOAD(blk->ffn_gate_w, "blk.%d.ffn_gate.weight", blk->ffn_gate_w_type);
    } else {
        blk->ffn_gate_w = NULL;
        blk->ffn_gate_w_type = 0;
    }
    LOAD(blk->ffn_up_w,   "blk.%d.ffn_up.weight",   blk->ffn_up_w_type);
    LOAD(blk->ffn_down_w, "blk.%d.ffn_down.weight",  blk->ffn_down_w_type);
#undef LOAD
#undef LOAD_NORM
#undef LOAD_NORM_OPTIONAL
    return 0;
}

static int load_layer_block_weights_mmap(const tensor_catalog *cat, const qwen_model_config *cfg, int li,
                                          const uint8_t *mmap_base,
                                          layer_block_weights_internal *blk) {
    char nm[256];
    blk->l_type = cfg->layer_types[li];

#define LOAD_MMAP(field, name_fmt, type_field) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (!_ti) { fprintf(stderr, "[ERROR] Missing layer tensor: %s\n", nm); return -1; } \
    field = (const void *)(mmap_base + _ti->absolute_offset); \
    type_field = _ti->type; \
} while(0)
#define LOAD_NORM_MMAP(field, name_fmt) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (!_ti) { fprintf(stderr, "[ERROR] Missing layer tensor: %s\n", nm); return -1; } \
    field = (const float *)(mmap_base + _ti->absolute_offset); \
} while(0)
#define LOAD_NORM_MMAP_OPTIONAL(field, name_fmt) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (_ti) { \
        field = (const float *)(mmap_base + _ti->absolute_offset); \
    } else { \
        field = NULL; \
    } \
} while(0)

    if (blk->l_type == LAYER_TYPE_ATTENTION) {
        LOAD_NORM_MMAP(blk->u.attn.attn_norm_w,               "blk.%d.attn_norm.weight");
        if (cfg->has_fused_qkv) {
            LOAD_MMAP(blk->u.attn.attn_qkv_w, "blk.%d.attn_qkv.weight", blk->u.attn.attn_qkv_w_type);
            blk->u.attn.attn_q_w = NULL;
            blk->u.attn.attn_k_w = NULL;
            blk->u.attn.attn_v_w = NULL;
            blk->u.attn.attn_q_norm_w = NULL;
            blk->u.attn.attn_k_norm_w = NULL;
            int head_dim  = cfg->key_length;
            int n_heads   = cfg->num_attn_heads;
            int n_kv      = cfg->num_kv_heads;
            blk->u.attn.q_total_dim = n_heads * head_dim;
            blk->u.attn.k_total_dim = n_kv    * head_dim;
            blk->u.attn.v_total_dim = n_kv    * head_dim;
        } else {
            LOAD_MMAP(blk->u.attn.attn_q_w,                       "blk.%d.attn_q.weight",   blk->u.attn.attn_q_w_type);
            LOAD_NORM_MMAP_OPTIONAL(blk->u.attn.attn_q_norm_w,   "blk.%d.attn_q_norm.weight");
            LOAD_MMAP(blk->u.attn.attn_k_w,                       "blk.%d.attn_k.weight",   blk->u.attn.attn_k_w_type);
            LOAD_NORM_MMAP_OPTIONAL(blk->u.attn.attn_k_norm_w,   "blk.%d.attn_k_norm.weight");
            LOAD_MMAP(blk->u.attn.attn_v_w,                       "blk.%d.attn_v.weight",   blk->u.attn.attn_v_w_type);
            blk->u.attn.attn_qkv_w = NULL;
            snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", li);
            const tensor_info *ti_q = find_tensor(cat, nm);
            blk->u.attn.q_total_dim = (ti_q && ti_q->n_dims >= 2) ? (int)ti_q->dims[1] : 0;
            snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", li);
            const tensor_info *ti_k = find_tensor(cat, nm);
            blk->u.attn.k_total_dim = (ti_k && ti_k->n_dims >= 2) ? (int)ti_k->dims[1] : 0;
            snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", li);
            const tensor_info *ti_v = find_tensor(cat, nm);
            blk->u.attn.v_total_dim = (ti_v && ti_v->n_dims >= 2) ? (int)ti_v->dims[1] : 0;
        }
        LOAD_MMAP(blk->u.attn.attn_output_w,                  "blk.%d.attn_output.weight", blk->u.attn.attn_output_w_type);
        snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", li);
        const tensor_info *ti_o = find_tensor(cat, nm);
        blk->u.attn.attn_out_dim = (ti_o && ti_o->n_dims >= 2) ? (int)ti_o->dims[0] : 0;
    } else {
        LOAD_NORM_MMAP(blk->u.ssm.attn_norm_w,     "blk.%d.attn_norm.weight");
        LOAD_MMAP(blk->u.ssm.attn_qkv_w,           "blk.%d.attn_qkv.weight", blk->u.ssm.attn_qkv_w_type);
        LOAD_NORM_MMAP(blk->u.ssm.ssm_conv1d_w,    "blk.%d.ssm_conv1d.weight");
        LOAD_NORM_MMAP(blk->u.ssm.ssm_a_w,         "blk.%d.ssm_a");
        LOAD_MMAP(blk->u.ssm.ssm_alpha_w,          "blk.%d.ssm_alpha.weight", blk->u.ssm.ssm_alpha_w_type);
        LOAD_MMAP(blk->u.ssm.ssm_beta_w,           "blk.%d.ssm_beta.weight",  blk->u.ssm.ssm_beta_w_type);
        LOAD_NORM_MMAP(blk->u.ssm.ssm_dt_bias,     "blk.%d.ssm_dt.bias");
        LOAD_NORM_MMAP(blk->u.ssm.ssm_norm_w,      "blk.%d.ssm_norm.weight");
        LOAD_MMAP(blk->u.ssm.ssm_out_w,            "blk.%d.ssm_out.weight",   blk->u.ssm.ssm_out_w_type);
        LOAD_MMAP(blk->u.ssm.attn_gate_w,          "blk.%d.attn_gate.weight", blk->u.ssm.attn_gate_w_type);
    }
    snprintf(nm, sizeof(nm), "blk.%d.post_attention_norm.weight", li);
    const tensor_info *_ti_post_m = find_tensor(cat, nm);
    if (!_ti_post_m) {
        snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight", li);
        _ti_post_m = find_tensor(cat, nm);
    }
    if (!_ti_post_m) { fprintf(stderr, "[ERROR] Missing layer norm tensor: blk.%d.(post_attention_norm|ffn_norm).weight\n", li); return -1; }
    blk->post_attn_norm_w = (const float *)(mmap_base + _ti_post_m->absolute_offset);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", li);
    if (find_tensor(cat, nm)) {
        LOAD_MMAP(blk->ffn_gate_w, "blk.%d.ffn_gate.weight", blk->ffn_gate_w_type);
    } else {
        blk->ffn_gate_w = NULL;
        blk->ffn_gate_w_type = 0;
    }
    LOAD_MMAP(blk->ffn_up_w,   "blk.%d.ffn_up.weight",   blk->ffn_up_w_type);
    LOAD_MMAP(blk->ffn_down_w, "blk.%d.ffn_down.weight",  blk->ffn_down_w_type);
#undef LOAD_MMAP
#undef LOAD_NORM_MMAP
#undef LOAD_NORM_MMAP_OPTIONAL
    return 0;
}

static void *prefetch_thread_fn(void *arg) {
    prefetch_args *a = (prefetch_args *)arg;
    a->status = load_layer_block_weights(a->fd, a->cat, a->cfg, a->layer_idx, a->buf, a->blk, &a->bytes_read);
    return NULL;
}

diskllm_context_params diskllm_context_params_default(void) {
    return (diskllm_context_params){
        .context_size = 4096,
        .cache_type = DISKLLM_CACHE_F16,
        .num_threads = 0
    };
}

diskllm_context *diskllm_context_init(diskllm_model *model, diskllm_context_params params) {
    if (!model) return NULL;

    diskllm_context *ctx = calloc(1, sizeof(diskllm_context));
    if (!ctx) return NULL;

    ctx->model = model;
    ctx->params = params;

    /* Initialize threads */
    int threads = (params.num_threads > 0) ? params.num_threads : model->params.num_threads;
    matvec_set_num_threads(threads);

    /* Allocate model state */
    cache_type_t ctype = CACHE_TYPE_F16;
    if (params.cache_type == DISKLLM_CACHE_Q8_0) ctype = CACHE_TYPE_Q8_0;
    else if (params.cache_type == DISKLLM_CACHE_Q4_0) ctype = CACHE_TYPE_Q4_0;

    ctx->state = allocate_model_state_ex(model->cfg, params.context_size, ctype);
    if (!ctx->state) {
        fprintf(stderr, "Error: Failed to allocate model state.\n");
        free(ctx);
        return NULL;
    }

    /* Allocate scratch buffers */
    ctx->scratch = allocate_scratch_buffers(model->cfg);
    if (!ctx->scratch) {
        fprintf(stderr, "Error: Failed to allocate scratch buffers.\n");
        free_model_state(ctx->state);
        free(ctx);
        return NULL;
    }

    size_t max_layer_bytes = 256 * 1024 * 1024;
    ctx->layer_buf_a = malloc(max_layer_bytes);
    ctx->layer_buf_b = malloc(max_layer_bytes);
    ctx->hidden_single = malloc(model->cfg->hidden_dim * sizeof(float));

    /* Init stream context if not pinned */
    if (!model->sctx) {
        io_mode_t io_mode = IO_MODE_PREAD;
        if (model->params.io_mode == DISKLLM_IO_MMAP) io_mode = IO_MODE_MMAP;
        else if (model->params.io_mode == DISKLLM_IO_DIRECT) io_mode = IO_MODE_DIRECT;
        else if (model->params.io_mode == DISKLLM_IO_IOURING) io_mode = IO_MODE_IOURING;

        model->sctx = init_stream_context_ex(model->model_path, ctx->scratch->stream_buffer, ctx->scratch->stream_buffer_size, io_mode);
        if (model->sctx) model->fd = model->sctx->fd;
    }

    /* Load output norm weight into model struct if not loaded */
    if (!model->output_norm) {
        const tensor_info *ti_onorm = find_tensor(model->catalog, "output_norm.weight");
        if (ti_onorm) {
            model->output_norm = malloc(ti_onorm->byte_size);
            if (model->g_mmap_full) {
                memcpy(model->output_norm, model->g_mmap_full + ti_onorm->absolute_offset, ti_onorm->byte_size);
            } else if (model->fd >= 0) {
                exact_pread(model->fd, model->output_norm, ti_onorm->byte_size, ti_onorm->absolute_offset);
            }
        }
    }

    return ctx;
}

void diskllm_context_free(diskllm_context *ctx) {
    if (!ctx) return;
    if (ctx->layer_buf_a) free(ctx->layer_buf_a);
    if (ctx->layer_buf_b) free(ctx->layer_buf_b);
    if (ctx->hidden_single) free(ctx->hidden_single);
    if (ctx->scratch) free_scratch_buffers(ctx->scratch);
    if (ctx->state) free_model_state(ctx->state);
    if (ctx->prompt_tokens) free(ctx->prompt_tokens);
    if (ctx->gen_tokens) free(ctx->gen_tokens);
    free(ctx);
}

int diskllm_eval(diskllm_context *ctx, const int *tokens, int n_tokens, float *logits) {
    if (!ctx || !tokens || n_tokens <= 0 || !logits) return -1;

    diskllm_model *model = ctx->model;
    qwen_model_config *cfg = model->cfg;
    scratch_buffers *scratch = ctx->scratch;

    ctx->prompt_len = n_tokens;
    ctx->cur_pos = n_tokens;
    ctx->t_prefill_start = diskllm_get_time_ms();

    /* Allocate hidden states buffer for prompt */
    float *hidden_states = malloc((size_t)n_tokens * cfg->hidden_dim * sizeof(float));
    if (!hidden_states) return -1;

    /* Embedding lookup */
    for (int p = 0; p < n_tokens; p++) {
        int tok = tokens[p];
        float *h = hidden_states + p * cfg->hidden_dim;
        if (model->g_mmap_full) {
            const uint8_t *row_ptr = model->g_mmap_full + model->ti_emb->absolute_offset + (uint64_t)tok * model->embed_row_bytes;
            dequantize_row(h, row_ptr, cfg->hidden_dim, model->ti_emb->type);
        } else {
            uint8_t *row_buf = malloc(model->embed_row_bytes);
            exact_pread(model->fd, row_buf, model->embed_row_bytes, model->ti_emb->absolute_offset + (uint64_t)tok * model->embed_row_bytes);
            dequantize_row(h, row_buf, cfg->hidden_dim, model->ti_emb->type);
            free(row_buf);
            ctx->bytes_read += model->embed_row_bytes;
        }
    }

    /* Layer-by-layer forward pass */
    if (model->g_mmap_full) {
        layer_block_weights_internal blk;
        for (int li = 0; li < cfg->block_count; li++) {
            load_layer_block_weights_mmap(model->catalog, cfg, li, model->g_mmap_full, &blk);
            if (model->arch_backend && model->arch_backend->prefill_layer) {
                model->arch_backend->prefill_layer(ctx, li, &blk, hidden_states, n_tokens);
            }
        }
    } else {
        uint8_t *bufs[2] = {ctx->layer_buf_a, ctx->layer_buf_b};
        layer_block_weights_internal blks[2];
        int abuf = 0;
        uint64_t lb = 0;

        load_layer_block_weights(model->fd, model->catalog, cfg, 0, bufs[0], &blks[0], &lb);
        ctx->bytes_read += lb;

        pthread_t pth;
        prefetch_args pargs;
        int pth_active = 0;

        for (int li = 0; li < cfg->block_count; li++) {
            if (li + 1 < cfg->block_count) {
                pargs = (prefetch_args){.fd=model->fd, .cat=model->catalog, .cfg=cfg, .layer_idx=li+1,
                                        .buf=bufs[1-abuf], .blk=&blks[1-abuf], .bytes_read=0, .status=-1};
                pth_active = (pthread_create(&pth, NULL, prefetch_thread_fn, &pargs) == 0);
                if (!pth_active) {
                    pargs.bytes_read = 0;
                    load_layer_block_weights(model->fd, model->catalog, cfg, li+1, bufs[1-abuf], &blks[1-abuf], &pargs.bytes_read);
                }
            } else { pth_active = 0; }

            layer_block_weights_internal *blk = &blks[abuf];
            if (model->arch_backend && model->arch_backend->prefill_layer) {
                model->arch_backend->prefill_layer(ctx, li, blk, hidden_states, n_tokens);
            }

            if (pth_active) {
                pthread_join(pth, NULL);
                ctx->bytes_read += pargs.bytes_read;
            } else if (li + 1 < cfg->block_count) {
                ctx->bytes_read += pargs.bytes_read;
            }
            abuf = 1 - abuf;
        }
    }

    /* Final norm & logits on last token position */
    float *last_h = hidden_states + (n_tokens - 1) * cfg->hidden_dim;
    int add_one = (cfg->model_type == MODEL_TYPE_GEMMA) ? 1 : 0;
    rmsnorm_ext(last_h, last_h, model->output_norm, cfg->hidden_dim, 1e-6f, add_one);

    if (model->mmap_output_weight) {
        matvec(logits, model->mmap_output_weight, last_h, cfg->hidden_dim, cfg->vocab_size, model->ti_outw->type, NULL);
    } else {
        int64_t rows_done = 0, chunk_rows = 15000;
        uint8_t *cbuf = scratch->stream_buffer;
        while (rows_done < cfg->vocab_size) {
            int64_t r = cfg->vocab_size - rows_done;
            if (r > chunk_rows) r = chunk_rows;
            exact_pread(model->fd, cbuf, r * model->logit_row_bytes, model->ti_outw->absolute_offset + rows_done * model->logit_row_bytes);
            ctx->bytes_read += r * model->logit_row_bytes;
            matvec(logits + rows_done, cbuf, last_h, cfg->hidden_dim, r, model->ti_outw->type, NULL);
            rows_done += r;
        }
    }

    if (cfg->final_logit_softcapping > 0.0f) {
        float cap = cfg->final_logit_softcapping;
        for (int i = 0; i < cfg->vocab_size; i++) logits[i] = cap * tanhf(logits[i] / cap);
    }

    free(hidden_states);
    ctx->t_prefill_end = diskllm_get_time_ms();
    ctx->decode_start_bytes_read = ctx->bytes_read;
    ctx->peak_rss = diskllm_read_rss_mb();
    return 0;
}

int diskllm_decode_step(diskllm_context *ctx, int token, float *logits) {
    if (!ctx || !logits) return -1;

    diskllm_model *model = ctx->model;
    qwen_model_config *cfg = model->cfg;
    scratch_buffers *scratch = ctx->scratch;

    if (ctx->gen_count == 0) {
        ctx->t_gen_start = diskllm_get_time_ms();
    }

    int cur_pos = ctx->prompt_len + ctx->gen_count;
    ctx->gen_count++;

    /* Embedding for token */
    float *h = ctx->hidden_single;
    if (model->g_mmap_full) {
        const uint8_t *row_ptr = model->g_mmap_full + model->ti_emb->absolute_offset + (uint64_t)token * model->embed_row_bytes;
        dequantize_row(h, row_ptr, cfg->hidden_dim, model->ti_emb->type);
    } else {
        uint8_t *row_buf = malloc(model->embed_row_bytes);
        exact_pread(model->fd, row_buf, model->embed_row_bytes, model->ti_emb->absolute_offset + (uint64_t)token * model->embed_row_bytes);
        dequantize_row(h, row_buf, cfg->hidden_dim, model->ti_emb->type);
        free(row_buf);
        ctx->bytes_read += model->embed_row_bytes;
    }

    /* Layer-by-layer decode */
    if (model->g_mmap_full) {
        layer_block_weights_internal blk;
        for (int li = 0; li < cfg->block_count; li++) {
            load_layer_block_weights_mmap(model->catalog, cfg, li, model->g_mmap_full, &blk);
            if (model->arch_backend && model->arch_backend->decode_layer) {
                model->arch_backend->decode_layer(ctx, li, &blk, h, cur_pos);
            }
        }
    } else {
        uint8_t *bufs[2] = {ctx->layer_buf_a, ctx->layer_buf_b};
        layer_block_weights_internal blks[2];
        int abuf = 0;
        uint64_t lb = 0;

        load_layer_block_weights(model->fd, model->catalog, cfg, 0, bufs[0], &blks[0], &lb);
        ctx->bytes_read += lb;

        pthread_t pth;
        prefetch_args pargs;
        int pth_active = 0;

        for (int li = 0; li < cfg->block_count; li++) {
            if (li + 1 < cfg->block_count) {
                pargs = (prefetch_args){.fd=model->fd, .cat=model->catalog, .cfg=cfg, .layer_idx=li+1,
                                        .buf=bufs[1-abuf], .blk=&blks[1-abuf], .bytes_read=0, .status=-1};
                pth_active = (pthread_create(&pth, NULL, prefetch_thread_fn, &pargs) == 0);
                if (!pth_active) {
                    pargs.bytes_read = 0;
                    load_layer_block_weights(model->fd, model->catalog, cfg, li+1, bufs[1-abuf], &blks[1-abuf], &pargs.bytes_read);
                }
            } else { pth_active = 0; }

            layer_block_weights_internal *blk = &blks[abuf];
            if (model->arch_backend && model->arch_backend->decode_layer) {
                model->arch_backend->decode_layer(ctx, li, blk, h, cur_pos);
            }

            if (pth_active) {
                pthread_join(pth, NULL);
                ctx->bytes_read += pargs.bytes_read;
            } else if (li + 1 < cfg->block_count) {
                ctx->bytes_read += pargs.bytes_read;
            }
            abuf = 1 - abuf;
        }
    }

    /* Final norm & logits */
    int add_one = (cfg->model_type == MODEL_TYPE_GEMMA) ? 1 : 0;
    rmsnorm_ext(h, h, model->output_norm, cfg->hidden_dim, 1e-6f, add_one);

    if (model->mmap_output_weight) {
        matvec(logits, model->mmap_output_weight, h, cfg->hidden_dim, cfg->vocab_size, model->ti_outw->type, NULL);
    } else {
        int64_t rows_done = 0, chunk_rows = 15000;
        uint8_t *cbuf = scratch->stream_buffer;
        while (rows_done < cfg->vocab_size) {
            int64_t r = cfg->vocab_size - rows_done;
            if (r > chunk_rows) r = chunk_rows;
            exact_pread(model->fd, cbuf, r * model->logit_row_bytes, model->ti_outw->absolute_offset + rows_done * model->logit_row_bytes);
            ctx->bytes_read += r * model->logit_row_bytes;
            matvec(logits + rows_done, cbuf, h, cfg->hidden_dim, r, model->ti_outw->type, NULL);
            rows_done += r;
        }
    }

    if (cfg->final_logit_softcapping > 0.0f) {
        float cap = cfg->final_logit_softcapping;
        for (int i = 0; i < cfg->vocab_size; i++) logits[i] = cap * tanhf(logits[i] / cap);
    }

    ctx->t_gen_end = diskllm_get_time_ms();
    long rss = diskllm_read_rss_mb();
    if (rss > ctx->peak_rss) ctx->peak_rss = rss;

    return 0;
}

diskllm_perf_metrics diskllm_get_perf_metrics(const diskllm_context *ctx) {
    diskllm_perf_metrics m = {0};
    if (!ctx) return m;

    m.prompt_tokens = ctx->prompt_len;
    m.gen_tokens = ctx->gen_count;
    m.prefill_time_ms = ctx->t_prefill_end - ctx->t_prefill_start;
    m.generation_time_ms = ctx->t_gen_end - ctx->t_gen_start;
    if (m.gen_tokens > 0) {
        m.gen_speed_ms_per_tok = m.generation_time_ms / m.gen_tokens;
    }
    m.bytes_read_total = ctx->bytes_read;
    m.bytes_read_decode = (ctx->bytes_read >= ctx->decode_start_bytes_read) ? (ctx->bytes_read - ctx->decode_start_bytes_read) : 0;
    m.peak_rss_mb = ctx->peak_rss;

    if (ctx->model->params.pin_weights) {
        m.weights_mode_str = "Pinned in RAM (Zero Decode I/O)";
    } else if (ctx->model->g_mmap_full) {
        m.weights_mode_str = "Memory Mapped";
    } else {
        m.weights_mode_str = "Disk Streaming";
    }

    return m;
}

/* ─── State Persistence ─────────────────────────────────────────────────────── */

typedef struct {
    char     magic[4];           // "DKST"
    uint32_t version;            // 1
    int32_t  pos;                // sequence position
    int32_t  prompt_len;         // total tokens processed
    int32_t  context_size;       // context capacity
    int32_t  hidden_dim;         // e.g. 5120 or 1024
    uint64_t kv_cache_bytes;     // kv cache size
    uint64_t ssm_state_bytes;    // ssm states size
    uint64_t ssm_conv_bytes;     // ssm conv history size
    int32_t  next_tok;           // next token ID
    uint32_t checksum;           // 0x5157454E
    int32_t  cache_type;         // 0 = F16, 1 = Q8_0, 2 = Q4_0
    int32_t  reserved[5];
} diskllm_state_header;

int diskllm_save_state(diskllm_context *ctx, const char *filepath) {
    if (!ctx || !filepath) return -1;
    diskllm_model *model = ctx->model;
    model_state *st = ctx->state;

    FILE *f = fopen(filepath, "wb");
    if (!f) return -1;

    diskllm_state_header hdr = {
        .magic = {'D','K','S','T'},
        .version = 1,
        .pos = ctx->cur_pos,
        .prompt_len = ctx->prompt_len,
        .context_size = st->context_length,
        .hidden_dim = model->cfg->hidden_dim,
        .kv_cache_bytes = (uint64_t)model->cfg->num_attn_layers * (uint64_t)st->context_length * st->kv_token_bytes,
        .ssm_state_bytes = (uint64_t)model->cfg->num_ssm_layers * (uint64_t)st->ssm_state_size * sizeof(float),
        .ssm_conv_bytes = (uint64_t)model->cfg->num_ssm_layers * (uint64_t)model->cfg->ssm_conv_kernel * sizeof(float),
        .next_tok = ctx->next_tok,
        .checksum = 0x5157454E,
        .cache_type = (int32_t)st->cache_type
    };

    fwrite(&hdr, sizeof(hdr), 1, f);
    if (hdr.kv_cache_bytes > 0 && st->kv_cache) fwrite(st->kv_cache, 1, hdr.kv_cache_bytes, f);
    if (hdr.ssm_state_bytes > 0 && st->ssm_states) fwrite(st->ssm_states, 1, hdr.ssm_state_bytes, f);
    if (hdr.ssm_conv_bytes > 0 && st->ssm_conv_histories) fwrite(st->ssm_conv_histories, 1, hdr.ssm_conv_bytes, f);
    fclose(f);
    return 0;
}

int diskllm_load_state(diskllm_context *ctx, const char *filepath) {
    if (!ctx || !filepath) return -1;
    model_state *st = ctx->state;

    FILE *f = fopen(filepath, "rb");
    if (!f) return -1;

    diskllm_state_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || strncmp(hdr.magic, "DKST", 4) != 0) {
        fclose(f);
        return -1;
    }

    ctx->cur_pos = hdr.pos;
    ctx->prompt_len = hdr.prompt_len;
    ctx->next_tok = hdr.next_tok;

    if (hdr.kv_cache_bytes > 0 && st->kv_cache) fread(st->kv_cache, 1, hdr.kv_cache_bytes, f);
    if (hdr.ssm_state_bytes > 0 && st->ssm_states) fread(st->ssm_states, 1, hdr.ssm_state_bytes, f);
    if (hdr.ssm_conv_bytes > 0 && st->ssm_conv_histories) fread(st->ssm_conv_histories, 1, hdr.ssm_conv_bytes, f);
    fclose(f);
    return 0;
}

int diskllm_print_state_info(const char *filepath) {
    if (!filepath) return -1;
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open state file: %s\n", filepath);
        return -1;
    }
    diskllm_state_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || strncmp(hdr.magic, "DKST", 4) != 0) {
        fprintf(stderr, "Invalid state header in file: %s\n", filepath);
        fclose(f);
        return -1;
    }
    fclose(f);

    const char *c_str = (hdr.cache_type == 1) ? "Q8_0" : ((hdr.cache_type == 2) ? "Q4_0" : "F16");
    printf("State Header Inspection (%s):\n", filepath);
    printf("  Magic            : %.4s\n", hdr.magic);
    printf("  Version          : %u\n", hdr.version);
    printf("  Position         : %d\n", hdr.pos);
    printf("  Prompt Tokens    : %d\n", hdr.prompt_len);
    printf("  Context Size     : %d\n", hdr.context_size);
    printf("  Hidden Dim       : %d\n", hdr.hidden_dim);
    printf("  KV Cache Bytes   : %llu\n", (unsigned long long)hdr.kv_cache_bytes);
    printf("  SSM State Bytes  : %llu\n", (unsigned long long)hdr.ssm_state_bytes);
    printf("  SSM Conv Bytes   : %llu\n", (unsigned long long)hdr.ssm_conv_bytes);
    printf("  Next Token ID    : %d\n", hdr.next_tok);
    printf("  Cache Format     : %s\n", c_str);
    return 0;
}
