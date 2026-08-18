#define _POSIX_C_SOURCE 200112L
#define _GNU_SOURCE

#include "state.h"
#include "scratch.h"
#include "kernels.h"
#include "dequant.h"
#include "attention.h"
#include "ssm.h"
#include "tensor_catalog.h"
#include "layer_map.h"
#include "sampler.h"
#include "stream.h"
#include "tokenizer.h"
#include "model_config.h"
#include "speculative.h"
#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <assert.h>
#include <time.h>
#include <pthread.h>
#include <sys/mman.h>

/* ─── Byte helpers ──────────────────────────────────────────────────────────── */

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

static int load_tensor_to_buf(int fd, const tensor_info *ti, void *dest,
                               uint64_t *bytes_counter) {
    if (!ti) return -1;
    ssize_t r = exact_pread(fd, dest, ti->byte_size, ti->absolute_offset);
    if (r < (ssize_t)ti->byte_size) {
        fprintf(stderr, "pread failed for tensor %s\n", ti->name);
        return -1;
    }
    if (bytes_counter) *bytes_counter += ti->byte_size;
    return 0;
}

/* ─── Layer weight block ───────────────────────────────────────────────────── */

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
} layer_block_weights;

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

static int load_layer_block_weights(int fd, const tensor_catalog *cat, const qwen_model_config *cfg, int li,
                                     uint8_t *buf, layer_block_weights *blk,
                                     uint64_t *ctr) {
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
        LOAD(blk->u.attn.attn_q_w,                       "blk.%d.attn_q.weight",   blk->u.attn.attn_q_w_type);
        LOAD_NORM_OPTIONAL(blk->u.attn.attn_q_norm_w,   "blk.%d.attn_q_norm.weight");
        LOAD(blk->u.attn.attn_k_w,                       "blk.%d.attn_k.weight",   blk->u.attn.attn_k_w_type);
        LOAD_NORM_OPTIONAL(blk->u.attn.attn_k_norm_w,   "blk.%d.attn_k_norm.weight");
        LOAD(blk->u.attn.attn_v_w,                       "blk.%d.attn_v.weight",   blk->u.attn.attn_v_w_type);
        LOAD(blk->u.attn.attn_output_w,                  "blk.%d.attn_output.weight", blk->u.attn.attn_output_w_type);

        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", li);
        const tensor_info *ti_q = find_tensor(cat, nm);
        blk->u.attn.q_total_dim = (ti_q && ti_q->n_dims >= 2) ? (int)ti_q->dims[1] : 0;
        snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", li);
        const tensor_info *ti_k = find_tensor(cat, nm);
        blk->u.attn.k_total_dim = (ti_k && ti_k->n_dims >= 2) ? (int)ti_k->dims[1] : 0;
        snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", li);
        const tensor_info *ti_v = find_tensor(cat, nm);
        blk->u.attn.v_total_dim = (ti_v && ti_v->n_dims >= 2) ? (int)ti_v->dims[1] : 0;
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
    LOAD(blk->ffn_gate_w, "blk.%d.ffn_gate.weight", blk->ffn_gate_w_type);
    LOAD(blk->ffn_up_w,   "blk.%d.ffn_up.weight",   blk->ffn_up_w_type);
    LOAD(blk->ffn_down_w, "blk.%d.ffn_down.weight",  blk->ffn_down_w_type);
#undef LOAD
#undef LOAD_NORM
#undef LOAD_NORM_OPTIONAL
    return 0;
}

static int load_layer_block_weights_mmap(const tensor_catalog *cat, const qwen_model_config *cfg, int li,
                                          const uint8_t *mmap_base,
                                          layer_block_weights *blk) {
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
        LOAD_MMAP(blk->u.attn.attn_q_w,                       "blk.%d.attn_q.weight",   blk->u.attn.attn_q_w_type);
        LOAD_NORM_MMAP_OPTIONAL(blk->u.attn.attn_q_norm_w,   "blk.%d.attn_q_norm.weight");
        LOAD_MMAP(blk->u.attn.attn_k_w,                       "blk.%d.attn_k.weight",   blk->u.attn.attn_k_w_type);
        LOAD_NORM_MMAP_OPTIONAL(blk->u.attn.attn_k_norm_w,   "blk.%d.attn_k_norm.weight");
        LOAD_MMAP(blk->u.attn.attn_v_w,                       "blk.%d.attn_v.weight",   blk->u.attn.attn_v_w_type);
        LOAD_MMAP(blk->u.attn.attn_output_w,                  "blk.%d.attn_output.weight", blk->u.attn.attn_output_w_type);

        snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", li);
        const tensor_info *ti_q = find_tensor(cat, nm);
        blk->u.attn.q_total_dim = (ti_q && ti_q->n_dims >= 2) ? (int)ti_q->dims[1] : 0;
        snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", li);
        const tensor_info *ti_k = find_tensor(cat, nm);
        blk->u.attn.k_total_dim = (ti_k && ti_k->n_dims >= 2) ? (int)ti_k->dims[1] : 0;
        snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", li);
        const tensor_info *ti_v = find_tensor(cat, nm);
        blk->u.attn.v_total_dim = (ti_v && ti_v->n_dims >= 2) ? (int)ti_v->dims[1] : 0;
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
    LOAD_MMAP(blk->ffn_gate_w, "blk.%d.ffn_gate.weight", blk->ffn_gate_w_type);
    LOAD_MMAP(blk->ffn_up_w,   "blk.%d.ffn_up.weight",   blk->ffn_up_w_type);
    LOAD_MMAP(blk->ffn_down_w, "blk.%d.ffn_down.weight",  blk->ffn_down_w_type);
#undef LOAD_MMAP
#undef LOAD_NORM_MMAP
#undef LOAD_NORM_MMAP_OPTIONAL
    return 0;
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint64_t get_proc_self_io_read_bytes(void) {
    FILE *f = fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[256];
    uint64_t bytes = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "read_bytes: %llu", (unsigned long long *)&bytes) == 1) {
            break;
        }
    }
    fclose(f);
    return bytes;
}

static void warm_os_page_cache(const char *model_path, int fd) {
    printf("[INFO] Warming OS page cache for %s...\n", model_path);
    double t0 = get_time_ms();

#if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
    posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
#endif

    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    size_t buf_size = 16 * 1024 * 1024;
    uint8_t *wbuf = malloc(buf_size);
    if (!wbuf) return;

    off_t offset = 0;
    uint64_t total_warmed = 0;
    while (offset < file_size) {
        size_t chunk = buf_size;
        if (offset + (off_t)chunk > file_size) chunk = file_size - offset;
        ssize_t r = pread(fd, wbuf, chunk, offset);
        if (r <= 0) break;
        total_warmed += r;
        offset += r;
    }
    free(wbuf);
    double t1 = get_time_ms();
    printf("[INFO] Warmed %.2f MB in page cache in %.2f ms (%.2f MB/s)\n",
           (double)total_warmed / (1024*1024), t1 - t0,
           ((double)total_warmed / (1024*1024)) / ((t1 - t0) / 1000.0));
}

/* ─── Utilities ────────────────────────────────────────────────────────────── */

static long read_rss_mb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256]; long kb = -1;
    while (fgets(line, sizeof(line), f))
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line, "VmRSS: %ld", &kb); break; }
    fclose(f);
    return kb >= 0 ? kb / 1024 : -1;
}

static float l2_norm(const float *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return (float)sqrt(s);
}

static void maybe_print_norm(const float *v, int n, int layer, int flag) {
    if (!flag) return;
    printf("[NORM Layer %2d] L2 = %.6f\n", layer, l2_norm(v, n));
}

static void dequantize_row(const uint8_t *src, float *dst, int dim, int type) {
    if (type == GGML_TYPE_Q4_K) {
        dequantize_q4_K(src, dst, dim);
    } else if (type == GGML_TYPE_Q6_K) {
        dequantize_q6_K(src, dst, dim);
    } else if (type == GGML_TYPE_Q8_0) {
        dequantize_q8_0(src, dst, dim);
    } else if (type == GGML_TYPE_F32) {
        dequantize_f32(src, dst, dim);
    } else {
        dequantize_q4_K(src, dst, dim);
    }
}

typedef struct {
    int                     fd;
    const tensor_catalog   *cat;
    const qwen_model_config *cfg;
    int                     layer_idx;
    uint8_t                *buf;
    layer_block_weights    *blk;
    uint64_t                bytes_read;
    int                     status;
} prefetch_args;

static void *prefetch_thread_fn(void *arg) {
    prefetch_args *a = (prefetch_args *)arg;
    a->status = load_layer_block_weights(a->fd, a->cat, a->cfg, a->layer_idx, a->buf, a->blk, &a->bytes_read);
    return NULL;
}

/* ─── Usage ─────────────────────────────────────────────────────────────────── */

static void print_usage(const char *prog) {
    printf("DiskLLM — Streaming / Disk-based LLM Inference Engine\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model <path>              Path to GGUF model file (required)\n");
    printf("  --arch <auto|qwen-hybrid|qwen-attention> Model architecture mode (default: auto)\n");
    printf("  --prompt <text>             Input prompt text\n");
    printf("  --system <text>             System prompt text (for --chat)\n");
    printf("  --prompt-ids <id1,id2,...>  Input prompt token IDs\n");
    printf("  --prompt-ids-file <file>    File containing prompt token IDs\n");
    printf("  --cache-type <f16|q8_0|q4_0> KV cache quantization format (default: f16)\n");
    printf("  --save-state <file>         File path to save execution state\n");
    printf("  --load-state <file>         File path to load execution state from\n");
    printf("  --state-info <file>         Print header information of a state file\n");
    printf("  --print-missing-tensors     Print missing expected tensors for model diagnostics\n");
    printf("  --context <size>            Context length capacity (default: 4096)\n");
    printf("  --max-tokens <N>            Maximum number of tokens to generate (default: 128)\n");
    printf("  --threads, -t <N>           Number of matvec threads (default: CPU cores)\n");
    printf("  --temp <val>                Sampling temperature (0.0 = greedy, default: 0.0)\n");
    printf("  --top-k <K>                 Top-K sampling\n");
    printf("  --top-p <P>                 Top-P nucleus sampling\n");
    printf("  --min-p <P>                 Min-P sampling\n");
    printf("  --repeat-penalty <val>      Repetition penalty factor (default: 1.0)\n");
    printf("  --stop-token <id>           Extra stop token ID (can be repeated)\n");
    printf("  --chat                      Auto-wrap prompt in Qwen chat template\n");
    printf("  --warm-cache                 Pre-warm OS page cache for model file\n");
    printf("  --io-mode <pread|mmap>      I/O streaming mode (default: pread)\n");
    printf("  --log-io-per-token          Log I/O wait vs compute breakdown per generated token\n");
    printf("  --profile-decode            Print detailed decode profiling breakdown\n");
    printf("  --greedy                    Shortcut for --temp 0.0\n");
    printf("  --gpu                       Enable Vulkan GPU acceleration for compute kernels\n");
    printf("  --quiet                     Suppress progress messages\n");
    printf("  --help, -h                  Show this help message\n");
}

static int print_state_info(const char *state_file) {
    FILE *f = fopen(state_file, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open state file %s\n", state_file);
        return 1;
    }
    diskllm_state_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1 || memcmp(hdr.magic, "DKST", 4) != 0) {
        fprintf(stderr, "Error: Invalid state header in %s\n", state_file);
        fclose(f);
        return 1;
    }
    fclose(f);

    const char *ctnames[] = {"F16", "Q8_0", "Q4_0"};
    const char *cname = (hdr.cache_type >= 0 && hdr.cache_type <= 2) ? ctnames[hdr.cache_type] : "Unknown";

    printf("State File Header Information for: %s\n", state_file);
    printf("  Magic          : %.4s\n", hdr.magic);
    printf("  Version        : %u\n", hdr.version);
    printf("  Position       : %d\n", hdr.pos);
    printf("  Prompt Length  : %d\n", hdr.prompt_len);
    printf("  Context Size   : %d\n", hdr.context_size);
    printf("  Hidden Dim     : %d\n", hdr.hidden_dim);
    printf("  Cache Type     : %s (%d)\n", cname, hdr.cache_type);
    printf("  KV Cache Size  : %.2f MB (%" PRIu64 " bytes)\n", (double)hdr.kv_cache_bytes / (1024*1024), hdr.kv_cache_bytes);
    printf("  SSM State Size : %.2f MB (%" PRIu64 " bytes)\n", (double)hdr.ssm_state_bytes / (1024*1024), hdr.ssm_state_bytes);
    printf("  SSM Conv Size  : %.2f MB (%" PRIu64 " bytes)\n", (double)hdr.ssm_conv_bytes / (1024*1024), hdr.ssm_conv_bytes);
    printf("  Next Token ID  : %d\n", hdr.next_tok);
    printf("  Checksum       : 0x%08X\n", hdr.checksum);
    return 0;
}

static void run_missing_tensor_diagnostics(const tensor_catalog *cat, const qwen_model_config *cfg) {
    printf("=== DiskLLM Missing Tensor Diagnostics ===\n");
    printf("Loaded GGUF catalog containing %d tensors.\n", cat->count);
    printf("Model Name : %s\n", cfg->model_name);
    printf("Architecture : %s\n\n", cfg->architecture);

    const char *core_tensors[] = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight"
    };
    int missing_core_count = 0;
    printf("--- Core Tensors Resolution ---\n");
    for (size_t i = 0; i < sizeof(core_tensors)/sizeof(core_tensors[0]); i++) {
        const tensor_info *ti = find_tensor(cat, core_tensors[i]);
        if (ti) {
            printf("  [FOUND]   %-32s (type %d, offset %llu)\n", core_tensors[i], ti->type, (unsigned long long)ti->absolute_offset);
        } else if (!strcmp(core_tensors[i], "output.weight") && cfg->is_tied_embedding) {
            printf("  [TIED]    %-32s (tied to token_embd.weight)\n", core_tensors[i]);
        } else {
            printf("  [MISSING] %-32s\n", core_tensors[i]);
            missing_core_count++;
        }
    }

    printf("\n--- Architecture Feature Inspection ---\n");
    printf("  Model Type           : %s\n",
           cfg->model_type == MODEL_TYPE_QWEN_HYBRID ? "MODEL_TYPE_QWEN_HYBRID" :
           cfg->model_type == MODEL_TYPE_QWEN_ATTENTION_ONLY ? "MODEL_TYPE_QWEN_ATTENTION_ONLY" :
           cfg->model_type == MODEL_TYPE_LLAMA ? "MODEL_TYPE_LLAMA" :
           cfg->model_type == MODEL_TYPE_MISTRAL ? "MODEL_TYPE_MISTRAL" : "MODEL_TYPE_UNSUPPORTED");
    printf("  Detected Block Count : %d (indices 0..%d)\n", cfg->block_count, cfg->block_count - 1);
    printf("  SSM Layers Count     : %d\n", cfg->num_ssm_layers);
    printf("  Attention Layers Count: %d\n", cfg->num_attn_layers);
    printf("  SSM Tensors Exist    : %s\n", cfg->num_ssm_layers > 0 ? "YES" : "NO");
    printf("  NextN Tensors Exist  : %s\n", cfg->has_nextn ? "YES" : "NO");

    printf("\n--- Per-Block Tensor Resolution ---\n");
    int missing_block_tensors = 0;
    for (int b = 0; b < cfg->block_count; b++) {
        char nm[256];
        if (cfg->layer_types[b] == LAYER_TYPE_SSM) {
            const char *expected_ssm[] = {
                "blk.%d.attn_norm.weight",
                "blk.%d.attn_qkv.weight",
                "blk.%d.ssm_conv1d.weight",
                "blk.%d.ssm_a",
                "blk.%d.ssm_alpha.weight",
                "blk.%d.ssm_beta.weight",
                "blk.%d.ssm_dt.bias",
                "blk.%d.ssm_norm.weight",
                "blk.%d.ssm_out.weight",
                "blk.%d.attn_gate.weight",
                "blk.%d.post_attention_norm.weight",
                "blk.%d.ffn_gate.weight",
                "blk.%d.ffn_up.weight",
                "blk.%d.ffn_down.weight"
            };
            for (size_t k = 0; k < sizeof(expected_ssm)/sizeof(expected_ssm[0]); k++) {
                snprintf(nm, sizeof(nm), expected_ssm[k], b);
                if (!find_tensor(cat, nm)) {
                    printf("  [MISSING] Block %d: %s\n", b, nm);
                    missing_block_tensors++;
                }
            }
        } else if (cfg->layer_types[b] == LAYER_TYPE_ATTENTION) {
            const char *expected_attn[] = {
                "blk.%d.attn_norm.weight",
                "blk.%d.attn_q.weight",
                "blk.%d.attn_k.weight",
                "blk.%d.attn_v.weight",
                "blk.%d.attn_output.weight",
                "blk.%d.post_attention_norm.weight",
                "blk.%d.ffn_gate.weight",
                "blk.%d.ffn_up.weight",
                "blk.%d.ffn_down.weight"
            };
            for (size_t k = 0; k < sizeof(expected_attn)/sizeof(expected_attn[0]); k++) {
                snprintf(nm, sizeof(nm), expected_attn[k], b);
                if (!find_tensor(cat, nm)) {
                    if (!strcmp(expected_attn[k], "blk.%d.post_attention_norm.weight")) {
                        char alt_nm[256];
                        snprintf(alt_nm, sizeof(alt_nm), "blk.%d.ffn_norm.weight", b);
                        if (find_tensor(cat, alt_nm)) continue;
                    }
                    printf("  [MISSING] Block %d: %s\n", b, nm);
                    missing_block_tensors++;
                }
            }
        }
    }

    printf("\n--- Diagnostic Summary ---\n");
    printf("  Total Core Tensors Missing  : %d\n", missing_core_count);
    printf("  Total Block Tensors Missing : %d\n", missing_block_tensors);
    if (missing_core_count == 0 && missing_block_tensors == 0) {
        printf("  Result: ALL EXPECTED TENSORS ARE PRESENT.\n");
    } else {
        printf("  Result: MISSING TENSORS DETECTED.\n");
    }
}

/* ─── Main Execution ────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    char    *model_path          = NULL;
    char    *arch_str            = "auto";
    char    *prompt_text         = NULL;
    char    *system_text         = NULL;
    char    *prompt_ids_str      = NULL;
    char    *prompt_ids_file     = NULL;
    char    *save_state_path     = NULL;
    char    *load_state_path     = NULL;
    char    *state_info_file     = NULL;
    char    *prefill_mode_str    = "stream";
    char    *cache_type_str      = "f16";
    int      num_threads         = 0;
    int      max_tokens          = 128;
    int      context_size        = 4096;
    int      decode_output       = 0;
    int      print_top_k         = 0;
    int      logits_summary      = 0;
    int      debug_hidden_norm   = 0;
    int      log_io              = 0;
    int      log_rss             = 0;
    int      profile_decode      = 0;
    int      warm_cache          = 0;
    char    *io_mode_str         = "pread";
    int      log_io_per_token    = 0;
    int      quiet               = 0;
    int      show_special_tokens = 0;
    int      dump_layer          = -1;
    float    repeat_penalty      = 1.0f;
    int      is_chat             = 0;
    int      lookup_id_val       = -1;
    char    *lookup_ids_arg      = NULL;
    char    *search_token_q      = NULL;
    int      print_missing_tensors_flag = 0;
    int      use_gpu             = 0;
    int      extra_stops[32]; int extra_stop_cnt = 0;

    sampler_config scfg = {
        .temperature = 0.0f,
        .top_k       = 0,
        .top_p       = 1.0f,
        .min_p       = 0.0f,
        .seed        = 0,
    };

    int      speculate_k         = 0;

    for (int i = 1; i < argc; i++) {
#define NEXTARG(dst) do { if (i+1>=argc){fprintf(stderr,"Missing arg for %s\n",argv[i]);return 1;} dst = argv[++i]; } while(0)
#define NEXTINT(dst) do { if (i+1>=argc){fprintf(stderr,"Missing arg for %s\n",argv[i]);return 1;} dst = atoi(argv[++i]); } while(0)
#define NEXTFLT(dst) do { if (i+1>=argc){fprintf(stderr,"Missing arg for %s\n",argv[i]);return 1;} dst = (float)atof(argv[++i]); } while(0)
        if      (!strcmp(argv[i],"--model"))            { NEXTARG(model_path); }
        else if (!strcmp(argv[i],"--arch"))             { NEXTARG(arch_str); }
        else if (!strcmp(argv[i],"--speculate"))        { NEXTINT(speculate_k); }
        else if (!strcmp(argv[i],"--gpu"))              { use_gpu = 1; }
        else if (!strcmp(argv[i],"--prompt"))           { NEXTARG(prompt_text); }
        else if (!strcmp(argv[i],"--system"))           { NEXTARG(system_text); }
        else if (!strcmp(argv[i],"--prompt-ids"))        { NEXTARG(prompt_ids_str); }
        else if (!strcmp(argv[i],"--prompt-ids-file"))   { NEXTARG(prompt_ids_file); }
        else if (!strcmp(argv[i],"--cache-type"))       { NEXTARG(cache_type_str); }
        else if (!strcmp(argv[i],"--save-state"))       { NEXTARG(save_state_path); }
        else if (!strcmp(argv[i],"--load-state"))       { NEXTARG(load_state_path); }
        else if (!strcmp(argv[i],"--state-info"))       { NEXTARG(state_info_file); }
        else if (!strcmp(argv[i],"--print-missing-tensors")) { print_missing_tensors_flag = 1; }
        else if (!strcmp(argv[i],"--prefill-mode"))     { NEXTARG(prefill_mode_str); }
        else if (!strcmp(argv[i],"--threads") || !strcmp(argv[i],"-t")) { NEXTINT(num_threads); }
        else if (!strcmp(argv[i],"--max-tokens"))        { NEXTINT(max_tokens); }
        else if (!strcmp(argv[i],"--context"))           { NEXTINT(context_size); }
        else if (!strcmp(argv[i],"--temp")||!strcmp(argv[i],"--temperature")) { NEXTFLT(scfg.temperature); }
        else if (!strcmp(argv[i],"--repeat-penalty")||!strcmp(argv[i],"--repetition-penalty")) { NEXTFLT(repeat_penalty); }
        else if (!strcmp(argv[i],"--top-k"))             { NEXTINT(scfg.top_k); }
        else if (!strcmp(argv[i],"--top-p"))             { NEXTFLT(scfg.top_p); }
        else if (!strcmp(argv[i],"--min-p"))             { NEXTFLT(scfg.min_p); }
        else if (!strcmp(argv[i],"--seed"))              { if(i+1<argc) scfg.seed=(uint64_t)atoll(argv[++i]); }
        else if (!strcmp(argv[i],"--stop-token"))        { if(extra_stop_cnt<32&&i+1<argc) extra_stops[extra_stop_cnt++]=atoi(argv[++i]); else i++; }
        else if (!strcmp(argv[i],"--dump-layer"))        { NEXTINT(dump_layer); }
        else if (!strcmp(argv[i],"--chat"))              { is_chat = 1; }
        else if (!strcmp(argv[i],"--quiet"))             { quiet = 1; }
        else if (!strcmp(argv[i],"--show-special-tokens")) { show_special_tokens = 1; }
        else if (!strcmp(argv[i],"--version"))           { printf("DiskLLM v1.0.0\n"); return 0; }
        else if (!strcmp(argv[i],"--lookup-id"))         { NEXTINT(lookup_id_val); }
        else if (!strcmp(argv[i],"--lookup-ids"))        { NEXTARG(lookup_ids_arg); }
        else if (!strcmp(argv[i],"--search-token"))      { NEXTARG(search_token_q); }
        else if (!strcmp(argv[i],"--find-token"))        { NEXTARG(search_token_q); }
        else if (!strcmp(argv[i],"--decode-output"))     { decode_output = 1; }
        else if (!strcmp(argv[i],"--print-top-k"))       { NEXTINT(print_top_k); }
        else if (!strcmp(argv[i],"--logits-summary"))    { logits_summary = 1; }
        else if (!strcmp(argv[i],"--debug-hidden-norm")) { debug_hidden_norm = 1; }
        else if (!strcmp(argv[i],"--log-io"))            { log_io = 1; }
        else if (!strcmp(argv[i],"--log-rss"))           { log_rss = 1; }
        else if (!strcmp(argv[i],"--profile-decode"))    { profile_decode = 1; }
        else if (!strcmp(argv[i],"--warm-cache"))        { warm_cache = 1; }
        else if (!strcmp(argv[i],"--io-mode"))          { NEXTARG(io_mode_str); }
        else if (!strcmp(argv[i],"--log-io-per-token")) { log_io_per_token = 1; }
        else if (!strcmp(argv[i],"--greedy"))            { scfg.temperature = 0.0f; }
        else if (!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")) { print_usage(argv[0]); return 0; }
        else { fprintf(stderr,"Unknown argument: %s\n", argv[i]); print_usage(argv[0]); return 1; }
#undef NEXTARG
#undef NEXTINT
#undef NEXTFLT
    }

    if (state_info_file) {
        return print_state_info(state_info_file);
    }

    (void)prefill_mode_str;
    (void)decode_output;
    (void)print_top_k;
    (void)logits_summary;
    (void)log_io;
    (void)show_special_tokens;

    if (use_gpu) {
        g_vulkan_ctx = vulkan_backend_init();
        if (!g_vulkan_ctx && !quiet) {
            fprintf(stderr, "[WARNING] Vulkan GPU backend initialization failed. Falling back to CPU.\n");
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model is required.\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ── Load catalog and dynamic config ── */
    tensor_catalog *catalog = load_tensor_catalog(model_path);
    if (!catalog) {
        fprintf(stderr, "Error: Failed to load tensor catalog for %s\n", model_path);
        return 1;
    }

    qwen_model_config *cfg = load_qwen_model_config(model_path, catalog, arch_str);
    if (!cfg) {
        fprintf(stderr, "Error: Failed to resolve model configuration for %s\n", model_path);
        free_tensor_catalog(catalog);
        return 1;
    }

    if (print_missing_tensors_flag) {
        run_missing_tensor_diagnostics(catalog, cfg);
        free_qwen_model_config(cfg);
        free_tensor_catalog(catalog);
        return 0;
    }

    if (cfg->model_type == MODEL_TYPE_UNSUPPORTED) {
        fprintf(stderr, "[ERROR] Model type unsupported: neither SSM nor Attention layers were recognized.\n");
        free_qwen_model_config(cfg);
        free_tensor_catalog(catalog);
        return 1;
    }

    /* ── Vocabulary-only commands ── */
    if (lookup_id_val >= 0) {
        char tok[1024];
        int r = lookup_token_by_id(model_path, lookup_id_val, tok, sizeof(tok));
        if (r == 0)  { printf("Token ID %d: \"%s\"\n", lookup_id_val, tok); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 0; }
        if (r == -2) { fprintf(stderr,"Token ID %d out of range.\n", lookup_id_val); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1; }
        fprintf(stderr,"Lookup failed.\n"); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1;
    }
    if (lookup_ids_arg) {
        char *dup = strdup(lookup_ids_arg);
        char *tok = strtok(dup, ",");
        while (tok) {
            int id = atoi(tok);
            char str[1024];
            if (lookup_token_by_id(model_path, id, str, sizeof(str)) == 0)
                printf("Token ID %-7d: \"%s\"\n", id, str);
            else
                printf("Token ID %-7d: [missing]\n", id);
            tok = strtok(NULL, ",");
        }
        free(dup);
        free_qwen_model_config(cfg);
        free_tensor_catalog(catalog);
        return 0;
    }
    if (search_token_q) {
        printf("Searching vocabulary for \"%s\"...\n", search_token_q);
        search_token(model_path, search_token_q);
        free_qwen_model_config(cfg);
        free_tensor_catalog(catalog);
        return 0;
    }

    /* ── Initialize Tokenizer ── */
    tokenizer *g_tok = tokenizer_init(model_path);

    /* ── Auto-Chat Prompt Formatting ── */
    if (is_chat && prompt_text) {
        size_t blen = strlen(prompt_text) + (system_text ? strlen(system_text) : 0) + 512;
        char *chat_buf = malloc(blen);
        if (chat_buf) {
            if (cfg->model_type == MODEL_TYPE_LLAMA) {
                if (system_text) {
                    snprintf(chat_buf, blen, "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", system_text, prompt_text);
                } else {
                    snprintf(chat_buf, blen, "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", prompt_text);
                }
            } else {
                if (system_text) {
                    snprintf(chat_buf, blen, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", system_text, prompt_text);
                } else {
                    snprintf(chat_buf, blen, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt_text);
                }
            }
            prompt_text = chat_buf;
        }
    }

    /* ── Parse prompt token IDs ── */
    int *prompt_tokens = NULL;
    int  prompt_len    = 0;

    if (prompt_text) {
        if (!quiet) fprintf(stderr, "[INFO] Loading GGUF tokenizer to encode --prompt...\n");
        if (!g_tok) {
            fprintf(stderr, "Error: Failed to initialize tokenizer from %s\n", model_path);
            free_qwen_model_config(cfg); free_tensor_catalog(catalog);
            return 1;
        }
        int max_prompt_toks = (int)strlen(prompt_text) * 2 + 128;
        prompt_tokens = malloc(max_prompt_toks * sizeof(int));
        int offset = 0;
        if (cfg->bos_token_id > 0 && cfg->bos_token_id < (uint32_t)cfg->vocab_size) {
            prompt_tokens[offset++] = (int)cfg->bos_token_id;
        }
        int enc_cnt = tokenizer_encode(g_tok, prompt_text, prompt_tokens + offset, max_prompt_toks - offset);
        prompt_len = offset + enc_cnt;
        if (prompt_len <= 0) {
            fprintf(stderr, "Error: Tokenizer produced 0 tokens for prompt text.\n");
            free(prompt_tokens); free_qwen_model_config(cfg); free_tensor_catalog(catalog);
            return 1;
        }
        if (!quiet) {
            fprintf(stderr, "[INFO] Tokenized --prompt into %d tokens:", prompt_len);
            for (int i = 0; i < prompt_len; i++) fprintf(stderr, " %d", prompt_tokens[i]);
            fprintf(stderr, "\n");
        }
    } else if (prompt_ids_str) {
        char *dup = strdup(prompt_ids_str);
        char *tok = strtok(dup, ",");
        while (tok) {
            int id = atoi(tok);
            if (id < 0 || id >= cfg->vocab_size) {
                fprintf(stderr,"Invalid token ID %d\n", id);
                free(dup); free(prompt_tokens); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1;
            }
            prompt_tokens = realloc(prompt_tokens, (prompt_len+1)*sizeof(int));
            prompt_tokens[prompt_len++] = id;
            tok = strtok(NULL, ",");
        }
        free(dup);
    } else if (prompt_ids_file) {
        FILE *f = fopen(prompt_ids_file, "r");
        if (!f) { fprintf(stderr,"Cannot open %s\n", prompt_ids_file); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1; }
        fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
        char *content = malloc(fsz + 1);
        if (fread(content, 1, fsz, f) != (size_t)fsz) {
            fprintf(stderr,"Read error on %s\n", prompt_ids_file); fclose(f); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1;
        }
        content[fsz] = '\0'; fclose(f);
        char *tok = strtok(content, ",\r\n\t ");
        while (tok) {
            if (strlen(tok)) {
                int id = atoi(tok);
                if (id < 0 || id >= cfg->vocab_size) {
                    fprintf(stderr,"Invalid token ID %d\n", id); free(content); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1;
                }
                prompt_tokens = realloc(prompt_tokens, (prompt_len+1)*sizeof(int));
                prompt_tokens[prompt_len++] = id;
            }
            tok = strtok(NULL, ",\r\n\t ");
        }
        free(content);
    }
    matvec_set_num_threads(num_threads);
    if (!quiet) fprintf(stderr, "[INFO] Configured %d matvec computing threads.\n", matvec_get_num_threads());

    if (!load_state_path) {
        if (prompt_len <= 0) {
            fprintf(stderr,"Error: empty prompt. Use --prompt, --prompt-ids, --prompt-ids-file, or --load-state.\n");
            free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1;
        }
        if (context_size < prompt_len + max_tokens)
            context_size = prompt_len + max_tokens;
    }

    uint32_t eos_id = cfg->eos_token_id;
    uint32_t bos_id = cfg->bos_token_id;
    if (!quiet) fprintf(stderr, "[INFO] Model EOS ID: %u, BOS ID: %u\n", eos_id, bos_id);

    scratch_buffers *scratch = allocate_scratch_buffers(cfg);
    if (!scratch) { fprintf(stderr, "Allocation failure for scratch buffers.\n"); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1; }

    io_mode_t io_mode = IO_MODE_PREAD;
    if (!strcmp(io_mode_str, "mmap")) {
        io_mode = IO_MODE_MMAP;
    } else if (!strcmp(io_mode_str, "direct")) {
        io_mode = IO_MODE_DIRECT;
    } else if (!strcmp(io_mode_str, "iouring") || !strcmp(io_mode_str, "io_uring")) {
        io_mode = IO_MODE_IOURING;
    }

    /* ── Streaming context ── */
    stream_context *sctx = init_stream_context_ex(model_path, scratch->stream_buffer, scratch->stream_buffer_size, io_mode);
    if (!sctx) { fprintf(stderr,"Failed to init stream context.\n"); free_scratch_buffers(scratch); free_qwen_model_config(cfg); free_tensor_catalog(catalog); return 1; }
    int fd = sctx->fd;

    /* ── Load output norm ── */
    uint64_t bytes_read = 0;
    const tensor_info *ti_onorm = find_tensor(catalog, "output_norm.weight");
    float *output_norm = NULL;
    if (ti_onorm) {
        output_norm = malloc(ti_onorm->byte_size);
        if (load_tensor_to_buf(fd, ti_onorm, output_norm, &bytes_read) != 0) return 1;
    }

    const tensor_info *ti_emb  = find_tensor(catalog, "token_embd.weight");
    const tensor_info *ti_outw = find_tensor(catalog, "output.weight");

    if (!ti_outw && ti_emb) {
        ti_outw = ti_emb; // Tied embedding fallback!
    }

    int missing_core = 0;
    if (!ti_emb)   { fprintf(stderr, "[ERROR] Missing core tensor: token_embd.weight\n"); missing_core++; }
    if (!ti_onorm) { fprintf(stderr, "[ERROR] Missing core tensor: output_norm.weight\n"); missing_core++; }
    if (!ti_outw)  { fprintf(stderr, "[ERROR] Missing core tensor: output.weight\n"); missing_core++; }
    if (missing_core > 0) {
        return 1;
    }

    size_t embed_row_bytes = ti_emb->byte_size / (ti_emb->dims[1] > 0 ? ti_emb->dims[1] : cfg->vocab_size);
    size_t logit_row_bytes = ti_outw->byte_size / (ti_outw->dims[1] > 0 ? ti_outw->dims[1] : cfg->vocab_size);

    int is_mmap_mode = (!strcmp(io_mode_str, "mmap"));
    const uint8_t *g_mmap_full = NULL;
    size_t g_mmap_full_size = 0;
    const uint8_t *mmap_output_weight = NULL;
    size_t mmap_output_weight_size = 0;
    void *mmap_base_ptr = NULL;

    if (is_mmap_mode) {
        off_t fsz = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        g_mmap_full_size = fsz;
        void *map_ptr = mmap(NULL, g_mmap_full_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map_ptr == MAP_FAILED) {
            fprintf(stderr, "Error: Failed to mmap full model file of size %llu bytes\n", (unsigned long long)fsz);
            return 1;
        }
        g_mmap_full = (const uint8_t *)map_ptr;
        madvise((void*)g_mmap_full, g_mmap_full_size, MADV_SEQUENTIAL);
        if (!quiet) fprintf(stderr, "[INFO] Memory-mapped FULL model file (%.2f MB)\n", (double)g_mmap_full_size / (1024*1024));
        mmap_output_weight = g_mmap_full + ti_outw->absolute_offset;
    } else if (ti_outw) {
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) page_size = 4096;
        off_t offset_aligned = (ti_outw->absolute_offset / page_size) * page_size;
        off_t offset_diff = ti_outw->absolute_offset - offset_aligned;
        mmap_output_weight_size = ti_outw->byte_size + offset_diff;
        mmap_base_ptr = mmap(NULL, mmap_output_weight_size, PROT_READ, MAP_SHARED, fd, offset_aligned);
        if (mmap_base_ptr != MAP_FAILED) {
            mmap_output_weight = (const uint8_t *)mmap_base_ptr + offset_diff;
            if (!quiet) fprintf(stderr, "[INFO] Memory-mapped output.weight tensor (%.2f MB)\n", (double)ti_outw->byte_size / (1024*1024));
        } else {
            if (!quiet) fprintf(stderr, "[INFO] mmap failed for output.weight, falling back to chunked pread streaming.\n");
        }
    }

    nextn_weights nweights = {0};
    int has_nextn_weights = 0;
    uint8_t *nextn_buf = NULL;

    if (cfg->has_nextn) {
        if (!is_mmap_mode) {
            nextn_buf = malloc(100ULL * 1024 * 1024);
        }
        if (load_nextn_weights(fd, catalog, cfg, g_mmap_full, &nweights, nextn_buf, &bytes_read) == 0) {
            has_nextn_weights = 1;
            if (!quiet) fprintf(stderr, "[INFO] Loaded NextN prediction head weights.\n");
        }
    }

    if (speculate_k > 0 && !has_nextn_weights) {
        if (!quiet) fprintf(stderr, "[WARNING] Model does not contain NextN prediction head. Speculative decoding disabled.\n");
        speculate_k = 0;
    }

    if (warm_cache) {
        if (is_mmap_mode && g_mmap_full) {
            if (!quiet) fprintf(stderr, "[INFO] Warming mmap pages using MADV_WILLNEED & page touches...\n");
            double tw0 = get_time_ms();
            madvise((void*)g_mmap_full, g_mmap_full_size, MADV_WILLNEED);
            volatile uint8_t dummy = 0;
            for (size_t i = 0; i < g_mmap_full_size; i += 4096) {
                dummy += g_mmap_full[i];
            }
            (void)dummy;
            double tw1 = get_time_ms();
            if (!quiet) fprintf(stderr, "[INFO] Warmed mmap pages in %.2f ms\n", tw1 - tw0);
        } else {
            warm_os_page_cache(model_path, fd);
        }
    }

    /* ── Sampler ── */
    sampler *smp = sampler_create(&scfg);
    if (!smp) { fprintf(stderr,"Failed to create sampler.\n"); return 1; }

    model_state    *state         = NULL;
    uint8_t        *buf_a         = NULL;
    uint8_t        *buf_b         = NULL;
    float          *hidden_states = NULL;
    float          *hidden_single = NULL;
    float          *logits        = NULL;
    int             is_loaded     = 0;
    (void)is_loaded;
    int             next_tok      = 0;

    layer_block_weights blks[2];
    uint8_t *bufs[2] = {NULL, NULL};
    int abuf = 0;
    uint64_t lb = 0;
    pthread_t pth;
    prefetch_args pargs;
    int pth_active = 0;

    float emb_scale = (cfg->model_type == MODEL_TYPE_GEMMA) ? sqrtf((float)cfg->hidden_dim) : 1.0f;
    int add_one = (cfg->model_type == MODEL_TYPE_GEMMA) ? 1 : 0;

    double t_prefill_start = get_time_ms();
    double t_prefill_end   = t_prefill_start;

    cache_type_t kv_cache_type = CACHE_TYPE_F16;
    if (!strcmp(cache_type_str, "q8_0") || !strcmp(cache_type_str, "q8")) {
        kv_cache_type = CACHE_TYPE_Q8_0;
    } else if (!strcmp(cache_type_str, "q4_0") || !strcmp(cache_type_str, "q4")) {
        kv_cache_type = CACHE_TYPE_Q4_0;
    }

    if (load_state_path) {
        FILE *sf = fopen(load_state_path, "rb");
        if (!sf) {
            fprintf(stderr, "Error: Cannot open state file %s\n", load_state_path);
            return 1;
        }
        diskllm_state_header hdr;
        if (fread(&hdr, sizeof(hdr), 1, sf) != 1 || memcmp(hdr.magic, "DKST", 4) != 0) {
            fprintf(stderr, "Error: Invalid state header in %s\n", load_state_path);
            fclose(sf);
            return 1;
        }
        if (hdr.checksum != 0x5157454E) {
            fprintf(stderr, "Error: Model checksum mismatch in state file %s\n", load_state_path);
            fclose(sf);
            return 1;
        }
        if (hdr.cache_type >= 0 && hdr.cache_type <= 2) {
            if (hdr.cache_type != (int32_t)kv_cache_type && !quiet) {
                fprintf(stderr, "[WARNING] State file cache_type (%d) differs from CLI cache_type (%d). Using state file cache_type format.\n", hdr.cache_type, (int)kv_cache_type);
            }
            kv_cache_type = (cache_type_t)hdr.cache_type;
        }
        int saved_pos = hdr.pos > 0 ? hdr.pos : hdr.prompt_len;
        prompt_len = saved_pos;
        if (context_size < prompt_len + max_tokens + 64)
            context_size = prompt_len + max_tokens + 64;

        state = allocate_model_state_ex(cfg, context_size, kv_cache_type);
        buf_a = malloc(300ULL * 1024 * 1024);
        buf_b = malloc(300ULL * 1024 * 1024);
        bufs[0] = buf_a; bufs[1] = buf_b;
        hidden_single = malloc(cfg->hidden_dim * sizeof(float));
        logits = malloc(cfg->vocab_size * sizeof(float));

        if (fread(state->kv_cache, 1, hdr.kv_cache_bytes, sf) != hdr.kv_cache_bytes ||
            fread(state->ssm_states, 1, hdr.ssm_state_bytes, sf) != hdr.ssm_state_bytes ||
            fread(state->ssm_conv_histories, 1, hdr.ssm_conv_bytes, sf) != hdr.ssm_conv_bytes ||
            fread(hidden_single, sizeof(float), cfg->hidden_dim, sf) != (size_t)cfg->hidden_dim) {
            fprintf(stderr, "Error: Failed to read state buffers from %s\n", load_state_path);
            fclose(sf);
            return 1;
        }
        next_tok = hdr.next_tok;
        fclose(sf);
        is_loaded = 1;
        t_prefill_end = get_time_ms();
        double loaded_mb = (double)(sizeof(hdr) + hdr.kv_cache_bytes + hdr.ssm_state_bytes + hdr.ssm_conv_bytes + cfg->hidden_dim * sizeof(float)) / (1024 * 1024);
        printf("[INFO] Loaded state from %s (%.2f MB, pos=%d, next_tok=%d) in %.2f ms\n",
               load_state_path, loaded_mb, prompt_len, next_tok, t_prefill_end - t_prefill_start);
    } else {
        state    = allocate_model_state_ex(cfg, context_size, kv_cache_type);
        if (!quiet) {
            const char *cnames[] = {"FP16", "Q8_0", "Q4_0"};
            fprintf(stderr, "[INFO] KV Cache format: %s (%.2f MB allocated for %d context tokens)\n",
                    cnames[state->cache_type], (double)state->kv_cache_size / (1024*1024), state->context_length);
        }
        buf_a    = malloc(300ULL * 1024 * 1024);
        buf_b    = malloc(300ULL * 1024 * 1024);
        hidden_states = malloc((size_t)prompt_len * cfg->hidden_dim * sizeof(float));
        hidden_single = malloc(cfg->hidden_dim * sizeof(float));
        logits        = malloc(cfg->vocab_size  * sizeof(float));
        if (!state || !buf_a || !buf_b || !hidden_states || !hidden_single || !logits) {
            fprintf(stderr,"Allocation failure.\n"); return 1;
        }

        /* ════════════════════════════════════════════════════════════════════════
           PREFILL PHASE
        ════════════════════════════════════════════════════════════════════════ */
        /* Embedding lookups */
        for (int pos = 0; pos < prompt_len; pos++) {
            uint8_t *row_buf = malloc(embed_row_bytes);
            if (exact_pread(fd, row_buf, embed_row_bytes,
                            ti_emb->absolute_offset + (uint64_t)prompt_tokens[pos] * embed_row_bytes)
                < (ssize_t)embed_row_bytes) { free(row_buf); return 1; }
            bytes_read += embed_row_bytes;
            float *h_vec = hidden_states + pos * cfg->hidden_dim;
            dequantize_row(row_buf, h_vec, cfg->hidden_dim, ti_emb->type);
            free(row_buf);
            if (emb_scale != 1.0f) {
                for (int i = 0; i < cfg->hidden_dim; i++) h_vec[i] *= emb_scale;
            }
        }

        /* Layer loop with double-buffered prefetch */
        bufs[0] = buf_a; bufs[1] = buf_b;
        abuf = 0; lb = 0; pth_active = 0;
        if (load_layer_block_weights(fd, catalog, cfg, 0, bufs[0], &blks[0], &lb) != 0) return 1;
        bytes_read += lb;

        for (int li = 0; li < cfg->block_count; li++) {
            if (li + 1 < cfg->block_count) {
                pargs = (prefetch_args){.fd=fd, .cat=catalog, .cfg=cfg, .layer_idx=li+1,
                                        .buf=bufs[1-abuf], .blk=&blks[1-abuf],
                                        .bytes_read=0, .status=-1};
                pth_active = (pthread_create(&pth, NULL, prefetch_thread_fn, &pargs) == 0);
                if (!pth_active) {
                    pargs.bytes_read = 0;
                    load_layer_block_weights(fd, catalog, cfg, li+1, bufs[1-abuf], &blks[1-abuf], &pargs.bytes_read);
                }
            } else { pth_active = 0; }

            layer_block_weights *blk = &blks[abuf];
            for (int pos = 0; pos < prompt_len; pos++) {
                float *h = hidden_states + pos * cfg->hidden_dim;
                if (blk->l_type == LAYER_TYPE_ATTENTION)
                    attention_forward(h, pos, li, &blk->u.attn, state, scratch, cfg);
                else
                    ssm_layer_forward(h, pos, li, &blk->u.ssm, state, scratch, cfg);

                rmsnorm_ext(scratch->hidden_state, h, blk->post_attn_norm_w, cfg->hidden_dim, 1e-6f, add_one);
                matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
                       cfg->hidden_dim, cfg->ffn_dim, blk->ffn_gate_w_type, scratch->ssm_qkv);
                matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
                       cfg->hidden_dim, cfg->ffn_dim, blk->ffn_up_w_type,   scratch->ssm_qkv);
                if (cfg->model_type == MODEL_TYPE_GEMMA) {
                    geglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
                } else {
                    swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
                }
                matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
                       cfg->ffn_dim, cfg->hidden_dim, blk->ffn_down_w_type, (float*)scratch->stream_buffer);
                add_residual(h, h, scratch->hidden_state, cfg->hidden_dim);
            }
            if (dump_layer == li || dump_layer == -2) {
                float *h_last = hidden_states + (prompt_len - 1) * cfg->hidden_dim;
                float norm = l2_norm(h_last, cfg->hidden_dim);
                printf("[DUMP-LAYER %d] pos=%d L2 norm: %.6f, first 10: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                       li, prompt_len - 1, norm,
                       h_last[0], h_last[1], h_last[2], h_last[3], h_last[4],
                       h_last[5], h_last[6], h_last[7], h_last[8], h_last[9]);
            }
            maybe_print_norm(hidden_states + (prompt_len-1)*cfg->hidden_dim, cfg->hidden_dim, li, debug_hidden_norm);

            if (pth_active) {
                pthread_join(pth, NULL);
                bytes_read += pargs.bytes_read;
            } else if (li + 1 < cfg->block_count) {
                bytes_read += pargs.bytes_read;
            }
            abuf = 1 - abuf;
        }

        /* Final norm on last position */
        float *last_h = hidden_states + (prompt_len - 1) * cfg->hidden_dim;
        rmsnorm_ext(last_h, last_h, output_norm, cfg->hidden_dim, 1e-6f, add_one);
        memcpy(hidden_single, last_h, cfg->hidden_dim * sizeof(float));

        /* Compute initial logits */
        if (mmap_output_weight) {
            matvec(logits, mmap_output_weight, last_h, cfg->hidden_dim, cfg->vocab_size, ti_outw->type, NULL);
        } else {
            int64_t rows_done = 0;
            int64_t chunk_rows = 15000;
            uint8_t *cbuf = scratch->stream_buffer;
            while (rows_done < cfg->vocab_size) {
                int64_t r = cfg->vocab_size - rows_done;
                if (r > chunk_rows) r = chunk_rows;
                if (exact_pread(fd, cbuf, r * logit_row_bytes,
                                ti_outw->absolute_offset + rows_done * logit_row_bytes)
                    < (ssize_t)(r * logit_row_bytes)) return 1;
                bytes_read += r * logit_row_bytes;
                matvec(logits + rows_done, cbuf, last_h, cfg->hidden_dim, r, ti_outw->type, NULL);
                rows_done += r;
            }
        }

        if (cfg->final_logit_softcapping > 0.0f) {
            float cap = cfg->final_logit_softcapping;
            for (int i = 0; i < cfg->vocab_size; i++) {
                logits[i] = cap * tanhf(logits[i] / cap);
            }
        }

        if (repeat_penalty > 1.0f && prompt_tokens && prompt_len > 0) {
            sampler_apply_repetition_penalty(logits, cfg->vocab_size, prompt_tokens, prompt_len, repeat_penalty);
        }
        next_tok = sampler_sample(smp, logits, cfg->vocab_size);
        t_prefill_end = get_time_ms();
    }

    long peak_rss = read_rss_mb();

    /* ════════════════════════════════════════════════════════════════════════
       AUTOREGRESSIVE GENERATION PHASE
    ════════════════════════════════════════════════════════════════════════ */
    int *gen_tokens = malloc(max_tokens * sizeof(int));
    int  gen_count  = 0;
    double t_gen_start = get_time_ms();

    long cur_rss = read_rss_mb();
    if (cur_rss > peak_rss) peak_rss = cur_rss;
    if (log_rss) printf("[RSS] after prefill: %ld MB\n", cur_rss);

    printf("\n[STREAM] ");
    if (g_tok) {
        char piece[256];
        tokenizer_decode_token(g_tok, next_tok, 1, piece, sizeof(piece));
        printf("%s", piece);
        fflush(stdout);
    }

    double total_decode_io_ms     = 0.0;
    double total_decode_attn_ms   = 0.0;
    double total_decode_ssm_ms    = 0.0;
    double total_decode_ffn_ms    = 0.0;
    double total_decode_out_io_ms = 0.0;
    double total_decode_out_cp_ms = 0.0;
    double total_decode_smp_ms    = 0.0;
    double total_decode_tok_ms    = 0.0;

    while (gen_count < max_tokens) {
        double t_tok_0 = get_time_ms();
        double tok_io_ms = 0.0;
        double tok_attn_ms = 0.0;
        double tok_ssm_ms = 0.0;
        double tok_ffn_ms = 0.0;
        double tok_out_io_ms = 0.0;
        double tok_out_cp_ms = 0.0;
        double tok_smp_ms = 0.0;

        /* Stop on EOS */
        if ((uint32_t)next_tok == eos_id) break;
        int stop = 0;
        for (int s = 0; s < extra_stop_cnt; s++)
            if (next_tok == extra_stops[s]) { stop = 1; break; }
        if (stop) break;

        gen_tokens[gen_count++] = next_tok;
        int cur_pos = prompt_len + gen_count - 1;

        /* Embedding for next_tok */
        {
            double t0 = get_time_ms();
            uint8_t *row_buf = malloc(embed_row_bytes);
            if (exact_pread(fd, row_buf, embed_row_bytes,
                            ti_emb->absolute_offset + (uint64_t)next_tok * embed_row_bytes)
                < (ssize_t)embed_row_bytes) { free(row_buf); return 1; }
            double t1 = get_time_ms();
            tok_io_ms += (t1 - t0);
            bytes_read += embed_row_bytes;
            dequantize_row(row_buf, hidden_single, cfg->hidden_dim, ti_emb->type);
            free(row_buf);
            if (emb_scale != 1.0f) {
                for (int i = 0; i < cfg->hidden_dim; i++) hidden_single[i] *= emb_scale;
            }
        }

        uint64_t phys_io_start = get_proc_self_io_read_bytes();
        uint64_t bytes_read_token_start = bytes_read;

        /* Layer loop */
        if (is_mmap_mode && g_mmap_full) {
            for (int li = 0; li < cfg->block_count; li++) {
                layer_block_weights *blk = &blks[0];
                load_layer_block_weights_mmap(catalog, cfg, li, g_mmap_full, blk);

                if (blk->l_type == LAYER_TYPE_ATTENTION) {
                    double t0 = get_time_ms();
                    attention_forward(hidden_single, cur_pos, li, &blk->u.attn, state, scratch, cfg);
                    double t1 = get_time_ms();
                    tok_attn_ms += (t1 - t0);
                } else {
                    double t0 = get_time_ms();
                    ssm_layer_forward(hidden_single, cur_pos, li, &blk->u.ssm, state, scratch, cfg);
                    double t1 = get_time_ms();
                    tok_ssm_ms += (t1 - t0);
                }

                {
                    double t0 = get_time_ms();
                    rmsnorm_ext(scratch->hidden_state, hidden_single, blk->post_attn_norm_w, cfg->hidden_dim, 1e-6f, add_one);
                    matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
                           cfg->hidden_dim, cfg->ffn_dim, blk->ffn_gate_w_type, scratch->ssm_qkv);
                    matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
                           cfg->hidden_dim, cfg->ffn_dim, blk->ffn_up_w_type,   scratch->ssm_qkv);
                    if (cfg->model_type == MODEL_TYPE_GEMMA) {
                        geglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
                    } else {
                        swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
                    }
                    matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
                           cfg->ffn_dim, cfg->hidden_dim, blk->ffn_down_w_type, (float*)scratch->stream_buffer);
                    add_residual(hidden_single, hidden_single, scratch->hidden_state, cfg->hidden_dim);
                    double t1 = get_time_ms();
                    tok_ffn_ms += (t1 - t0);
                }
            }
        } else {
            abuf = 0; lb = 0;
            {
                double t0 = get_time_ms();
                if (load_layer_block_weights(fd, catalog, cfg, 0, bufs[0], &blks[0], &lb) != 0) return 1;
                double t1 = get_time_ms();
                tok_io_ms += (t1 - t0);
                bytes_read += lb;
            }

            for (int li = 0; li < cfg->block_count; li++) {
                if (li + 1 < cfg->block_count) {
                    pargs = (prefetch_args){.fd=fd, .cat=catalog, .cfg=cfg, .layer_idx=li+1,
                                            .buf=bufs[1-abuf], .blk=&blks[1-abuf],
                                            .bytes_read=0, .status=-1};
                    pth_active = (pthread_create(&pth, NULL, prefetch_thread_fn, &pargs) == 0);
                    if (!pth_active) {
                        double t0 = get_time_ms();
                        pargs.bytes_read = 0;
                        load_layer_block_weights(fd, catalog, cfg, li+1, bufs[1-abuf], &blks[1-abuf], &pargs.bytes_read);
                        double t1 = get_time_ms();
                        tok_io_ms += (t1 - t0);
                    }
                } else { pth_active = 0; }

                layer_block_weights *blk = &blks[abuf];
                if (blk->l_type == LAYER_TYPE_ATTENTION) {
                    double t0 = get_time_ms();
                    attention_forward(hidden_single, cur_pos, li, &blk->u.attn, state, scratch, cfg);
                    double t1 = get_time_ms();
                    tok_attn_ms += (t1 - t0);
                } else {
                    double t0 = get_time_ms();
                    ssm_layer_forward(hidden_single, cur_pos, li, &blk->u.ssm, state, scratch, cfg);
                    double t1 = get_time_ms();
                    tok_ssm_ms += (t1 - t0);
                }

                {
                    double t0 = get_time_ms();
                    rmsnorm_ext(scratch->hidden_state, hidden_single, blk->post_attn_norm_w, cfg->hidden_dim, 1e-6f, add_one);
                    matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
                           cfg->hidden_dim, cfg->ffn_dim, blk->ffn_gate_w_type, scratch->ssm_qkv);
                    matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
                           cfg->hidden_dim, cfg->ffn_dim, blk->ffn_up_w_type,   scratch->ssm_qkv);
                    if (cfg->model_type == MODEL_TYPE_GEMMA) {
                        geglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
                    } else {
                        swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
                    }
                    matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
                           cfg->ffn_dim, cfg->hidden_dim, blk->ffn_down_w_type, (float*)scratch->stream_buffer);
                    add_residual(hidden_single, hidden_single, scratch->hidden_state, cfg->hidden_dim);
                    double t1 = get_time_ms();
                    tok_ffn_ms += (t1 - t0);
                }

                if (pth_active) {
                    double t0 = get_time_ms();
                    pthread_join(pth, NULL);
                    double t1 = get_time_ms();
                    tok_io_ms += (t1 - t0);
                    bytes_read += pargs.bytes_read;
                } else if (li + 1 < cfg->block_count) {
                    bytes_read += pargs.bytes_read;
                }
                abuf = 1 - abuf;
            }
        }

        rmsnorm_ext(hidden_single, hidden_single, output_norm, cfg->hidden_dim, 1e-6f, add_one);

        /* Logits */
        if (mmap_output_weight) {
            double t0 = get_time_ms();
            matvec(logits, mmap_output_weight, hidden_single, cfg->hidden_dim, cfg->vocab_size, ti_outw->type, NULL);
            double t1 = get_time_ms();
            tok_out_cp_ms += (t1 - t0);
        } else {
            int64_t rows_done = 0, chunk_rows = 15000;
            uint8_t *cbuf = scratch->stream_buffer;
            while (rows_done < cfg->vocab_size) {
                int64_t r = cfg->vocab_size - rows_done;
                if (r > chunk_rows) r = chunk_rows;
                double t0 = get_time_ms();
                if (exact_pread(fd, cbuf, r * logit_row_bytes,
                                ti_outw->absolute_offset + rows_done * logit_row_bytes)
                    < (ssize_t)(r * logit_row_bytes)) return 1;
                double t1 = get_time_ms();
                tok_out_io_ms += (t1 - t0);
                bytes_read += r * logit_row_bytes;

                double t2 = get_time_ms();
                matvec(logits + rows_done, cbuf, hidden_single, cfg->hidden_dim, r, ti_outw->type, NULL);
                double t3 = get_time_ms();
                tok_out_cp_ms += (t3 - t2);

                rows_done += r;
            }
        }

        if (cfg->final_logit_softcapping > 0.0f) {
            float cap = cfg->final_logit_softcapping;
            for (int i = 0; i < cfg->vocab_size; i++) {
                logits[i] = cap * tanhf(logits[i] / cap);
            }
        }

        {
            double t0 = get_time_ms();
            if (repeat_penalty > 1.0f) {
                int num_seen = prompt_len + gen_count;
                int *seen = malloc(num_seen * sizeof(int));
                if (seen) {
                    memcpy(seen, prompt_tokens, prompt_len * sizeof(int));
                    memcpy(seen + prompt_len, gen_tokens, gen_count * sizeof(int));
                    sampler_apply_repetition_penalty(logits, cfg->vocab_size, seen, num_seen, repeat_penalty);
                    free(seen);
                }
            }
            next_tok = sampler_sample(smp, logits, cfg->vocab_size);
            double t1 = get_time_ms();
            tok_smp_ms += (t1 - t0);
        }

        if (g_tok) {
            char piece[256];
            tokenizer_decode_token(g_tok, next_tok, 0, piece, sizeof(piece));
            printf("%s", piece);
            fflush(stdout);
        }

        cur_rss = read_rss_mb();
        if (cur_rss > peak_rss) peak_rss = cur_rss;

        double tok_total_ms = get_time_ms() - t_tok_0;
        total_decode_io_ms     += tok_io_ms;
        total_decode_attn_ms   += tok_attn_ms;
        total_decode_ssm_ms    += tok_ssm_ms;
        total_decode_ffn_ms    += tok_ffn_ms;
        total_decode_out_io_ms += tok_out_io_ms;
        total_decode_out_cp_ms += tok_out_cp_ms;
        total_decode_smp_ms    += tok_smp_ms;
        total_decode_tok_ms    += tok_total_ms;

        if (log_io_per_token) {
            uint64_t phys_io_end = get_proc_self_io_read_bytes();
            uint64_t tok_logical_bytes = bytes_read - bytes_read_token_start;
            uint64_t tok_phys_bytes = (phys_io_end >= phys_io_start) ? (phys_io_end - phys_io_start) : 0;
            double comp_ms = tok_attn_ms + tok_ssm_ms + tok_ffn_ms + tok_out_cp_ms + tok_smp_ms;
            printf("\n[IO-TOKEN %d] Logical read: %.2f MB | Physical read: %.2f MB | I/O wait: %.2f ms | Compute: %.2f ms | Total: %.2f ms\n",
                   gen_count,
                   (double)tok_logical_bytes / (1024*1024),
                   (double)tok_phys_bytes / (1024*1024),
                   tok_io_ms + tok_out_io_ms,
                   comp_ms,
                   tok_total_ms);
        }

        if (profile_decode) {
            double layer_comp = tok_attn_ms + tok_ssm_ms + tok_ffn_ms;
            double out_head   = tok_out_io_ms + tok_out_cp_ms;
            printf("\n[PROFILE Token %d]\n", gen_count);
            printf("  Total token time      : %8.2f ms\n", tok_total_ms);
            printf("  I/O wait time         : %8.2f ms (%5.1f%%)\n", tok_io_ms, 100.0 * tok_io_ms / tok_total_ms);
            printf("  Layer compute time    : %8.2f ms (%5.1f%%)\n", layer_comp, 100.0 * layer_comp / tok_total_ms);
            printf("    - Attention compute : %8.2f ms (%5.1f%%)\n", tok_attn_ms, 100.0 * tok_attn_ms / tok_total_ms);
            printf("    - SSM compute       : %8.2f ms (%5.1f%%)\n", tok_ssm_ms, 100.0 * tok_ssm_ms / tok_total_ms);
            printf("    - FFN compute       : %8.2f ms (%5.1f%%)\n", tok_ffn_ms, 100.0 * tok_ffn_ms / tok_total_ms);
            printf("  Output head time      : %8.2f ms (%5.1f%%)\n", out_head, 100.0 * out_head / tok_total_ms);
            printf("    - Output head I/O   : %8.2f ms (%5.1f%%)\n", tok_out_io_ms, 100.0 * tok_out_io_ms / tok_total_ms);
            printf("    - Output head comp  : %8.2f ms (%5.1f%%)\n", tok_out_cp_ms, 100.0 * tok_out_cp_ms / tok_total_ms);
            printf("  Sampler time          : %8.2f ms (%5.1f%%)\n", tok_smp_ms, 100.0 * tok_smp_ms / tok_total_ms);
        }
    }
    fprintf(stderr, "\n");

    if (!quiet && profile_decode && gen_count > 0) {
        double avg_tok_ms = total_decode_tok_ms / gen_count;
        double avg_io_ms  = total_decode_io_ms / gen_count;
        double avg_attn   = total_decode_attn_ms / gen_count;
        double avg_ssm    = total_decode_ssm_ms / gen_count;
        double avg_ffn    = total_decode_ffn_ms / gen_count;
        double avg_out_io = total_decode_out_io_ms / gen_count;
        double avg_out_cp = total_decode_out_cp_ms / gen_count;
        double avg_smp    = total_decode_smp_ms / gen_count;
        double avg_layer  = avg_attn + avg_ssm + avg_ffn;
        double avg_out    = avg_out_io + avg_out_cp;

        fprintf(stderr, "\n=================== DECODE PROFILING SUMMARY (%d tokens) ===================\n", gen_count);
        fprintf(stderr, "Average Token Time      : %8.2f ms\n", avg_tok_ms);
        fprintf(stderr, "  I/O Wait Time         : %8.2f ms (%5.1f%%)\n", avg_io_ms, 100.0 * avg_io_ms / avg_tok_ms);
        fprintf(stderr, "  Layer Compute Time    : %8.2f ms (%5.1f%%)\n", avg_layer, 100.0 * avg_layer / avg_tok_ms);
        fprintf(stderr, "    - Attention Compute : %8.2f ms (%5.1f%%)\n", avg_attn, 100.0 * avg_attn / avg_tok_ms);
        fprintf(stderr, "    - SSM Compute       : %8.2f ms (%5.1f%%)\n", avg_ssm, 100.0 * avg_ssm / avg_tok_ms);
        fprintf(stderr, "    - FFN Compute       : %8.2f ms (%5.1f%%)\n", avg_ffn, 100.0 * avg_ffn / avg_tok_ms);
        fprintf(stderr, "  Output Head Time      : %8.2f ms (%5.1f%%)\n", avg_out, 100.0 * avg_out / avg_tok_ms);
        fprintf(stderr, "    - Output Head I/O   : %8.2f ms (%5.1f%%)\n", avg_out_io, 100.0 * avg_out_io / avg_tok_ms);
        fprintf(stderr, "    - Output Head Compute: %8.2f ms (%5.1f%%)\n", avg_out_cp, 100.0 * avg_out_cp / avg_tok_ms);
        fprintf(stderr, "  Sampler Time          : %8.2f ms (%5.1f%%)\n", avg_smp, 100.0 * avg_smp / avg_tok_ms);
        fprintf(stderr, "============================================================================\n");
    }

    double t_gen_end = get_time_ms();

    if (save_state_path) {
        char tmp_path[512];
        snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", save_state_path);
        FILE *sf = fopen(tmp_path, "wb");
        if (sf) {
            diskllm_state_header hdr = {
                .magic = {'D', 'K', 'S', 'T'},
                .version = 1,
                .pos = prompt_len + gen_count,
                .prompt_len = prompt_len + gen_count,
                .context_size = context_size,
                .hidden_dim = cfg->hidden_dim,
                .kv_cache_bytes = state->kv_cache_size,
                .ssm_state_bytes = state->ssm_states_size,
                .ssm_conv_bytes = state->ssm_conv_histories_size,
                .next_tok = next_tok,
                .checksum = 0x5157454E,
                .cache_type = (int32_t)state->cache_type
            };
            fwrite(&hdr, sizeof(hdr), 1, sf);
            fwrite(state->kv_cache, 1, state->kv_cache_size, sf);
            fwrite(state->ssm_states, 1, state->ssm_states_size, sf);
            fwrite(state->ssm_conv_histories, 1, state->ssm_conv_histories_size, sf);
            fwrite(hidden_single, sizeof(float), cfg->hidden_dim, sf);
            fflush(sf);
            fclose(sf);
            if (rename(tmp_path, save_state_path) == 0) {
                double total_mb = (double)(sizeof(hdr) + state->kv_cache_size + state->ssm_states_size + state->ssm_conv_histories_size + cfg->hidden_dim * sizeof(float)) / (1024 * 1024);
                if (!quiet) fprintf(stderr, "[INFO] Saved multi-turn state (%.2f MB) to %s (pos=%d)\n", total_mb, save_state_path, hdr.pos);
            } else {
                fprintf(stderr, "Error: Failed to rename temporary state file %s to %s\n", tmp_path, save_state_path);
            }
        } else {
            fprintf(stderr, "Error: Could not open %s for saving state.\n", tmp_path);
        }
    }

    if (!quiet) {
        fprintf(stderr, "\n=== Execution Summary ===\n");
        fprintf(stderr, "Prompt tokens  : %d\n", prompt_len);

        if (g_tok) {
            fprintf(stderr, "Input text     : \"");
            for (int i = 0; i < prompt_len; i++) {
                char piece[256];
                tokenizer_decode_token(g_tok, prompt_tokens[i], (i == 0), piece, sizeof(piece));
                fprintf(stderr, "%s", piece);
            }
            fprintf(stderr, "\"\n");
        }

        fprintf(stderr, "Generated IDs  :");
        for (int i = 0; i < gen_count; i++) fprintf(stderr, " %d", gen_tokens[i]);
        fprintf(stderr, "\n");

        if (g_tok) {
            fprintf(stderr, "Generated text : \"");
            for (int i = 0; i < gen_count; i++) {
                char piece[256];
                tokenizer_decode_token(g_tok, gen_tokens[i], (i == 0), piece, sizeof(piece));
                fprintf(stderr, "%s", piece);
            }
            fprintf(stderr, "\"\n");
        }

        fprintf(stderr, "Token count    : %d\n", gen_count);
        fprintf(stderr, "Prefill time   : %.2f ms\n", t_prefill_end - t_prefill_start);
        fprintf(stderr, "Generation time: %.2f ms\n", t_gen_end - t_gen_start);
        if (gen_count > 0)
            fprintf(stderr, "Gen speed      : %.2f ms/tok\n",
                   (t_gen_end - t_gen_start) / gen_count);
        if (bytes_read >= (1024ULL * 1024ULL * 1024ULL)) {
            fprintf(stderr, "Bytes read     : %.2f GB (%llu bytes)\n", (double)bytes_read / (1024.0 * 1024.0 * 1024.0), (unsigned long long)bytes_read);
        } else if (bytes_read >= (1024ULL * 1024ULL)) {
            fprintf(stderr, "Bytes read     : %.2f MB (%llu bytes)\n", (double)bytes_read / (1024.0 * 1024.0), (unsigned long long)bytes_read);
        } else {
            fprintf(stderr, "Bytes read     : %llu bytes\n", (unsigned long long)bytes_read);
        }
        fprintf(stderr, "Peak RSS       : %ld MB\n", peak_rss);
    }

    /* ─── Cleanup ─────────────────────────────────────────────────────────── */
    if (g_vulkan_ctx) {
        vulkan_backend_free(g_vulkan_ctx);
        g_vulkan_ctx = NULL;
    }
    if (g_tok) tokenizer_free(g_tok);
    sampler_free(smp);
    free(gen_tokens); free(hidden_states); free(hidden_single); free(logits);
    free(output_norm); free(buf_a); free(buf_b);
    close_stream_context(sctx);
    free_model_state(state);
    free_scratch_buffers(scratch);
    free_qwen_model_config(cfg);
    free_tensor_catalog(catalog);
    free(prompt_tokens);
    return 0;
}
