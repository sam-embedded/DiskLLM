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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    int32_t  hidden_dim;         // 5120
    uint64_t kv_cache_bytes;     // kv cache size
    uint64_t ssm_state_bytes;    // ssm states size
    uint64_t ssm_conv_bytes;     // ssm conv history size
    int32_t  next_tok;           // next token ID
    uint32_t checksum;           // 0x5157454E
    int32_t  reserved[6];
} diskllm_state_header;

static int load_layer_block_weights(int fd, const tensor_catalog *cat, int li,
                                     uint8_t *buf, layer_block_weights *blk,
                                     uint64_t *ctr) {
    char nm[256];
    uint8_t *p = buf;
    blk->l_type = get_layer_type(li);

#define LOAD(field, name_fmt, type_field) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (load_tensor_to_buf(fd, _ti, p, ctr) != 0) return -1; \
    field = (const void *)p; type_field = _ti->type; p += _ti->byte_size; \
} while(0)
#define LOAD_NORM(field, name_fmt) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (load_tensor_to_buf(fd, _ti, p, ctr) != 0) return -1; \
    field = (const float *)p; p += _ti->byte_size; \
} while(0)

    if (blk->l_type == LAYER_TYPE_ATTENTION) {
        LOAD_NORM(blk->u.attn.attn_norm_w,    "blk.%d.attn_norm.weight");
        LOAD(blk->u.attn.attn_q_w,            "blk.%d.attn_q.weight",   blk->u.attn.attn_q_w_type);
        LOAD_NORM(blk->u.attn.attn_q_norm_w,  "blk.%d.attn_q_norm.weight");
        LOAD(blk->u.attn.attn_k_w,            "blk.%d.attn_k.weight",   blk->u.attn.attn_k_w_type);
        LOAD_NORM(blk->u.attn.attn_k_norm_w,  "blk.%d.attn_k_norm.weight");
        LOAD(blk->u.attn.attn_v_w,            "blk.%d.attn_v.weight",   blk->u.attn.attn_v_w_type);
        LOAD(blk->u.attn.attn_output_w,       "blk.%d.attn_output.weight", blk->u.attn.attn_output_w_type);
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
    LOAD_NORM(blk->post_attn_norm_w,          "blk.%d.post_attention_norm.weight");
    LOAD(blk->ffn_gate_w, "blk.%d.ffn_gate.weight", blk->ffn_gate_w_type);
    LOAD(blk->ffn_up_w,   "blk.%d.ffn_up.weight",   blk->ffn_up_w_type);
    LOAD(blk->ffn_down_w, "blk.%d.ffn_down.weight",  blk->ffn_down_w_type);
#undef LOAD
#undef LOAD_NORM
    return 0;
}

static int load_layer_block_weights_mmap(const tensor_catalog *cat, int li,
                                          const uint8_t *mmap_base,
                                          layer_block_weights *blk) {
    char nm[256];
    blk->l_type = get_layer_type(li);

#define LOAD_MMAP(field, name_fmt, type_field) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (!_ti) return -1; \
    field = (const void *)(mmap_base + _ti->absolute_offset); \
    type_field = _ti->type; \
} while(0)
#define LOAD_NORM_MMAP(field, name_fmt) do { \
    snprintf(nm, sizeof(nm), name_fmt, li); \
    const tensor_info *_ti = find_tensor(cat, nm); \
    if (!_ti) return -1; \
    field = (const float *)(mmap_base + _ti->absolute_offset); \
} while(0)

    if (blk->l_type == LAYER_TYPE_ATTENTION) {
        LOAD_NORM_MMAP(blk->u.attn.attn_norm_w,    "blk.%d.attn_norm.weight");
        LOAD_MMAP(blk->u.attn.attn_q_w,            "blk.%d.attn_q.weight",   blk->u.attn.attn_q_w_type);
        LOAD_NORM_MMAP(blk->u.attn.attn_q_norm_w,  "blk.%d.attn_q_norm.weight");
        LOAD_MMAP(blk->u.attn.attn_k_w,            "blk.%d.attn_k.weight",   blk->u.attn.attn_k_w_type);
        LOAD_NORM_MMAP(blk->u.attn.attn_k_norm_w,  "blk.%d.attn_k_norm.weight");
        LOAD_MMAP(blk->u.attn.attn_v_w,            "blk.%d.attn_v.weight",   blk->u.attn.attn_v_w_type);
        LOAD_MMAP(blk->u.attn.attn_output_w,       "blk.%d.attn_output.weight", blk->u.attn.attn_output_w_type);
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
    LOAD_NORM_MMAP(blk->post_attn_norm_w,          "blk.%d.post_attention_norm.weight");
    LOAD_MMAP(blk->ffn_gate_w, "blk.%d.ffn_gate.weight", blk->ffn_gate_w_type);
    LOAD_MMAP(blk->ffn_up_w,   "blk.%d.ffn_up.weight",   blk->ffn_up_w_type);
    LOAD_MMAP(blk->ffn_down_w, "blk.%d.ffn_down.weight",  blk->ffn_down_w_type);
#undef LOAD_MMAP
#undef LOAD_NORM_MMAP
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

static void maybe_print_norm(const float *h, int n, int lid, int flag) {
    if (!flag) return;
    static const int print_at[] = {-1, 0, 1, 2, 3, 4, 7, 15, 31, 63, 100};
    for (int k = 0; k < (int)(sizeof(print_at)/sizeof(print_at[0])); k++) {
        if (print_at[k] == lid) {
            if (lid == -1) printf("[NORM] emb lookup: %.6f\n", l2_norm(h, n));
            else if (lid == 100) printf("[NORM] final output_norm: %.6f\n", l2_norm(h, n));
            else printf("[NORM] after layer %d: %.6f\n", lid, l2_norm(h, n));
        }
    }
}

/* ─── Decode token to UTF-8 string ─────────────────────────────────────────── */
/* BPE artifacts: Ġ = space prefix (U+0120), Ċ = newline (U+010A), etc.
   We translate these back to their ASCII equivalents. */
static void __attribute__((unused)) decode_token_string(const char *raw, char *out, size_t out_len) {
    size_t wi = 0;
    size_t ri = 0;
    size_t raw_len = strlen(raw);
    while (ri < raw_len && wi + 1 < out_len) {
        unsigned char c = (unsigned char)raw[ri];
        if (c == 0xC4 && ri + 1 < raw_len) {
            unsigned char c2 = (unsigned char)raw[ri + 1];
            if (c2 == 0xA0) { out[wi++] = ' ';  ri += 2; continue; }  // Ġ → space
            if (c2 == 0x8A) { out[wi++] = '\n'; ri += 2; continue; }  // Ċ → newline
            if (c2 == 0x89) { out[wi++] = '\t'; ri += 2; continue; }  // ĉ → tab
        }
        out[wi++] = (char)c;
        ri++;
    }
    out[wi] = '\0';
}

/* ─── Prefetch thread ───────────────────────────────────────────────────────── */

typedef struct {
    int fd;
    const tensor_catalog *cat;
    int layer_idx;
    uint8_t *buf;
    layer_block_weights *blk;
    uint64_t bytes_read;
    int status;
} prefetch_args;

static void *prefetch_thread_fn(void *arg) {
    prefetch_args *a = (prefetch_args *)arg;
    a->status = load_layer_block_weights(a->fd, a->cat, a->layer_idx, a->buf, a->blk, &a->bytes_read);
    return NULL;
}

/* ─── Logits helpers ─────────────────────────────────────────────────────────── */

static void __attribute__((unused)) print_logits_summary(const float *logits, int n, int step) {
    float mn = logits[0], mx = logits[0];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (logits[i] < mn) mn = logits[i];
        if (logits[i] > mx) mx = logits[i];
        sum += logits[i];
    }
    double mean = sum / n;
    double ssq = 0.0;
    for (int i = 0; i < n; i++) { double d = logits[i] - mean; ssq += d*d; }
    printf("[LOGITS-SUMMARY] step %d: max=%.6f min=%.6f mean=%.6f stddev=%.6f\n",
           step, mx, mn, (float)mean, (float)sqrt(ssq / n));
}

/* ─── Usage & State Info ────────────────────────────────────────────────────── */

static int print_state_info(const char *filepath) {
    FILE *sf = fopen(filepath, "rb");
    if (!sf) {
        fprintf(stderr, "Error: Cannot open state file %s\n", filepath);
        return 1;
    }
    diskllm_state_header hdr;
    if (fread(&hdr, sizeof(hdr), 1, sf) != 1) {
        fprintf(stderr, "Error: Failed to read state header from %s\n", filepath);
        fclose(sf);
        return 1;
    }
    fclose(sf);

    if (memcmp(hdr.magic, "DKST", 4) != 0) {
        fprintf(stderr, "Error: Invalid magic '%.4s' in %s (expected 'DKST')\n", hdr.magic, filepath);
        return 1;
    }

    printf("=== DiskLLM State File Info ===\n");
    printf("File            : %s\n", filepath);
    printf("Magic           : %.4s\n", hdr.magic);
    printf("Version         : %u\n", hdr.version);
    printf("Checksum        : 0x%08X\n", hdr.checksum);
    printf("Sequence Pos    : %d\n", hdr.pos);
    printf("Prompt Length   : %d\n", hdr.prompt_len);
    printf("Context Size    : %d\n", hdr.context_size);
    printf("Hidden Dim      : %d\n", hdr.hidden_dim);
    printf("Next Token ID   : %d\n", hdr.next_tok);
    printf("KV Cache Size   : %llu bytes (%.2f MB)\n", (unsigned long long)hdr.kv_cache_bytes, (double)hdr.kv_cache_bytes / (1024*1024));
    printf("SSM State Size  : %llu bytes (%.2f MB)\n", (unsigned long long)hdr.ssm_state_bytes, (double)hdr.ssm_state_bytes / (1024*1024));
    printf("SSM Conv Size   : %llu bytes (%.2f MB)\n", (unsigned long long)hdr.ssm_conv_bytes, (double)hdr.ssm_conv_bytes / (1024*1024));
    double total_mb = (double)(sizeof(hdr) + hdr.kv_cache_bytes + hdr.ssm_state_bytes + hdr.ssm_conv_bytes + hdr.hidden_dim * sizeof(float)) / (1024 * 1024);
    printf("Total State Size: %.2f MB\n", total_mb);
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "DiskLLM v1.0.0 — Pure C Disk-Streaming LLM Inference Engine\n\n");
    fprintf(stderr, "Usage: %s --model <PATH.gguf> [options]\n\n", prog);
    fprintf(stderr, "Input Options:\n");
    fprintf(stderr, "  --prompt \"text\"              UTF-8 text prompt to encode and run\n");
    fprintf(stderr, "  --prompt-ids \"id1,id2\"       Comma-separated prompt token IDs\n");
    fprintf(stderr, "  --prompt-ids-file <FILE>     File containing prompt token IDs\n");
    fprintf(stderr, "  --system \"text\"              System prompt (used with --chat)\n");
    fprintf(stderr, "  --chat                       Apply ChatML formatting tags\n\n");
    fprintf(stderr, "Generation & Sampling Options:\n");
    fprintf(stderr, "  --max-tokens N               Max tokens to generate (default: 16)\n");
    fprintf(stderr, "  --threads N, -t N            Number of compute worker threads (default: 4)\n");
    fprintf(stderr, "  --context N                  KV cache context size (default: 256)\n");
    fprintf(stderr, "  --greedy                     Greedy decoding (default)\n");
    fprintf(stderr, "  --temp T, --temperature T    Sampling temperature (0 = greedy)\n");
    fprintf(stderr, "  --top-k K                    Top-K filtering (0 = disabled)\n");
    fprintf(stderr, "  --top-p P                    Top-P nucleus filtering (1.0 = disabled)\n");
    fprintf(stderr, "  --min-p P                    Min-P filtering (0.0 = disabled)\n");
    fprintf(stderr, "  --repeat-penalty R           Repetition penalty factor (default: 1.0)\n");
    fprintf(stderr, "  --seed N                     RNG seed for sampling\n");
    fprintf(stderr, "  --stop-token ID              Add extra stop token ID\n");
    fprintf(stderr, "  --show-special-tokens        Print special tokens during decoding\n\n");
    fprintf(stderr, "State Cache & Persistence:\n");
    fprintf(stderr, "  --save-state FILE            Save model SSM & KV state to binary file after generation\n");
    fprintf(stderr, "  --load-state FILE            Load saved SSM & KV state binary file (skips prefill)\n");
    fprintf(stderr, "  --state-info FILE            Print state file header metadata and exit\n\n");
    fprintf(stderr, "Storage & Profiling Options:\n");
    fprintf(stderr, "  --io-mode <pread|mmap>       I/O mode for model weight access (default: pread)\n");
    fprintf(stderr, "  --prefill-mode <batch|seq>   Prefill mode (default: batch)\n");
    fprintf(stderr, "  --warm-cache                 Pre-warm OS page cache before inference\n");
    fprintf(stderr, "  --profile-decode             Print fine-grained per-token decode timing breakdown\n");
    fprintf(stderr, "  --log-io-per-token           Log logical vs physical disk reads per token\n");
    fprintf(stderr, "  --log-rss                    Print resident memory usage (RSS) at key checkpoints\n");
    fprintf(stderr, "  --quiet                      Suppress informational logs; print ONLY generated text\n");
    fprintf(stderr, "  --version                    Print version info and exit\n");
    fprintf(stderr, "  -h, --help                   Show this help message\n\n");
}

/* ─── Main ───────────────────────────────────────────────────────────────────── */

#define MODEL_HIDDEN 5120
#define MODEL_FFN    17408
#define MODEL_VOCAB  248320
#define MODEL_EMBED_ROW_Q4K  2880   /* 5120/256 blocks * 144 bytes */
#define MODEL_LOGIT_ROW_Q6K  4200   /* 5120/256 blocks * 210 bytes */
#define MODEL_LAYERS 64

static void run_bench_matvec(void) {
    int in_features = 5120;
    int out_features = 17408;
    int blocks_per_row = in_features / 256;
    size_t num_blocks = (size_t)out_features * blocks_per_row;

    printf("=== Matvec Kernel Benchmark ===\n");
    printf("Shape: in_features = %d, out_features = %d\n", in_features, out_features);
    printf("FLOPs per call: %.2f MFLOPs\n\n", (2.0 * in_features * out_features) / 1e6);

    block_q4_K *w = calloc(num_blocks, sizeof(block_q4_K));
    float *x = malloc(in_features * sizeof(float));
    float *out = malloc(out_features * sizeof(float));

    if (!w || !x || !out) {
        fprintf(stderr, "Failed to allocate memory for benchmark.\n");
        return;
    }

    for (int i = 0; i < in_features; i++) x[i] = 0.01f * (i % 100);

    int thread_counts[] = {1, 2, 4, 6};
    int num_benchmarks = 4;
    int warmup = 5;
    int iterations = 20;

    for (int b = 0; b < num_benchmarks; b++) {
        int threads = thread_counts[b];
        matvec_set_num_threads(threads);

        for (int i = 0; i < warmup; i++) {
            matvec(out, w, x, in_features, out_features, GGML_TYPE_Q4_K, NULL);
        }

        double t0 = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            matvec(out, w, x, in_features, out_features, GGML_TYPE_Q4_K, NULL);
        }
        double t1 = get_time_ms();

        double total_ms = t1 - t0;
        double avg_ms = total_ms / iterations;
        double gflops = (2.0 * in_features * out_features * 1e-9) / (avg_ms * 1e-3);

        printf("  Threads: %2d | Time: %6.2f ms/call | Performance: %6.2f GFLOPS\n",
               threads, avg_ms, gflops);
    }

    free(w);
    free(x);
    free(out);
}

int main(int argc, char **argv) {
    /* ── CLI state ── */
    char    *model_path          = NULL;
    char    *prompt_text         = NULL;
    char    *system_text         = NULL;
    char    *prompt_ids_str      = NULL;
    char    *prompt_ids_file     = NULL;
    char    *save_state_path     = NULL;
    char    *load_state_path     = NULL;
    char    *state_info_file     = NULL;
    char    *prefill_mode_str    = "batch";
    int      num_threads         = 4;
    int      max_tokens          = 16;
    int      context_size        = 256;
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
    int      extra_stops[32]; int extra_stop_cnt = 0;

    sampler_config scfg = {
        .temperature = 0.0f,
        .top_k       = 0,
        .top_p       = 1.0f,
        .min_p       = 0.0f,
        .seed        = 0,
    };

    for (int i = 1; i < argc; i++) {
#define NEXTARG(dst) do { if (i+1>=argc){fprintf(stderr,"Missing arg for %s\n",argv[i]);return 1;} dst = argv[++i]; } while(0)
#define NEXTINT(dst) do { if (i+1>=argc){fprintf(stderr,"Missing arg for %s\n",argv[i]);return 1;} dst = atoi(argv[++i]); } while(0)
#define NEXTFLT(dst) do { if (i+1>=argc){fprintf(stderr,"Missing arg for %s\n",argv[i]);return 1;} dst = (float)atof(argv[++i]); } while(0)
        if      (!strcmp(argv[i],"--model"))            { NEXTARG(model_path); }
        else if (!strcmp(argv[i],"--prompt"))           { NEXTARG(prompt_text); }
        else if (!strcmp(argv[i],"--system"))           { NEXTARG(system_text); }
        else if (!strcmp(argv[i],"--prompt-ids"))        { NEXTARG(prompt_ids_str); }
        else if (!strcmp(argv[i],"--prompt-ids-file"))   { NEXTARG(prompt_ids_file); }
        else if (!strcmp(argv[i],"--save-state"))       { NEXTARG(save_state_path); }
        else if (!strcmp(argv[i],"--load-state"))       { NEXTARG(load_state_path); }
        else if (!strcmp(argv[i],"--state-info"))       { NEXTARG(state_info_file); }
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
        else if (!strcmp(argv[i],"--bench-matvec"))     {
            run_bench_matvec();
            return 0;
        }
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

    if (!model_path) {
        fprintf(stderr, "Error: --model is required.\n");
        print_usage(argv[0]);
        return 1;
    }

    /* ── Vocabulary-only commands ── */
    if (lookup_id_val >= 0) {
        char tok[1024];
        int r = lookup_token_by_id(model_path, lookup_id_val, tok, sizeof(tok));
        if (r == 0)  { printf("Token ID %d: \"%s\"\n", lookup_id_val, tok); return 0; }
        if (r == -2) { fprintf(stderr,"Token ID %d out of range.\n", lookup_id_val); return 1; }
        fprintf(stderr,"Lookup failed.\n"); return 1;
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
        return 0;
    }
    if (search_token_q) {
        printf("Searching vocabulary for \"%s\"...\n", search_token_q);
        search_token(model_path, search_token_q);
        return 0;
    }

    /* ── Initialize Tokenizer ── */
    tokenizer *g_tok = tokenizer_init(model_path);

    /* ── Auto-Chat Prompt Formatting ── */
    if (is_chat && prompt_text) {
        size_t blen = strlen(prompt_text) + (system_text ? strlen(system_text) : 0) + 512;
        char *chat_buf = malloc(blen);
        if (chat_buf) {
            if (system_text) {
                snprintf(chat_buf, blen, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", system_text, prompt_text);
            } else {
                snprintf(chat_buf, blen, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt_text);
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
            return 1;
        }
        int max_prompt_toks = (int)strlen(prompt_text) * 2 + 128;
        prompt_tokens = malloc(max_prompt_toks * sizeof(int));
        prompt_len = tokenizer_encode(g_tok, prompt_text, prompt_tokens, max_prompt_toks);
        if (prompt_len <= 0) {
            fprintf(stderr, "Error: Tokenizer produced 0 tokens for prompt text.\n");
            free(prompt_tokens);
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
            if (id < 0 || id >= MODEL_VOCAB) {
                fprintf(stderr,"Invalid token ID %d\n", id);
                free(dup); free(prompt_tokens); return 1;
            }
            prompt_tokens = realloc(prompt_tokens, (prompt_len+1)*sizeof(int));
            prompt_tokens[prompt_len++] = id;
            tok = strtok(NULL, ",");
        }
        free(dup);
    } else if (prompt_ids_file) {
        FILE *f = fopen(prompt_ids_file, "r");
        if (!f) { fprintf(stderr,"Cannot open %s\n", prompt_ids_file); return 1; }
        fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
        char *content = malloc(fsz + 1);
        if (fread(content, 1, fsz, f) != (size_t)fsz) {
            fprintf(stderr,"Read error on %s\n", prompt_ids_file); fclose(f); return 1;
        }
        content[fsz] = '\0'; fclose(f);
        char *tok = strtok(content, ",\r\n\t ");
        while (tok) {
            if (strlen(tok)) {
                int id = atoi(tok);
                if (id < 0 || id >= MODEL_VOCAB) {
                    fprintf(stderr,"Invalid token ID %d\n", id); free(content); return 1;
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
            fprintf(stderr,"Error: empty prompt. Use --prompt, --prompt-ids, --prompt-ids-file, or --load-state.\n"); return 1;
        }
        if (context_size < prompt_len + max_tokens)
            context_size = prompt_len + max_tokens;
    }

    /* ── Read EOS/BOS from GGUF metadata ── */
    uint32_t eos_id = 248046, bos_id = 248044;
    get_metadata_uint32(model_path, "tokenizer.ggml.eos_token_id", &eos_id);
    get_metadata_uint32(model_path, "tokenizer.ggml.bos_token_id", &bos_id);
    if (!quiet) fprintf(stderr, "[INFO] EOS token ID: %u\n", eos_id);

    /* ── Load tensor catalog ── */
    tensor_catalog *catalog = load_tensor_catalog(model_path);
    if (!catalog) { fprintf(stderr,"Failed to load tensor catalog.\n"); return 1; }

    scratch_buffers *scratch = allocate_scratch_buffers();
    if (!scratch) { fprintf(stderr, "Allocation failure for scratch buffers.\n"); return 1; }

    /* ── Streaming context ── */
    stream_context *sctx = init_stream_context(model_path, scratch->stream_buffer, scratch->stream_buffer_size);
    if (!sctx) { fprintf(stderr,"Failed to init stream context.\n"); return 1; }
    int fd = sctx->fd;

    /* ── Load output norm ── */
    const tensor_info *ti_onorm = find_tensor(catalog, "output_norm.weight");
    float *output_norm = malloc(ti_onorm->byte_size);
    uint64_t bytes_read = 0;
    if (load_tensor_to_buf(fd, ti_onorm, output_norm, &bytes_read) != 0) return 1;

    const tensor_info *ti_emb  = find_tensor(catalog, "token_embd.weight");
    const tensor_info *ti_outw = find_tensor(catalog, "output.weight");
    if (!ti_emb || !ti_outw) { fprintf(stderr,"Core tensors missing.\n"); return 1; }

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

    double t_prefill_start = get_time_ms();
    double t_prefill_end   = t_prefill_start;

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
        int saved_pos = hdr.pos > 0 ? hdr.pos : hdr.prompt_len;
        prompt_len = saved_pos;
        if (context_size < prompt_len + max_tokens + 64)
            context_size = prompt_len + max_tokens + 64;

        state = allocate_model_state(context_size);
        buf_a = malloc(300ULL * 1024 * 1024);
        buf_b = malloc(300ULL * 1024 * 1024);
        bufs[0] = buf_a; bufs[1] = buf_b;
        hidden_single = malloc(MODEL_HIDDEN * sizeof(float));
        logits = malloc(MODEL_VOCAB * sizeof(float));

        if (fread(state->kv_cache, 1, hdr.kv_cache_bytes, sf) != hdr.kv_cache_bytes ||
            fread(state->ssm_states, 1, hdr.ssm_state_bytes, sf) != hdr.ssm_state_bytes ||
            fread(state->ssm_conv_histories, 1, hdr.ssm_conv_bytes, sf) != hdr.ssm_conv_bytes ||
            fread(hidden_single, sizeof(float), MODEL_HIDDEN, sf) != (size_t)MODEL_HIDDEN) {
            fprintf(stderr, "Error: Failed to read state buffers from %s\n", load_state_path);
            fclose(sf);
            return 1;
        }
        next_tok = hdr.next_tok;
        fclose(sf);
        is_loaded = 1;
        t_prefill_end = get_time_ms();
        double loaded_mb = (double)(sizeof(hdr) + hdr.kv_cache_bytes + hdr.ssm_state_bytes + hdr.ssm_conv_bytes + MODEL_HIDDEN * sizeof(float)) / (1024 * 1024);
        printf("[INFO] Loaded state from %s (%.2f MB, pos=%d, next_tok=%d) in %.2f ms\n",
               load_state_path, loaded_mb, prompt_len, next_tok, t_prefill_end - t_prefill_start);
    } else {
        state    = allocate_model_state(context_size);
        buf_a    = malloc(300ULL * 1024 * 1024);
        buf_b    = malloc(300ULL * 1024 * 1024);
        hidden_states = malloc((size_t)prompt_len * MODEL_HIDDEN * sizeof(float));
        hidden_single = malloc(MODEL_HIDDEN * sizeof(float));
        logits        = malloc(MODEL_VOCAB  * sizeof(float));
        if (!state || !buf_a || !buf_b || !hidden_states || !hidden_single || !logits) {
            fprintf(stderr,"Allocation failure.\n"); return 1;
        }

        /* ════════════════════════════════════════════════════════════════════════
           PREFILL PHASE
        ════════════════════════════════════════════════════════════════════════ */
        if (log_rss) printf("[RSS] before prefill: %ld MB\n", read_rss_mb());

        /* Embedding lookups */
        for (int pos = 0; pos < prompt_len; pos++) {
            uint8_t row_buf[MODEL_EMBED_ROW_Q4K];
            if (exact_pread(fd, row_buf, MODEL_EMBED_ROW_Q4K,
                            ti_emb->absolute_offset + (uint64_t)prompt_tokens[pos] * MODEL_EMBED_ROW_Q4K)
                < MODEL_EMBED_ROW_Q4K) return 1;
            bytes_read += MODEL_EMBED_ROW_Q4K;
            dequantize_q4_K(row_buf, hidden_states + pos * MODEL_HIDDEN, MODEL_HIDDEN);
        }

        /* Layer loop with double-buffered prefetch */
        bufs[0] = buf_a; bufs[1] = buf_b;
        abuf = 0; lb = 0; pth_active = 0;
        if (load_layer_block_weights(fd, catalog, 0, bufs[0], &blks[0], &lb) != 0) return 1;
        bytes_read += lb;

        for (int li = 0; li < MODEL_LAYERS; li++) {
            if (li + 1 < MODEL_LAYERS) {
                pargs = (prefetch_args){.fd=fd, .cat=catalog, .layer_idx=li+1,
                                        .buf=bufs[1-abuf], .blk=&blks[1-abuf],
                                        .bytes_read=0, .status=-1};
                pth_active = (pthread_create(&pth, NULL, prefetch_thread_fn, &pargs) == 0);
                if (!pth_active) {
                    pargs.bytes_read = 0;
                    load_layer_block_weights(fd, catalog, li+1, bufs[1-abuf], &blks[1-abuf], &pargs.bytes_read);
                }
            } else { pth_active = 0; }

            layer_block_weights *blk = &blks[abuf];
            for (int pos = 0; pos < prompt_len; pos++) {
                float *h = hidden_states + pos * MODEL_HIDDEN;
                if (blk->l_type == LAYER_TYPE_ATTENTION)
                    attention_forward(h, pos, li, &blk->u.attn, state, scratch, 10000000.0, 64);
                else
                    ssm_layer_forward(h, pos, li, &blk->u.ssm, state, scratch);

                rmsnorm(scratch->hidden_state, h, blk->post_attn_norm_w, MODEL_HIDDEN, 1e-6f);
                matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
                       MODEL_HIDDEN, MODEL_FFN, blk->ffn_gate_w_type, scratch->ssm_qkv);
                matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
                       MODEL_HIDDEN, MODEL_FFN, blk->ffn_up_w_type,   scratch->ssm_qkv);
                swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, MODEL_FFN);
                matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
                       MODEL_FFN, MODEL_HIDDEN, blk->ffn_down_w_type, (float*)scratch->stream_buffer);
                add_residual(h, h, scratch->hidden_state, MODEL_HIDDEN);
            }
            if (dump_layer == li || dump_layer == -2) {
                float *h_last = hidden_states + (prompt_len - 1) * MODEL_HIDDEN;
                float norm = l2_norm(h_last, MODEL_HIDDEN);
                printf("[DUMP-LAYER %d] pos=%d L2 norm: %.6f, first 10: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
                       li, prompt_len - 1, norm,
                       h_last[0], h_last[1], h_last[2], h_last[3], h_last[4],
                       h_last[5], h_last[6], h_last[7], h_last[8], h_last[9]);
            }
            maybe_print_norm(hidden_states + (prompt_len-1)*MODEL_HIDDEN, MODEL_HIDDEN, li, debug_hidden_norm);

            if (pth_active) {
                pthread_join(pth, NULL);
                bytes_read += pargs.bytes_read;
            } else if (li + 1 < MODEL_LAYERS) {
                bytes_read += pargs.bytes_read;
            }
            abuf = 1 - abuf;
        }

        /* Final norm on last position */
        float *last_h = hidden_states + (prompt_len - 1) * MODEL_HIDDEN;
        rmsnorm(last_h, last_h, output_norm, MODEL_HIDDEN, 1e-6f);
        memcpy(hidden_single, last_h, MODEL_HIDDEN * sizeof(float));

        /* Compute initial logits */
        if (mmap_output_weight) {
            matvec(logits, mmap_output_weight, last_h, MODEL_HIDDEN, MODEL_VOCAB, GGML_TYPE_Q6_K, NULL);
        } else {
            int64_t rows_done = 0;
            int64_t chunk_rows = 15000;
            uint8_t *cbuf = scratch->stream_buffer;
            while (rows_done < MODEL_VOCAB) {
                int64_t r = MODEL_VOCAB - rows_done;
                if (r > chunk_rows) r = chunk_rows;
                if (exact_pread(fd, cbuf, r * MODEL_LOGIT_ROW_Q6K,
                                ti_outw->absolute_offset + rows_done * MODEL_LOGIT_ROW_Q6K)
                    < r * MODEL_LOGIT_ROW_Q6K) return 1;
                bytes_read += r * MODEL_LOGIT_ROW_Q6K;
                matvec(logits + rows_done, cbuf, last_h, MODEL_HIDDEN, r, GGML_TYPE_Q6_K, NULL);
                rows_done += r;
            }
        }

        if (repeat_penalty > 1.0f && prompt_tokens && prompt_len > 0) {
            sampler_apply_repetition_penalty(logits, MODEL_VOCAB, prompt_tokens, prompt_len, repeat_penalty);
        }
        next_tok = sampler_sample(smp, logits, MODEL_VOCAB);
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
            uint8_t row_buf[MODEL_EMBED_ROW_Q4K];
            if (exact_pread(fd, row_buf, MODEL_EMBED_ROW_Q4K,
                            ti_emb->absolute_offset + (uint64_t)next_tok * MODEL_EMBED_ROW_Q4K)
                < MODEL_EMBED_ROW_Q4K) return 1;
            double t1 = get_time_ms();
            tok_io_ms += (t1 - t0);
            bytes_read += MODEL_EMBED_ROW_Q4K;
            dequantize_q4_K(row_buf, hidden_single, MODEL_HIDDEN);
        }

        uint64_t phys_io_start = get_proc_self_io_read_bytes();
        uint64_t bytes_read_token_start = bytes_read;

        /* Layer loop */
        if (is_mmap_mode && g_mmap_full) {
            for (int li = 0; li < MODEL_LAYERS; li++) {
                layer_block_weights *blk = &blks[0];
                load_layer_block_weights_mmap(catalog, li, g_mmap_full, blk);

                if (blk->l_type == LAYER_TYPE_ATTENTION) {
                    double t0 = get_time_ms();
                    attention_forward(hidden_single, cur_pos, li, &blk->u.attn, state, scratch, 10000000.0, 64);
                    double t1 = get_time_ms();
                    tok_attn_ms += (t1 - t0);
                } else {
                    double t0 = get_time_ms();
                    ssm_layer_forward(hidden_single, cur_pos, li, &blk->u.ssm, state, scratch);
                    double t1 = get_time_ms();
                    tok_ssm_ms += (t1 - t0);
                }

                {
                    double t0 = get_time_ms();
                    rmsnorm(scratch->hidden_state, hidden_single, blk->post_attn_norm_w, MODEL_HIDDEN, 1e-6f);
                    matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
                           MODEL_HIDDEN, MODEL_FFN, blk->ffn_gate_w_type, scratch->ssm_qkv);
                    matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
                           MODEL_HIDDEN, MODEL_FFN, blk->ffn_up_w_type,   scratch->ssm_qkv);
                    swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, MODEL_FFN);
                    matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
                           MODEL_FFN, MODEL_HIDDEN, blk->ffn_down_w_type, (float*)scratch->stream_buffer);
                    add_residual(hidden_single, hidden_single, scratch->hidden_state, MODEL_HIDDEN);
                    double t1 = get_time_ms();
                    tok_ffn_ms += (t1 - t0);
                }
            }
        } else {
            abuf = 0; lb = 0;
            {
                double t0 = get_time_ms();
                if (load_layer_block_weights(fd, catalog, 0, bufs[0], &blks[0], &lb) != 0) return 1;
                double t1 = get_time_ms();
                tok_io_ms += (t1 - t0);
                bytes_read += lb;
            }

            for (int li = 0; li < MODEL_LAYERS; li++) {
                if (li + 1 < MODEL_LAYERS) {
                    pargs = (prefetch_args){.fd=fd, .cat=catalog, .layer_idx=li+1,
                                            .buf=bufs[1-abuf], .blk=&blks[1-abuf],
                                            .bytes_read=0, .status=-1};
                    pth_active = (pthread_create(&pth, NULL, prefetch_thread_fn, &pargs) == 0);
                    if (!pth_active) {
                        double t0 = get_time_ms();
                        pargs.bytes_read = 0;
                        load_layer_block_weights(fd, catalog, li+1, bufs[1-abuf], &blks[1-abuf], &pargs.bytes_read);
                        double t1 = get_time_ms();
                        tok_io_ms += (t1 - t0);
                    }
                } else { pth_active = 0; }

                layer_block_weights *blk = &blks[abuf];
                if (blk->l_type == LAYER_TYPE_ATTENTION) {
                    double t0 = get_time_ms();
                    attention_forward(hidden_single, cur_pos, li, &blk->u.attn, state, scratch, 10000000.0, 64);
                    double t1 = get_time_ms();
                    tok_attn_ms += (t1 - t0);
                } else {
                    double t0 = get_time_ms();
                    ssm_layer_forward(hidden_single, cur_pos, li, &blk->u.ssm, state, scratch);
                    double t1 = get_time_ms();
                    tok_ssm_ms += (t1 - t0);
                }

                {
                    double t0 = get_time_ms();
                    rmsnorm(scratch->hidden_state, hidden_single, blk->post_attn_norm_w, MODEL_HIDDEN, 1e-6f);
                    matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
                           MODEL_HIDDEN, MODEL_FFN, blk->ffn_gate_w_type, scratch->ssm_qkv);
                    matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
                           MODEL_HIDDEN, MODEL_FFN, blk->ffn_up_w_type,   scratch->ssm_qkv);
                    swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, MODEL_FFN);
                    matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
                           MODEL_FFN, MODEL_HIDDEN, blk->ffn_down_w_type, (float*)scratch->stream_buffer);
                    add_residual(hidden_single, hidden_single, scratch->hidden_state, MODEL_HIDDEN);
                    double t1 = get_time_ms();
                    tok_ffn_ms += (t1 - t0);
                }

                if (pth_active) {
                    double t0 = get_time_ms();
                    pthread_join(pth, NULL);
                    double t1 = get_time_ms();
                    tok_io_ms += (t1 - t0);
                    bytes_read += pargs.bytes_read;
                } else if (li + 1 < MODEL_LAYERS) {
                    bytes_read += pargs.bytes_read;
                }
                abuf = 1 - abuf;
            }
        }

        rmsnorm(hidden_single, hidden_single, output_norm, MODEL_HIDDEN, 1e-6f);

        /* Logits */
        if (mmap_output_weight) {
            double t0 = get_time_ms();
            matvec(logits, mmap_output_weight, hidden_single, MODEL_HIDDEN, MODEL_VOCAB, GGML_TYPE_Q6_K, NULL);
            double t1 = get_time_ms();
            tok_out_cp_ms += (t1 - t0);
        } else {
            int64_t rows_done = 0, chunk_rows = 15000;
            uint8_t *cbuf = scratch->stream_buffer;
            while (rows_done < MODEL_VOCAB) {
                int64_t r = MODEL_VOCAB - rows_done;
                if (r > chunk_rows) r = chunk_rows;
                double t0 = get_time_ms();
                if (exact_pread(fd, cbuf, r * MODEL_LOGIT_ROW_Q6K,
                                ti_outw->absolute_offset + rows_done * MODEL_LOGIT_ROW_Q6K)
                    < r * MODEL_LOGIT_ROW_Q6K) return 1;
                double t1 = get_time_ms();
                tok_out_io_ms += (t1 - t0);
                bytes_read += r * MODEL_LOGIT_ROW_Q6K;

                double t2 = get_time_ms();
                matvec(logits + rows_done, cbuf, hidden_single, MODEL_HIDDEN, r, GGML_TYPE_Q6_K, NULL);
                double t3 = get_time_ms();
                tok_out_cp_ms += (t3 - t2);

                rows_done += r;
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
                    sampler_apply_repetition_penalty(logits, MODEL_VOCAB, seen, num_seen, repeat_penalty);
                    free(seen);
                }
            }
            next_tok = sampler_sample(smp, logits, MODEL_VOCAB);
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
                .hidden_dim = MODEL_HIDDEN,
                .kv_cache_bytes = state->kv_cache_size,
                .ssm_state_bytes = state->ssm_states_size,
                .ssm_conv_bytes = state->ssm_conv_histories_size,
                .next_tok = next_tok,
                .checksum = 0x5157454E
            };
            fwrite(&hdr, sizeof(hdr), 1, sf);
            fwrite(state->kv_cache, 1, state->kv_cache_size, sf);
            fwrite(state->ssm_states, 1, state->ssm_states_size, sf);
            fwrite(state->ssm_conv_histories, 1, state->ssm_conv_histories_size, sf);
            fwrite(hidden_single, sizeof(float), MODEL_HIDDEN, sf);
            fflush(sf);
            fclose(sf);
            if (rename(tmp_path, save_state_path) == 0) {
                double total_mb = (double)(sizeof(hdr) + state->kv_cache_size + state->ssm_states_size + state->ssm_conv_histories_size + MODEL_HIDDEN * sizeof(float)) / (1024 * 1024);
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
        fprintf(stderr, "Bytes read     : %llu\n", (unsigned long long)bytes_read);
        fprintf(stderr, "Peak RSS       : %ld MB\n", peak_rss);
    }

    /* ─── Cleanup ─────────────────────────────────────────────────────────── */
    if (g_tok) tokenizer_free(g_tok);
    sampler_free(smp);
    free(gen_tokens); free(hidden_states); free(hidden_single); free(logits);
    free(output_norm); free(buf_a); free(buf_b);
    close_stream_context(sctx);
    free_model_state(state);
    free_scratch_buffers(scratch);
    free_tensor_catalog(catalog);
    free(prompt_tokens);
    return 0;
}
