#ifndef DISKLLM_H
#define DISKLLM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Opaque Data Types ─────────────────────────────────────────────────────── */

typedef struct diskllm_model diskllm_model;
typedef struct diskllm_context diskllm_context;
typedef struct diskllm_sampler diskllm_sampler;
typedef struct diskllm_tokenizer diskllm_tokenizer;

/* ─── Parameter Structures ────────────────────────────────────────────────── */

typedef enum {
    DISKLLM_CACHE_F16 = 0,
    DISKLLM_CACHE_Q8_0 = 1,
    DISKLLM_CACHE_Q4_0 = 2
} diskllm_cache_type;

typedef enum {
    DISKLLM_IO_PREAD = 0,
    DISKLLM_IO_MMAP = 1,
    DISKLLM_IO_DIRECT = 2,
    DISKLLM_IO_IOURING = 3
} diskllm_io_mode;

typedef struct {
    const char *arch_flag;       /* "auto", "llama", "phi3", "mistral", "gemma", "qwen-hybrid" */
    bool        pin_weights;     /* Pin entire model into physical RAM (zero decode I/O) */
    diskllm_io_mode io_mode;     /* I/O engine mode */
    int         num_threads;     /* Computing threads (0 = auto) */
    bool        quiet;           /* Suppress verbose logs */
} diskllm_model_params;

typedef struct {
    int                context_size;   /* Max context window tokens */
    diskllm_cache_type cache_type;     /* KV cache format */
    int                num_threads;    /* Computing threads */
} diskllm_context_params;

typedef struct {
    float temp;                 /* Temperature (0.0 = greedy) */
    float top_p;                /* Top-P (nucleus) sampling */
    int   top_k;                /* Top-K sampling */
    float min_p;                /* Min-P sampling */
    float repeat_penalty;       /* Repetition penalty (1.0 = disabled) */
    float presence_penalty;     /* Presence penalty (0.0 = disabled) */
    uint64_t seed;              /* Random seed */
} diskllm_sampler_params;

typedef struct {
    double   prefill_time_ms;
    double   generation_time_ms;
    int      prompt_tokens;
    int      gen_tokens;
    double   gen_speed_ms_per_tok;
    uint64_t bytes_read_total;
    uint64_t bytes_read_decode;
    long     peak_rss_mb;
    const char *weights_mode_str;
} diskllm_perf_metrics;

/* ─── Default Parameters Constructors ─────────────────────────────────────── */

diskllm_model_params   diskllm_model_params_default(void);
diskllm_context_params diskllm_context_params_default(void);
diskllm_sampler_params diskllm_sampler_params_default(void);

/* ─── Model Management API ─────────────────────────────────────────────────── */

diskllm_model *diskllm_model_load(const char *model_path, diskllm_model_params params);
void           diskllm_model_free(diskllm_model *model);

int            diskllm_model_get_vocab_size(const diskllm_model *model);
uint32_t       diskllm_model_get_eos_id(const diskllm_model *model);
uint32_t       diskllm_model_get_bos_id(const diskllm_model *model);
const char    *diskllm_model_get_arch_name(const diskllm_model *model);
int            diskllm_model_get_hidden_dim(const diskllm_model *model);
int            diskllm_model_get_block_count(const diskllm_model *model);

/* ─── Tokenizer API ────────────────────────────────────────────────────────── */

diskllm_tokenizer *diskllm_model_get_tokenizer(diskllm_model *model);
int                diskllm_tokenize(const diskllm_tokenizer *tok, const char *text, int *out_tokens, int max_tokens, bool add_bos);
int                diskllm_decode_token(const diskllm_tokenizer *tok, int token, bool is_first, char *buf, size_t buf_size);
char              *diskllm_format_chat_prompt(const diskllm_model *model, const char *system_prompt, const char *user_prompt);
char              *diskllm_format_chat_prompt_ex(const diskllm_model *model, const char *system_prompt, const char *user_prompt, bool enable_thinking);
char              *diskllm_format_agent_prompt(const diskllm_model *model, const char *tools_json, const char *system_instructions, const char *user_prompt, bool enable_thinking);
char              *diskllm_format_image_prompt(const diskllm_model *model, const char *image_path_or_desc, const char *user_prompt, bool enable_thinking);
char              *diskllm_format_tool_response(const diskllm_model *model, const char *tool_output_json);
char              *diskllm_format_fim_prompt(const diskllm_model *model, const char *prefix, const char *suffix, const char *repo_name, const char *file_name);
char              *diskllm_strip_think_tags(const char *text);

/* ─── Context & Inference API ──────────────────────────────────────────────── */

diskllm_context *diskllm_context_init(diskllm_model *model, diskllm_context_params params);
void             diskllm_context_free(diskllm_context *ctx);

/* Prefill prompt tokens, returns 0 on success. Logits buffer size must be vocab_size. */
int              diskllm_eval(diskllm_context *ctx, const int *tokens, int n_tokens, float *logits);

/* Prefill multimodal prompt (text tokens + visual patch embeddings injected at image_pad position). */
int              diskllm_eval_multimodal(diskllm_context *ctx, const int *tokens, int n_tokens, int img_pad_pos, const float *visual_embeddings, int n_patches, float *logits);

/* Run single autoregressive decode step for token, returns 0 on success. */
int              diskllm_decode_step(diskllm_context *ctx, int token, float *logits);

/* Get performance metrics for execution summary */
diskllm_perf_metrics diskllm_get_perf_metrics(const diskllm_context *ctx);

/* ─── Sampler API ──────────────────────────────────────────────────────────── */

diskllm_sampler *diskllm_sampler_init(diskllm_sampler_params params);
void             diskllm_sampler_free(diskllm_sampler *smp);
int              diskllm_sample(diskllm_sampler *smp, float *logits, int vocab_size, const int *seen_tokens, int n_seen);

/* ─── State Persistence API ────────────────────────────────────────────────── */

int              diskllm_save_state(diskllm_context *ctx, const char *filepath);
int              diskllm_load_state(diskllm_context *ctx, const char *filepath);
int              diskllm_print_state_info(const char *filepath);

#ifdef __cplusplus
}
#endif

#endif /* DISKLLM_H */
