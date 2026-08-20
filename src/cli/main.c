#include "diskllm.h"
#include "diskllm_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <unistd.h>

static void print_usage(const char *prog_name) {
    printf("DiskLLM CLI — High Performance Edge LLM Engine\n\n");
    printf("Usage: %s --model <path> [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  --model <path>              Path to GGUF model file (required)\n");
    printf("  --arch <type>               Architecture hint (auto, llama, phi3, mistral, gemma, qwen-hybrid)\n");
    printf("  --prompt <text>             Input text prompt\n");
    printf("  --system <text>             System prompt (for --chat)\n");
    printf("  --prompt-ids <id1,id2,...>  Comma-separated token IDs\n");
    printf("  --prompt-ids-file <path>    Path to text file containing token IDs\n");
    printf("  --max-tokens <int>          Max generated tokens (default: 128)\n");
    printf("  --ctx <int>                 Context capacity (default: 4096)\n");
    printf("  --threads <int>             Matvec computing threads (default: CPU cores)\n");
    printf("  --temp <float>              Sampling temperature (default: 0.7)\n");
    printf("  --top-p <float>             Top-P nucleus threshold (default: 0.9)\n");
    printf("  --top-k <int>               Top-K threshold (default: 40)\n");
    printf("  --min-p <float>             Min-P threshold (default: 0.05)\n");
    printf("  --repeat-penalty <float>    Repetition penalty (default: 1.0)\n");
    printf("  --greedy                    Greedy sampling (temp=0.0)\n");
    printf("  --chat                      Auto-wrap prompt in Chat template\n");
    printf("  --interactive, -i           Interactive multi-turn REPL chat mode\n");
    printf("  --pin-weights               Pin model into RAM for zero decode I/O\n");
    printf("  --io-mode <mode>            I/O engine mode (pread, mmap, direct, iouring)\n");
    printf("  --save-state <path>         Save KV state after prefill\n");
    printf("  --load-state <path>         Load KV state before decode\n");
    printf("  --state-info <path>         Inspect saved state file and exit\n");
    printf("  --quiet                     Quiet execution summary\n");
    printf("  --help, -h                  Display help\n");
}

int main(int argc, char **argv) {
    char *model_path = NULL;
    char *arch_str = "auto";
    char *prompt_text = NULL;
    char *system_text = NULL;
    char *prompt_ids_str = NULL;
    char *prompt_ids_file = NULL;
    char *save_state_path = NULL;
    char *load_state_path = NULL;
    char *state_info_file = NULL;
    char *io_mode_str = "pread";

    int max_tokens = 128;
    int context_size = 4096;
    int num_threads = 0;
    float temp = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float min_p = 0.05f;
    float repeat_penalty = 1.0f;
    bool greedy = false;
    bool is_chat = false;
    bool is_interactive = false;
    bool pin_weights = false;
    bool quiet = false;
    bool debug_hidden_norm = false;
    bool logits_summary = false;

    static struct option long_options[] = {
        {"model",             required_argument, 0, 'm'},
        {"arch",              required_argument, 0, 'a'},
        {"prompt",            required_argument, 0, 'p'},
        {"system",            required_argument, 0, 's'},
        {"prompt-ids",        required_argument, 0, '1'},
        {"prompt-ids-file",   required_argument, 0, '2'},
        {"max-tokens",        required_argument, 0, 'n'},
        {"ctx",               required_argument, 0, 'c'},
        {"threads",           required_argument, 0, 't'},
        {"temp",              required_argument, 0, 'T'},
        {"top-p",             required_argument, 0, 'P'},
        {"top-k",             required_argument, 0, 'K'},
        {"min-p",             required_argument, 0, 'M'},
        {"repeat-penalty",    required_argument, 0, 'R'},
        {"greedy",            no_argument,       0, 'g'},
        {"chat",              no_argument,       0, 'C'},
        {"interactive",       no_argument,       0, 'i'},
        {"pin-weights",       no_argument,       0, 'W'},
        {"io-mode",           required_argument, 0, 'I'},
        {"save-state",        required_argument, 0, 'S'},
        {"load-state",        required_argument, 0, 'L'},
        {"state-info",        required_argument, 0, 'f'},
        {"quiet",             no_argument,       0, 'q'},
        {"debug-hidden-norm", no_argument,       0, 1001},
        {"logits-summary",    no_argument,       0, 1002},
        {"help",              no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:a:p:s:n:c:t:iWqh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'a': arch_str = optarg; break;
            case 'p': prompt_text = optarg; break;
            case 's': system_text = optarg; break;
            case '1': prompt_ids_str = optarg; break;
            case '2': prompt_ids_file = optarg; break;
            case 'n': max_tokens = atoi(optarg); break;
            case 'c': context_size = atoi(optarg); break;
            case 't': num_threads = atoi(optarg); break;
            case 'T': temp = atof(optarg); break;
            case 'P': top_p = atof(optarg); break;
            case 'K': top_k = atoi(optarg); break;
            case 'M': min_p = atof(optarg); break;
            case 'R': repeat_penalty = atof(optarg); break;
            case 'g': greedy = true; temp = 0.0f; break;
            case 'C': is_chat = true; break;
            case 'i': is_interactive = true; is_chat = true; break;
            case 'W': pin_weights = true; break;
            case 'I': io_mode_str = optarg; break;
            case 'S': save_state_path = optarg; break;
            case 'L': load_state_path = optarg; break;
            case 'f': state_info_file = optarg; break;
            case 'q': quiet = true; break;
            case 1001: debug_hidden_norm = true; break;
            case 1002: logits_summary = true; break;
            case 'h': print_usage(argv[0]); return 0;
            default: break;
        }
    }

    if (state_info_file) {
        return diskllm_print_state_info(state_info_file);
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model <path> is required.\n\n");
        print_usage(argv[0]);
        return 1;
    }

    diskllm_io_mode io_mode = DISKLLM_IO_PREAD;
    if (!strcmp(io_mode_str, "mmap")) io_mode = DISKLLM_IO_MMAP;
    else if (!strcmp(io_mode_str, "direct")) io_mode = DISKLLM_IO_DIRECT;
    else if (!strcmp(io_mode_str, "iouring")) io_mode = DISKLLM_IO_IOURING;

    diskllm_model_params mparams = diskllm_model_params_default();
    mparams.arch_flag = arch_str;
    mparams.pin_weights = pin_weights;
    mparams.io_mode = io_mode;
    mparams.num_threads = num_threads;
    mparams.quiet = quiet;

    diskllm_model *model = diskllm_model_load(model_path, mparams);
    if (!model) return 1;

    diskllm_context_params cparams = diskllm_context_params_default();
    cparams.context_size = context_size;
    cparams.num_threads = num_threads;

    diskllm_context *ctx = diskllm_context_init(model, cparams);
    if (!ctx) {
        diskllm_model_free(model);
        return 1;
    }
    ctx->debug_hidden_norm = debug_hidden_norm;

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);

    /* Format chat prompt if requested */
    char *formatted_prompt = NULL;
    if (is_chat && prompt_text) {
        formatted_prompt = diskllm_format_chat_prompt(model, system_text, prompt_text);
        if (formatted_prompt) prompt_text = formatted_prompt;
    }

    /* Parse or tokenize prompt */
    int prompt_tokens[4096];
    int prompt_len = 0;

    int vocab_size = diskllm_model_get_vocab_size(model);

    if (load_state_path) {
        if (!quiet) printf("[INFO] Loading saved KV state from %s...\n", load_state_path);
        if (diskllm_load_state(ctx, load_state_path) != 0) {
            fprintf(stderr, "Failed to load state from %s\n", load_state_path);
            diskllm_context_free(ctx); diskllm_model_free(model);
            return 1;
        }
    } else if (prompt_text) {
        if (!quiet) fprintf(stderr, "[INFO] Loading GGUF tokenizer to encode --prompt...\n");
        uint32_t bos_id = diskllm_model_get_bos_id(model);
        int offset = 0;
        if (bos_id > 0 && bos_id < (uint32_t)vocab_size) {
            prompt_tokens[offset++] = (int)bos_id;
        }
        int enc_cnt = diskllm_tokenize(tok, prompt_text, prompt_tokens + offset, 4096 - offset, false);
        prompt_len = offset + enc_cnt;
        if (!quiet) {
            fprintf(stderr, "[INFO] Tokenized --prompt into %d tokens:", prompt_len);
            for (int i = 0; i < prompt_len; i++) fprintf(stderr, " %d", prompt_tokens[i]);
            fprintf(stderr, "\n");
        }
    }

    float *logits = malloc(vocab_size * sizeof(float));

    if (!load_state_path && prompt_len > 0) {
        diskllm_eval(ctx, prompt_tokens, prompt_len, logits);

        if (logits_summary) {
            double sum = 0.0;
            float max_v = -1e30f;
            float min_v = 1e30f;
            int best_i = 0;
            for (int i = 0; i < vocab_size; i++) {
                float v = logits[i];
                uint32_t u; memcpy(&u, &v, 4);
                if ((u & 0x7F800000U) != 0x7F800000U) {
                    sum += v;
                    if (v > max_v) { max_v = v; best_i = i; }
                    if (v < min_v) min_v = v;
                }
            }
            double mean = sum / vocab_size;
            double var = 0.0;
            for (int i = 0; i < vocab_size; i++) {
                float v = logits[i];
                uint32_t u; memcpy(&u, &v, 4);
                if ((u & 0x7F800000U) != 0x7F800000U) {
                    double diff = v - mean;
                    var += diff * diff;
                }
            }
            double stddev = sqrt(var / vocab_size);
            char pbuf[128] = {0};
            diskllm_decode_token(tok, best_i, true, pbuf, sizeof(pbuf));
            fprintf(stderr, "[LOGITS-SUMMARY] step 0: max=%.4f, min=%.4f, mean=%.4f, stddev=%.4f\n", max_v, min_v, (float)mean, (float)stddev);
            fprintf(stderr, "Top 1: token_id=%d logit=%.4f piece='%s'\n", best_i, max_v, pbuf);
        }

        if (save_state_path) {
            if (!quiet) printf("[INFO] Saving KV state to %s...\n", save_state_path);
            diskllm_save_state(ctx, save_state_path);
        }
    }

    /* Initialize sampler */
    diskllm_sampler_params sparams = diskllm_sampler_params_default();
    sparams.temp = greedy ? 0.0f : temp;
    sparams.top_p = top_p;
    sparams.top_k = top_k;
    sparams.min_p = min_p;
    sparams.repeat_penalty = repeat_penalty;

    diskllm_sampler *smp = diskllm_sampler_init(sparams);

    /* Sample first token from prefill logits */
    int next_tok = diskllm_sample(smp, logits, vocab_size, prompt_tokens, prompt_len);

    uint32_t eos_id = diskllm_model_get_eos_id(model);

    printf("\n\033[1;32m[STREAM]\033[0m ");
    char piece[256];
    diskllm_decode_token(tok, next_tok, true, piece, sizeof(piece));
    printf("%s", piece);
    fflush(stdout);

    int gen_tokens[4096];
    int gen_count = 0;

    while (gen_count < max_tokens) {
        if ((uint32_t)next_tok == eos_id || next_tok == 32007 || next_tok == 32000) break;
        gen_tokens[gen_count++] = next_tok;

        diskllm_decode_step(ctx, next_tok, logits);
        next_tok = diskllm_sample(smp, logits, vocab_size, gen_tokens, gen_count);

        diskllm_decode_token(tok, next_tok, false, piece, sizeof(piece));
        printf("%s", piece);
        fflush(stdout);
    }
    printf("\n");

    /* Execution Summary */
    if (!quiet) {
        diskllm_perf_metrics m = diskllm_get_perf_metrics(ctx);
        fprintf(stderr, "\n=== Execution Summary ===\n");
        fprintf(stderr, "Weights Mode   : %s\n", m.weights_mode_str);
        fprintf(stderr, "Prompt tokens  : %d\n", m.prompt_tokens);
        fprintf(stderr, "Token count    : %d\n", m.gen_tokens);
        fprintf(stderr, "Prefill time   : %.2f ms\n", m.prefill_time_ms);
        fprintf(stderr, "Generation time: %.2f ms\n", m.generation_time_ms);
        if (m.gen_tokens > 0)
            fprintf(stderr, "Gen speed      : %.2f ms/tok\n", m.gen_speed_ms_per_tok);
        fprintf(stderr, "Bytes read     : %.2f MB (%llu bytes, Decode: %llu bytes)\n",
                (double)m.bytes_read_total / (1024.0 * 1024.0),
                (unsigned long long)m.bytes_read_total,
                (unsigned long long)m.bytes_read_decode);
        fprintf(stderr, "Peak RSS       : %ld MB\n", m.peak_rss_mb);
    }

    if (formatted_prompt) free(formatted_prompt);
    free(logits);
    diskllm_sampler_free(smp);
    diskllm_context_free(ctx);
    diskllm_model_free(model);
    return 0;
}
