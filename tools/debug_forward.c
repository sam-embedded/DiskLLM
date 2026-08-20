#define _GNU_SOURCE
#include "diskllm.h"
#include "diskllm_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    const char *model_path = argc > 1 ? argv[1] : "/home/sam/models/gemma-4-E4B-it-Q4_K_M.gguf";
    printf("Loading model %s...\n", model_path);

    diskllm_model_params mparams = diskllm_model_params_default();
    mparams.io_mode = DISKLLM_IO_PREAD;
    diskllm_model *model = diskllm_model_load(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    diskllm_context_params cparams = diskllm_context_params_default();
    diskllm_context *ctx = diskllm_context_init(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);
    const char *prompt = argc > 2 ? argv[2] : "The capital of France is";
    int tokens[128];
    int n_tok = diskllm_tokenize(tok, prompt, tokens, 128, true);
    printf("Tokenized %d tokens: ", n_tok);
    for (int i = 0; i < n_tok; i++) printf("%d ", tokens[i]);
    printf("\n");

    int vocab_size = diskllm_model_get_vocab_size(model);
    float *logits = malloc(vocab_size * sizeof(float));
    int ret = diskllm_eval(ctx, tokens, n_tok, logits);
    printf("diskllm_eval returned %d\n", ret);

    // Print top 10 tokens
    printf("\nTop 10 predicted tokens:\n");
    for (int rank = 0; rank < 10; rank++) {
        int best_i = -1;
        float best_v = -1e30f;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > best_v) {
                best_v = logits[i];
                best_i = i;
            }
        }
        char piece[128] = {0};
        diskllm_decode_token(tok, best_i, true, piece, sizeof(piece));
        printf("Rank %d: token_id=%d logit=%.4f piece='%s'\n", rank + 1, best_i, best_v, piece);
        logits[best_i] = -1e30f; // mask for next rank
    }

    free(logits);
    diskllm_context_free(ctx);
    diskllm_model_free(model);
    return 0;
}
