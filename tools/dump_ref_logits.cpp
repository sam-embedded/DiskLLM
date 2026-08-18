#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

struct logit_pair {
    int id;
    float val;
};

int main(int argc, char **argv) {
    std::string model_path = "/data/data/com.termux/files/home/models/Qwen3.8-27B-Q4_K_M.gguf";
    std::vector<int32_t> prompt_ids = {248045,846,198,760,6511,314,9338,369,248046,198,248045,74455,198};
    int top_k = 8;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt-ids") == 0 && i + 1 < argc) {
            prompt_ids.clear();
            char *dup = strdup(argv[++i]);
            char *tok = strtok(dup, ",");
            while (tok) {
                prompt_ids.push_back(std::atoi(tok));
                tok = strtok(NULL, ",");
            }
            free(dup);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = std::atoi(argv[++i]);
        }
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model %s\n", model_path.c_str());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = prompt_ids.size() + 16;
    cparams.n_batch = prompt_ids.size() + 16;
    cparams.n_ubatch = prompt_ids.size() + 16;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    int vocab_size = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_get_one(prompt_ids.data(), prompt_ids.size());
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    float *logits = llama_get_logits_ith(ctx, -1);
    if (!logits) {
        logits = llama_get_logits(ctx);
    }

    if (!logits) {
        fprintf(stderr, "Failed to get logits\n");
        return 1;
    }

    std::vector<logit_pair> pairs(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        pairs[i] = {i, logits[i]};
    }
    std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(), [](const logit_pair &a, const logit_pair &b) {
        return a.val > b.val;
    });

    printf("=== LLAMA.CPP REFERENCE LOGITS ===\n");
    printf("Prompt tokens count: %zu\n", prompt_ids.size());
    printf("Top %d Logits for next token:\n", top_k);
    for (int r = 0; r < top_k; r++) {
        char buf[256];
        int len = llama_token_to_piece(vocab, pairs[r].id, buf, sizeof(buf) - 1, 0, true);
        if (len < 0) len = 0;
        buf[len] = '\0';
        printf("  Rank %d: ID %-7d  logit %10.6f  piece: \"%s\"\n", r + 1, pairs[r].id, pairs[r].val, buf);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
