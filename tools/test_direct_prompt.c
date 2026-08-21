#include "diskllm.h"
#include "diskllm_internal.h"
#include "vision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    diskllm_model_params mparams = diskllm_model_params_default();
    mparams.pin_weights = true;
    mparams.num_threads = 4;
    diskllm_model *model = diskllm_model_load("/home/sam/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", mparams);
    if (!model) return 1;

    diskllm_context_params cparams = diskllm_context_params_default();
    cparams.num_threads = 4;
    cparams.context_size = 2048;
    diskllm_context *ctx = diskllm_context_init(model, cparams);

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);

    /* Test 1: Plain text prompt without image to verify LLM base inference quality */
    const char *prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2 + 2? Answer in one word.<|im_end|>\n<|im_start|>assistant\n";
    printf("Prompt:\n%s\n", prompt);

    int prompt_tokens[1024];
    int prompt_len = diskllm_tokenize(tok, prompt, prompt_tokens, 1024, false);
    printf("Prompt tokens count: %d\n", prompt_len);

    float *logits = malloc(model->cfg->vocab_size * sizeof(float));
    diskllm_eval(ctx, prompt_tokens, prompt_len, logits);

    diskllm_sampler_params sparams = diskllm_sampler_params_default();
    sparams.temp = 0.0f; // greedy
    diskllm_sampler *smp = diskllm_sampler_init(sparams);

    printf("\n[STREAM] ");
    int cur_tok = diskllm_sample(smp, logits, model->cfg->vocab_size, NULL, 0);

    for (int step = 0; step < 16; step++) {
        if (cur_tok == 151645 || cur_tok == 151643) break;
        char buf[64] = {0};
        diskllm_decode_token(tok, cur_tok, step == 0, buf, sizeof(buf));
        printf("%s", buf);
        fflush(stdout);
        diskllm_decode_step(ctx, cur_tok, logits);
        cur_tok = diskllm_sample(smp, logits, model->cfg->vocab_size, NULL, 0);
    }
    printf("\n\n");

    diskllm_perf_metrics perf = diskllm_get_perf_metrics(ctx);
    printf("Prefill Time: %.2f ms\n", perf.prefill_time_ms);
    printf("Gen Speed   : %.2f ms/tok\n", perf.gen_speed_ms_per_tok);

    diskllm_sampler_free(smp);
    diskllm_context_free(ctx);
    diskllm_model_free(model);
    return 0;
}
