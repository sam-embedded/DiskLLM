#include "diskllm.h"
#include "diskllm_internal.h"
#include "vision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    printf("=== Testing Qwen2.5-VL ViT + LLM Prompt Formats ===\n");
    diskllm_model_params mparams = diskllm_model_params_default();
    mparams.pin_weights = true;
    mparams.num_threads = 4;
    diskllm_model *model = diskllm_model_load("/home/sam/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", mparams);
    if (!model) return 1;

    diskllm_vision_model *vm = diskllm_vision_load("/home/sam/models/mmproj-Qwen2.5-VL-3B.gguf");
    if (!vm) return 1;

    /* Test 280x280 downscaled to produce 100 patches -> 5x5 = 25 tokens */
    int img_w = 0, img_h = 0;
    int target_res = 280;
    float *rgb = diskllm_image_load_rgb("/home/sam/test.jpeg", target_res, target_res, &img_w, &img_h, vm->cfg.image_mean, vm->cfg.image_std);
    if (!rgb) return 1;

    int n_patches = 0;
    float *vis_emb = diskllm_vision_encode(vm, rgb, img_w, img_h, &n_patches, 4);
    diskllm_image_free(rgb);
    diskllm_vision_free(vm);
    if (!vis_emb) return 1;

    printf("ViT Generated %d visual tokens (proj_dim=%d)\n", n_patches, model->cfg->hidden_dim);

    diskllm_context_params cparams = diskllm_context_params_default();
    cparams.num_threads = 4;
    cparams.context_size = 1024;
    diskllm_context *ctx = diskllm_context_init(model, cparams);

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);
    char *prompt_text = diskllm_format_image_prompt(model, "/home/sam/test.jpeg", "Extract the text and numbers from this screenshot.", false);

    int prompt_tokens[1024];
    int prompt_len = diskllm_tokenize(tok, prompt_text, prompt_tokens, 1024, false);
    printf("Prompt tokens count: %d\n", prompt_len);

    int img_pad_pos = -1;
    for (int i = 0; i < prompt_len; i++) {
        if (prompt_tokens[i] == 151655 || prompt_tokens[i] == 248056) {
            img_pad_pos = i;
            break;
        }
    }

    float *logits = malloc(model->cfg->vocab_size * sizeof(float));
    printf("Prefilling with %d visual patches at index %d...\n", n_patches, img_pad_pos);
    diskllm_eval_multimodal(ctx, prompt_tokens, prompt_len, img_pad_pos, vis_emb, n_patches, logits);
    free(vis_emb);

    diskllm_sampler_params sparams = diskllm_sampler_params_default();
    sparams.temp = 0.2f;
    diskllm_sampler *smp = diskllm_sampler_init(sparams);

    printf("\n[STREAM] ");
    int seen[64];
    int cur_tok = diskllm_sample(smp, logits, model->cfg->vocab_size, NULL, 0);

    for (int step = 0; step < 32; step++) {
        if (cur_tok == 151645 || cur_tok == 151643) break;
        char buf[64] = {0};
        diskllm_decode_token(tok, cur_tok, step == 0, buf, sizeof(buf));
        printf("%s", buf);
        fflush(stdout);
        seen[step] = cur_tok;
        diskllm_decode_step(ctx, cur_tok, logits);
        cur_tok = diskllm_sample(smp, logits, model->cfg->vocab_size, seen, step + 1);
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
