#include "diskllm.h"
#include "diskllm_internal.h"
#include "vision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *img_file = (argc > 1) ? argv[1] : "/home/sam/test.jpeg";
    const char *user_q = (argc > 2) ? argv[2] : "Extract the text and numbers from this screenshot.";

    printf("=== Testing Qwen3.5-0.8B Native Multimodal Vision Support ===\n");
    printf("Model : /home/sam/models/Qwen3.5-0.8B-Q4_K_M.gguf\n");
    printf("ViT   : /home/sam/models/mmproj-Qwen3.5-0.8B-F16.gguf\n");
    printf("Image : %s\n", img_file);
    printf("Query : %s\n\n", user_q);

    diskllm_model_params mparams = diskllm_model_params_default();
    mparams.pin_weights = true;
    mparams.num_threads = 4;
    diskllm_model *model = diskllm_model_load("/home/sam/models/Qwen3.5-0.8B-Q4_K_M.gguf", mparams);
    if (!model) return 1;

    diskllm_vision_model *vm = diskllm_vision_load("/home/sam/models/mmproj-Qwen3.5-0.8B-F16.gguf");
    if (!vm) return 1;

    int img_w = 0, img_h = 0;
    /* Qwen3.5 ViT image_size is 768, patch_size 16 */
    int target_res = 256; /* 256x256 -> 16x16 = 256 patches -> 2x2 merged to 64 tokens */
    float *rgb = diskllm_image_load_rgb(img_file, target_res, target_res, &img_w, &img_h, vm->cfg.image_mean, vm->cfg.image_std);
    if (!rgb) {
        fprintf(stderr, "Failed to load image: %s\n", img_file);
        return 1;
    }

    int n_patches = 0;
    float *vis_emb = diskllm_vision_encode(vm, rgb, img_w, img_h, &n_patches, 4);
    diskllm_image_free(rgb);
    diskllm_vision_free(vm);
    if (!vis_emb) {
        fprintf(stderr, "ViT encoding failed\n");
        return 1;
    }

    printf("[INFO] ViT generated %d visual tokens (proj_dim=%d).\n", n_patches, model->cfg->hidden_dim);

    diskllm_context_params cparams = diskllm_context_params_default();
    cparams.num_threads = 4;
    cparams.context_size = 2048;
    diskllm_context *ctx = diskllm_context_init(model, cparams);

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);
    char *prompt_text = diskllm_format_image_prompt(model, img_file, user_q, false);

    int prompt_tokens[2048];
    int prompt_len = diskllm_tokenize(tok, prompt_text, prompt_tokens, 2048, false);
    printf("[INFO] Prompt tokens count: %d\n", prompt_len);

    int img_pad_pos = -1;
    for (int i = 0; i < prompt_len; i++) {
        if (prompt_tokens[i] == 248056 || prompt_tokens[i] == 151655) {
            img_pad_pos = i;
            break;
        }
    }

    float *logits = malloc(model->cfg->vocab_size * sizeof(float));
    printf("[INFO] Prefilling multimodal tokens with visual embeddings at index %d...\n", img_pad_pos);
    diskllm_eval_multimodal(ctx, prompt_tokens, prompt_len, img_pad_pos, vis_emb, n_patches, logits);
    free(vis_emb);

    diskllm_sampler_params sparams = diskllm_sampler_params_default();
    sparams.temp = 0.2f;
    sparams.repeat_penalty = 1.1f;
    diskllm_sampler *smp = diskllm_sampler_init(sparams);

    printf("\n[RESPONSE] ");
    int seen[128];
    int cur_tok = diskllm_sample(smp, logits, model->cfg->vocab_size, NULL, 0);

    for (int step = 0; step < 64; step++) {
        if (cur_tok == model->cfg->eos_token_id || cur_tok == 248046 || cur_tok == 151645) break;
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
