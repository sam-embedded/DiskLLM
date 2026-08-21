#include "diskllm.h"
#include "diskllm_internal.h"
#include "vision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *img_file = (argc > 1) ? argv[1] : "/home/sam/cat_dog.jpg";
    const char *user_q = (argc > 2) ? argv[2] : "Describe what is in this image.";

    printf("=== Testing Multimodal Image Understanding ===\n");
    printf("Image: %s\n", img_file);
    printf("Query: %s\n\n", user_q);

    diskllm_model_params mparams = diskllm_model_params_default();
    mparams.pin_weights = true;
    mparams.num_threads = 4;
    diskllm_model *model = diskllm_model_load("/home/sam/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", mparams);
    if (!model) return 1;

    diskllm_vision_model *vm = diskllm_vision_load("/home/sam/models/mmproj-Qwen2.5-VL-3B.gguf");
    if (!vm) return 1;

    /* Downscale to 280x280 for fast responsive prefill on CPU */
    int img_w = 0, img_h = 0;
    int target_res = 280;
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

    printf("ViT generated %d visual tokens.\n", n_patches);

    diskllm_context_params cparams = diskllm_context_params_default();
    cparams.num_threads = 4;
    cparams.context_size = 1024;
    diskllm_context *ctx = diskllm_context_init(model, cparams);

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);
    char *prompt_text = diskllm_format_image_prompt(model, img_file, user_q, false);

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
    printf("Prefilling multimodal tokens with visual patch embeddings...\n");
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

    diskllm_sampler_free(smp);
    diskllm_context_free(ctx);
    diskllm_model_free(model);
    return 0;
}
