#include "diskllm.h"
#include "diskllm_internal.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    diskllm_model_params params = diskllm_model_params_default();
    diskllm_model *m = diskllm_model_load("/home/sam/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", params);
    if (!m) return 1;
    diskllm_tokenizer *t = diskllm_model_get_tokenizer(m);
    
    char *p = diskllm_format_image_prompt(m, "test.jpeg", "Extract the text and numbers from this screenshot.", false);
    printf("Formatted Prompt:\n---\n%s\n---\n", p);
    
    int tokens[1024];
    int n = diskllm_tokenize(t, p, tokens, 1024, false);
    printf("Tokens (%d):\n", n);
    for (int i = 0; i < n; i++) {
        char buf[64] = {0};
        diskllm_decode_token(t, tokens[i], i==0, buf, sizeof(buf));
        printf("[%d]: ID=%d -> '%s'\n", i, tokens[i], buf);
    }
    return 0;
}
