#include "diskllm_internal.h"

diskllm_tokenizer *diskllm_model_get_tokenizer(diskllm_model *model) {
    if (!model || !model->tok) return NULL;
    diskllm_tokenizer *wrap = malloc(sizeof(diskllm_tokenizer));
    if (wrap) wrap->tok = model->tok;
    return wrap;
}

int diskllm_tokenize(const diskllm_tokenizer *tok, const char *text, int *out_tokens, int max_tokens, bool add_bos) {
    (void)add_bos;
    if (!tok || !tok->tok || !text || !out_tokens || max_tokens <= 0) return 0;
    int enc_cnt = tokenizer_encode(tok->tok, text, out_tokens, max_tokens);
    return enc_cnt;
}

int diskllm_decode_token(const diskllm_tokenizer *tok, int token, bool is_first, char *buf, size_t buf_size) {
    if (!tok || !tok->tok || !buf || buf_size == 0) return -1;
    return tokenizer_decode_token(tok->tok, token, is_first ? 1 : 0, buf, buf_size);
}

char *diskllm_format_chat_prompt(const diskllm_model *model, const char *system_prompt, const char *user_prompt) {
    if (!model || !user_prompt) return NULL;

    const qwen_model_config *cfg = model->cfg;
    size_t blen = strlen(user_prompt) + (system_prompt ? strlen(system_prompt) : 0) + 512;
    char *chat_buf = malloc(blen);
    if (!chat_buf) return NULL;

    if (cfg->model_type == MODEL_TYPE_LLAMA) {
        if (system_prompt) {
            snprintf(chat_buf, blen, "<|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", system_prompt, user_prompt);
        } else {
            snprintf(chat_buf, blen, "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", user_prompt);
        }
    } else if (cfg->model_type == MODEL_TYPE_MISTRAL) {
        if (system_prompt) {
            snprintf(chat_buf, blen, "[INST] %s\n\n%s [/INST]", system_prompt, user_prompt);
        } else {
            snprintf(chat_buf, blen, "[INST] %s [/INST]", user_prompt);
        }
    } else if (cfg->model_type == MODEL_TYPE_PHI3) {
        if (system_prompt) {
            snprintf(chat_buf, blen, "<|system|>\n%s<|end|>\n<|user|>\n%s<|end|>\n<|assistant|>\n", system_prompt, user_prompt);
        } else {
            snprintf(chat_buf, blen, "<|user|>\n%s<|end|>\n<|assistant|>\n", user_prompt);
        }
    } else {
        if (system_prompt) {
            snprintf(chat_buf, blen, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", system_prompt, user_prompt);
        } else {
            snprintf(chat_buf, blen, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user_prompt);
        }
    }

    return chat_buf;
}
