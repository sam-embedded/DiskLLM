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

char *diskllm_format_chat_prompt_ex(const diskllm_model *model, const char *system_prompt, const char *user_prompt, bool enable_thinking) {
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
    } else if (cfg->model_type == MODEL_TYPE_GEMMA || cfg->model_type == MODEL_TYPE_GEMMA2 || cfg->model_type == MODEL_TYPE_GEMMA3 || cfg->model_type == MODEL_TYPE_GEMMA4) {
        if (system_prompt) {
            snprintf(chat_buf, blen, "<start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n", system_prompt, user_prompt);
        } else {
            snprintf(chat_buf, blen, "<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user_prompt);
        }
    } else if (cfg->model_type == MODEL_TYPE_QWEN_HYBRID) {
        const char *gen_suffix = enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
        if (system_prompt) {
            snprintf(chat_buf, blen, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s", system_prompt, user_prompt, gen_suffix);
        } else {
            snprintf(chat_buf, blen, "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s", user_prompt, gen_suffix);
        }
    } else {
        const char *sys = system_prompt ? system_prompt : "You are a helpful assistant.";
        snprintf(chat_buf, blen, "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", sys, user_prompt);
    }

    return chat_buf;
}

char *diskllm_format_chat_prompt(const diskllm_model *model, const char *system_prompt, const char *user_prompt) {
    return diskllm_format_chat_prompt_ex(model, system_prompt, user_prompt, false);
}

char *diskllm_format_agent_prompt(const diskllm_model *model, const char *tools_json, const char *system_instructions, const char *user_prompt, bool enable_thinking) {
    if (!model || !user_prompt) return NULL;

    const char *tool_header =
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n";

    const char *tool_footer =
        "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        "</IMPORTANT>";

    size_t tools_len = tools_json ? strlen(tools_json) : 0;
    size_t inst_len = system_instructions ? strlen(system_instructions) : 0;
    size_t full_sys_len = strlen(tool_header) + tools_len + strlen(tool_footer) + (inst_len > 0 ? (inst_len + 4) : 0) + 128;

    char *full_system = malloc(full_sys_len);
    if (!full_system) return NULL;

    if (tools_json && tools_len > 0) {
        if (system_instructions && inst_len > 0) {
            snprintf(full_system, full_sys_len, "%s%s%s\n\n%s", tool_header, tools_json, tool_footer, system_instructions);
        } else {
            snprintf(full_system, full_sys_len, "%s%s%s", tool_header, tools_json, tool_footer);
        }
    } else {
        if (system_instructions && inst_len > 0) {
            snprintf(full_system, full_sys_len, "%s", system_instructions);
        } else {
            full_system[0] = '\0';
        }
    }

    char *res = diskllm_format_chat_prompt_ex(model, full_system[0] ? full_system : NULL, user_prompt, enable_thinking);
    free(full_system);
    return res;
}

char *diskllm_format_image_prompt(const diskllm_model *model, const char *image_path_or_desc, const char *user_prompt, bool enable_thinking) {
    (void)image_path_or_desc;
    if (!model || !user_prompt) return NULL;

    size_t u_len = strlen(user_prompt);
    size_t blen = u_len + 512;
    char *user_content = malloc(blen);
    if (!user_content) return NULL;

    snprintf(user_content, blen, "Picture 1: <|vision_start|><|image_pad|><|vision_end|>\n%s", user_prompt);

    char *res = diskllm_format_chat_prompt_ex(model, NULL, user_content, enable_thinking);
    free(user_content);
    return res;
}

char *diskllm_format_tool_response(const diskllm_model *model, const char *tool_output_json) {
    if (!model || !tool_output_json) return NULL;
    size_t blen = strlen(tool_output_json) + 128;
    char *buf = malloc(blen);
    if (!buf) return NULL;
    snprintf(buf, blen, "<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n", tool_output_json);
    return buf;
}

char *diskllm_format_fim_prompt(const diskllm_model *model, const char *prefix, const char *suffix, const char *repo_name, const char *file_name) {
    (void)model;
    if (!prefix && !suffix) return NULL;

    size_t rlen = (repo_name ? strlen(repo_name) : 0) + (file_name ? strlen(file_name) : 0) +
                  (prefix ? strlen(prefix) : 0) + (suffix ? strlen(suffix) : 0) + 256;
    char *buf = malloc(rlen);
    if (!buf) return NULL;

    char extra_meta[256] = {0};
    if (repo_name && file_name) {
        snprintf(extra_meta, sizeof(extra_meta), "<|repo_name|>%s<|file_sep|>%s\n", repo_name, file_name);
    } else if (file_name) {
        snprintf(extra_meta, sizeof(extra_meta), "<|file_sep|>%s\n", file_name);
    }

    snprintf(buf, rlen, "<|fim_prefix|>%s%s<|fim_suffix|>%s<|fim_middle|>",
             extra_meta,
             prefix ? prefix : "",
             suffix ? suffix : "");

    return buf;
}

char *diskllm_strip_think_tags(const char *text) {
    if (!text) return NULL;
    const char *open_tag = "<think>";
    const char *close_tag = "</think>";

    const char *start = strstr(text, open_tag);
    const char *end = strstr(text, close_tag);

    if (start && end && end > start) {
        const char *after_close = end + strlen(close_tag);
        while (*after_close == '\n' || *after_close == '\r' || *after_close == ' ') {
            after_close++;
        }
        size_t pre_len = (size_t)(start - text);
        size_t post_len = strlen(after_close);
        char *cleaned = malloc(pre_len + post_len + 1);
        if (!cleaned) return strdup(text);

        if (pre_len > 0) memcpy(cleaned, text, pre_len);
        memcpy(cleaned + pre_len, after_close, post_len);
        cleaned[pre_len + post_len] = '\0';
        return cleaned;
    }

    return strdup(text);
}
