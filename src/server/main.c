#define _GNU_SOURCE
#include "diskllm.h"
#include "cJSON.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define BUFFER_SIZE 65536

static void print_server_usage(const char *prog_name) {
    printf("DiskLLM Server — OpenAI Compatible Edge LLM API Server\n\n");
    printf("Usage: %s --model <path> [options]\n\n", prog_name);
    printf("Options:\n");
    printf("  --model <path>       Path to GGUF model file (required)\n");
    printf("  --port <int>         HTTP server port (default: 8080)\n");
    printf("  --host <ip>          HTTP server host IP (default: 0.0.0.0)\n");
    printf("  --arch <type>        Architecture hint (auto, llama, phi3, mistral, gemma)\n");
    printf("  --ctx <int>          Context capacity (default: 4096)\n");
    printf("  --threads <int>      Computing threads\n");
    printf("  --pin-weights        Pin weights in RAM (zero decode I/O)\n");
    printf("  --io-mode <mode>     I/O engine mode (pread, mmap, direct)\n");
    printf("  --help, -h           Display help\n");
}

static void escape_json_string(const char *src, char *dst, size_t dst_size) {
    size_t d = 0;
    while (*src && d + 6 < dst_size) {
        unsigned char c = (unsigned char)*src++;
        if (c == '"') { dst[d++] = '\\'; dst[d++] = '"'; }
        else if (c == '\\') { dst[d++] = '\\'; dst[d++] = '\\'; }
        else if (c == '\n') { dst[d++] = '\\'; dst[d++] = 'n'; }
        else if (c == '\r') { dst[d++] = '\\'; dst[d++] = 'r'; }
        else if (c == '\t') { dst[d++] = '\\'; dst[d++] = 't'; }
        else if (c < 32) {
            d += snprintf(dst + d, dst_size - d, "\\u%04x", c);
        } else {
            dst[d++] = c;
        }
    }
    dst[d] = '\0';
}

static const char *extract_message_content(cJSON *content_item, char *buf, size_t buf_size) {
    if (!content_item) return "";
    if (content_item->valuestring) return content_item->valuestring;
    if (cJSON_IsArray(content_item)) {
        int count = cJSON_GetArraySize(content_item);
        buf[0] = '\0';
        size_t cur = 0;
        int img_count = 0;
        for (int i = 0; i < count; i++) {
            cJSON *part = cJSON_GetArrayItem(content_item, i);
            cJSON *type_item = cJSON_GetObjectItemCaseSensitive(part, "type");
            const char *type_str = type_item ? type_item->valuestring : NULL;

            if (type_str && (!strcmp(type_str, "image_url") || !strcmp(type_str, "image"))) {
                img_count++;
                char img_tag[128];
                snprintf(img_tag, sizeof(img_tag), "Picture %d: <|vision_start|><|image_pad|><|vision_end|>\n", img_count);
                size_t it_len = strlen(img_tag);
                if (cur + it_len < buf_size - 1) {
                    memcpy(buf + cur, img_tag, it_len);
                    cur += it_len;
                    buf[cur] = '\0';
                }
            }

            cJSON *text_item = cJSON_GetObjectItemCaseSensitive(part, "text");
            if (text_item && text_item->valuestring) {
                size_t t_len = strlen(text_item->valuestring);
                if (cur + t_len < buf_size - 1) {
                    memcpy(buf + cur, text_item->valuestring, t_len);
                    cur += t_len;
                    buf[cur] = '\0';
                }
            }
        }
        return buf;
    }
    return "";
}

static char *read_http_request(int client_fd, size_t *out_size) {
    size_t cap = BUFFER_SIZE;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    struct timeval tv = {.tv_sec = 2, .tv_usec = 0};
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    while (1) {
        if (len + 4096 >= cap) {
            cap *= 2;
            char *new_buf = realloc(buf, cap);
            if (!new_buf) { free(buf); return NULL; }
            buf = new_buf;
        }

        ssize_t n = recv(client_fd, buf + len, cap - len - 1, 0);
        if (n <= 0) break;
        len += n;
        buf[len] = '\0';

        char *header_end = strstr(buf, "\r\n\r\n");
        if (header_end) {
            char *cl_pos = strcasestr(buf, "Content-Length:");
            if (cl_pos && cl_pos < header_end) {
                const char *val = cl_pos + 15;
                while (*val == ' ' || *val == '\t') val++;
                int content_len = atoi(val);
                size_t body_received = len - (header_end + 4 - buf);
                if (body_received >= (size_t)content_len) {
                    break;
                }
            } else {
                break;
            }
        }
    }

    if (len == 0) {
        free(buf);
        return NULL;
    }

    if (out_size) *out_size = len;
    return buf;
}

static void handle_client(int client_fd, diskllm_model *model, diskllm_context *ctx) {
    size_t req_size = 0;
    char *buffer = read_http_request(client_fd, &req_size);
    if (!buffer) {
        close(client_fd);
        return;
    }

    /* Check HTTP Method & Path */
    char method[16], path[256];
    sscanf(buffer, "%15s %255s", method, path);

    fprintf(stderr, "[SERVER] Request: %s %s (%zu bytes)\n", method, path, req_size);

    /* Handle CORS Preflight */
    if (!strcmp(method, "OPTIONS")) {
        const char *cors_res =
            "HTTP/1.1 200 OK\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: *\r\n"
            "Content-Length: 0\r\n\r\n";
        send(client_fd, cors_res, strlen(cors_res), 0);
        free(buffer);
        close(client_fd);
        return;
    }

    /* Handle /v1/models */
    if (!strcmp(method, "GET") && (!strcmp(path, "/v1/models") || !strcmp(path, "/models"))) {
        const char *models_json =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n\r\n"
            "{\"object\":\"list\",\"data\":[{\"id\":\"qwen35\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"diskllm\"},{\"id\":\"qwen\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"diskllm\"}]}";
        send(client_fd, models_json, strlen(models_json), 0);
        free(buffer);
        close(client_fd);
        return;
    }

    /* Check if Chat Completions or Completions */
    bool is_chat_endpoint = (!strcmp(path, "/v1/chat/completions") || !strcmp(path, "/chat/completions"));
    bool is_completion_endpoint = (!strcmp(path, "/v1/completions") || !strcmp(path, "/completions"));

    if (strcmp(method, "POST") != 0 || (!is_chat_endpoint && !is_completion_endpoint)) {
        const char *not_found = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: 9\r\n\r\nNot Found";
        send(client_fd, not_found, strlen(not_found), 0);
        free(buffer);
        close(client_fd);
        return;
    }

    /* Locate JSON Payload */
    char *body = strstr(buffer, "\r\n\r\n");
    if (!body) {
        free(buffer);
        close(client_fd);
        return;
    }
    body += 4;

    cJSON *json = cJSON_Parse(body);
    if (!json) {
        const char *bad_req = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: 15\r\n\r\nInvalid Payload";
        send(client_fd, bad_req, strlen(bad_req), 0);
        free(buffer);
        close(client_fd);
        return;
    }

    cJSON *messages = cJSON_GetObjectItemCaseSensitive(json, "messages");
    cJSON *prompt_item = cJSON_GetObjectItemCaseSensitive(json, "prompt");
    cJSON *tools_item = cJSON_GetObjectItemCaseSensitive(json, "tools");
    cJSON *extra_body = cJSON_GetObjectItemCaseSensitive(json, "extra_body");
    cJSON *max_tokens_item = cJSON_GetObjectItemCaseSensitive(json, "max_tokens");
    cJSON *temp_item = cJSON_GetObjectItemCaseSensitive(json, "temperature");
    cJSON *top_p_item = cJSON_GetObjectItemCaseSensitive(json, "top_p");
    cJSON *top_k_item = cJSON_GetObjectItemCaseSensitive(json, "top_k");
    cJSON *presence_item = cJSON_GetObjectItemCaseSensitive(json, "presence_penalty");
    cJSON *repeat_item = cJSON_GetObjectItemCaseSensitive(json, "repetition_penalty");
    cJSON *stream_item = cJSON_GetObjectItemCaseSensitive(json, "stream");
    cJSON *thinking_item = cJSON_GetObjectItemCaseSensitive(json, "enable_thinking");

    if (extra_body) {
        if (!thinking_item) thinking_item = cJSON_GetObjectItemCaseSensitive(extra_body, "enable_thinking");
        if (!top_k_item) top_k_item = cJSON_GetObjectItemCaseSensitive(extra_body, "top_k");
    }

    int max_tokens = max_tokens_item ? max_tokens_item->valueint : 512;
    float temp = temp_item ? (float)temp_item->valuedouble : 0.7f;
    float top_p = top_p_item ? (float)top_p_item->valuedouble : 0.95f;
    int top_k = top_k_item ? top_k_item->valueint : 40;
    float presence_penalty = presence_item ? (float)presence_item->valuedouble : 0.0f;
    float repeat_penalty = repeat_item ? (float)repeat_item->valuedouble : 1.0f;
    bool is_streaming = stream_item ? ((stream_item->type & cJSON_True) != 0) : true;
    bool enable_thinking = thinking_item ? ((thinking_item->type & cJSON_True) != 0) : false;

    char *formatted_prompt = NULL;

    if (is_chat_endpoint) {
        char *tools_str = NULL;
        if (tools_item) {
            if (tools_item->valuestring) {
                tools_str = strdup(tools_item->valuestring);
            } else {
                const char *t_pos = strstr(body, "\"tools\"");
                if (t_pos) {
                    const char *colon = strchr(t_pos, ':');
                    if (colon) {
                        while (*colon == ' ' || *colon == ':') colon++;
                        tools_str = strdup(colon);
                    }
                }
            }
        }

        if (cJSON_IsArray(messages)) {
            int count = cJSON_GetArraySize(messages);
            size_t buf_cap = 65536;
            formatted_prompt = malloc(buf_cap);
            if (formatted_prompt) {
                formatted_prompt[0] = '\0';
                size_t cur_len = 0;
                char piece_buf[8192];

                for (int i = 0; i < count; i++) {
                    cJSON *item = cJSON_GetArrayItem(messages, i);
                    cJSON *role = cJSON_GetObjectItemCaseSensitive(item, "role");
                    cJSON *content = cJSON_GetObjectItemCaseSensitive(item, "content");
                    const char *r_str = (role && role->valuestring) ? role->valuestring : "user";
                    const char *c_str = extract_message_content(content, piece_buf, sizeof(piece_buf));

                    char turn_buf[8192];
                    if (!strcmp(r_str, "system")) {
                        if (tools_str) {
                            char *agent_sys = diskllm_format_agent_prompt(model, tools_str, c_str, "PLACEHOLDER", enable_thinking);
                            if (agent_sys) {
                                char *user_mark = strstr(agent_sys, "<|im_start|>user\nPLACEHOLDER");
                                if (user_mark) *user_mark = '\0';
                                snprintf(turn_buf, sizeof(turn_buf), "%s", agent_sys);
                                free(agent_sys);
                            } else {
                                snprintf(turn_buf, sizeof(turn_buf), "<|im_start|>system\n%s<|im_end|>\n", c_str);
                            }
                        } else {
                            snprintf(turn_buf, sizeof(turn_buf), "<|im_start|>system\n%s<|im_end|>\n", c_str);
                        }
                    } else if (!strcmp(r_str, "assistant")) {
                        char *clean_c = diskllm_strip_think_tags(c_str);
                        snprintf(turn_buf, sizeof(turn_buf), "<|im_start|>assistant\n%s<|im_end|>\n", clean_c ? clean_c : c_str);
                        if (clean_c) free(clean_c);
                    } else if (!strcmp(r_str, "tool")) {
                        snprintf(turn_buf, sizeof(turn_buf), "<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n", c_str);
                    } else {
                        snprintf(turn_buf, sizeof(turn_buf), "<|im_start|>user\n%s<|im_end|>\n", c_str);
                    }

                    size_t t_len = strlen(turn_buf);
                    if (cur_len + t_len + 256 < buf_cap) {
                        memcpy(formatted_prompt + cur_len, turn_buf, t_len);
                        cur_len += t_len;
                        formatted_prompt[cur_len] = '\0';
                    }
                }

                const char *gen_suffix = enable_thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think>\n\n</think>\n\n";
                size_t s_len = strlen(gen_suffix);
                if (cur_len + s_len < buf_cap) {
                    memcpy(formatted_prompt + cur_len, gen_suffix, s_len);
                    cur_len += s_len;
                    formatted_prompt[cur_len] = '\0';
                }
            }
        }

        if (tools_str) free(tools_str);
    } else {
        if (prompt_item && prompt_item->valuestring) {
            formatted_prompt = strdup(prompt_item->valuestring);
        } else {
            formatted_prompt = strdup("Hello");
        }
    }

    if (!formatted_prompt) {
        cJSON_Delete(json);
        close(client_fd);
        return;
    }

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);
    int prompt_tokens[4096];
    int prompt_len = diskllm_tokenize(tok, formatted_prompt, prompt_tokens, 4096, true);

    int vocab_size = diskllm_model_get_vocab_size(model);
    float *logits = malloc(vocab_size * sizeof(float));

    /* Prefill */
    diskllm_eval(ctx, prompt_tokens, prompt_len, logits);

    /* Initialize sampler */
    diskllm_sampler_params sparams = diskllm_sampler_params_default();
    sparams.temp = temp;
    sparams.top_p = top_p;
    sparams.top_k = top_k;
    sparams.presence_penalty = presence_penalty;
    sparams.repeat_penalty = repeat_penalty;
    diskllm_sampler *smp = diskllm_sampler_init(sparams);

    int next_tok = diskllm_sample(smp, logits, vocab_size, prompt_tokens, prompt_len);
    uint32_t eos_id = diskllm_model_get_eos_id(model);

    char piece_buf[256];
    char escaped_buf[1024];
    char sse_chunk[2048];

    if (is_streaming) {
        /* Send SSE HTTP Headers */
        const char *sse_header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: keep-alive\r\n"
            "Access-Control-Allow-Origin: *\r\n\r\n";
        send(client_fd, sse_header, strlen(sse_header), 0);

        /* Role Header Chunk */
        const char *role_chunk = "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"qwen35\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
        send(client_fd, role_chunk, strlen(role_chunk), 0);

        diskllm_decode_token(tok, next_tok, true, piece_buf, sizeof(piece_buf));
        escape_json_string(piece_buf, escaped_buf, sizeof(escaped_buf));

        int len = snprintf(sse_chunk, sizeof(sse_chunk),
                           "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"qwen35\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                           escaped_buf);
        send(client_fd, sse_chunk, len, 0);

        int gen_tokens[4096];
        int gen_count = 0;

        while (gen_count < max_tokens) {
            if ((uint32_t)next_tok == eos_id || next_tok == 32007 || next_tok == 32000) break;
            gen_tokens[gen_count++] = next_tok;

            diskllm_decode_step(ctx, next_tok, logits);
            next_tok = diskllm_sample(smp, logits, vocab_size, gen_tokens, gen_count);

            diskllm_decode_token(tok, next_tok, false, piece_buf, sizeof(piece_buf));
            escape_json_string(piece_buf, escaped_buf, sizeof(escaped_buf));

            len = snprintf(sse_chunk, sizeof(sse_chunk),
                           "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"qwen35\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                           escaped_buf);
            send(client_fd, sse_chunk, len, 0);
        }

        const char *done_chunk =
            "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"qwen35\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
            "data: [DONE]\n\n";
        send(client_fd, done_chunk, strlen(done_chunk), 0);
    } else {
        /* Non-streaming response buffer */
        char full_content[65536];
        size_t c_len = 0;
        full_content[0] = '\0';

        diskllm_decode_token(tok, next_tok, true, piece_buf, sizeof(piece_buf));
        size_t p_len = strlen(piece_buf);
        if (c_len + p_len < sizeof(full_content) - 1) {
            memcpy(full_content + c_len, piece_buf, p_len);
            c_len += p_len;
            full_content[c_len] = '\0';
        }

        int gen_tokens[4096];
        int gen_count = 0;

        while (gen_count < max_tokens) {
            if ((uint32_t)next_tok == eos_id || next_tok == 32007 || next_tok == 32000) break;
            gen_tokens[gen_count++] = next_tok;

            diskllm_decode_step(ctx, next_tok, logits);
            next_tok = diskllm_sample(smp, logits, vocab_size, gen_tokens, gen_count);

            diskllm_decode_token(tok, next_tok, false, piece_buf, sizeof(piece_buf));
            p_len = strlen(piece_buf);
            if (c_len + p_len < sizeof(full_content) - 1) {
                memcpy(full_content + c_len, piece_buf, p_len);
                c_len += p_len;
                full_content[c_len] = '\0';
            }
        }

        char escaped_full[65536 * 2];
        escape_json_string(full_content, escaped_full, sizeof(escaped_full));

        char json_response[65536 * 2 + 512];
        int json_body_len = snprintf(json_response, sizeof(json_response),
            "{\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion\",\"created\":1700000000,\"model\":\"qwen35\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"%s\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
            escaped_full, prompt_len, gen_count, prompt_len + gen_count);

        char http_resp[65536 * 2 + 1024];
        int http_len = snprintf(http_resp, sizeof(http_resp),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: application/json\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Content-Length: %d\r\n"
            "Connection: close\r\n\r\n"
            "%s",
            json_body_len, json_response);

        send(client_fd, http_resp, http_len, 0);
    }

    free(formatted_prompt);
    free(logits);
    diskllm_sampler_free(smp);
    cJSON_Delete(json);
    free(buffer);
    close(client_fd);
}

int main(int argc, char **argv) {
    char *model_path = NULL;
    char *arch_str = "auto";
    char *host = "0.0.0.0";
    int port = 8080;
    int context_size = 4096;
    int num_threads = 0;
    bool pin_weights = false;
    char *io_mode_str = "pread";

    static struct option long_options[] = {
        {"model",       required_argument, 0, 'm'},
        {"port",        required_argument, 0, 'p'},
        {"host",        required_argument, 0, 'H'},
        {"arch",        required_argument, 0, 'a'},
        {"ctx",         required_argument, 0, 'c'},
        {"threads",     required_argument, 0, 't'},
        {"pin-weights", no_argument,       0, 'W'},
        {"io-mode",     required_argument, 0, 'I'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "m:p:H:a:c:t:WI:h", long_options, NULL)) != -1) {
        switch (opt) {
            case 'm': model_path = optarg; break;
            case 'p': port = atoi(optarg); break;
            case 'H': host = optarg; break;
            case 'a': arch_str = optarg; break;
            case 'c': context_size = atoi(optarg); break;
            case 't': num_threads = atoi(optarg); break;
            case 'W': pin_weights = true; break;
            case 'I': io_mode_str = optarg; break;
            case 'h': print_server_usage(argv[0]); return 0;
            default: break;
        }
    }

    if (!model_path) {
        fprintf(stderr, "Error: --model <path> is required.\n\n");
        print_server_usage(argv[0]);
        return 1;
    }

    diskllm_io_mode io_mode = DISKLLM_IO_PREAD;
    if (!strcmp(io_mode_str, "mmap")) io_mode = DISKLLM_IO_MMAP;
    else if (!strcmp(io_mode_str, "direct")) io_mode = DISKLLM_IO_DIRECT;

    diskllm_model_params mparams = diskllm_model_params_default();
    mparams.arch_flag = arch_str;
    mparams.pin_weights = pin_weights;
    mparams.io_mode = io_mode;
    mparams.num_threads = num_threads;

    printf("[SERVER] Loading model %s...\n", model_path);
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

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("Socket creation failed");
        diskllm_context_free(ctx); diskllm_model_free(model);
        return 1;
    }

    int opt_val = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
#ifdef SO_REUSEPORT
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEPORT, &opt_val, sizeof(opt_val));
#endif

    struct sockaddr_in address;
    memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(host);
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        close(server_fd);
        diskllm_context_free(ctx); diskllm_model_free(model);
        return 1;
    }

    if (listen(server_fd, 10) < 0) {
        perror("Listen failed");
        close(server_fd);
        diskllm_context_free(ctx); diskllm_model_free(model);
        return 1;
    }

    printf("\033[1;32m[SERVER]\033[0m DiskLLM OpenAI Compatible Server listening on http://%s:%d/v1/chat/completions\n", host, port);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addr_len);
        if (client_fd >= 0) {
            handle_client(client_fd, model, ctx);
        }
    }

    close(server_fd);
    diskllm_context_free(ctx);
    diskllm_model_free(model);
    return 0;
}
