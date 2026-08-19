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

static void handle_client(int client_fd, diskllm_model *model, diskllm_context *ctx) {
    char buffer[BUFFER_SIZE];
    ssize_t bytes_received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    if (bytes_received <= 0) {
        close(client_fd);
        return;
    }
    buffer[bytes_received] = '\0';

    /* Check HTTP Method & Path */
    char method[16], path[256];
    sscanf(buffer, "%15s %255s", method, path);

    if (strcmp(method, "POST") != 0 || strcmp(path, "/v1/chat/completions") != 0) {
        const char *not_found = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 9\r\n\r\nNot Found";
        send(client_fd, not_found, strlen(not_found), 0);
        close(client_fd);
        return;
    }

    /* Locate JSON Payload */
    char *body = strstr(buffer, "\r\n\r\n");
    if (!body) {
        close(client_fd);
        return;
    }
    body += 4;

    cJSON *json = cJSON_Parse(body);
    if (!json) {
        const char *bad_req = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: 15\r\n\r\nInvalid Payload";
        send(client_fd, bad_req, strlen(bad_req), 0);
        close(client_fd);
        return;
    }

    cJSON *messages = cJSON_GetObjectItemCaseSensitive(json, "messages");
    cJSON *max_tokens_item = cJSON_GetObjectItemCaseSensitive(json, "max_tokens");
    cJSON *temp_item = cJSON_GetObjectItemCaseSensitive(json, "temperature");

    int max_tokens = max_tokens_item ? max_tokens_item->valueint : 128;
    float temp = temp_item ? (float)temp_item->valuedouble : 0.7f;

    const char *sys_prompt = NULL;
    const char *user_prompt = NULL;

    if (cJSON_IsArray(messages)) {
        int count = cJSON_GetArraySize(messages);
        for (int i = 0; i < count; i++) {
            cJSON *item = cJSON_GetArrayItem(messages, i);
            cJSON *role = cJSON_GetObjectItemCaseSensitive(item, "role");
            cJSON *content = cJSON_GetObjectItemCaseSensitive(item, "content");
            if (role && content && role->valuestring && content->valuestring) {
                if (!strcmp(role->valuestring, "system")) sys_prompt = content->valuestring;
                else if (!strcmp(role->valuestring, "user")) user_prompt = content->valuestring;
            }
        }
    }

    if (!user_prompt) {
        user_prompt = "Hello";
    }

    char *chat_prompt = diskllm_format_chat_prompt(model, sys_prompt, user_prompt);
    if (!chat_prompt) {
        cJSON_Delete(json);
        close(client_fd);
        return;
    }

    diskllm_tokenizer *tok = diskllm_model_get_tokenizer(model);
    int prompt_tokens[4096];
    int prompt_len = diskllm_tokenize(tok, chat_prompt, prompt_tokens, 4096, true);

    int vocab_size = diskllm_model_get_vocab_size(model);
    float *logits = malloc(vocab_size * sizeof(float));

    /* Prefill */
    diskllm_eval(ctx, prompt_tokens, prompt_len, logits);

    /* Initialize sampler */
    diskllm_sampler_params sparams = diskllm_sampler_params_default();
    sparams.temp = temp;
    diskllm_sampler *smp = diskllm_sampler_init(sparams);

    int next_tok = diskllm_sample(smp, logits, vocab_size, prompt_tokens, prompt_len);

    /* Send SSE HTTP Headers */
    const char *sse_header = 
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n\r\n";
    send(client_fd, sse_header, strlen(sse_header), 0);

    /* Role Header Chunk */
    const char *role_chunk = "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"diskllm\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
    send(client_fd, role_chunk, strlen(role_chunk), 0);

    uint32_t eos_id = diskllm_model_get_eos_id(model);
    char piece_buf[256];
    char escaped_buf[1024];
    char sse_chunk[2048];

    diskllm_decode_token(tok, next_tok, true, piece_buf, sizeof(piece_buf));
    escape_json_string(piece_buf, escaped_buf, sizeof(escaped_buf));

    int len = snprintf(sse_chunk, sizeof(sse_chunk),
                       "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"diskllm\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
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
                       "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"diskllm\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n",
                       escaped_buf);
        send(client_fd, sse_chunk, len, 0);
    }

    const char *done_chunk = 
        "data: {\"id\":\"chatcmpl-diskllm\",\"object\":\"chat.completion.chunk\",\"created\":1700000000,\"model\":\"diskllm\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n";
    send(client_fd, done_chunk, strlen(done_chunk), 0);

    free(chat_prompt);
    free(logits);
    diskllm_sampler_free(smp);
    cJSON_Delete(json);
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
