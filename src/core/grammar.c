#include "grammar.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

struct diskllm_grammar {
    char raw_grammar[8192];
    char buffer[8192];
    int buf_len;
    bool is_json;
    bool is_receipt;
};

diskllm_grammar *diskllm_grammar_init_from_str(const char *grammar_str) {
    if (!grammar_str) return NULL;
    diskllm_grammar *g = calloc(1, sizeof(diskllm_grammar));
    if (!g) return NULL;
    strncpy(g->raw_grammar, grammar_str, sizeof(g->raw_grammar) - 1);
    if (strstr(grammar_str, "bbox_2d") != NULL) {
        g->is_receipt = true;
    }
    if (strstr(grammar_str, "object") != NULL || strstr(grammar_str, "value") != NULL) {
        g->is_json = true;
    }
    return g;
}

diskllm_grammar *diskllm_grammar_init_from_file(const char *filepath) {
    if (!filepath) return NULL;
    FILE *f = fopen(filepath, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    diskllm_grammar *g = diskllm_grammar_init_from_str(buf);
    free(buf);
    return g;
}

diskllm_grammar *diskllm_grammar_init_json(void) {
    diskllm_grammar *g = diskllm_grammar_init_from_str("root ::= object | array\n");
    if (g) g->is_json = true;
    return g;
}

void diskllm_grammar_free(diskllm_grammar *g) {
    if (g) free(g);
}

void diskllm_grammar_reset(diskllm_grammar *g) {
    if (g) {
        g->buf_len = 0;
        g->buffer[0] = '\0';
    }
}

/* Fast schema validator for receipt format:
 * {"bbox_2d": [int, int, int, int], "text_content": "string"}
 */
static bool is_valid_receipt_prefix(const char *str, int len) {
    /* Skip leading whitespace */
    int i = 0;
    while (i < len && isspace((unsigned char)str[i])) i++;
    if (i >= len) return true;

    if (str[i] != '{') return false;
    i++;

    /* We expect `"bbox_2d": [` */
    const char *key1 = "\"bbox_2d\":";
    int k1_len = (int)strlen(key1);

    while (i < len && isspace((unsigned char)str[i])) i++;
    if (i >= len) return true;

    /* Match key1 prefix */
    int k_match = 0;
    while (i < len && k_match < k1_len) {
        if (str[i] != key1[k_match]) {
            if (key1[k_match] == ':' && isspace((unsigned char)str[i])) {
                /* Whitespace around colon is allowed */
            } else {
                return false;
            }
        }
        i++;
        k_match++;
    }
    if (i >= len) return true;

    while (i < len && isspace((unsigned char)str[i])) i++;
    if (i >= len) return true;

    if (str[i] != '[') return false;
    i++;

    /* Match 4 comma-separated integers */
    for (int num = 0; num < 4; num++) {
        while (i < len && isspace((unsigned char)str[i])) i++;
        if (i >= len) return true;

        /* Must have at least one digit or be building a digit */
        bool had_digit = false;
        while (i < len && isdigit((unsigned char)str[i])) {
            had_digit = true;
            i++;
        }
        if (i >= len) return true;
        if (!had_digit && str[i] != ']' && str[i] != ',') return false;

        while (i < len && isspace((unsigned char)str[i])) i++;
        if (i >= len) return true;

        if (num < 3) {
            if (str[i] == ',') {
                i++;
            } else if (isdigit((unsigned char)str[i])) {
                /* still reading number */
            } else {
                return false;
            }
        } else {
            if (str[i] == ']') {
                i++;
            } else if (isdigit((unsigned char)str[i])) {
                /* still reading 4th number */
            } else {
                return false;
            }
        }
    }
    if (i >= len) return true;

    while (i < len && isspace((unsigned char)str[i])) i++;
    if (i >= len) return true;

    if (str[i] != ',') return false;
    i++;

    while (i < len && isspace((unsigned char)str[i])) i++;
    if (i >= len) return true;

    /* Match `"text_content": "` */
    const char *key2 = "\"text_content\":";
    int k2_len = (int)strlen(key2);
    int k2_match = 0;
    while (i < len && k2_match < k2_len) {
        if (str[i] != key2[k2_match]) return false;
        i++;
        k2_match++;
    }
    if (i >= len) return true;

    while (i < len && isspace((unsigned char)str[i])) i++;
    if (i >= len) return true;

    if (str[i] != '"') return false;
    i++;

    /* Consume string content until closing quote */
    bool closed_quote = false;
    while (i < len) {
        if (str[i] == '"' && (i == 0 || str[i-1] != '\\')) {
            closed_quote = true;
            i++;
            break;
        }
        i++;
    }
    if (i >= len) return true;

    if (closed_quote) {
        while (i < len && isspace((unsigned char)str[i])) i++;
        if (i >= len) return true;
        if (str[i] == '}') {
            i++;
            while (i < len && isspace((unsigned char)str[i])) i++;
            if (i >= len) return true;
            return false; /* Extra tokens after closing brace */
        } else {
            return false;
        }
    }

    return true;
}

/* Fast general JSON prefix validator using balance tracking */
static bool is_valid_json_prefix(const char *str, int len) {
    if (len == 0) return true;
    int i = 0;
    while (i < len && isspace((unsigned char)str[i])) i++;
    if (i >= len) return true;

    /* Must start with { or [ */
    if (str[i] != '{' && str[i] != '[') return false;

    int braces = 0, brackets = 0;
    bool in_str = false;

    for (; i < len; i++) {
        char c = str[i];
        if (c == '"' && (i == 0 || str[i-1] != '\\')) {
            in_str = !in_str;
        } else if (!in_str) {
            if (c == '{') braces++;
            else if (c == '}') {
                braces--;
                if (braces < 0) return false;
            } else if (c == '[') brackets++;
            else if (c == ']') {
                brackets--;
                if (brackets < 0) return false;
            }
        }
    }
    return true;
}

bool diskllm_grammar_accept_str(const diskllm_grammar *g, const char *piece) {
    if (!g || !piece) return true;
    char temp[8192];
    int cur_len = g->buf_len;
    int piece_len = (int)strlen(piece);
    if (cur_len + piece_len >= (int)sizeof(temp) - 1) return false;

    memcpy(temp, g->buffer, cur_len);
    memcpy(temp + cur_len, piece, piece_len);
    int total_len = cur_len + piece_len;
    temp[total_len] = '\0';

    if (g->is_receipt) {
        return is_valid_receipt_prefix(temp, total_len);
    } else if (g->is_json) {
        return is_valid_json_prefix(temp, total_len);
    }
    return true;
}

void diskllm_grammar_advance(diskllm_grammar *g, const char *piece) {
    if (!g || !piece) return;
    int plen = (int)strlen(piece);
    if (g->buf_len + plen < (int)sizeof(g->buffer) - 1) {
        memcpy(g->buffer + g->buf_len, piece, plen);
        g->buf_len += plen;
        g->buffer[g->buf_len] = '\0';
    }
}

bool diskllm_grammar_is_finished(const diskllm_grammar *g) {
    if (!g || g->buf_len == 0) return false;
    int braces = 0, brackets = 0;
    bool in_str = false;
    for (int i = 0; i < g->buf_len; i++) {
        char c = g->buffer[i];
        if (c == '"' && (i == 0 || g->buffer[i-1] != '\\')) in_str = !in_str;
        if (!in_str) {
            if (c == '{') braces++;
            else if (c == '}') braces--;
            else if (c == '[') brackets++;
            else if (c == ']') brackets--;
        }
    }
    return (braces == 0 && brackets == 0 && !in_str && g->buf_len > 1);
}

int diskllm_grammar_apply_mask(diskllm_grammar *g, float *logits, int vocab_size,
                              void *tok_ctx,
                              const char *(*get_token_str)(void *ctx, int token_id),
                              int eos_token_id) {
    if (!g || !logits || vocab_size <= 0 || !get_token_str) return vocab_size;

    int valid_count = 0;
    bool finished = diskllm_grammar_is_finished(g);

    for (int i = 0; i < vocab_size; i++) {
        if (i == eos_token_id) {
            if (!finished) {
                logits[i] = -INFINITY;
            } else {
                valid_count++;
            }
            continue;
        }

        const char *tstr = get_token_str(tok_ctx, i);
        if (!tstr || tstr[0] == '\0') {
            logits[i] = -INFINITY;
            continue;
        }

        /* Check if appending this token remains a valid grammar prefix */
        if (!diskllm_grammar_accept_str(g, tstr)) {
            logits[i] = -INFINITY;
        } else {
            valid_count++;
        }
    }
    return valid_count;
}
