#define _GNU_SOURCE
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#ifndef _LARGEFILE_SOURCE
#define _LARGEFILE_SOURCE
#endif
#ifndef _LARGEFILE64_SOURCE
#define _LARGEFILE64_SOURCE
#endif

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <assert.h>

/* ─── Hash Tables & Data Structures ────────────────────────────────────────── */

typedef struct {
    char *str;
    int32_t id;
} vocab_entry;

typedef struct {
    uint64_t key;       /* ((uint64_t)idA << 32) | (uint32_t)idB */
    int32_t merged_id;
    int32_t rank;
} merge_entry;

typedef struct {
    char *str;
    int32_t id;
    size_t len;
} special_token_entry;

struct tokenizer {
    char **tokens;
    int vocab_size;

    vocab_entry *vocab_table;
    size_t vocab_capacity;

    merge_entry *merge_table;
    size_t merge_capacity;
    int num_merges;

    special_token_entry special_tokens[128];
    int num_special;
};

/* ─── GPT-2 Byte-to-Unicode Mapping ───────────────────────────────────────── */

static uint32_t gpt2_bytes_to_unicode[256];
static int gpt2_table_initialized = 0;

static void init_gpt2_bytes_to_unicode(void) {
    if (gpt2_table_initialized) return;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            gpt2_bytes_to_unicode[b] = (uint32_t)b;
        } else {
            gpt2_bytes_to_unicode[b] = (uint32_t)(256 + n);
            n++;
        }
    }
    gpt2_table_initialized = 1;
}

static size_t codepoint_to_utf8(uint32_t cp, char *out) {
    if (cp < 128) {
        out[0] = (char)cp;
        out[1] = '\0';
        return 1;
    } else if (cp < 2048) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        out[2] = '\0';
        return 2;
    } else {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        out[3] = '\0';
        return 3;
    }
}

/* ─── Hash Map Utilities ────────────────────────────────────────────────────── */

static uint32_t hash_string(const char *str) {
    uint32_t hash = 2166136261u;
    for (; *str; str++) {
        hash ^= (uint8_t)*str;
        hash *= 16777619u;
    }
    return hash;
}

static void vocab_table_insert(tokenizer *tok, const char *str, int32_t id) {
    uint32_t h = hash_string(str);
    size_t mask = tok->vocab_capacity - 1;
    size_t idx = h & mask;
    while (tok->vocab_table[idx].str != NULL) {
        if (strcmp(tok->vocab_table[idx].str, str) == 0) return;
        idx = (idx + 1) & mask;
    }
    tok->vocab_table[idx].str = strdup(str);
    tok->vocab_table[idx].id = id;
}

static int32_t vocab_table_lookup(const tokenizer *tok, const char *str) {
    uint32_t h = hash_string(str);
    size_t mask = tok->vocab_capacity - 1;
    size_t idx = h & mask;
    while (tok->vocab_table[idx].str != NULL) {
        if (strcmp(tok->vocab_table[idx].str, str) == 0) {
            return tok->vocab_table[idx].id;
        }
        idx = (idx + 1) & mask;
    }
    return -1;
}

static uint32_t hash_uint64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)k;
}

static void merge_table_insert(tokenizer *tok, int32_t id_A, int32_t id_B, int32_t merged_id, int32_t rank) {
    uint64_t key = ((uint64_t)(uint32_t)id_A << 32) | (uint32_t)id_B;
    uint32_t h = hash_uint64(key);
    size_t mask = tok->merge_capacity - 1;
    size_t idx = h & mask;
    while (tok->merge_table[idx].key != 0xFFFFFFFFFFFFFFFFULL) {
        if (tok->merge_table[idx].key == key) return;
        idx = (idx + 1) & mask;
    }
    tok->merge_table[idx].key = key;
    tok->merge_table[idx].merged_id = merged_id;
    tok->merge_table[idx].rank = rank;
}

static int merge_table_lookup(const tokenizer *tok, int32_t id_A, int32_t id_B, int32_t *out_merged_id, int32_t *out_rank) {
    uint64_t key = ((uint64_t)(uint32_t)id_A << 32) | (uint32_t)id_B;
    uint32_t h = hash_uint64(key);
    size_t mask = tok->merge_capacity - 1;
    size_t idx = h & mask;
    while (tok->merge_table[idx].key != 0xFFFFFFFFFFFFFFFFULL) {
        if (tok->merge_table[idx].key == key) {
            *out_merged_id = tok->merge_table[idx].merged_id;
            *out_rank = tok->merge_table[idx].rank;
            return 1;
        }
        idx = (idx + 1) & mask;
    }
    return 0;
}

/* ─── GGUF Metadata Parsing ───────────────────────────────────────────────── */

static void skip_val(FILE *f, uint32_t val_type) {
    switch (val_type) {
        case 0: case 1: case 7: fseeko(f, 1, SEEK_CUR); break;
        case 2: case 3: fseeko(f, 2, SEEK_CUR); break;
        case 4: case 5: case 6: fseeko(f, 4, SEEK_CUR); break;
        case 10: case 11: case 12: fseeko(f, 8, SEEK_CUR); break;
        case 8: {
            uint64_t len;
            if (fread(&len, 8, 1, f) == 1) fseeko(f, len, SEEK_CUR);
            break;
        }
        case 9: {
            uint32_t elem_type; uint64_t len;
            if (fread(&elem_type, 4, 1, f) == 1 && fread(&len, 8, 1, f) == 1) {
                for (uint64_t i = 0; i < len; i++) skip_val(f, elem_type);
            }
            break;
        }
        default: break;
    }
}

static char *read_gguf_string(FILE *f) {
    uint64_t len;
    if (fread(&len, 8, 1, f) != 1) return NULL;
    char *s = malloc(len + 1);
    if (!s) return NULL;
    if (len > 0 && fread(s, 1, len, f) != len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

/* ─── Special Tokens Registration ─────────────────────────────────────────── */

static void register_special_token(tokenizer *tok, const char *str, int32_t id) {
    if (!str || tok->num_special >= 128) return;
    for (int i = 0; i < tok->num_special; i++) {
        if (strcmp(tok->special_tokens[i].str, str) == 0) return;
    }
    tok->special_tokens[tok->num_special].str = strdup(str);
    tok->special_tokens[tok->num_special].id = id;
    tok->special_tokens[tok->num_special].len = strlen(str);
    tok->num_special++;
}

static const struct {
    const char *str;
    int32_t fallback_id;
} DEFAULT_SPECIAL_TOKENS[] = {
    {"<|endoftext|>", 248044},
    {"<|im_start|>", 248045},
    {"<|im_end|>", 248046},
    {"<|object_ref_start|>", 248047},
    {"<|object_ref_end|>", 248048},
    {"<|box_start|>", 248049},
    {"<|box_end|>", 248050},
    {"<|quad_start|>", 248051},
    {"<|quad_end|>", 248052},
    {"<|vision_start|>", 248053},
    {"<|vision_end|>", 248054},
    {"<|vision_pad|>", 248055},
    {NULL, -1}
};

/* ─── Initialization ───────────────────────────────────────────────────────── */

tokenizer *tokenizer_init(const char *gguf_path) {
    init_gpt2_bytes_to_unicode();

    FILE *f = fopen(gguf_path, "rb");
    if (!f) return NULL;

    struct {
        char magic[4];
        uint32_t version;
        uint64_t tensor_count;
        uint64_t metadata_kv_count;
    } header;

    if (fread(&header, 1, sizeof(header), f) != sizeof(header) || strncmp(header.magic, "GGUF", 4) != 0) {
        fclose(f);
        return NULL;
    }

    tokenizer *tok = calloc(1, sizeof(tokenizer));
    if (!tok) { fclose(f); return NULL; }

    tok->vocab_capacity = 1048576; // 2^20
    tok->vocab_table = calloc(tok->vocab_capacity, sizeof(vocab_entry));

    tok->merge_capacity = 524288;  // 2^19
    tok->merge_table = malloc(tok->merge_capacity * sizeof(merge_entry));
    for (size_t i = 0; i < tok->merge_capacity; i++) {
        tok->merge_table[i].key = 0xFFFFFFFFFFFFFFFFULL;
    }

    char **raw_merges = NULL;
    int num_raw_merges = 0;

    for (uint64_t i = 0; i < header.metadata_kv_count; i++) {
        char *key = read_gguf_string(f);
        if (!key) break;

        uint32_t val_type;
        if (fread(&val_type, 4, 1, f) != 1) { free(key); break; }

        if (strcmp(key, "tokenizer.ggml.tokens") == 0 && val_type == 9) {
            uint32_t elem_type; uint64_t array_len;
            if (fread(&elem_type, 4, 1, f) == 1 && fread(&array_len, 8, 1, f) == 1 && elem_type == 8) {
                tok->vocab_size = (int)array_len;
                tok->tokens = calloc(tok->vocab_size, sizeof(char *));
                for (uint64_t idx = 0; idx < array_len; idx++) {
                    tok->tokens[idx] = read_gguf_string(f);
                    if (tok->tokens[idx]) {
                        vocab_table_insert(tok, tok->tokens[idx], (int32_t)idx);
                    }
                }
            } else skip_val(f, val_type);
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && val_type == 9) {
            uint32_t elem_type; uint64_t array_len;
            if (fread(&elem_type, 4, 1, f) == 1 && fread(&array_len, 8, 1, f) == 1 && elem_type == 8) {
                num_raw_merges = (int)array_len;
                raw_merges = calloc(num_raw_merges, sizeof(char *));
                for (uint64_t idx = 0; idx < array_len; idx++) {
                    raw_merges[idx] = read_gguf_string(f);
                }
            } else skip_val(f, val_type);
        } else {
            skip_val(f, val_type);
        }
        free(key);
    }
    fclose(f);

    /* Process raw merge rules after vocab table is built */
    if (raw_merges && tok->vocab_table) {
        for (int r = 0; r < num_raw_merges; r++) {
            if (!raw_merges[r]) continue;
            char *sp = strchr(raw_merges[r], ' ');
            if (sp) {
                *sp = '\0';
                const char *tokA = raw_merges[r];
                const char *tokB = sp + 1;
                int32_t idA = vocab_table_lookup(tok, tokA);
                int32_t idB = vocab_table_lookup(tok, tokB);

                char merged_str[512];
                snprintf(merged_str, sizeof(merged_str), "%s%s", tokA, tokB);
                int32_t merged_id = vocab_table_lookup(tok, merged_str);

                if (idA >= 0 && idB >= 0 && merged_id >= 0) {
                    merge_table_insert(tok, idA, idB, merged_id, r);
                }
            }
            free(raw_merges[r]);
        }
        free(raw_merges);
        tok->num_merges = num_raw_merges;
    }

    /* Register special tokens */
    for (int i = 0; DEFAULT_SPECIAL_TOKENS[i].str != NULL; i++) {
        int32_t id = vocab_table_lookup(tok, DEFAULT_SPECIAL_TOKENS[i].str);
        if (id < 0) id = DEFAULT_SPECIAL_TOKENS[i].fallback_id;
        register_special_token(tok, DEFAULT_SPECIAL_TOKENS[i].str, id);
    }

    return tok;
}

void tokenizer_free(tokenizer *tok) {
    if (!tok) return;
    if (tok->tokens) {
        for (int i = 0; i < tok->vocab_size; i++) free(tok->tokens[i]);
        free(tok->tokens);
    }
    if (tok->vocab_table) {
        for (size_t i = 0; i < tok->vocab_capacity; i++) free(tok->vocab_table[i].str);
        free(tok->vocab_table);
    }
    free(tok->merge_table);
    for (int i = 0; i < tok->num_special; i++) free(tok->special_tokens[i].str);
    free(tok);
}

const char *tokenizer_decode_raw(const tokenizer *tok, int32_t token_id) {
    if (!tok || token_id < 0 || token_id >= tok->vocab_size) return NULL;
    return tok->tokens[token_id];
}

int tokenizer_decode_token(const tokenizer *tok, int32_t token_id, int is_first_token, char *out_buf, size_t max_len) {
    if (!tok || !out_buf || max_len == 0) return 0;
    out_buf[0] = '\0';
    if (token_id < 0 || token_id >= tok->vocab_size) return 0;

    const char *raw = tok->tokens[token_id];
    if (!raw) return 0;

    size_t rlen = strlen(raw);

    // Handle byte fallback tokens like "<0x0A>" or "<0x20>"
    if (rlen == 6 && strncmp(raw, "<0x", 3) == 0 && raw[5] == '>') {
        char hex[3] = { raw[3], raw[4], '\0' };
        unsigned int byte_val = (unsigned int)strtoul(hex, NULL, 16);
        if (max_len > 1) {
            out_buf[0] = (char)byte_val;
            out_buf[1] = '\0';
            return 1;
        }
        return 0;
    }

    size_t out_idx = 0;
    size_t i = 0;

    // If first token and starts with GPT-2 space 'Ġ' (0xC4 0xA0), skip the leading Ġ
    if (is_first_token && rlen >= 2 && (unsigned char)raw[0] == 0xC4 && (unsigned char)raw[1] == 0xA0) {
        i += 2;
    }

    while (i < rlen && out_idx + 1 < max_len) {
        // GPT-2 'Ġ' (U+0120 = 0xC4 0xA0) -> ASCII space ' '
        if (i + 1 < rlen && (unsigned char)raw[i] == 0xC4 && (unsigned char)raw[i+1] == 0xA0) {
            out_buf[out_idx++] = ' ';
            i += 2;
        }
        // GPT-2 'Ċ' (U+010A = 0xC4 0x8A) -> ASCII newline '\n'
        else if (i + 1 < rlen && (unsigned char)raw[i] == 0xC4 && (unsigned char)raw[i+1] == 0x8A) {
            out_buf[out_idx++] = '\n';
            i += 2;
        }
        // GPT-2 'ĉ' (U+0109 = 0xC4 0x89) -> ASCII tab '\t'
        else if (i + 1 < rlen && (unsigned char)raw[i] == 0xC4 && (unsigned char)raw[i+1] == 0x89) {
            out_buf[out_idx++] = '\t';
            i += 2;
        }
        else {
            out_buf[out_idx++] = raw[i++];
        }
    }
    out_buf[out_idx] = '\0';
    return (int)out_idx;
}

int tokenizer_vocab_size(const tokenizer *tok) {
    return tok ? tok->vocab_size : 0;
}

/* ─── Pre-tokenizer Chunking & Encoding ─────────────────────────────────────── */

typedef enum {
    CHAR_TYPE_END = 0,
    CHAR_TYPE_SPACE,
    CHAR_TYPE_WHITESPACE,
    CHAR_TYPE_PUNCT,
    CHAR_TYPE_WORD
} char_type;

static char_type get_char_type(const char *p, size_t rem, size_t *out_len) {
    if (rem == 0) { *out_len = 0; return CHAR_TYPE_END; }
    unsigned char c = (unsigned char)*p;

    /* Multi-byte UTF-8 */
    if ((c & 0x80) != 0) {
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (len > rem) len = rem;
        *out_len = len;
        return CHAR_TYPE_WORD;
    }

    *out_len = 1;
    if (c == ' ') return CHAR_TYPE_SPACE;
    if (c == '\n' || c == '\r' || c == '\t') return CHAR_TYPE_WHITESPACE;
    if (ispunct(c)) return CHAR_TYPE_PUNCT;
    return CHAR_TYPE_WORD;
}

static void encode_chunk(const tokenizer *tok, const char *chunk_str, size_t chunk_len,
                         int32_t *out_tokens, int *token_cnt, int max_tokens) {
    if (chunk_len == 0) return;

    /* 1. Convert raw UTF-8 bytes of chunk into GPT-2 Unicode character token IDs */
    int32_t ids[1024];
    int n_ids = 0;

    for (size_t i = 0; i < chunk_len; i++) {
        uint8_t byte_val = (uint8_t)chunk_str[i];
        uint32_t cp = gpt2_bytes_to_unicode[byte_val];
        char u_str[8];
        codepoint_to_utf8(cp, u_str);

        int32_t id = vocab_table_lookup(tok, u_str);
        if (id < 0) {
            /* Fallback byte token <0xXX> */
            snprintf(u_str, sizeof(u_str), "<0x%02X>", byte_val);
            id = vocab_table_lookup(tok, u_str);
        }
        if (id >= 0 && n_ids < 1024) {
            ids[n_ids++] = id;
        }
    }

    /* 2. BPE Iterative Merge Loop */
    while (n_ids >= 2) {
        int32_t best_rank = 0x7FFFFFFF;
        int best_idx = -1;
        int32_t best_merged_id = -1;

        for (int i = 0; i < n_ids - 1; i++) {
            int32_t merged_id, rank;
            if (merge_table_lookup(tok, ids[i], ids[i+1], &merged_id, &rank)) {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_idx = i;
                    best_merged_id = merged_id;
                }
            }
        }

        if (best_idx == -1) break; /* No more merges possible */

        ids[best_idx] = best_merged_id;
        for (int i = best_idx + 1; i < n_ids - 1; i++) {
            ids[i] = ids[i+1];
        }
        n_ids--;
    }

    /* 3. Emit tokens */
    for (int i = 0; i < n_ids; i++) {
        if (out_tokens && *token_cnt < max_tokens) {
            out_tokens[*token_cnt] = ids[i];
        }
        (*token_cnt)++;
    }
}

static void encode_text_segment(const tokenizer *tok, const char *text, size_t text_len,
                                int32_t *out_tokens, int *token_cnt, int max_tokens) {
    size_t pos = 0;
    while (pos < text_len) {
        size_t clen = 0;
        char_type ctype = get_char_type(text + pos, text_len - pos, &clen);

        size_t chunk_start = pos;
        size_t chunk_len = clen;

        if (ctype == CHAR_TYPE_SPACE) {
            /* Peek character following space */
            size_t nlen = 0;
            char_type ntype = get_char_type(text + pos + clen, text_len - (pos + clen), &nlen);
            if (ntype == CHAR_TYPE_WORD || ntype == CHAR_TYPE_PUNCT) {
                /* Include space with leading word or punctuation */
                pos += clen;
                while (pos < text_len) {
                    size_t cur_len = 0;
                    char_type cur_type = get_char_type(text + pos, text_len - pos, &cur_len);
                    if (cur_type != ntype) break;
                    pos += cur_len;
                }
                chunk_len = pos - chunk_start;
            } else {
                /* Contiguous space / whitespace */
                pos += clen;
                while (pos < text_len) {
                    size_t cur_len = 0;
                    char_type cur_type = get_char_type(text + pos, text_len - pos, &cur_len);
                    if (cur_type != CHAR_TYPE_SPACE && cur_type != CHAR_TYPE_WHITESPACE) break;
                    pos += cur_len;
                }
                chunk_len = pos - chunk_start;
            }
        } else {
            /* Contiguous chars of same type */
            pos += clen;
            while (pos < text_len) {
                size_t cur_len = 0;
                char_type cur_type = get_char_type(text + pos, text_len - pos, &cur_len);
                if (cur_type != ctype) break;
                pos += cur_len;
            }
            chunk_len = pos - chunk_start;
        }

        encode_chunk(tok, text + chunk_start, chunk_len, out_tokens, token_cnt, max_tokens);
    }
}

/* ─── Main Tokenizer Encode API ───────────────────────────────────────────── */

int tokenizer_encode(const tokenizer *tok, const char *text, int32_t *out_tokens, int max_tokens) {
    if (!tok || !text) return 0;

    int token_cnt = 0;
    size_t text_len = strlen(text);
    size_t pos = 0;
    size_t segment_start = 0;

    while (pos < text_len) {
        int match_idx = -1;
        size_t match_len = 0;

        for (int s = 0; s < tok->num_special; s++) {
            size_t slen = tok->special_tokens[s].len;
            if (pos + slen <= text_len && strncmp(text + pos, tok->special_tokens[s].str, slen) == 0) {
                if (slen > match_len) {
                    match_len = slen;
                    match_idx = s;
                }
            }
        }

        if (match_idx >= 0) {
            /* Encode text before special token */
            if (pos > segment_start) {
                encode_text_segment(tok, text + segment_start, pos - segment_start,
                                    out_tokens, &token_cnt, max_tokens);
            }
            /* Emit special token ID */
            if (out_tokens && token_cnt < max_tokens) {
                out_tokens[token_cnt] = tok->special_tokens[match_idx].id;
            }
            token_cnt++;

            pos += match_len;
            segment_start = pos;
        } else {
            pos++;
        }
    }

    if (pos > segment_start) {
        encode_text_segment(tok, text + segment_start, pos - segment_start,
                            out_tokens, &token_cnt, max_tokens);
    }

    return token_cnt;
}
