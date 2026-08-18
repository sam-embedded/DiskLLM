#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer tokenizer;

// Initialize tokenizer by reading vocabulary and merge rules from GGUF file.
// Returns NULL on failure.
tokenizer *tokenizer_init(const char *gguf_path);

// Free tokenizer memory.
void tokenizer_free(tokenizer *tok);

// Encode UTF-8 text into token IDs.
// Returns number of tokens produced.
// If out_tokens is NULL, returns total number of tokens needed without writing.
int tokenizer_encode(const tokenizer *tok, const char *text, int32_t *out_tokens, int max_tokens);

// Retrieve raw token string for a given token ID.
// Returns NULL if token ID is out of range.
const char *tokenizer_decode_raw(const tokenizer *tok, int32_t token_id);

// Decode token ID to clean UTF-8 text. Replaces GPT-2 BPE space ('Ġ') with ASCII space.
// If is_first_token is non-zero, strips any leading space.
// Returns length of output string written into out_buf.
int tokenizer_decode_token(const tokenizer *tok, int32_t token_id, int is_first_token, char *out_buf, size_t max_len);

// Get vocabulary size.
int tokenizer_vocab_size(const tokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_H
