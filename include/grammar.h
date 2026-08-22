#ifndef DISKLLM_GRAMMAR_H
#define DISKLLM_GRAMMAR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct diskllm_grammar diskllm_grammar;

/* Initialize grammar constraint from a GBNF grammar file or string */
diskllm_grammar *diskllm_grammar_init_from_file(const char *filepath);
diskllm_grammar *diskllm_grammar_init_from_str(const char *grammar_str);
diskllm_grammar *diskllm_grammar_init_json(void);

/* Free grammar instance */
void diskllm_grammar_free(diskllm_grammar *g);

/* Reset grammar to initial state */
void diskllm_grammar_reset(diskllm_grammar *g);

/* Check if appending a token string keeps the sequence valid under grammar */
bool diskllm_grammar_accept_str(const diskllm_grammar *g, const char *piece);

/* Advance grammar state upon accepting a chosen token string */
void diskllm_grammar_advance(diskllm_grammar *g, const char *piece);

/* Check if the grammar is currently in a valid end state (where EOS is allowed) */
bool diskllm_grammar_is_finished(const diskllm_grammar *g);

/* Filter/mask logits for all vocabulary tokens according to current grammar state.
 * Sets invalid tokens to -INFINITY.
 * Returns the number of valid tokens remaining.
 */
int diskllm_grammar_apply_mask(diskllm_grammar *g, float *logits, int vocab_size,
                              void *tok_ctx,
                              const char *(*get_token_str)(void *ctx, int token_id),
                              int eos_token_id);

#ifdef __cplusplus
}
#endif

#endif /* DISKLLM_GRAMMAR_H */
