#include "diskllm_internal.h"

diskllm_sampler_params diskllm_sampler_params_default(void) {
    return (diskllm_sampler_params){
        .temp = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .min_p = 0.05f,
        .repeat_penalty = 1.0f,
        .presence_penalty = 0.0f,
        .seed = 0
    };
}

diskllm_sampler *diskllm_sampler_init(diskllm_sampler_params params) {
    diskllm_sampler *s = calloc(1, sizeof(diskllm_sampler));
    if (!s) return NULL;

    s->params = params;
    sampler_config sc = {
        .temperature = params.temp,
        .top_p = params.top_p,
        .top_k = params.top_k,
        .min_p = params.min_p,
        .seed = params.seed
    };

    s->smp = sampler_create(&sc);
    if (!s->smp) {
        free(s);
        return NULL;
    }
    return s;
}

void diskllm_sampler_free(diskllm_sampler *smp) {
    if (!smp) return;
    if (smp->smp) sampler_free(smp->smp);
    free(smp);
}

static const char *get_token_string_cb(void *ctx, int token_id) {
    const diskllm_tokenizer *tok = (const diskllm_tokenizer *)ctx;
    if (!tok || !tok->tok) return NULL;
    return tokenizer_decode_raw(tok->tok, token_id);
}

int diskllm_sample_grammar(diskllm_sampler *smp, float *logits, int vocab_size, const int *seen_tokens, int n_seen, diskllm_grammar *grammar, const diskllm_tokenizer *tok, int eos_token_id) {
    if (!smp || !smp->smp || !logits || vocab_size <= 0) return 0;

    if (grammar && tok) {
        diskllm_grammar_apply_mask(grammar, logits, vocab_size, (void *)tok, get_token_string_cb, eos_token_id);
    }

    if (smp->params.presence_penalty != 0.0f && seen_tokens && n_seen > 0) {
        sampler_apply_presence_penalty(logits, vocab_size, seen_tokens, n_seen, smp->params.presence_penalty);
    }

    if (smp->params.repeat_penalty > 1.0f && seen_tokens && n_seen > 0) {
        sampler_apply_repetition_penalty(logits, vocab_size, seen_tokens, n_seen, smp->params.repeat_penalty);
    }

    int chosen_tok = sampler_sample(smp->smp, logits, vocab_size);

    if (grammar && tok && chosen_tok != eos_token_id) {
        const char *tstr = tokenizer_decode_raw(tok->tok, chosen_tok);
        if (tstr) {
            diskllm_grammar_advance(grammar, tstr);
        }
    }

    return chosen_tok;
}

int diskllm_sample(diskllm_sampler *smp, float *logits, int vocab_size, const int *seen_tokens, int n_seen) {
    return diskllm_sample_grammar(smp, logits, vocab_size, seen_tokens, n_seen, NULL, NULL, -1);
}
