#include "arch/registry.h"
#include "diskllm_internal.h"
#include "attention.h"
#include "kernels.h"
#include <strings.h>

typedef struct {
    const float *ffn_norm_w;
    const float *post_attn_norm_w;
    const float *post_ffw_norm_w;
    const void  *ffn_gate_w; int ffn_gate_w_type;
    const void  *ffn_up_w;   int ffn_up_w_type;
    const void  *ffn_down_w; int ffn_down_w_type;
    const void  *inp_gate_w; int inp_gate_w_type;
    const void  *proj_w;     int proj_w_type;
    const float *post_norm_w;
    const float *layer_output_scale;
    layer_type   l_type;
    union {
        attention_layer_weights attn;
    } u;
} layer_block_weights_internal;

static bool gemma4_init(diskllm_model *model) {
    (void)model;
    return true;
}

static bool gemma4_prefill_layer(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_states, int prompt_len) {
    diskllm_model *model = ctx->model;
    const qwen_model_config *cfg = model->cfg;
    const layer_block_weights_internal *blk = (const layer_block_weights_internal *)layer_weights;
    scratch_buffers *scratch = ctx->scratch;
    model_state *state = ctx->state;
    int n_embd_per_layer = (model->n_embd_per_layer > 0) ? model->n_embd_per_layer : 256;

    for (int pos = 0; pos < prompt_len; pos++) {
        float *h = hidden_states + pos * cfg->hidden_dim;
        
        /* 1. Self-Attention (with unscaled attention, V RMSNorm, post-attn norm & residual) */
        attention_forward(h, pos, layer_idx, &blk->u.attn, state, scratch, cfg);

        /* 2. Feed-Forward Network (GeGLU) */
        if (blk->ffn_norm_w) {
            rmsnorm_ext(scratch->hidden_state, h, blk->ffn_norm_w, cfg->hidden_dim, 1e-6f, 0);
        } else {
            memcpy(scratch->hidden_state, h, cfg->hidden_dim * sizeof(float));
        }

        matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
               cfg->hidden_dim, cfg->ffn_dim, blk->ffn_gate_w_type, scratch->ssm_qkv);
        matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
               cfg->hidden_dim, cfg->ffn_dim, blk->ffn_up_w_type,   scratch->ssm_qkv);

        geglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);

        matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
               cfg->ffn_dim, cfg->hidden_dim, blk->ffn_down_w_type, (float*)scratch->stream_buffer);

        if (blk->post_ffw_norm_w) {
            rmsnorm_ext(scratch->hidden_state, scratch->hidden_state, blk->post_ffw_norm_w, cfg->hidden_dim, 1e-6f, 0);
        }

        add_residual(h, h, scratch->hidden_state, cfg->hidden_dim);

        /* 3. Per-Layer Embedding (PLE / Laurel) Injection */
        if (blk->inp_gate_w && blk->proj_w) {
            const float *ple_layer = ctx->ple_prompt_cache ? 
                (ctx->ple_prompt_cache + ((size_t)pos * cfg->block_count + layer_idx) * n_embd_per_layer) :
                (ctx->ple_cache + layer_idx * n_embd_per_layer);

            matvec(scratch->ple_gate, blk->inp_gate_w, h, cfg->hidden_dim, n_embd_per_layer, blk->inp_gate_w_type, scratch->ssm_qkv);
            gelu(scratch->ple_gate, scratch->ple_gate, n_embd_per_layer);
            for (int j = 0; j < n_embd_per_layer; j++) {
                scratch->ple_mult[j] = scratch->ple_gate[j] * ple_layer[j];
            }
            matvec(scratch->ple_out, blk->proj_w, scratch->ple_mult, n_embd_per_layer, cfg->hidden_dim, blk->proj_w_type, (float*)scratch->stream_buffer);
            if (blk->post_norm_w) {
                rmsnorm_ext(scratch->ple_out, scratch->ple_out, blk->post_norm_w, cfg->hidden_dim, 1e-6f, 0);
            }
            add_residual(h, h, scratch->ple_out, cfg->hidden_dim);
        }

        /* 4. Layer Output Scaling */
        if (blk->layer_output_scale) {
            float s = *blk->layer_output_scale;
            for (int j = 0; j < cfg->hidden_dim; j++) {
                h[j] *= s;
            }
        }
    }
    return true;
}

static bool gemma4_decode_layer(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_single, int cur_pos) {
    diskllm_model *model = ctx->model;
    const qwen_model_config *cfg = model->cfg;
    const layer_block_weights_internal *blk = (const layer_block_weights_internal *)layer_weights;
    scratch_buffers *scratch = ctx->scratch;
    model_state *state = ctx->state;
    int n_embd_per_layer = (model->n_embd_per_layer > 0) ? model->n_embd_per_layer : 256;

    /* 1. Self-Attention (with unscaled attention, V RMSNorm, post-attn norm & residual) */
    attention_forward(hidden_single, cur_pos, layer_idx, &blk->u.attn, state, scratch, cfg);

    /* 2. Feed-Forward Network (GeGLU) */
    if (blk->ffn_norm_w) {
        rmsnorm_ext(scratch->hidden_state, hidden_single, blk->ffn_norm_w, cfg->hidden_dim, 1e-6f, 0);
    } else {
        memcpy(scratch->hidden_state, hidden_single, cfg->hidden_dim * sizeof(float));
    }

    matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
           cfg->hidden_dim, cfg->ffn_dim, blk->ffn_gate_w_type, scratch->ssm_qkv);
    matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
           cfg->hidden_dim, cfg->ffn_dim, blk->ffn_up_w_type,   scratch->ssm_qkv);

    geglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);

    matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
           cfg->ffn_dim, cfg->hidden_dim, blk->ffn_down_w_type, (float*)scratch->stream_buffer);

    if (blk->post_ffw_norm_w) {
        rmsnorm_ext(scratch->hidden_state, scratch->hidden_state, blk->post_ffw_norm_w, cfg->hidden_dim, 1e-6f, 0);
    }

    add_residual(hidden_single, hidden_single, scratch->hidden_state, cfg->hidden_dim);

    /* 3. Per-Layer Embedding (PLE / Laurel) Injection */
    if (blk->inp_gate_w && blk->proj_w && ctx->ple_cache) {
        const float *ple_layer = ctx->ple_cache + layer_idx * n_embd_per_layer;

        matvec(scratch->ple_gate, blk->inp_gate_w, hidden_single, cfg->hidden_dim, n_embd_per_layer, blk->inp_gate_w_type, scratch->ssm_qkv);
        gelu(scratch->ple_gate, scratch->ple_gate, n_embd_per_layer);
        for (int j = 0; j < n_embd_per_layer; j++) {
            scratch->ple_mult[j] = scratch->ple_gate[j] * ple_layer[j];
        }
        matvec(scratch->ple_out, blk->proj_w, scratch->ple_mult, n_embd_per_layer, cfg->hidden_dim, blk->proj_w_type, (float*)scratch->stream_buffer);
        if (blk->post_norm_w) {
            rmsnorm_ext(scratch->ple_out, scratch->ple_out, blk->post_norm_w, cfg->hidden_dim, 1e-6f, 0);
        }
        add_residual(hidden_single, hidden_single, scratch->ple_out, cfg->hidden_dim);
    }

    /* 4. Layer Output Scaling */
    if (blk->layer_output_scale) {
        float s = *blk->layer_output_scale;
        for (int j = 0; j < cfg->hidden_dim; j++) {
            hidden_single[j] *= s;
        }
    }

    return true;
}

const diskllm_arch_backend diskllm_arch_gemma4 = {
    .name = "Gemma4",
    .init = gemma4_init,
    .prefill_layer = gemma4_prefill_layer,
    .decode_layer = gemma4_decode_layer
};
