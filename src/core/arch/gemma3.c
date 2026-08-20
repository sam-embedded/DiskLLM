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

static bool gemma3_init(diskllm_model *model) {
    (void)model;
    return true;
}

static bool gemma3_prefill_layer(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_states, int prompt_len) {
    diskllm_model *model = ctx->model;
    const qwen_model_config *cfg = model->cfg;
    const layer_block_weights_internal *blk = (const layer_block_weights_internal *)layer_weights;
    scratch_buffers *scratch = ctx->scratch;
    model_state *state = ctx->state;
    int add_one = 0;

    for (int pos = 0; pos < prompt_len; pos++) {
        float *h = hidden_states + pos * cfg->hidden_dim;
        attention_forward(h, pos, layer_idx, &blk->u.attn, state, scratch, cfg);

        if (blk->ffn_norm_w) {
            rmsnorm_ext(scratch->hidden_state, h, blk->ffn_norm_w, cfg->hidden_dim, 1e-6f, add_one);
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
            rmsnorm_ext(scratch->hidden_state, scratch->hidden_state, blk->post_ffw_norm_w, cfg->hidden_dim, 1e-6f, add_one);
        }

        add_residual(h, h, scratch->hidden_state, cfg->hidden_dim);
    }
    return true;
}

static bool gemma3_decode_layer(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_single, int cur_pos) {
    diskllm_model *model = ctx->model;
    const qwen_model_config *cfg = model->cfg;
    const layer_block_weights_internal *blk = (const layer_block_weights_internal *)layer_weights;
    scratch_buffers *scratch = ctx->scratch;
    model_state *state = ctx->state;
    int add_one = 0;

    attention_forward(hidden_single, cur_pos, layer_idx, &blk->u.attn, state, scratch, cfg);

    if (blk->ffn_norm_w) {
        rmsnorm_ext(scratch->hidden_state, hidden_single, blk->ffn_norm_w, cfg->hidden_dim, 1e-6f, add_one);
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
        rmsnorm_ext(scratch->hidden_state, scratch->hidden_state, blk->post_ffw_norm_w, cfg->hidden_dim, 1e-6f, add_one);
    }

    add_residual(hidden_single, hidden_single, scratch->hidden_state, cfg->hidden_dim);
    return true;
}

const diskllm_arch_backend diskllm_arch_gemma3 = {
    .name = "Gemma3",
    .init = gemma3_init,
    .prefill_layer = gemma3_prefill_layer,
    .decode_layer = gemma3_decode_layer
};
