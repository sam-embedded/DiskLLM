#include "arch/registry.h"
#include "diskllm_internal.h"
#include "attention.h"
#include "ssm.h"
#include "kernels.h"
#include "dequant.h"

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
        ssm_layer_weights       ssm;
    } u;
} layer_block_weights_internal;

static bool qwen_init(diskllm_model *model) {
    (void)model;
    return true;
}

static bool qwen_prefill_layer(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_states, int prompt_len) {
    diskllm_model *model = ctx->model;
    const qwen_model_config *cfg = model->cfg;
    const layer_block_weights_internal *blk = (const layer_block_weights_internal *)layer_weights;
    scratch_buffers *scratch = ctx->scratch;
    model_state *state = ctx->state;

    int add_one = (cfg->model_type == MODEL_TYPE_GEMMA) ? 1 : 0;

    for (int pos = 0; pos < prompt_len; pos++) {
        float *h = hidden_states + pos * cfg->hidden_dim;
        if (blk->l_type == LAYER_TYPE_ATTENTION) {
            attention_forward(h, pos, layer_idx, &blk->u.attn, state, scratch, cfg);
        } else {
            ssm_layer_forward(h, pos, layer_idx, &blk->u.ssm, state, scratch, cfg);
        }

        rmsnorm_ext(scratch->hidden_state, h, blk->ffn_norm_w, cfg->hidden_dim, 1e-6f, add_one);
        if (blk->ffn_gate_w) {
            matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
                   cfg->hidden_dim, cfg->ffn_dim, blk->ffn_gate_w_type, scratch->ssm_qkv);
            matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
                   cfg->hidden_dim, cfg->ffn_dim, blk->ffn_up_w_type,   scratch->ssm_qkv);
        } else {
            matvec(scratch->ffn_gate, blk->ffn_up_w, scratch->hidden_state,
                   cfg->hidden_dim, 2 * cfg->ffn_dim, blk->ffn_up_w_type, scratch->ssm_qkv);
            memcpy(scratch->ffn_up, scratch->ffn_gate + cfg->ffn_dim, cfg->ffn_dim * sizeof(float));
        }

        if (cfg->model_type == MODEL_TYPE_GEMMA) {
            geglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
        } else {
            swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
        }

        matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
               cfg->ffn_dim, cfg->hidden_dim, blk->ffn_down_w_type, (float*)scratch->stream_buffer);

        add_residual(h, h, scratch->hidden_state, cfg->hidden_dim);
    }
    return true;
}

static bool qwen_decode_layer(diskllm_context *ctx, int layer_idx, const void *layer_weights, float *hidden_single, int cur_pos) {
    diskllm_model *model = ctx->model;
    const qwen_model_config *cfg = model->cfg;
    const layer_block_weights_internal *blk = (const layer_block_weights_internal *)layer_weights;
    scratch_buffers *scratch = ctx->scratch;
    model_state *state = ctx->state;

    int add_one = (cfg->model_type == MODEL_TYPE_GEMMA) ? 1 : 0;

    if (blk->l_type == LAYER_TYPE_ATTENTION) {
        attention_forward(hidden_single, cur_pos, layer_idx, &blk->u.attn, state, scratch, cfg);
    } else {
        ssm_layer_forward(hidden_single, cur_pos, layer_idx, &blk->u.ssm, state, scratch, cfg);
    }

    rmsnorm_ext(scratch->hidden_state, hidden_single, blk->ffn_norm_w, cfg->hidden_dim, 1e-6f, add_one);
    if (blk->ffn_gate_w) {
        matvec(scratch->ffn_gate, blk->ffn_gate_w, scratch->hidden_state,
               cfg->hidden_dim, cfg->ffn_dim, blk->ffn_gate_w_type, scratch->ssm_qkv);
        matvec(scratch->ffn_up,   blk->ffn_up_w,   scratch->hidden_state,
               cfg->hidden_dim, cfg->ffn_dim, blk->ffn_up_w_type,   scratch->ssm_qkv);
    } else {
        matvec(scratch->ffn_gate, blk->ffn_up_w, scratch->hidden_state,
               cfg->hidden_dim, 2 * cfg->ffn_dim, blk->ffn_up_w_type, scratch->ssm_qkv);
        memcpy(scratch->ffn_up, scratch->ffn_gate + cfg->ffn_dim, cfg->ffn_dim * sizeof(float));
    }

    if (cfg->model_type == MODEL_TYPE_GEMMA) {
        geglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
    } else {
        swiglu(scratch->ffn_gate, scratch->ffn_gate, scratch->ffn_up, cfg->ffn_dim);
    }
    matvec(scratch->hidden_state, blk->ffn_down_w, scratch->ffn_gate,
           cfg->ffn_dim, cfg->hidden_dim, blk->ffn_down_w_type, (float*)scratch->stream_buffer);
    add_residual(hidden_single, hidden_single, scratch->hidden_state, cfg->hidden_dim);
    return true;
}

const diskllm_arch_backend diskllm_arch_qwen = {
    .name = "Qwen",
    .init = qwen_init,
    .prefill_layer = qwen_prefill_layer,
    .decode_layer = qwen_decode_layer
};
