#ifndef SPECULATIVE_H
#define SPECULATIVE_H

#include "state.h"
#include "scratch.h"
#include "kernels.h"
#include "model_config.h"
#include "tensor_catalog.h"
#include "sampler.h"

typedef struct {
    const float *enorm_w;             // RMSNorm for token embedding e_t [hidden_dim]
    const float *hnorm_w;             // RMSNorm for backbone hidden state h_t [hidden_dim]
    const float *shared_head_norm_w;  // RMSNorm for h_next before output projection [hidden_dim]
    const void  *eh_proj_w;           // Projection matrix [2 * hidden_dim, hidden_dim]
    int          eh_proj_w_type;      // Tensor quantization type (e.g. Q8_0)
} nextn_weights;

typedef struct {
    int total_drafted;
    int total_accepted;
    int backbone_sweeps;
} speculative_stats;

/* Load NextN prediction head weights */
int load_nextn_weights(
    int fd,
    const tensor_catalog *cat,
    const qwen_model_config *cfg,
    const uint8_t *mmap_base,
    nextn_weights *weights,
    uint8_t *buf,
    uint64_t *bytes_read
);

/* Run a single NextN draft forward step to predict h_next from e_t and h_t */
void nextn_draft_step(
    const float *embed_t,
    const float *hidden_t,
    float *out_hidden_next,
    const nextn_weights *weights,
    scratch_buffers *scratch,
    const qwen_model_config *cfg
);

#endif // SPECULATIVE_H
