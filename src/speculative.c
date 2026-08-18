#define _POSIX_C_SOURCE 200809L

#include "speculative.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

int load_nextn_weights(
    int fd,
    const tensor_catalog *cat,
    const qwen_model_config *cfg,
    const uint8_t *mmap_base,
    nextn_weights *weights,
    uint8_t *buf,
    uint64_t *bytes_read
) {
    if (!cat || !cfg || !weights) return -1;

    int blk_idx = cfg->block_count; // Block 64 for 27B model
    char nm[256];

    snprintf(nm, sizeof(nm), "blk.%d.nextn.eh_proj.weight", blk_idx);
    const tensor_info *ti_eh = find_tensor(cat, nm);
    if (!ti_eh) {
        // Fallback: check blk.N-1 if blk_idx was 0-indexed
        snprintf(nm, sizeof(nm), "blk.%d.nextn.eh_proj.weight", cfg->block_count - 1);
        ti_eh = find_tensor(cat, nm);
        if (ti_eh) blk_idx = cfg->block_count - 1;
    }

    if (!ti_eh) {
        return -1; // NextN tensors not present in this model
    }

    snprintf(nm, sizeof(nm), "blk.%d.nextn.enorm.weight", blk_idx);
    const tensor_info *ti_enorm = find_tensor(cat, nm);

    snprintf(nm, sizeof(nm), "blk.%d.nextn.hnorm.weight", blk_idx);
    const tensor_info *ti_hnorm = find_tensor(cat, nm);

    snprintf(nm, sizeof(nm), "blk.%d.nextn.shared_head_norm.weight", blk_idx);
    const tensor_info *ti_snorm = find_tensor(cat, nm);

    if (!ti_enorm || !ti_hnorm || !ti_snorm) {
        fprintf(stderr, "[ERROR] Incomplete NextN layer tensors in block %d.\n", blk_idx);
        return -1;
    }

    if (mmap_base) {
        weights->eh_proj_w          = (const void *)(mmap_base + ti_eh->absolute_offset);
        weights->enorm_w            = (const float *)(mmap_base + ti_enorm->absolute_offset);
        weights->hnorm_w            = (const float *)(mmap_base + ti_hnorm->absolute_offset);
        weights->shared_head_norm_w = (const float *)(mmap_base + ti_snorm->absolute_offset);
        weights->eh_proj_w_type     = ti_eh->type;
    } else {
        if (!buf) return -1;
        uint8_t *p = buf;

        ssize_t r = pread(fd, p, ti_enorm->byte_size, ti_enorm->absolute_offset);
        if (r < (ssize_t)ti_enorm->byte_size) return -1;
        weights->enorm_w = (const float *)p; p += ti_enorm->byte_size;
        if (bytes_read) *bytes_read += ti_enorm->byte_size;

        r = pread(fd, p, ti_hnorm->byte_size, ti_hnorm->absolute_offset);
        if (r < (ssize_t)ti_hnorm->byte_size) return -1;
        weights->hnorm_w = (const float *)p; p += ti_hnorm->byte_size;
        if (bytes_read) *bytes_read += ti_hnorm->byte_size;

        r = pread(fd, p, ti_snorm->byte_size, ti_snorm->absolute_offset);
        if (r < (ssize_t)ti_snorm->byte_size) return -1;
        weights->shared_head_norm_w = (const float *)p; p += ti_snorm->byte_size;
        if (bytes_read) *bytes_read += ti_snorm->byte_size;

        r = pread(fd, p, ti_eh->byte_size, ti_eh->absolute_offset);
        if (r < (ssize_t)ti_eh->byte_size) return -1;
        weights->eh_proj_w = (const void *)p; p += ti_eh->byte_size;
        weights->eh_proj_w_type = ti_eh->type;
        if (bytes_read) *bytes_read += ti_eh->byte_size;
    }

    return 0;
}

void nextn_draft_step(
    const float *embed_t,
    const float *hidden_t,
    float *out_hidden_next,
    const nextn_weights *weights,
    scratch_buffers *scratch,
    const qwen_model_config *cfg
) {
    int hidden_dim = cfg->hidden_dim;

    /* 1. Normalize token embedding e_t */
    rmsnorm(scratch->hidden_state, embed_t, weights->enorm_w, hidden_dim, 1e-6f);

    /* 2. Normalize backbone hidden state h_t */
    rmsnorm(scratch->hidden_state + hidden_dim, hidden_t, weights->hnorm_w, hidden_dim, 1e-6f);

    /* 3. Project concatenated [e_t, h_t] (size 2 * hidden_dim) to h_next (size hidden_dim) */
    matvec(
        out_hidden_next,
        weights->eh_proj_w,
        scratch->hidden_state,
        2 * hidden_dim,
        hidden_dim,
        weights->eh_proj_w_type,
        scratch->ssm_qkv
    );

    /* 4. Add residual connection h_t */
    add_residual(out_hidden_next, out_hidden_next, hidden_t, hidden_dim);
}
