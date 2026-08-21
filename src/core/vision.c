#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "vision.h"
#include "diskllm.h"
#include "diskllm_internal.h"
#include "dequant.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static ssize_t v_exact_pread(int fd, void *buf, size_t count, off_t offset) {
    size_t done = 0;
    uint8_t *ptr = (uint8_t *)buf;
    while (done < count) {
        ssize_t r = pread(fd, ptr + done, count - done, offset + done);
        if (r < 0) return -1;
        if (r == 0) break;
        done += r;
    }
    return done;
}

float *diskllm_image_load_rgb(const char *filepath, int target_w, int target_h, int *out_w, int *out_h, const float mean[3], const float std[3]) {
    if (!filepath) return NULL;
    int w, h, channels;
    unsigned char *img = stbi_load(filepath, &w, &h, &channels, 3);
    if (!img) {
        fprintf(stderr, "[ERROR] Vision: Failed to load image from %s\n", filepath);
        return NULL;
    }

    int tw = (target_w > 0) ? target_w : w;
    int th = (target_h > 0) ? target_h : h;

    float *rgb_data = malloc((size_t)3 * tw * th * sizeof(float));
    if (!rgb_data) {
        stbi_image_free(img);
        return NULL;
    }

    float m[3] = {mean ? mean[0] : 0.481455f, mean ? mean[1] : 0.457828f, mean ? mean[2] : 0.408211f};
    float s[3] = {std ? std[0] : 0.26863f, std ? std[1] : 0.261303f, std ? std[2] : 0.275777f};
    if (s[0] == 0.0f) s[0] = 1.0f;
    if (s[1] == 0.0f) s[1] = 1.0f;
    if (s[2] == 0.0f) s[2] = 1.0f;

    /* Bilinear interpolation resize and channel normalization */
    for (int y = 0; y < th; y++) {
        float src_y = (float)y * (float)h / (float)th;
        int y0 = (int)src_y;
        int y1 = (y0 + 1 < h) ? y0 + 1 : y0;
        float dy = src_y - y0;

        for (int x = 0; x < tw; x++) {
            float src_x = (float)x * (float)w / (float)tw;
            int x0 = (int)src_x;
            int x1 = (x0 + 1 < w) ? x0 + 1 : x0;
            float dx = src_x - x0;

            for (int c = 0; c < 3; c++) {
                float v00 = img[(y0 * w + x0) * 3 + c] / 255.0f;
                float v01 = img[(y0 * w + x1) * 3 + c] / 255.0f;
                float v10 = img[(y1 * w + x0) * 3 + c] / 255.0f;
                float v11 = img[(y1 * w + x1) * 3 + c] / 255.0f;

                float val = (1.0f - dy) * ((1.0f - dx) * v00 + dx * v01) + dy * ((1.0f - dx) * v10 + dx * v11);
                val = (val - m[c]) / s[c];

                /* Planar layout [3, th, tw] */
                rgb_data[(c * th + y) * tw + x] = val;
            }
        }
    }

    stbi_image_free(img);
    if (out_w) *out_w = tw;
    if (out_h) *out_h = th;
    return rgb_data;
}

void diskllm_image_free(float *rgb_data) {
    if (rgb_data) free(rgb_data);
}

diskllm_vision_model *diskllm_vision_load(const char *mmproj_path) {
    if (!mmproj_path) return NULL;
    tensor_catalog *cat = load_tensor_catalog(mmproj_path);
    if (!cat) {
        fprintf(stderr, "[ERROR] Vision: Failed to load tensor catalog from %s\n", mmproj_path);
        return NULL;
    }

    diskllm_vision_model *vm = calloc(1, sizeof(diskllm_vision_model));
    if (!vm) {
        free_tensor_catalog(cat);
        return NULL;
    }

    snprintf(vm->mmproj_path, sizeof(vm->mmproj_path), "%s", mmproj_path);
    vm->catalog = cat;

    /* Parse metadata */
    uint32_t val32 = 0;
    if (get_metadata_uint32(mmproj_path, "clip.vision.image_size", &val32) == 0) vm->cfg.image_size = (int)val32;
    else vm->cfg.image_size = 560;

    if (get_metadata_uint32(mmproj_path, "clip.vision.patch_size", &val32) == 0) vm->cfg.patch_size = (int)val32;
    else vm->cfg.patch_size = 14;

    if (get_metadata_uint32(mmproj_path, "clip.vision.embedding_length", &val32) == 0) vm->cfg.hidden_dim = (int)val32;
    else vm->cfg.hidden_dim = 1280;

    if (get_metadata_uint32(mmproj_path, "clip.vision.attention.head_count", &val32) == 0) vm->cfg.num_heads = (int)val32;
    else vm->cfg.num_heads = 16;

    if (get_metadata_uint32(mmproj_path, "clip.vision.block_count", &val32) == 0) vm->cfg.num_layers = (int)val32;
    else vm->cfg.num_layers = 32;

    if (get_metadata_uint32(mmproj_path, "clip.vision.projection_dim", &val32) == 0) vm->cfg.projection_dim = (int)val32;
    else vm->cfg.projection_dim = 2048;

    if (get_metadata_uint32(mmproj_path, "clip.vision.feed_forward_length", &val32) == 0) vm->cfg.ffn_dim = (int)val32;
    else vm->cfg.ffn_dim = 3420;

    float feps = 1e-6f;
    if (get_metadata_float(mmproj_path, "clip.vision.attention.layer_norm_epsilon", &feps) == 0) vm->cfg.eps = feps;
    else vm->cfg.eps = 1e-6f;

    get_metadata_string(mmproj_path, "clip.projector_type", vm->cfg.projector_type, sizeof(vm->cfg.projector_type));

    /* Memory map the projector file */
    vm->fd = open(mmproj_path, O_RDONLY);
    if (vm->fd >= 0) {
        struct stat st;
        if (fstat(vm->fd, &st) == 0) {
            vm->mmap_size = st.st_size;
            void *ptr = mmap(NULL, vm->mmap_size, PROT_READ, MAP_SHARED, vm->fd, 0);
            if (ptr != MAP_FAILED) {
                vm->mmap_base = (uint8_t *)ptr;
            }
        }
    }

    printf("[INFO] Vision Model Loaded: %s (img_size=%d, patch=%d, hidden=%d, layers=%d, proj_dim=%d)\n",
           mmproj_path, vm->cfg.image_size, vm->cfg.patch_size, vm->cfg.hidden_dim, vm->cfg.num_layers, vm->cfg.projection_dim);

    return vm;
}

void diskllm_vision_free(diskllm_vision_model *vmodel) {
    if (!vmodel) return;
    if (vmodel->mmap_base && vmodel->mmap_size > 0) {
        munmap(vmodel->mmap_base, vmodel->mmap_size);
    }
    if (vmodel->fd >= 0) close(vmodel->fd);
    if (vmodel->catalog) free_tensor_catalog(vmodel->catalog);
    free(vmodel);
}

static const void *v_get_tensor_ptr(diskllm_vision_model *vm, const tensor_info *ti, void *fallback_buf) {
    if (!ti) return NULL;
    if (vm->mmap_base) {
        return vm->mmap_base + ti->absolute_offset;
    }
    if (fallback_buf && vm->fd >= 0) {
        v_exact_pread(vm->fd, fallback_buf, ti->byte_size, ti->absolute_offset);
        return fallback_buf;
    }
    return NULL;
}

static void add_bias(float *out, const void *bias_ptr, int dim, int type) {
    if (!bias_ptr) return;
    if (type == GGML_TYPE_F32) {
        const float *b = (const float *)bias_ptr;
        for (int i = 0; i < dim; i++) out[i] += b[i];
    } else if (type == GGML_TYPE_F16) {
        const uint16_t *b = (const uint16_t *)bias_ptr;
        for (int i = 0; i < dim; i++) {
            out[i] += fp16_to_fp32(b[i]);
        }
    }
}

float *diskllm_vision_encode(diskllm_vision_model *vm, const float *image_rgb, int img_w, int img_h, int *out_num_patches, int num_threads) {
    (void)num_threads;
    if (!vm || !image_rgb) return NULL;

    int patch_size = vm->cfg.patch_size > 0 ? vm->cfg.patch_size : 14;
    int v_hidden = vm->cfg.hidden_dim > 0 ? vm->cfg.hidden_dim : 1280;
    int proj_dim = vm->cfg.projection_dim > 0 ? vm->cfg.projection_dim : 2048;
    int n_heads = vm->cfg.num_heads > 0 ? vm->cfg.num_heads : 16;
    int head_dim = v_hidden / n_heads;
    int n_layers = vm->cfg.num_layers > 0 ? vm->cfg.num_layers : 32;
    int ffn_dim = vm->cfg.ffn_dim > 0 ? vm->cfg.ffn_dim : 3420;

    int patches_w = img_w / patch_size;
    int patches_h = img_h / patch_size;
    int num_patches = patches_w * patches_h;
    if (num_patches <= 0) return NULL;

    /* 1. Extract patch embeddings via Conv2D / linear projection */
    const tensor_info *ti_patch = find_tensor(vm->catalog, "v.patch_embd.weight");
    if (!ti_patch) ti_patch = find_tensor(vm->catalog, "visual.patch_embed.proj.weight");

    float *patch_tokens = malloc((size_t)num_patches * v_hidden * sizeof(float));
    if (!patch_tokens) return NULL;
    memset(patch_tokens, 0, (size_t)num_patches * v_hidden * sizeof(float));

    int patch_pixels = 3 * patch_size * patch_size;

    if (ti_patch) {
        const void *w_patch = v_get_tensor_ptr(vm, ti_patch, NULL);
        if (w_patch) {
            float *patch_flat = malloc((size_t)patch_pixels * sizeof(float));
            for (int py = 0; py < patches_h; py++) {
                for (int px = 0; px < patches_w; px++) {
                    int p_idx = py * patches_w + px;
                    /* Flatten patch into [3, patch_size, patch_size] */
                    int f = 0;
                    for (int c = 0; c < 3; c++) {
                        for (int dy = 0; dy < patch_size; dy++) {
                            for (int dx = 0; dx < patch_size; dx++) {
                                int sy = py * patch_size + dy;
                                int sx = px * patch_size + dx;
                                patch_flat[f++] = image_rgb[(c * img_h + sy) * img_w + sx];
                            }
                        }
                    }
                    /* Matvec: [v_hidden, patch_pixels] x [patch_pixels] -> [v_hidden] */
                    matvec(patch_tokens + p_idx * v_hidden, w_patch, patch_flat, patch_pixels, v_hidden, ti_patch->type, NULL);
                }
            }
            free(patch_flat);
        }
    }

    /* 2. Positional Embedding if present */
    const tensor_info *ti_pos = find_tensor(vm->catalog, "v.position_embd.weight");
    if (!ti_pos) ti_pos = find_tensor(vm->catalog, "visual.pos_embed");
    if (ti_pos) {
        const void *w_pos = v_get_tensor_ptr(vm, ti_pos, NULL);
        if (w_pos && ti_pos->type == GGML_TYPE_F32) {
            const float *pos_f32 = (const float *)w_pos;
            for (int i = 0; i < num_patches && i < (int)(ti_pos->byte_size / (v_hidden * sizeof(float))); i++) {
                for (int d = 0; d < v_hidden; d++) {
                    patch_tokens[i * v_hidden + d] += pos_f32[i * v_hidden + d];
                }
            }
        }
    }

    /* 3. Run Vision Transformer Blocks */
    float *attn_out = malloc((size_t)v_hidden * sizeof(float));
    float *norm_buf = malloc((size_t)v_hidden * sizeof(float));
    float *scores = malloc((size_t)num_patches * sizeof(float));
    float *ffn_gate = malloc((size_t)ffn_dim * sizeof(float));
    float *ffn_up = malloc((size_t)ffn_dim * sizeof(float));
    float *ffn_out = malloc((size_t)v_hidden * sizeof(float));

    for (int l = 0; l < n_layers; l++) {
        char name[128];
        snprintf(name, sizeof(name), "v.blk.%d.ln1.weight", l);
        const tensor_info *ti_ln1 = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.ln1.bias", l);
        const tensor_info *ti_ln1_b = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.weight", l);
        const tensor_info *ti_qkv = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.attn_qkv.bias", l);
        const tensor_info *ti_qkv_b = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.attn_q.weight", l);
        const tensor_info *ti_q = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.attn_q.bias", l);
        const tensor_info *ti_qb = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.attn_k.weight", l);
        const tensor_info *ti_k = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.attn_k.bias", l);
        const tensor_info *ti_kb = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.attn_v.weight", l);
        const tensor_info *ti_v = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.attn_v.bias", l);
        const tensor_info *ti_vb = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.attn_out.weight", l);
        const tensor_info *ti_o = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.attn_out.bias", l);
        const tensor_info *ti_ob = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.ln2.weight", l);
        const tensor_info *ti_ln2 = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.ln2.bias", l);
        const tensor_info *ti_ln2_b = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.weight", l);
        const tensor_info *ti_ffn_down = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_down.bias", l);
        const tensor_info *ti_ffn_down_b = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.weight", l);
        const tensor_info *ti_ffn_up = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_up.bias", l);
        const tensor_info *ti_ffn_up_b = find_tensor(vm->catalog, name);

        snprintf(name, sizeof(name), "v.blk.%d.ffn_gate.weight", l);
        const tensor_info *ti_ffn_gate = find_tensor(vm->catalog, name);
        snprintf(name, sizeof(name), "v.blk.%d.ffn_gate.bias", l);
        const tensor_info *ti_ffn_gate_b = find_tensor(vm->catalog, name);

        if (!ti_o || (!ti_qkv && (!ti_q || !ti_k || !ti_v))) continue;

        const void *w_ln1 = v_get_tensor_ptr(vm, ti_ln1, NULL);
        const void *w_ln1_b = v_get_tensor_ptr(vm, ti_ln1_b, NULL);
        const void *w_qkv = v_get_tensor_ptr(vm, ti_qkv, NULL);
        const void *w_qkv_b = v_get_tensor_ptr(vm, ti_qkv_b, NULL);
        const void *w_q = v_get_tensor_ptr(vm, ti_q, NULL);
        const void *w_qb = v_get_tensor_ptr(vm, ti_qb, NULL);
        const void *w_k = v_get_tensor_ptr(vm, ti_k, NULL);
        const void *w_kb = v_get_tensor_ptr(vm, ti_kb, NULL);
        const void *w_v = v_get_tensor_ptr(vm, ti_v, NULL);
        const void *w_vb = v_get_tensor_ptr(vm, ti_vb, NULL);
        const void *w_o = v_get_tensor_ptr(vm, ti_o, NULL);
        const void *w_ob = v_get_tensor_ptr(vm, ti_ob, NULL);
        const void *w_ln2 = v_get_tensor_ptr(vm, ti_ln2, NULL);
        const void *w_ln2_b = v_get_tensor_ptr(vm, ti_ln2_b, NULL);
        const void *w_ffn_down = v_get_tensor_ptr(vm, ti_ffn_down, NULL);
        const void *w_ffn_down_b = v_get_tensor_ptr(vm, ti_ffn_down_b, NULL);
        const void *w_ffn_up = v_get_tensor_ptr(vm, ti_ffn_up, NULL);
        const void *w_ffn_up_b = v_get_tensor_ptr(vm, ti_ffn_up_b, NULL);
        const void *w_ffn_gate = v_get_tensor_ptr(vm, ti_ffn_gate, NULL);
        const void *w_ffn_gate_b = v_get_tensor_ptr(vm, ti_ffn_gate_b, NULL);

        /* Multi-Head Self-Attention over all patches */
        float *layer_q = malloc((size_t)num_patches * v_hidden * sizeof(float));
        float *layer_k = malloc((size_t)num_patches * v_hidden * sizeof(float));
        float *layer_v = malloc((size_t)num_patches * v_hidden * sizeof(float));
        float *qkv_fused = w_qkv ? malloc((size_t)3 * v_hidden * sizeof(float)) : NULL;

        for (int i = 0; i < num_patches; i++) {
            float *h = patch_tokens + i * v_hidden;
            if (w_ln1 && ti_ln1->type == GGML_TYPE_F32) {
                rmsnorm(norm_buf, h, (const float *)w_ln1, v_hidden, vm->cfg.eps);
                if (w_ln1_b) add_bias(norm_buf, w_ln1_b, v_hidden, ti_ln1_b->type);
            } else {
                memcpy(norm_buf, h, v_hidden * sizeof(float));
            }

            if (w_qkv) {
                matvec(qkv_fused, w_qkv, norm_buf, v_hidden, 3 * v_hidden, ti_qkv->type, NULL);
                if (w_qkv_b) add_bias(qkv_fused, w_qkv_b, 3 * v_hidden, ti_qkv_b->type);
                memcpy(layer_q + i * v_hidden, qkv_fused, v_hidden * sizeof(float));
                memcpy(layer_k + i * v_hidden, qkv_fused + v_hidden, v_hidden * sizeof(float));
                memcpy(layer_v + i * v_hidden, qkv_fused + 2 * v_hidden, v_hidden * sizeof(float));
            } else {
                matvec(layer_q + i * v_hidden, w_q, norm_buf, v_hidden, v_hidden, ti_q->type, NULL);
                if (w_qb) add_bias(layer_q + i * v_hidden, w_qb, v_hidden, ti_qb->type);

                matvec(layer_k + i * v_hidden, w_k, norm_buf, v_hidden, v_hidden, ti_k->type, NULL);
                if (w_kb) add_bias(layer_k + i * v_hidden, w_kb, v_hidden, ti_kb->type);

                matvec(layer_v + i * v_hidden, w_v, norm_buf, v_hidden, v_hidden, ti_v->type, NULL);
                if (w_vb) add_bias(layer_v + i * v_hidden, w_vb, v_hidden, ti_vb->type);
            }
        }
        if (qkv_fused) free(qkv_fused);

        float scale = 1.0f / sqrtf((float)head_dim);
        for (int h_idx = 0; h_idx < n_heads; h_idx++) {
            int h_offset = h_idx * head_dim;
            for (int i = 0; i < num_patches; i++) {
                const float *qi = layer_q + i * v_hidden + h_offset;
                float max_score = -1e9f;
                for (int j = 0; j < num_patches; j++) {
                    const float *kj = layer_k + j * v_hidden + h_offset;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > max_score) max_score = scores[j];
                }
                float sum_exp = 0.0f;
                for (int j = 0; j < num_patches; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
                for (int d = 0; d < head_dim; d++) {
                    float out_val = 0.0f;
                    for (int j = 0; j < num_patches; j++) {
                        out_val += (scores[j] * inv_sum) * layer_v[j * v_hidden + h_offset + d];
                    }
                    attn_out[h_offset + d] = out_val;
                }
            }
        }

        free(layer_q);
        free(layer_k);
        free(layer_v);

        /* Project attention output & add residual */
        for (int i = 0; i < num_patches; i++) {
            float *h = patch_tokens + i * v_hidden;
            float *out_proj = malloc((size_t)v_hidden * sizeof(float));
            matvec(out_proj, w_o, attn_out, v_hidden, v_hidden, ti_o->type, NULL);
            if (w_ob) add_bias(out_proj, w_ob, v_hidden, ti_ob->type);
            for (int d = 0; d < v_hidden; d++) h[d] += out_proj[d];
            free(out_proj);

            /* FFN block */
            if (w_ffn_down && w_ffn_up) {
                if (w_ln2 && ti_ln2->type == GGML_TYPE_F32) {
                    rmsnorm(norm_buf, h, (const float *)w_ln2, v_hidden, vm->cfg.eps);
                    if (w_ln2_b) add_bias(norm_buf, w_ln2_b, v_hidden, ti_ln2_b->type);
                } else {
                    memcpy(norm_buf, h, v_hidden * sizeof(float));
                }

                matvec(ffn_up, w_ffn_up, norm_buf, v_hidden, ffn_dim, ti_ffn_up->type, NULL);
                if (w_ffn_up_b) add_bias(ffn_up, w_ffn_up_b, ffn_dim, ti_ffn_up_b->type);

                if (w_ffn_gate) {
                    matvec(ffn_gate, w_ffn_gate, norm_buf, v_hidden, ffn_dim, ti_ffn_gate->type, NULL);
                    if (w_ffn_gate_b) add_bias(ffn_gate, w_ffn_gate_b, ffn_dim, ti_ffn_gate_b->type);
                    swiglu(ffn_gate, ffn_gate, ffn_up, ffn_dim);
                    matvec(ffn_out, w_ffn_down, ffn_gate, ffn_dim, v_hidden, ti_ffn_down->type, NULL);
                } else {
                    gelu(ffn_up, ffn_up, ffn_dim);
                    matvec(ffn_out, w_ffn_down, ffn_up, ffn_dim, v_hidden, ti_ffn_down->type, NULL);
                }
                if (w_ffn_down_b) add_bias(ffn_out, w_ffn_down_b, v_hidden, ti_ffn_down_b->type);

                for (int d = 0; d < v_hidden; d++) h[d] += ffn_out[d];
            }
        }
    }

    free(attn_out);
    free(norm_buf);
    free(scores);
    free(ffn_gate);
    free(ffn_up);
    free(ffn_out);

    /* 4. Post LN if present */
    const tensor_info *ti_post_ln = find_tensor(vm->catalog, "v.post_ln.weight");
    if (ti_post_ln) {
        const void *w_pln = v_get_tensor_ptr(vm, ti_post_ln, NULL);
        if (w_pln && ti_post_ln->type == GGML_TYPE_F32) {
            float *tmp = malloc((size_t)v_hidden * sizeof(float));
            for (int i = 0; i < num_patches; i++) {
                float *h = patch_tokens + i * v_hidden;
                rmsnorm(tmp, h, (const float *)w_pln, v_hidden, vm->cfg.eps);
                memcpy(h, tmp, v_hidden * sizeof(float));
            }
            free(tmp);
        }
    }

    /* 5. Final Multimodal Projector (MLP / Merger) into LLM Hidden Dimension */
    /* Qwen2.5-VL Merger: 2x2 spatial merger + mm.0 MLP -> mm.2 Linear */
    const tensor_info *ti_mm0 = find_tensor(vm->catalog, "mm.0.weight");
    const tensor_info *ti_mm0_b = find_tensor(vm->catalog, "mm.0.bias");
    const tensor_info *ti_mm2 = find_tensor(vm->catalog, "mm.2.weight");
    const tensor_info *ti_mm2_b = find_tensor(vm->catalog, "mm.2.bias");

    int out_patches = num_patches;
    float *visual_embeddings = NULL;

    if (ti_mm0 && ti_mm2) {
        /* 2x2 spatial pooling merger: (patches_h/2) * (patches_w/2) */
        int mw = patches_w / 2;
        int mh = patches_h / 2;
        if (mw > 0 && mh > 0) {
            out_patches = mw * mh;
            int concat_dim = 4 * v_hidden; /* 4 * 1280 = 5120 */
            float *merged = malloc((size_t)out_patches * concat_dim * sizeof(float));
            float *mlp_mid = malloc((size_t)concat_dim * sizeof(float));
            visual_embeddings = malloc((size_t)out_patches * proj_dim * sizeof(float));

            const void *w_mm0 = v_get_tensor_ptr(vm, ti_mm0, NULL);
            const void *w_mm0_b = v_get_tensor_ptr(vm, ti_mm0_b, NULL);
            const void *w_mm2 = v_get_tensor_ptr(vm, ti_mm2, NULL);
            const void *w_mm2_b = v_get_tensor_ptr(vm, ti_mm2_b, NULL);

            for (int my = 0; my < mh; my++) {
                for (int mx = 0; mx < mw; mx++) {
                    int m_idx = my * mw + mx;
                    /* Concatenate 2x2 patches: (2*my, 2*mx), (2*my, 2*mx+1), (2*my+1, 2*mx), (2*my+1, 2*mx+1) */
                    int p00 = (2 * my) * patches_w + (2 * mx);
                    int p01 = (2 * my) * patches_w + (2 * mx + 1);
                    int p10 = (2 * my + 1) * patches_w + (2 * mx);
                    int p11 = (2 * my + 1) * patches_w + (2 * mx + 1);

                    memcpy(merged + m_idx * concat_dim, patch_tokens + p00 * v_hidden, v_hidden * sizeof(float));
                    memcpy(merged + m_idx * concat_dim + v_hidden, patch_tokens + p01 * v_hidden, v_hidden * sizeof(float));
                    memcpy(merged + m_idx * concat_dim + 2 * v_hidden, patch_tokens + p10 * v_hidden, v_hidden * sizeof(float));
                    memcpy(merged + m_idx * concat_dim + 3 * v_hidden, patch_tokens + p11 * v_hidden, v_hidden * sizeof(float));

                    /* mm.0: [5120, 5120] -> GELU -> mm.2: [5120, 2048] */
                    matvec(mlp_mid, w_mm0, merged + m_idx * concat_dim, concat_dim, concat_dim, ti_mm0->type, NULL);
                    if (w_mm0_b) add_bias(mlp_mid, w_mm0_b, concat_dim, ti_mm0_b->type);
                    gelu(mlp_mid, mlp_mid, concat_dim);

                    matvec(visual_embeddings + m_idx * proj_dim, w_mm2, mlp_mid, concat_dim, proj_dim, ti_mm2->type, NULL);
                    if (w_mm2_b) add_bias(visual_embeddings + m_idx * proj_dim, w_mm2_b, proj_dim, ti_mm2_b->type);
                }
            }
            free(merged);
            free(mlp_mid);
        }
    }

    if (!visual_embeddings) {
        visual_embeddings = malloc((size_t)num_patches * proj_dim * sizeof(float));
        const tensor_info *ti_proj = find_tensor(vm->catalog, "mm.input_projection.weight");
        if (ti_proj) {
            const void *w_p = v_get_tensor_ptr(vm, ti_proj, NULL);
            for (int i = 0; i < num_patches; i++) {
                matvec(visual_embeddings + i * proj_dim, w_p, patch_tokens + i * v_hidden, v_hidden, proj_dim, ti_proj->type, NULL);
            }
        } else {
            for (int i = 0; i < num_patches; i++) {
                int copy_d = (v_hidden < proj_dim) ? v_hidden : proj_dim;
                memcpy(visual_embeddings + i * proj_dim, patch_tokens + i * v_hidden, copy_d * sizeof(float));
                if (proj_dim > copy_d) memset(visual_embeddings + i * proj_dim + copy_d, 0, (proj_dim - copy_d) * sizeof(float));
            }
        }
    }

    free(patch_tokens);
    if (out_num_patches) *out_num_patches = out_patches;
    return visual_embeddings;
}
