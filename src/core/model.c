#include "diskllm_internal.h"
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

double diskllm_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

long diskllm_read_rss_mb(void) {
    FILE *f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long pages = 0;
    if (fscanf(f, "%*s %ld", &pages) != 1) {
        fclose(f);
        return 0;
    }
    fclose(f);
    long page_size = sysconf(_SC_PAGESIZE);
    return (pages * page_size) / (1024 * 1024);
}

uint64_t diskllm_get_available_memory_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t avail_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %llu kB", (unsigned long long *)&avail_kb) == 1) {
            break;
        }
    }
    fclose(f);
    return avail_kb * 1024ULL;
}

diskllm_model_params diskllm_model_params_default(void) {
    return (diskllm_model_params){
        .arch_flag = "auto",
        .pin_weights = false,
        .io_mode = DISKLLM_IO_PREAD,
        .num_threads = 0,
        .quiet = false
    };
}

diskllm_model *diskllm_model_load(const char *model_path, diskllm_model_params params) {
    if (!model_path) return NULL;

    diskllm_model *model = calloc(1, sizeof(diskllm_model));
    if (!model) return NULL;

    strncpy(model->model_path, model_path, sizeof(model->model_path) - 1);
    model->params = params;

    /* 1. Load Tensor Catalog */
    model->catalog = load_tensor_catalog(model_path);
    if (!model->catalog) {
        fprintf(stderr, "Error: Failed to load tensor catalog from %s\n", model_path);
        free(model);
        return NULL;
    }

    /* 2. Load Model Config */
    model->cfg = load_qwen_model_config(model_path, model->catalog, params.arch_flag);
    if (!model->cfg) {
        fprintf(stderr, "Error: Failed to load model config from %s\n", model_path);
        free_tensor_catalog(model->catalog);
        free(model);
        return NULL;
    }

    if (!params.quiet) {
        fprintf(stderr, "[INFO] Model arch: %s, type: %d, blocks: %d, hidden: %d, heads: %d/%d, key_len: %d, attn_layers: %d\n",
                model->cfg->architecture, model->cfg->model_type, model->cfg->block_count,
                model->cfg->hidden_dim, model->cfg->num_attn_heads, model->cfg->num_kv_heads,
                model->cfg->key_length, model->cfg->num_attn_layers);
    }

    /* 3. Initialize Tokenizer */
    model->tok = tokenizer_init(model_path);

    /* 4. Match Core Tensors */
    model->ti_emb = find_tensor(model->catalog, "token_embd.weight");
    model->ti_outw = find_tensor(model->catalog, "output.weight");

    if (!model->ti_emb) {
        fprintf(stderr, "[ERROR] Missing core tensor: token_embd.weight\n");
        free_qwen_model_config(model->cfg);
        free_tensor_catalog(model->catalog);
        if (model->tok) tokenizer_free(model->tok);
        free(model);
        return NULL;
    }

    if (!model->ti_outw && model->cfg->is_tied_embedding) {
        if (!params.quiet) fprintf(stderr, "[INFO] Tied embedding detected: output.weight maps to token_embd.weight\n");
        model->ti_outw = model->ti_emb;
    }

    if (!model->ti_outw) {
        fprintf(stderr, "[ERROR] Missing core tensor: output.weight\n");
        free_qwen_model_config(model->cfg);
        free_tensor_catalog(model->catalog);
        if (model->tok) tokenizer_free(model->tok);
        free(model);
        return NULL;
    }

    model->embed_row_bytes = model->ti_emb->byte_size / (model->ti_emb->dims[1] > 0 ? model->ti_emb->dims[1] : model->cfg->vocab_size);
    model->logit_row_bytes = model->ti_outw->byte_size / (model->ti_outw->dims[1] > 0 ? model->ti_outw->dims[1] : model->cfg->vocab_size);

    /* 5. Handle mmap / pin_weights */
    int is_mmap_mode = (params.io_mode == DISKLLM_IO_MMAP || params.pin_weights);
    if (is_mmap_mode) {
        int mfd = open(model_path, O_RDONLY);
        if (mfd >= 0) {
            off_t fsz = lseek(mfd, 0, SEEK_END);
            lseek(mfd, 0, SEEK_SET);
            model->g_mmap_full_size = fsz;

            if (params.pin_weights) {
                uint64_t avail_ram = diskllm_get_available_memory_bytes();
                if (avail_ram > 0 && (uint64_t)fsz > avail_ram / 2) {
                    fprintf(stderr, "[WARN] Model size (%.2f MB) > 50%% available RAM (%.2f MB). Pinning weights may cause OOM.\n",
                            (double)fsz / (1024.0 * 1024.0), (double)avail_ram / (1024.0 * 1024.0));
                }
            }

            void *mptr = mmap(NULL, fsz, PROT_READ, MAP_SHARED, mfd, 0);
            close(mfd);

            if (mptr != MAP_FAILED) {
                model->g_mmap_full = (uint8_t *)mptr;
                model->mmap_output_weight = model->g_mmap_full + model->ti_outw->absolute_offset;

                if (params.pin_weights) {
                    if (mlock(model->g_mmap_full, model->g_mmap_full_size) != 0) {
                        fprintf(stderr, "[WARN] mlock failed, falling back to madvise. OS may still page out weights.\n");
#if defined(MADV_WILLNEED)
                        madvise((void *)model->g_mmap_full, model->g_mmap_full_size, MADV_WILLNEED);
#endif
                    } else if (!params.quiet) {
                        fprintf(stderr, "[INFO] Successfully locked %llu MB of model weights into physical RAM (Zero Decode I/O).\n",
                                (unsigned long long)(fsz / (1024 * 1024)));
                    }
                } else {
#if defined(MADV_WILLNEED)
                    madvise((void *)model->g_mmap_full, model->g_mmap_full_size, MADV_WILLNEED);
#endif
                }
            } else {
                fprintf(stderr, "[WARN] mmap failed. Falling back to pread streaming.\n");
            }
        }
    }

    /* 6. Find Architecture Backend */
    model->arch_backend = diskllm_arch_find(model->cfg->model_type);
    if (model->arch_backend && model->arch_backend->init) {
        model->arch_backend->init(model);
    }

    return model;
}

void diskllm_model_free(diskllm_model *model) {
    if (!model) return;

    if (model->g_mmap_full) {
        if (model->params.pin_weights) {
            munlock(model->g_mmap_full, model->g_mmap_full_size);
        }
        munmap((void *)model->g_mmap_full, model->g_mmap_full_size);
    }

    if (model->sctx) close_stream_context(model->sctx);
    if (model->output_norm) free(model->output_norm);
    if (model->tok) tokenizer_free(model->tok);
    if (model->cfg) free_qwen_model_config(model->cfg);
    if (model->catalog) free_tensor_catalog(model->catalog);
    free(model);
}

int diskllm_model_get_vocab_size(const diskllm_model *model) {
    return (model && model->cfg) ? model->cfg->vocab_size : 0;
}

uint32_t diskllm_model_get_eos_id(const diskllm_model *model) {
    return (model && model->cfg) ? model->cfg->eos_token_id : 0;
}

uint32_t diskllm_model_get_bos_id(const diskllm_model *model) {
    return (model && model->cfg) ? model->cfg->bos_token_id : 0;
}

const char *diskllm_model_get_arch_name(const diskllm_model *model) {
    return (model && model->cfg) ? model->cfg->architecture : "unknown";
}

int diskllm_model_get_hidden_dim(const diskllm_model *model) {
    return (model && model->cfg) ? model->cfg->hidden_dim : 0;
}

int diskllm_model_get_block_count(const diskllm_model *model) {
    return (model && model->cfg) ? model->cfg->block_count : 0;
}
