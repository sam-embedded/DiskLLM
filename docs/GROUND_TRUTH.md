# Qwen-3.5-Stream-C Ground Truth

This document lists the ground-truth specifications, metadata, topology, and tensor counts of the target model: `Qwen3.8-27B-Q4_K_M.gguf`.

## 1. GGUF File Summary
* **GGUF Version:** 3
* **Tensor Count:** 866
* **Metadata KV Count:** 51
* **Alignment:** 32 bytes
* **Total File Size:** ~15.93 GB (17,106,775,008 bytes)

## 2. Model Metadata
The following metadata values are extracted from the GGUF file:

| Metadata Key | Value | Description |
|---|---|---|
| `general.architecture` | `"qwen35"` | Model family / architecture name |
| `general.name` | `"Qwen3.8-27B"` | Human-readable name |
| `general.quantized_by` | `"Unsloth"` | Quantizing entity |
| `qwen35.block_count` | `65` | Total number of blocks (0-63 main, 64 NextN) |
| `qwen35.context_length` | `262144` | Maximum model context length |
| `qwen35.embedding_length` | `5120` | Hidden dimension size ($d_{model}$) |
| `qwen35.feed_forward_length`| `17408` | FFN intermediate dimension |
| `qwen35.attention.head_count` | `24` | Number of attention heads for Q |
| `qwen35.attention.head_count_kv` | `4` | Number of attention heads for K/V (GQA) |
| `qwen35.attention.key_length` | `256` | Dimension of key heads ($d_k$) |
| `qwen35.attention.value_length` | `256` | Dimension of value heads ($d_v$) |
| `qwen35.attention.layer_norm_rms_epsilon` | `0.000001` | RMSNorm epsilon |
| `qwen35.rope.freq_base` | `10000000.0` | RoPE frequency base |
| `qwen35.rope.dimension_count` | `64` | RoPE dimension |
| `qwen35.rope.dimension_sections` | `[11, 11, 10, 0]` | Section partitions for RoPE |
| `qwen35.nextn_predict_layers` | `1` | Number of next-token prediction heads |
| `qwen35.full_attention_interval` | `4` | Interval at which full self-attention layers occur |
| `qwen35.ssm.conv_kernel` | `4` | SSM 1D convolution kernel width |
| `qwen35.ssm.state_size` | `128` | SSM latent state dimension ($d_{state}$) |
| `qwen35.ssm.group_count` | `16` | SSM block structure groups |
| `qwen35.ssm.time_step_rank` | `48` | SSM input projection rank |
| `qwen35.ssm.inner_size` | `6144` | SSM internal channel expansion size |

## 3. Tokenizer Configuration
* **Model Type:** `"gpt2"` (BPE)
* **Pre-tokenizer:** `"qwen35"`
* **Vocabulary Size:** `248320` entries
* **Merges Count:** `247587` entries
* **Special Tokens:**
  * BOS (Beginning of Sequence): `248044`
  * EOS (End of Sequence): `248046`
  * Padding: `248055`

> [!NOTE]
> The tokenizer will NOT be implemented in the initial runtime stages. Pre-tokenized input (array of `int32_t`) is accepted directly.

## 4. Model Topology
The model consists of 65 blocks (`blk.0` to `blk.64`).
* `blk.0` through `blk.63` are the main backbone (64 layers total).
* `blk.64` contains the Next-Token-Prediction (NextN) head (which is ignored for the initial milestones).

### Layer Types
* **Full Attention Layers** (16 layers):
  * Pattern: `layer_index % 4 == 3`
  * Layer indices: `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63`
* **SSM (State Space Model) Layers** (48 layers):
  * Pattern: `layer_index % 4 != 3`
  * Layer indices: all other main layer indices

## 5. Tensor Count Breakdown
The file contains exactly **866** tensors, partitioned as follows:

1. **Global Tensors (3):**
   * `token_embd.weight`
   * `output_norm.weight`
   * `output.weight`
2. **SSM Layers (672):**
   * 48 layers × 14 tensors per layer = 672 tensors
3. **Full Attention Layers (176):**
   * 16 layers × 11 tensors per layer = 176 tensors
4. **NextN Layer `blk.64` (15):**
   * 11 standard attention layer tensors + 4 custom nextn projection/norm tensors
5. **Total Tensors:** `3 + 672 + 176 + 15 = 866`
