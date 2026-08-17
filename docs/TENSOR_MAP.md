# Qwen-3.5-Stream-C Tensor Map

This document defines the expected roles, dimensions, quantizations, and RAM-residency/streaming policies for all tensors in the model.

## 1. Global Tensors

| Tensor Name | Shape (in, out) | Type | Role | RAM Policy |
|---|---|---|---|---|
| `token_embd.weight` | `[5120, 248320]` | `Q4_K` | Token embedding lookup table | Stream selected row(s) (or Pin in RAM if budget allows) |
| `output_norm.weight` | `[5120]` | `F32` | Final RMSNorm scale factor | Pin in RAM (resident) |
| `output.weight` | `[5120, 248320]` | `Q6_K` | LM head output projection | Stream from disk in chunks |

---

## 2. SSM Layer Tensors (48 Layers)
These tensors appear in layers where `layer_index % 4 != 3` (e.g. `blk.0`, `blk.1`, `blk.2`, `blk.4`, ...).
There are 14 tensors per SSM layer.

| Tensor Name | Shape | Type | Role | RAM Policy |
|---|---|---|---|---|
| `blk.N.attn_norm.weight` | `[5120]` | `F32` | Layer input norm | Pin in RAM |
| `blk.N.attn_qkv.weight` | `[5120, 10240]` | `Q4_K` or `Q6_K` | Input projection for SSM/Conv path | Stream from disk |
| `blk.N.ssm_conv1d.weight` | `[4, 10240]` | `F32` | SSM 1D convolution weights | Pin in RAM (or Stream if tight) |
| `blk.N.ssm_a` | `[48]` | `F32` | SSM recurrent decay term | Pin in RAM |
| `blk.N.ssm_alpha.weight` | `[5120, 48]` | `F32` | SSM input projection | Pin in RAM (or Stream) |
| `blk.N.ssm_beta.weight` | `[5120, 48]` | `F32` | SSM gating projection | Pin in RAM (or Stream) |
| `blk.N.ssm_dt.bias` | `[48]` | `F32` | SSM time-step bias | Pin in RAM |
| `blk.N.ssm_norm.weight` | `[128]` | `F32` | SSM state output RMSNorm | Pin in RAM |
| `blk.N.ssm_out.weight` | `[6144, 5120]` | `Q5_K` | SSM projection output | Stream from disk |
| `blk.N.attn_gate.weight` | `[5120, 6144]` | `Q4_K` | Output gating for SSM path | Stream from disk |
| `blk.N.post_attention_norm.weight`| `[5120]` | `F32` | Post-SSM block RMSNorm | Pin in RAM |
| `blk.N.ffn_gate.weight` | `[5120, 17408]` | `Q4_K` | FFN gate projection | Stream from disk |
| `blk.N.ffn_up.weight` | `[5120, 17408]` | `Q4_K` | FFN up projection | Stream from disk |
| `blk.N.ffn_down.weight` | `[17408, 5120]` | `Q4_K` or `Q6_K` | FFN down projection | Stream from disk |

---

## 3. Full Attention Layer Tensors (16 Layers)
These tensors appear in layers where `layer_index % 4 == 3` (e.g. `blk.3`, `blk.7`, ...).
There are 11 tensors per Full Attention layer.

| Tensor Name | Shape | Type | Role | RAM Policy |
|---|---|---|---|---|
| `blk.N.attn_norm.weight` | `[5120]` | `F32` | Attention input RMSNorm | Pin in RAM |
| `blk.N.attn_q.weight` | `[5120, 12288]` | `Q4_K` | Query projection (24 heads × 512) | Stream from disk |
| `blk.N.attn_q_norm.weight` | `[256]` | `F32` | Per-head Query RMSNorm scale | Pin in RAM |
| `blk.N.attn_k.weight` | `[5120, 1024]` | `Q4_K` | Key projection (4 heads × 256) | Stream from disk |
| `blk.N.attn_k_norm.weight` | `[256]` | `F32` | Per-head Key RMSNorm scale | Pin in RAM |
| `blk.N.attn_v.weight` | `[5120, 1024]` | `Q4_K` or `Q6_K` | Value projection (4 heads × 256) | Stream from disk |
| `blk.N.attn_output.weight` | `[6144, 5120]` | `Q4_K` | Attention output projection | Stream from disk |
| `blk.N.post_attention_norm.weight`| `[5120]` | `F32` | Post-attention RMSNorm | Pin in RAM |
| `blk.N.ffn_gate.weight` | `[5120, 17408]` | `Q4_K` | FFN gate projection | Stream from disk |
| `blk.N.ffn_up.weight` | `[5120, 17408]` | `Q4_K` | FFN up projection | Stream from disk |
| `blk.N.ffn_down.weight` | `[17408, 5120]` | `Q4_K` or `Q6_K` | FFN down projection | Stream from disk |

---

## 4. NextN Prediction Layer Tensors (`blk.64` - 15 Tensors)
These tensors are located in the final speculative head block.
* 11 standard attention layer tensors (same as Section 3 above, with name `blk.64.*`).
* 4 unique NextN head tensors:

| Tensor Name | Shape | Type | Role | RAM Policy |
|---|---|---|---|---|
| `blk.64.nextn.eh_proj.weight` | `[10240, 5120]` | `Q8_0` | NextN state projection | Ignored (Initial stage) |
| `blk.64.nextn.enorm.weight` | `[5120]` | `F32` | NextN embedding norm | Ignored (Initial stage) |
| `blk.64.nextn.hnorm.weight` | `[5120]` | `F32` | NextN hidden norm | Ignored (Initial stage) |
| `blk.64.nextn.shared_head_norm.weight` | `[5120]` | `F32` | NextN shared head norm | Ignored (Initial stage) |

---

## 5. Tensor Shape Conventions (GGML vs. Standard)
* GGML storage transposes 2D matrices. A tensor stored in GGUF with dimensions `[A, B]` has:
  * `A` = input features / columns of weight matrix (contiguous in memory)
  * `B` = output features / rows of weight matrix
* **Example:** `ffn_gate.weight` has shape `[5120, 17408]`. In a forward pass, it acts on an input of size `5120` to produce an activation of size `17408`.
