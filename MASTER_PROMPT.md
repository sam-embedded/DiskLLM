MASTER PROJECT PROMPT
Project: qwen35-stream-c
Goal: Pure C disk-streaming inference engine for Qwen3.8-27B-Q4_K_M.gguf

======================================================================
1. ROLE AND OBJECTIVE
======================================================================

You are a systems programmer building a pure C inference engine.

Your objective is to build a minimal, correct, low-memory C project that can run inference on:

    Qwen3.8-27B-Q4_K_M.gguf

The core design constraint is:

    Do NOT load the full model into RAM.

Instead:

    1. Load only essential tensors into RAM.
    2. Stream large weight tensors from the GGUF file on disk just-in-time.
    3. Keep KV cache, SSM recurrent state, activation buffers, and small frequent tensors resident.
    4. Use double-buffered streaming where possible.

This is a research/engineering project focused on correctness, low memory usage, and streaming architecture. Performance matters, but correctness and clear architecture come first.

======================================================================
2. HARD CONSTRAINTS
======================================================================

Language:

    - C11.
    - No C++.
    - No Rust.
    - No Python in the runtime.
    - No external ML frameworks.
    - No llama.cpp runtime dependency.
    - No ggml runtime dependency.

Allowed dependencies:

    - libc
    - libm
    - POSIX APIs: open, pread, mmap, fseeko, ftello, posix_memalign, fadvise, madvise
    - pthreads optional for prefetching
    - CMake or plain Makefile

Coding rules:

    - Use fixed-width integer types.
    - Check every allocation.
    - Check every file read.
    - Avoid global mutable state where possible.
    - Use explicit structs for model, layers, state, scratch, and streaming.
    - Prefer simple, testable modules.
    - No hidden large allocations.
    - No floating-point assumptions that break on different libc implementations.
    - Accumulate reductions in float or double where numerically appropriate.
    - All code should compile with:

        cc -std=c11 -Wall -Wextra -O2

Portability:

    - GGUF is little-endian.
    - Initially target little-endian hosts.
    - If host is big-endian, byte-swap explicitly.

======================================================================
3. TARGET MODEL GROUND TRUTH
======================================================================

The target file is:

    Qwen3.8-27B-Q4_K_M.gguf

Important metadata already dumped from the file:

    GGUF version:                  3
    Tensor count:                  866
    Metadata KV count:             51
    Alignment:                     32 bytes

    general.architecture:          "qwen35"
    general.name:                  "Qwen3.8-27B"
    general.quantized_by:          "Unsloth"

    qwen35.block_count:            65
    qwen35.context_length:         262144
    qwen35.embedding_length:       5120
    qwen35.feed_forward_length:    17408

    qwen35.attention.head_count:   24
    qwen35.attention.head_count_kv: 4
    qwen35.attention.key_length:   256
    qwen35.attention.value_length: 256
    qwen35.attention.layer_norm_rms_epsilon: 0.000001

    qwen35.rope.freq_base:         10000000.0
    qwen35.rope.dimension_count:   64
    qwen35.rope.dimension_sections: [11, 11, 10, 0]

    qwen35.nextn_predict_layers:   1
    qwen35.full_attention_interval: 4

    qwen35.ssm.conv_kernel:        4
    qwen35.ssm.state_size:         128
    qwen35.ssm.group_count:        16
    qwen35.ssm.time_step_rank:     48
    qwen35.ssm.inner_size:         6144

Tokenizer:

    tokenizer.ggml.model:          "gpt2"
    tokenizer.ggml.pre:            "qwen35"
    tokenizer.ggml.tokens:         248320 entries
    tokenizer.ggml.merges:         247587 entries
    tokenizer.ggml.eos_token_id:   248046
    tokenizer.ggml.padding_token_id: 248055
    tokenizer.ggml.bos_token_id:   248044

For initial bring-up, do NOT implement the tokenizer.

Accept pre-tokenized input as an array of int32 token IDs.

======================================================================
4. MODEL TOPOLOGY
======================================================================

The model has 65 tensor blocks: blk.0 through blk.64.

Interpretation:

    - blk.0 through blk.63 are the main backbone.
    - blk.64 contains an additional Next-Token-Prediction / speculative head.
    - For first correct generation, target blk.0 through blk.63 only.
    - Disable or ignore blk.64.nextn.* initially unless explicitly enabled.

Main backbone layer pattern:

    - Total main layers: 64
    - Full attention layers: layer_index % 4 == 3
    - SSM / DeltaNet-style layers: all other main layers

Therefore:

    Full attention layers:
        3, 7, 11, 15, 19, 23, 27, 31,
        35, 39, 43, 47, 51, 55, 59, 63

    SSM layers:
        all other indices from 0 to 63

Expected tensor counts:

    Global tensors:
        output.weight
        output_norm.weight
        token_embd.weight
        => 3 tensors

    SSM layers:
        48 layers * 14 tensors = 672 tensors

    Main attention layers:
        16 layers * 11 tensors = 176 tensors

    NextN layer blk.64:
        15 tensors
        includes attention tensors and nextn tensors

    Total:
        3 + 672 + 176 + 15 = 866 tensors

The project must verify this count.

======================================================================
5. TENSOR SHAPES AND ROLES
======================================================================

Global tensors:

    output.weight
        shape: [5120, 248320]
        type: Q6_K
        role: final LM head
        policy: stream from disk, chunked

    output_norm.weight
        shape: [5120]
        type: F32
        role: final RMSNorm
        policy: pin in RAM

    token_embd.weight
        shape: [5120, 248320]
        type: Q4_K
        role: token embedding table
        policy: pin if RAM budget allows, otherwise stream selected row

SSM layer tensors, example blk.0:

    blk.N.attn_norm.weight
        [5120]
        F32
        pin in RAM

    blk.N.attn_qkv.weight
        [5120, 10240]
        Q6_K or Q4_K depending on layer
        input projection into SSM/Conv path
        stream from disk

    blk.N.ssm_conv1d.weight
        [4, 10240]
        F32
        Conv1D weights
        pin if memory allows, else stream

    blk.N.ssm_a
        [48]
        F32
        pin in RAM

    blk.N.ssm_alpha.weight
        [5120, 48]
        F32
        pin if memory allows, else stream

    blk.N.ssm_beta.weight
        [5120, 48]
        F32
        pin if memory allows, else stream

    blk.N.ssm_dt.bias
        [48]
        F32
        pin in RAM

    blk.N.ssm_norm.weight
        [128]
        F32
        pin in RAM

    blk.N.ssm_out.weight
        [6144, 5120]
        Q5_K
        stream from disk

    blk.N.attn_gate.weight
        [5120, 6144]
        Q4_K
        output gate for SSM path
        stream from disk

    blk.N.post_attention_norm.weight
        [5120]
        F32
        pin in RAM

    blk.N.ffn_gate.weight
        [5120, 17408]
        Q4_K
        stream from disk

    blk.N.ffn_up.weight
        [5120, 17408]
        Q4_K
        stream from disk

    blk.N.ffn_down.weight
        [17408, 5120]
        Q6_K or Q4_K depending on layer
        stream from disk

Full attention layer tensors, example blk.3:

    blk.N.attn_norm.weight
        [5120]
        F32
        pin in RAM

    blk.N.attn_q.weight
        [5120, 12288]
        Q4_K
        stream from disk

    blk.N.attn_q_norm.weight
        [256]
        F32
        pin in RAM

    blk.N.attn_k.weight
        [5120, 1024]
        Q4_K
        stream from disk

    blk.N.attn_k_norm.weight
        [256]
        F32
        pin in RAM

    blk.N.attn_v.weight
        [5120, 1024]
        Q4_K or Q6_K depending on layer
        stream from disk

    blk.N.attn_output.weight
        [6144, 5120]
        Q4_K
        stream from disk

    blk.N.post_attention_norm.weight
        [5120]
        F32
        pin in RAM

    blk.N.ffn_gate.weight
        [5120, 17408]
        Q4_K
        stream from disk

    blk.N.ffn_up.weight
        [5120, 17408]
        Q4_K
        stream from disk

    blk.N.ffn_down.weight
        [17408, 5120]
        Q6_K or Q4_K depending on layer
        stream from disk

NextN layer blk.64:

    blk.64.attn_k.weight
    blk.64.attn_k_norm.weight
    blk.64.attn_norm.weight
    blk.64.attn_output.weight
    blk.64.attn_q.weight
    blk.64.attn_q_norm.weight
    blk.64.attn_v.weight
    blk.64.ffn_down.weight
    blk.64.ffn_gate.weight
    blk.64.ffn_up.weight
    blk.64.post_attention_norm.weight

    blk.64.nextn.eh_proj.weight
        [10240, 5120]
        Q8_0

    blk.64.nextn.enorm.weight
        [5120]
        F32

    blk.64.nextn.hnorm.weight
        [5120]
        F32

    blk.64.nextn.shared_head_norm.weight
        [5120]
        F32

Initial policy:

    Ignore NextN head.
    Do not use blk.64.nextn.* for first generation milestone.

======================================================================
6. CRITICAL ARCHITECTURAL FACTS
======================================================================

1. Do NOT assume every layer is attention.

   This is a hybrid SSM/Attention model.

2. Do NOT assume `attn_qkv.weight` in SSM layers is normal Q/K/V.

   In SSM layers, `attn_qkv.weight` is an input projection into the SSM/Conv path.

3. Do NOT assume metadata `head_count` fully describes Q shape.

   Attention Q projection is:

       [5120, 12288]

   K/V projections are:

       [5120, 1024]

   Attention output projection expects:

       [6144, 5120]

   Buffer sizes must be derived from tensor shapes, not only metadata.

4. QK norms exist.

   Full attention layers have:

       attn_q_norm.weight
       attn_k_norm.weight

   These are size [256].

   They likely apply per-head normalization before or around RoPE. Verify against reference behavior.

5. RoPE is nonstandard.

   Metadata contains:

       rope.freq_base = 10000000.0
       rope.dimension_count = 64
       rope.dimension_sections = [11, 11, 10, 0]

   Do not blindly use generic Llama-style RoPE.

   Write a RoPE plan first.

6. SSM math is bleeding edge.

   Do not hallucinate the DeltaNet/SSM update.

   Before implementing SSM kernels, write a spec using:

       qwen35.ssm.conv_kernel
       qwen35.ssm.state_size
       qwen35.ssm.group_count
       qwen35.ssm.time_step_rank
       qwen35.ssm.inner_size

   and the tensor shapes above.

7. Mixed quantization is present.

   The project must support at least:

       F32
       Q4_K
       Q5_K
       Q6_K
       Q8_0

   The streaming engine must dispatch by tensor type.

8. GGML weight shape convention:

   For 2D weights printed as [A, B], treat:

       A = input features
       B = output features

   Example:

       ffn_gate.weight [5120, 17408]
           in = 5120
           out = 17408

       ffn_down.weight [17408, 5120]
           in = 17408
           out = 5120

   Verify this assumption with known tensors and reference math.

======================================================================
7. MEMORY POLICY
======================================================================

Initial RAM target:

    Under 4 GB for 8192 context.
    Stretch target under 2 GB.

Default context for first milestones:

    8192 tokens

Do not attempt 262144 context initially.

Resident RAM should include:

    - model metadata
    - layer map
    - tensor catalog
    - all norm weights
    - all small biases
    - small SSM control tensors
    - token embedding table if budget allows
    - KV cache for attention layers only
    - SSM recurrent state
    - SSM conv history
    - activation buffers
    - streaming buffers

Stream from disk:

    - large Q4_K/Q5_K/Q6_K/Q8_0 projections
    - FFN weights
    - attention Q/K/V/output weights
    - SSM input/output/gate projections
    - final output.weight

KV cache:

    Only full attention layers require growing KV cache.
    There are 16 such main layers.

SSM state:

    Fixed-size.
    Does not grow with context.
    Must remain resident.

======================================================================
8. PROJECT STRUCTURE
======================================================================

Create the repository with this structure:

    qwen35-stream-c/
        CMakeLists.txt
        Makefile
        README.md
        MASTER_PROMPT.md
        GROUND_TRUTH.md

        docs/
            ARCHITECTURE.md
            TENSOR_MAP.md
            MEMORY_PLAN.md
            SSM_SPEC.md
            ROPE_PLAN.md
            ATTENTION_PLAN.md
            STREAMING_PLAN.md
            VALIDATION_PLAN.md

        include/
            gguf.h
            tensor_catalog.h
            layer_map.h
            model_config.h
            state.h
            scratch.h
            stream.h
            dequant.h
            kernels.h
            attention.h
            ssm.h
            sampler.h
            log.h
            util.h

        src/
            gguf.c
            tensor_catalog.c
            layer_map.c
            model_config.c
            state.c
            scratch.c
            stream.c
            dequant.c
            kernels.c
            attention.c
            ssm.c
            sampler.c
            log.c
            util.c

        tools/
            gguf_dump.c
            tensor_map.c
            probe_model.c
            bench_stream.c
            test_dequant.c

        main/
            qwen_stream_main.c

        scripts/
            run_dump.sh
            check_tensor_counts.sh
            measure_rss.sh

======================================================================
9. MILESTONES
======================================================================

Milestone 0: Documentation and ground truth

    Create docs.
    Record the model dump and tensor topology.
    Do not write inference kernels yet.

    Acceptance:
        GROUND_TRUTH.md contains metadata, layer pattern, tensor counts, and shapes.
        TENSOR_MAP.md explains expected tensor roles.

Milestone 1: GGUF parser and dumper

    Implement pure C GGUF header/metadata/tensor-table parser.

    Required output:
        magic
        version
        tensor_count
        metadata_kv_count
        alignment
        metadata keys/values, truncated safely
        tensor name
        tensor dims
        tensor ggml type
        tensor byte size
        tensor absolute file offset

    Acceptance:
        ./gguf_dump "$MODEL_GGUF" prints all 866 tensors.
        Total tensor count is exactly 866.
        Parser handles large files with 64-bit offsets.

Milestone 2: Tensor catalog and layer map

    Build a runtime catalog from GGUF.

    Classify layers:

        SSM layer:
            has blk.N.ssm_a

        Main attention layer:
            has blk.N.attn_q.weight
            layer index < 64
            no SSM tensors

        NextN layer:
            has blk.64.nextn.* tensors

    Populate layer structs by tensor name.

    Acceptance:
        48 SSM layers detected.
        16 main attention layers detected.
        1 NextN layer detected.
        No tensor is unassigned except intentionally ignored tokenizer/chat-template tensors.
        Program aborts cleanly if an expected tensor is missing.

Milestone 3: State and scratch allocation

    Allocate:

        KV cache for 16 attention layers.
        SSM recurrent state.
        SSM conv history.
        activation buffers.
        streaming buffers.

    Use the previously established dimensions:

        hidden_size = 5120
        ffn_hidden = 17408
        q_features = 12288
        kv_features = 1024
        attn_out_features = 6144
        ssm_inner_size = 6144
        ssm_state_size = 128
        ssm_conv_kernel = 4

    Acceptance:
        Allocation succeeds for 8192 context.
        RSS is measured and logged.
        Freeing works without leaks.
        Allocation failure is handled gracefully.

Milestone 4: Streaming engine

    Implement disk streaming using pread.

    Requirements:

        - 64-bit file offsets.
        - exact read loop.
        - aligned optional I/O.
        - chunked reads.
        - double buffering.
        - optional prefetch thread.
        - tensor streaming by offset and byte size.
        - do not read whole output.weight at once.
        - do not read whole token_embd.weight at once unless explicitly pinned.

    Initial buffer size:

        128 MiB total streaming buffers, split into two 64 MiB buffers.

    Acceptance:
        Can stream blk.N.ffn_down.weight in chunks.
        Can stream output.weight in chunks.
        Bytes read are logged.
        No full-model load occurs.

Milestone 5: Dequantization kernels

    Implement dequantizers or fused dot kernels for:

        F32
        Q4_K
        Q5_K
        Q6_K
        Q8_0

    Start with correctness, not SIMD.

    Acceptance:
        Q4_K block decoder matches known GGML layout.
        Q6_K block decoder matches known GGML layout.
        Q8_0 block decoder matches known GGML layout.
        Unit tests compare dequantized values against expected values.

Milestone 6: Basic math kernels

    Implement:

        rmsnorm
        add_residual
        silu
        swiglu gate/up combine
        matvec dispatcher
        fused dequantize-matvec for quantized weights

    Acceptance:
        rmsnorm produces stable results.
        matvec on F32 test matrix is correct.
        matvec on Q4_K test matrix is correct.

Milestone 7: Attention layer kernel

    Implement full attention for the 16 attention layers.

    Requirements:

        derive buffer sizes from tensor shapes
        support QK norms
        support RoPE according to ROPE_PLAN.md
        support grouped-query attention
        update KV cache
        handle Q feature size 12288
        handle K/V feature size 1024
        handle attention output feature size 6144

    Acceptance:
        A single attention layer can run on random input.
        Output shape is hidden_size = 5120 after residual path.
        No NaNs for small sequences.

Milestone 8: SSM layer kernel

    Implement SSM/DeltaNet layer according to SSM_SPEC.md.

    Requirements:

        use ssm_conv1d
        update conv history
        update recurrent state
        use ssm_a, ssm_alpha, ssm_beta, ssm_dt.bias, ssm_norm
        use ssm_out and attn_gate
        do not grow state with sequence length

    Acceptance:
        A single SSM layer can run on random input.
        State updates in place.
        State remains bounded.
        No NaNs after many steps.

Milestone 9: Full forward pass without tokenizer

    Implement:

        embedding lookup
        64 main layers
        final output_norm
        output logits computation
        greedy sampler

    Input:

        array of token IDs

    Output:

        next token ID
        optional top-k logits

    Acceptance:
        The model can generate one deterministic next token from a small prompt.
        RSS stays within target.
        Disk bytes read are logged.

Milestone 10: Optimization

    Only after correctness:

        prefetch next tensor while computing current tensor
        sort streaming order by file offset where safe
        use fused quantized dot products
        reduce temporary float buffers
        optionally use SIMD intrinsics behind #ifdef
        optionally use O_DIRECT or io_uring
        measure tokens/sec and bytes/token

======================================================================
10. ACCEPTANCE CRITERIA FOR THE FIRST SESSION
======================================================================

For the first coding session, do NOT implement the full model.

Deliver only:

    1. Repository skeleton.
    2. docs/GROUND_TRUTH.md.
    3. docs/TENSOR_MAP.md.
    4. tools/gguf_dump.c.
    5. tools/tensor_map.c.
    6. scripts/check_tensor_counts.sh.

The first session is successful if:

    The project compiles.
    gguf_dump prints the tensor table.
    tensor_map classifies:
        48 SSM layers
        16 attention layers
        1 NextN layer
    Total tensor count is 866.
    No tensor role is ambiguous.

======================================================================
11. IMPORTANT SAFETY AND CORRECTNESS RULES
======================================================================

Do not guess tensor names.

Use the ground-truth names:

    blk.N.attn_gate.weight
    blk.N.attn_norm.weight
    blk.N.attn_qkv.weight
    blk.N.ffn_down.weight
    blk.N.ffn_gate.weight
    blk.N.ffn_up.weight
    blk.N.post_attention_norm.weight
    blk.N.ssm_a
    blk.N.ssm_alpha.weight
    blk.N.ssm_beta.weight
    blk.N.ssm_conv1d.weight
    blk.N.ssm_dt.bias
    blk.N.ssm_norm.weight
    blk.N.ssm_out.weight

    blk.N.attn_q.weight
    blk.N.attn_q_norm.weight
    blk.N.attn_k.weight
    blk.N.attn_k_norm.weight
    blk.N.attn_v.weight
    blk.N.attn_output.weight

    blk.64.nextn.eh_proj.weight
    blk.64.nextn.enorm.weight
    blk.64.nextn.hnorm.weight
    blk.64.nextn.shared_head_norm.weight

If a tensor is missing, abort with a clear error.

Do not silently fall back to a different architecture.

Do not assume Qwen2, Qwen2.5, Llama, Mistral, or standard Mamba behavior without verification.

This architecture is hybrid and bleeding edge. Prefer explicit validation over assumptions.

======================================================================
12. IMMEDIATE ACTIONS
======================================================================

Start now by doing the following:

    1. Create the project directory:

        qwen35-stream-c

    2. Initialize a git repository.

    3. Create MASTER_PROMPT.md containing this prompt.

    4. Create docs/GROUND_TRUTH.md from the model facts above.

    5. Create tools/gguf_dump.c.

    6. Build it.

    7. Run it against:

        $MODEL_GGUF

    8. Save output to:

        build/gguf_dump.txt

    9. Create tools/tensor_map.c.

    10. Run tensor_map and verify:

        total tensors = 866
        SSM layers = 48
        attention layers = 16
        nextn layers = 1

    11. Do not begin compute kernels until the tensor map is verified.

======================================================================
13. DEFINITION OF DONE FOR THE PROJECT
======================================================================

The project is considered successful when:

    It parses the target GGUF file.
    It maps all 866 tensors.
    It allocates low-memory runtime state.
    It streams weights from disk.
    It runs the hybrid layer dispatcher.
    It generates at least one token greedily from token IDs.
    It does not load the full 15.92 GB model into RAM.
    RSS and disk bytes read are measurable.
    The code is pure C and builds with -std=c11 -Wall -Wextra.

End of master prompt.
