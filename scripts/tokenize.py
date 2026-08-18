#!/usr/bin/env python3
"""
DiskLLM Tokenizer Encode Script — Phase 17 & 18B
Encodes a plain text string into token IDs using Qwen3 BPE vocabulary
from GGUF metadata. Preserves special chat tokens (<|im_start|>, <|im_end|>, etc.).

Usage:
    python3 scripts/tokenize.py \
        --model /path/to/model.gguf \
        --text "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant"
"""

import argparse
import struct
import sys
import os
import re

SPECIAL_TOKENS = {
    "<|endoftext|>": 248044,
    "<|im_start|>": 248045,
    "<|im_end|>": 248046,
    "<|object_ref_start|>": 248047,
    "<|object_ref_end|>": 248048,
    "<|box_start|>": 248049,
    "<|box_end|>": 248050,
    "<|quad_start|>": 248051,
    "<|quad_end|>": 248052,
    "<|vision_start|>": 248053,
    "<|vision_end|>": 248054,
    "<|vision_pad|>": 248055,
}


def read_gguf_string_array(filepath, key_target):
    """Read a string array from GGUF metadata."""
    with open(filepath, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", "Not a GGUF file"
        version = struct.unpack("<I", f.read(4))[0]
        tc = struct.unpack("<Q", f.read(8))[0]
        mkvc = struct.unpack("<Q", f.read(8))[0]

        def skip_val(vt):
            sizes = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
                     10: 8, 11: 8, 12: 8}
            if vt in sizes:
                f.read(sizes[vt])
            elif vt == 8:
                l = struct.unpack("<Q", f.read(8))[0]
                f.read(l)
            elif vt == 9:
                et = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                for _ in range(alen):
                    skip_val(et)

        def read_string():
            l = struct.unpack("<Q", f.read(8))[0]
            return f.read(l).decode("utf-8", errors="replace")

        for _ in range(mkvc):
            kl = struct.unpack("<Q", f.read(8))[0]
            k = f.read(kl).decode("utf-8")
            vt = struct.unpack("<I", f.read(4))[0]

            if k == key_target:
                if vt != 9:
                    raise ValueError(f"Key {k} is not an array")
                et = struct.unpack("<I", f.read(4))[0]
                alen = struct.unpack("<Q", f.read(8))[0]
                assert et == 8, "Expected string array"
                return [read_string() for _ in range(alen)]
            else:
                skip_val(vt)

    return None


def bpe_decode_char(s):
    """Convert BPE Ġ/Ċ artifacts back to ASCII for lookup."""
    return s.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def build_vocab(tokens):
    """Build token→id map from vocabulary list."""
    vocab = {bpe_decode_char(tok): i for i, tok in enumerate(tokens)}
    # Ensure special tokens map directly
    for tok_str, tok_id in SPECIAL_TOKENS.items():
        vocab[tok_str] = tok_id
    return vocab


def encode_greedy(text, vocab, tokens_raw):
    """
    Greedy maximal-munch BPE encoder with special token preservation.
    """
    ids = []
    pos = 0

    # Build regex pattern to match any special token
    special_pattern = "|".join(re.escape(k) for k in SPECIAL_TOKENS.keys())

    while pos < len(text):
        # 1. Check if special token matches at current position
        match = re.match(special_pattern, text[pos:])
        if match:
            spec_str = match.group(0)
            ids.append(SPECIAL_TOKENS[spec_str])
            pos += len(spec_str)
            continue

        # 2. Match standard vocabulary substrings
        best_len = 0
        best_id = None
        for length in range(min(len(text) - pos, 30), 0, -1):
            candidate = text[pos:pos + length]
            # Ensure candidate doesn't cross into a special token
            if any(k in candidate for k in SPECIAL_TOKENS.keys()) and candidate not in SPECIAL_TOKENS:
                continue
            if candidate in vocab:
                best_len = length
                best_id = vocab[candidate]
                break

        if best_id is not None:
            ids.append(best_id)
            pos += best_len
        else:
            # Fallback byte encoding
            char = text[pos]
            byte_tok = f"<0x{ord(char):02X}>"
            if byte_tok in vocab:
                ids.append(vocab[byte_tok])
            else:
                print(f"[WARN] Cannot encode character '{char}' (pos {pos}), skipping", file=sys.stderr)
            pos += 1
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to GGUF model file")
    ap.add_argument("--text", required=True, help="Text to tokenize")
    ap.add_argument("--output-file", help="Write token IDs to file instead of stdout")
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading vocabulary from {args.model}...", file=sys.stderr)
    tokens = read_gguf_string_array(args.model, "tokenizer.ggml.tokens")
    if tokens is None:
        print("ERROR: tokenizer.ggml.tokens not found in GGUF metadata", file=sys.stderr)
        sys.exit(1)
    print(f"Vocabulary size: {len(tokens)}", file=sys.stderr)

    vocab = build_vocab(tokens)
    ids = encode_greedy(args.text, vocab, tokens)
    result = ",".join(str(i) for i in ids)

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(result + "\n")
        print(f"Written {len(ids)} token IDs to {args.output_file}", file=sys.stderr)
    else:
        print(result)

    # Print decoded tokens for verification
    print(f"\nTokenization of: '{args.text}'", file=sys.stderr)
    for i, tid in enumerate(ids):
        raw = tokens[tid] if tid < len(tokens) else "[SPECIAL]"
        decoded = bpe_decode_char(raw)
        print(f"  [{i}] ID={tid:6d}  raw={repr(raw):20s}  decoded={repr(decoded)}", file=sys.stderr)


if __name__ == "__main__":
    main()
