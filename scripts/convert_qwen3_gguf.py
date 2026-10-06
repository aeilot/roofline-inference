"""Convert Qwen3 safetensors (BF16) to a BF16 GGUF for llama.cpp, without torch.

The upstream converter (llama.cpp convert_hf_to_gguf.py) needs PyTorch; this
one reads safetensors directly and writes with the `gguf` package. It covers
only what Qwen3 dense models need: architecture metadata, the GPT-2 style BPE
vocabulary (tokenizer.json via transformers), merges and special tokens
(gguf.SpecialVocab), and the tensors under their GGUF names. Weights are copied
bit for bit as BF16 (1-D norm weights widened to F32, as
upstream does); Qwen uses NeoX RoPE, so Q/K are not permuted. With tied
embeddings there is no output.weight and llama.cpp reuses token_embd.

    python scripts/convert_qwen3_gguf.py ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<rev> \
        data/qwen3-0.6b-bf16.gguf
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import gguf
import numpy as np


def read_safetensors(path: Path):
    """Yield (name, shape, raw uint16 array) for every BF16 tensor."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
    data = np.memmap(path, dtype=np.uint8, mode="r")
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if meta["dtype"] != "BF16":
            raise ValueError(f"{name}: expected BF16, got {meta['dtype']}")
        lo, hi = meta["data_offsets"]
        yield name, tuple(meta["shape"]), data[base + lo: base + hi].view(np.uint16).reshape(meta["shape"])


def vocab(model_dir: Path, n_vocab: int) -> tuple[list[str], list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    reverse = {i: t for t, i in tok.get_vocab().items()}
    added = tok.get_added_vocab()
    special = set(tok.all_special_tokens)
    tokens, types = [], []
    for i in range(n_vocab):
        if i not in reverse:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
            continue
        t = reverse[i]
        tokens.append(t)
        if t in added:
            types.append(gguf.TokenType.CONTROL if t in special or t.startswith("<|") else gguf.TokenType.USER_DEFINED)
        else:
            types.append(gguf.TokenType.NORMAL)
    return tokens, types


def main() -> None:
    model_dir, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
    cfg = json.loads((model_dir / "config.json").read_text())
    n_layers = cfg["num_hidden_layers"]
    rope_theta = cfg.get("rope_theta") or cfg.get("rope_parameters", {}).get("rope_theta")

    w = gguf.GGUFWriter(out, "qwen3")
    w.add_name(model_dir.parent.parent.name.removeprefix("models--").replace("--", "/"))
    w.add_block_count(n_layers)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_rope_freq_base(float(rope_theta))
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)

    tokens, types = vocab(model_dir, cfg["vocab_size"])
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    special = gguf.SpecialVocab(model_dir, load_merges=True, n_vocab=len(tokens))
    special.add_to_gguf(w)

    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, n_layers)
    count = 0
    for f in sorted(model_dir.glob("*.safetensors")):
        for name, shape, raw in read_safetensors(f):
            if name == "lm_head.weight" and cfg.get("tie_word_embeddings"):
                continue
            new = names.get_name(name, try_suffixes=(".weight", ".bias"))
            if new is None:
                raise ValueError(f"no GGUF name for {name}")
            if len(shape) == 1:  # norm weights: ggml wants F32, as upstream writes them
                w.add_tensor(new, (raw.astype(np.uint32) << 16).view(np.float32))
            else:
                w.add_tensor(new, raw, raw_shape=shape, raw_dtype=gguf.GGMLQuantizationType.BF16)
            count += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out} ({count} tensors, {out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
