"""Model spec for dense decoder-only transformers (GQA attention, SwiGLU FFN)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    qk_norm: bool
    weight_dtype: str
    kv_dtype: str

    def __post_init__(self) -> None:
        for dt in (self.weight_dtype, self.kv_dtype):
            if dt not in DTYPE_BYTES:
                raise ValueError(f"unsupported dtype {dt!r}")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")

    @property
    def weight_bytes_per_param(self) -> int:
        return DTYPE_BYTES[self.weight_dtype]

    @property
    def kv_bytes_per_elem(self) -> int:
        return DTYPE_BYTES[self.kv_dtype]

    @property
    def q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def layer_linear_params(self) -> int:
        """Matmul weights in one layer: Q, K, V, O projections and gate/up/down."""
        d = self.hidden_size
        attn = d * self.q_dim + 2 * d * self.kv_dim + self.q_dim * d
        ffn = 3 * d * self.intermediate_size
        return attn + ffn

    @property
    def layer_norm_params(self) -> int:
        norms = 2 * self.hidden_size
        if self.qk_norm:
            norms += 2 * self.head_dim
        return norms

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_params(self) -> int:
        """Separate LM head weights; 0 when tied to the embedding table."""
        return 0 if self.tie_word_embeddings else self.vocab_size * self.hidden_size

    @property
    def total_params(self) -> int:
        per_layer = self.layer_linear_params + self.layer_norm_params
        final_norm = self.hidden_size
        return self.num_layers * per_layer + final_norm + self.embedding_params + self.lm_head_params

    @property
    def weight_bytes(self) -> int:
        return self.total_params * self.weight_bytes_per_param

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V for one token across all layers."""
        return 2 * self.num_layers * self.kv_dim * self.kv_bytes_per_elem


def load_model(path: str | Path) -> ModelSpec:
    data = yaml.safe_load(Path(path).read_text())
    known = {f.name for f in fields(ModelSpec)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown model spec fields: {sorted(unknown)}")
    return ModelSpec(**data)
