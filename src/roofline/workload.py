"""FLOPs, minimum DRAM traffic and memory footprint for one inference step.

Traffic assumes ideal fusion (the physical bound): every weight is read once
per step, each cached K/V is read once, new K/V is written once, and
intermediate activations and logits never leave the chip. Elementwise work
(norms, RoPE, SiLU, softmax, residuals) is not counted in FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass

from roofline.model import ModelSpec


@dataclass(frozen=True)
class Workload:
    phase: str  # "decode" | "prefill"
    batch: int
    tokens: int  # tokens processed this step
    flops: dict[str, float]
    bytes: dict[str, float]  # DRAM traffic
    memory: dict[str, float]  # resident footprint

    @property
    def total_flops(self) -> float:
        return sum(self.flops.values())

    @property
    def total_bytes(self) -> float:
        return sum(self.bytes.values())

    @property
    def total_memory(self) -> float:
        return sum(self.memory.values())

    @property
    def arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_bytes


def _weights_read_bytes(m: ModelSpec) -> int:
    """Weights streamed once per forward pass.

    The LM head matrix is read in full. With tied embeddings that matrix is
    the embedding table; otherwise the embedding table is only gathered.
    """
    layers = m.num_layers * (m.layer_linear_params + m.layer_norm_params)
    final_norm = m.hidden_size
    head = m.vocab_size * m.hidden_size
    return (layers + final_norm + head) * m.weight_bytes_per_param


def decode(m: ModelSpec, batch: int, context: int) -> Workload:
    """One decode step: each of `batch` sequences has `context` cached tokens
    and generates one new token, which attends over context + 1 positions."""
    if batch < 1 or context < 0:
        raise ValueError("batch must be >= 1 and context >= 0")
    attended = context + 1
    flops = {
        "linear": 2 * m.num_layers * m.layer_linear_params * batch,
        "attention": 4 * m.num_layers * m.num_heads * m.head_dim * attended * batch,
        "lm_head": 2 * m.vocab_size * m.hidden_size * batch,
    }
    traffic = {
        "weights": _weights_read_bytes(m),
        "embedding_gather": batch * m.hidden_size * m.weight_bytes_per_param,
        "kv_read": batch * context * m.kv_bytes_per_token,
        "kv_write": batch * m.kv_bytes_per_token,
    }
    memory = {
        "weights": m.weight_bytes,
        "kv_cache": batch * attended * m.kv_bytes_per_token,
    }
    return Workload("decode", batch, batch, flops, traffic, memory)


def prefill(m: ModelSpec, batch: int, prompt_len: int) -> Workload:
    """Prefill `batch` prompts of `prompt_len` tokens with causal attention.
    Logits are only needed for the last position of each prompt."""
    if batch < 1 or prompt_len < 1:
        raise ValueError("batch and prompt_len must be >= 1")
    p = prompt_len
    # Causal: position i attends to i keys; QK^T and AV each cost 2*i*head_dim.
    causal_pairs = p * (p + 1) // 2
    flops = {
        "linear": 2 * m.num_layers * m.layer_linear_params * batch * p,
        "attention": 4 * m.num_layers * m.num_heads * m.head_dim * causal_pairs * batch,
        "lm_head": 2 * m.vocab_size * m.hidden_size * batch,
    }
    traffic = {
        "weights": _weights_read_bytes(m),
        "embedding_gather": batch * p * m.hidden_size * m.weight_bytes_per_param,
        "kv_read": 0,
        "kv_write": batch * p * m.kv_bytes_per_token,
    }
    memory = {
        "weights": m.weight_bytes,
        "kv_cache": batch * p * m.kv_bytes_per_token,
    }
    return Workload("prefill", batch, batch * p, flops, traffic, memory)
