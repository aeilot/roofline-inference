"""FLOPs, minimum DRAM traffic and memory footprint for one inference step.

Traffic assumes ideal fusion (the physical bound): every weight is read once
per step, each cached K/V is read once, new K/V is written once, token ids
are read and the sampled ids written, and intermediate activations and
logits never leave the chip. Elementwise work
(norms, RoPE, SiLU, softmax, residuals) is not counted in FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass

from roofline.model import ModelSpec

TOKEN_BYTES = 4  # int32 token ids


@dataclass(frozen=True)
class Gemm:
    """C[m x n] = A[m x k] @ W[k x n], repeated `count` times per step.

    A is the activation operand, W the weight. Shapes are what the I/O bound
    in pebble.py needs; the physical bound only uses the totals above.
    """

    name: str
    m: int
    n: int
    k: int
    count: int
    elem_bytes: int  # A, W and C share the weight dtype
    persistent_out_bytes: int = 0  # output that must reach DRAM (new K/V)


@dataclass(frozen=True)
class Attention:
    """Score/value attention of one layer, repeated `count` times per step.

    `scores` counts the QK^T entries actually computed, over all query heads
    and sequences (causal entries only in prefill). FLOPs are 4 * scores *
    head_dim (QK^T and PV).
    """

    scores: int
    head_dim: int
    count: int
    elem_bytes: int


@dataclass(frozen=True)
class Workload:
    phase: str  # "decode" | "prefill"
    batch: int
    tokens: int  # tokens processed this step
    flops: dict[str, float]
    bytes: dict[str, float]  # DRAM traffic
    memory: dict[str, float]  # resident footprint
    gemms: tuple[Gemm, ...] = ()
    attention: Attention | None = None

    @property
    def special(self) -> dict[str, float]:
        """Special-function counts: one exp per attention score."""
        if self.attention is None:
            return {}
        return {"exp": self.attention.scores * self.attention.count}

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


def _embedding_gather_bytes(m: ModelSpec, rows: int) -> int:
    """Embedding rows gathered for `rows` tokens. With tied embeddings the
    rows are part of the LM-head read, which already reads the whole table."""
    return 0 if m.tie_word_embeddings else rows * m.hidden_size * m.weight_bytes_per_param


def _gemms(m: ModelSpec, rows: int, head_rows: int) -> tuple[Gemm, ...]:
    """Matmuls of one forward pass over `rows` tokens; the LM head sees
    `head_rows`. Projections that share an input are one GEMM (QKV, gate+up);
    SwiGLU runs in the gate+up epilogue, so down reads `intermediate_size`
    columns."""
    d, L, e = m.hidden_size, m.num_layers, m.weight_bytes_per_param
    new_kv = rows * 2 * m.kv_dim * m.kv_bytes_per_elem
    return (
        Gemm("qkv", rows, m.q_dim + 2 * m.kv_dim, d, L, e, persistent_out_bytes=new_kv),
        Gemm("o_proj", rows, d, m.q_dim, L, e),
        Gemm("gate_up", rows, 2 * m.intermediate_size, d, L, e),
        Gemm("down", rows, d, m.intermediate_size, L, e),
        Gemm("lm_head", head_rows, m.vocab_size, d, 1, e),
    )


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
        "embedding_gather": _embedding_gather_bytes(m, batch),
        "token_io": 2 * batch * TOKEN_BYTES,
        "kv_read": batch * context * m.kv_bytes_per_token,
        "kv_write": batch * m.kv_bytes_per_token,
    }
    memory = {
        "weights": m.weight_bytes,
        "kv_cache": batch * attended * m.kv_bytes_per_token,
    }
    attention = Attention(m.num_heads * attended * batch, m.head_dim, m.num_layers, m.kv_bytes_per_elem)
    return Workload("decode", batch, batch, flops, traffic, memory, _gemms(m, batch, batch), attention)


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
        "embedding_gather": _embedding_gather_bytes(m, batch * p),
        "token_io": (batch * p + batch) * TOKEN_BYTES,
        "kv_read": 0,
        "kv_write": batch * p * m.kv_bytes_per_token,
    }
    memory = {
        "weights": m.weight_bytes,
        "kv_cache": batch * p * m.kv_bytes_per_token,
    }
    attention = Attention(m.num_heads * causal_pairs * batch, m.head_dim, m.num_layers, m.kv_bytes_per_elem)
    return Workload(
        "prefill", batch, batch * p, flops, traffic, memory, _gemms(m, batch * p, batch), attention
    )
