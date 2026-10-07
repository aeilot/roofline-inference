"""Fixed prompt sets for v3: random tokens and real text from a frozen corpus.

A prompt is identified by (kind, length, variant). Random prompts use a seed
derived from (length, variant); text prompts cycle through a fixed corpus
(Chinese report, English README, Python source) starting at an offset derived
from the variant, so different pairs see different content.

The corpus is a snapshot (corpus/, files as of commit 8094be0) rather than the
live docs: editing the docs must not change the prompts of v3 or of
benchmarks/kv_quality.py, which records the same corpus_sha256.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
CORPUS_FILES = sorted((p for p in CORPUS_DIR.iterdir() if p.name[0].isdigit()), key=lambda p: int(p.name.split("-")[0]))


@lru_cache(maxsize=1)
def _tokenizer():
    from mlx_lm import load
    _, tok = load("Qwen/Qwen3-0.6B")
    return tok


@lru_cache(maxsize=1)
def corpus_tokens() -> tuple[int, ...]:
    tok = _tokenizer()
    text = "\n\n".join(f.read_text() for f in CORPUS_FILES)
    return tuple(tok.encode(text))


def corpus_sha256() -> str:
    h = hashlib.sha256()
    for f in CORPUS_FILES:
        h.update(f.read_bytes())
    return h.hexdigest()


def make_prompt(kind: str, length: int, variant: int) -> list[int]:
    if kind == "random":
        import mlx.core as mx
        key = mx.random.key(1_000_003 * length + variant)
        return mx.random.randint(1000, 150000, (length,), key=key).tolist()
    if kind == "text":
        toks = corpus_tokens()
        start = (variant * 7919) % len(toks)
        out = []
        while len(out) < length:
            out.extend(toks[start:start + length - len(out)])
            start = 0
        return out
    raise ValueError(kind)


def prompt_sha256(prompt: list[int]) -> str:
    return hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
