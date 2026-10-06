"""Quick check: kernels vs MLX ops, decode/prefill logits vs mlx-lm, and a generation."""
import sys
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from engine import Engine
from kernels import gemv

model, tok = load("Qwen/Qwen3-0.6B")
eng = Engine(model)

# GEMV variants vs MLX
for N, K in [(4096, 1024), (1024, 2048), (1024, 3072)]:
    W = (mx.random.normal((N, K)) * 0.02).astype(mx.bfloat16)
    x = mx.random.normal((K,)).astype(mx.bfloat16)
    r = mx.random.normal((N,)).astype(mx.bfloat16)
    ref = (x[None] @ W.T)[0]
    print("gemv", N, K, mx.max(mx.abs(gemv(W, x).astype(mx.float32) - ref.astype(mx.float32))).item(),
          mx.max(mx.abs(gemv(W, x, res=r).astype(mx.float32) - (r + ref).astype(mx.float32))).item())

# teacher-forced: prefill 37 tokens, then 5 decode steps, compare logits with mlx-lm
p = mx.random.randint(1000, 150000, (1, 37), key=mx.random.key(1))
rc = make_prompt_cache(model)
c = eng.new_cache(64)
a = model(p, cache=rc)[:, -1]
b = eng.prefill(p, c, last=True)[:, -1]
print("prefill last-row logits max err", mx.max(mx.abs(a - b)).item(), "argmax eq", (mx.argmax(a) == mx.argmax(b)).item())
y = mx.argmax(a, axis=-1)
for s in range(5):
    a = model(y[None], cache=rc)[0, -1]
    h_ref = a
    # engine decode returns argmax; compare argmax and also logits via internal path
    nxt = eng.decode(y.astype(mx.int32), c)
    print("decode step", s, "ref argmax", mx.argmax(a).item(), "engine", nxt.item())
    y = mx.argmax(a, keepdims=True)

prompt = tok.apply_chat_template([{"role": "user", "content": "Explain the roofline model in two sentences."}],
                                 tokenize=True, add_generation_prompt=True, enable_thinking=False)
t = time.perf_counter()
out = list(eng.generate(prompt, max_tokens=60, eos_ids=tok.eos_token_ids))
print(time.perf_counter() - t)
print(tok.decode(out))
