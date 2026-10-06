"""Unit tests for the v2 kernels and engine (run: ../../.venv/bin/python -m pytest -q test_engine.py)."""
import mlx.core as mx
import pytest
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

import kernels as K
from attention import attention
from engine import STREAM, Engine


@pytest.fixture(scope="module")
def setup():
    model, tok = load("Qwen/Qwen3-0.6B")
    return model, tok, Engine(model)


def f32(a):
    return a.astype(mx.float32)


@pytest.mark.parametrize("N,Kd", [(4096, 1024), (1024, 2048), (1024, 3072), (151936, 1024)])
def test_gemv_matches_mlx(N, Kd):
    W = (mx.random.normal((N, Kd), key=mx.random.key(1)) * 0.02).astype(mx.bfloat16)
    x = mx.random.normal((Kd,), key=mx.random.key(2)).astype(mx.bfloat16)
    r = mx.random.normal((N,), key=mx.random.key(3)).astype(mx.bfloat16)
    ref = (x[None] @ W.T)[0]
    assert mx.max(mx.abs(f32(K.gemv(W, x)) - f32(ref))).item() <= 2e-2
    assert mx.max(mx.abs(f32(K.gemv(W, x, res=r)) - f32(r + ref))).item() <= 4e-2


def test_gemv_norm_and_swiglu():
    W = (mx.random.normal((6144, 1024), key=mx.random.key(4)) * 0.03).astype(mx.bfloat16)
    h = mx.random.normal((1024,), key=mx.random.key(5)).astype(mx.bfloat16)
    nw = (1 + 0.1 * mx.random.normal((1024,), key=mx.random.key(6))).astype(mx.bfloat16)
    xn = mx.fast.rms_norm(h, nw, 1e-6)
    gate, up = mx.split(xn[None] @ W.T, 2, axis=-1)
    ref = (gate * mx.sigmoid(gate) * up)[0]
    out = K.gemv(W, h, nw=nw, swiglu=True)
    assert mx.max(mx.abs(f32(out) - f32(ref))).item() <= 6.25e-2  # <= 1 BF16 ulp at |y| < 16


def test_add_rms_bitwise():
    a = mx.random.normal((37, 1024), key=mx.random.key(7)).astype(mx.bfloat16)
    b = mx.random.normal((37, 1024), key=mx.random.key(8)).astype(mx.bfloat16)
    w = mx.random.normal((1024,), key=mx.random.key(9)).astype(mx.bfloat16)
    r, n = K.add_rms(a, b, w)
    assert mx.array_equal(r, a + b).item()
    assert mx.array_equal(n, mx.fast.rms_norm(a + b, w, 1e-6)).item()


def _qk_reference(qkv, qw, kw, offset, L):
    q, k, v = mx.split(qkv.reshape(1, L, -1), [2048, 3072], axis=-1)
    q = mx.fast.rms_norm(q.reshape(1, L, 16, 128), qw, 1e-6).transpose(0, 2, 1, 3)
    k = mx.fast.rms_norm(k.reshape(1, L, 8, 128), kw, 1e-6).transpose(0, 2, 1, 3)
    v = v.reshape(1, L, 8, 128).transpose(0, 2, 1, 3)
    q = mx.fast.rope(q, 128, traditional=False, base=1e6, scale=1.0, offset=offset)
    k = mx.fast.rope(k, 128, traditional=False, base=1e6, scale=1.0, offset=offset)
    return q, k, v


@pytest.mark.parametrize("L,offset", [(1, 0), (1, 5000), (37, 0), (64, 300)])
def test_qk_norm_rope(L, offset):
    qkv = mx.random.normal((L, 4096), key=mx.random.key(10)).astype(mx.bfloat16)
    qw = (1 + 0.1 * mx.random.normal((128,), key=mx.random.key(11))).astype(mx.bfloat16)
    kw = (1 + 0.1 * mx.random.normal((128,), key=mx.random.key(12))).astype(mx.bfloat16)
    q, k, v = _qk_reference(qkv, qw, kw, offset, L)
    if L == 1:
        q2, kv = K.qk_norm_rope(qkv[0], qw, kw, mx.array([offset], dtype=mx.int32), 16, 8, 1e6)
    else:
        Lp = (L + 31) // 32 * 32
        q2, kv = K.qk_norm_rope_prefill(qkv, qw, kw, offset, 16, 8, 1e6, Lp)
        assert mx.all(q2[:, L:] == 0).item()
        q2 = q2[None, :, :L]
    # RoPE angle evaluation may differ by float rounding; one BF16 ulp at |x| <~ 4.
    assert mx.max(mx.abs(f32(q2) - f32(q))).item() <= 3.2e-2
    assert mx.max(mx.abs(f32(kv[0:1]) - f32(k))).item() <= 3.2e-2
    assert mx.array_equal(kv[1:2], v).item()


@pytest.mark.parametrize("qL,kL", [(37, 37), (128, 300), (1000, 1000), (512, 2600)])
def test_gqa_attention_vs_fp32(qL, kL):
    H, NKV, D = 16, 8, 128
    cap = (kL + 255) // 256 * 256
    k = mx.random.normal((1, NKV, kL, D), key=mx.random.key(13)).astype(mx.bfloat16)
    v = mx.random.normal((1, NKV, kL, D), key=mx.random.key(14)).astype(mx.bfloat16)
    q = (1.5 * mx.random.normal((1, H, qL, D), key=mx.random.key(15))).astype(mx.bfloat16)
    buf = mx.zeros((2, NKV, cap, D), mx.bfloat16)
    buf[0:1, :, :kL] = k
    buf[1:2, :, :kL] = v
    qLp = (qL + 31) // 32 * 32
    Q = mx.pad(q[0], [(0, 0), (0, qLp - qL), (0, 0)])
    out = attention(Q, buf[0:1], buf[1:2], qL, kL, D ** -0.5)
    ref32 = mx.fast.scaled_dot_product_attention(f32(q), f32(k), f32(v), scale=D ** -0.5, mask="causal")
    ref32 = ref32[0].transpose(1, 0, 2).reshape(qL, H * D)
    mlx = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask="causal")[0].transpose(1, 0, 2).reshape(qL, -1)
    e_ours = mx.max(mx.abs(f32(out) - ref32)).item()
    e_mlx = mx.max(mx.abs(f32(mlx) - ref32)).item()
    assert e_ours <= 1.1 * e_mlx + 1e-6


@pytest.mark.parametrize("M,N,Kd", [(9, 4096, 1024), (16, 6144, 1024), (40, 1024, 2048), (56, 1024, 3072)])
def test_skinny_matches_matmul(M, N, Kd):
    x = mx.random.normal((M, Kd), key=mx.random.key(16)).astype(mx.bfloat16)
    W = (mx.random.normal((N, Kd), key=mx.random.key(17)) * 0.03).astype(mx.bfloat16)
    ref32 = f32(x) @ f32(W).T
    e = mx.max(mx.abs(f32(K.skinny_matmul(x, W)) - ref32)).item()
    e_mlx = mx.max(mx.abs(f32(x @ W.T) - ref32)).item()
    assert e <= 1.1 * e_mlx + 1e-6


@pytest.mark.parametrize("chunks", [[16], [37], [100, 1], [255, 2, 1], [64, 64]])
def test_prefill_all_logits_close_to_mlx(setup, chunks):
    model, _, eng = setup
    rc, ec = make_prompt_cache(model), eng.new_cache(sum(chunks) + 8)
    for n in chunks:
        x = mx.random.randint(1000, 150000, (1, n), key=mx.random.key(n + rc[0].offset))
        a = model(x, cache=rc)
        with mx.stream(STREAM):
            b = eng.prefill(x, ec, last=True, all_logits=True)
        mx.eval(a, b)
        d = f32(a) - f32(b)
        assert mx.sqrt(mx.mean(d * d)).item() < 0.08
        # Where greedy choices differ they must be near-ties of the reference logits
        # (threshold = mlx-lm's own p99 BF16 top-2 margin error, see check.json).
        ia, ib = mx.argmax(a, -1), mx.argmax(b, -1)
        gap = mx.take_along_axis(a, ia[..., None], -1) - mx.take_along_axis(a, ib[..., None], -1)
        assert mx.max(f32(gap)).item() < 0.25
        assert rc[0].offset == ec.offset


def test_chunked_prefill_matches_single(setup):
    _, _, eng = setup
    prompt = mx.random.randint(1000, 150000, (300,), key=mx.random.key(20)).tolist()
    a = list(eng.generate(prompt, max_tokens=12, chunk=4096))
    b = list(eng.generate(prompt, max_tokens=12, chunk=128))
    assert a[:4] == b[:4]  # later tokens may legitimately diverge only at FP32 near-ties


def test_real_prompt_greedy(setup):
    model, tok, eng = setup
    from mlx_lm.generate import generate_step
    prompt = tok.apply_chat_template([{"role": "user", "content": "What is 2 + 2? Answer with one number."}],
                                     tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ref = [t for t, _ in generate_step(mx.array(prompt), model, max_tokens=8)]
    out = list(eng.generate(prompt, max_tokens=8))
    assert out[:3] == ref[:3]
    assert "4" in tok.decode(out)


def test_eos_and_limits(setup):
    _, _, eng = setup
    prompt = [1000, 2000, 3000]
    assert list(eng.generate(prompt, max_tokens=0)) == []
    first = next(eng.generate(prompt, max_tokens=5))
    assert list(eng.generate(prompt, max_tokens=5, eos_ids={first})) == [first]
    with pytest.raises(ValueError):
        list(eng.generate([], max_tokens=3))
    with pytest.raises(ValueError):
        list(eng.generate([10 ** 7], max_tokens=3))
    cache = eng.new_cache(4)
    with pytest.raises(ValueError):
        cache.update(0, mx.zeros((2, 8, 300, 128), mx.bfloat16))
