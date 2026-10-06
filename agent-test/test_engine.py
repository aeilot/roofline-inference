"""Run with: .venv/bin/python -m pytest -q agent-test/test_engine.py"""
import mlx.core as mx
import mlx.nn as nn
import pytest
import weakref
from mlx_lm import load
from mlx_lm.generate import generate_step, generation_stream

from benchmark import check_forward
from engine import Engine
from kernels import add_rms
import engine as engine_module


@pytest.fixture(scope="module")
def models():
    ref, tok = load("Qwen/Qwen3-0.6B")
    return ref, Engine(ref, fused_norm=True), tok


@pytest.mark.parametrize("rows", [1, 2, 16, 257])
def test_add_rms_preserves_bf16_rounding(rows):
    norm = nn.RMSNorm(1024, eps=1e-6)
    norm.weight = mx.random.uniform(0.5, 1.5, (1024,), key=mx.random.key(8)).astype(mx.bfloat16)
    a = mx.random.normal((1, rows, 1024), key=mx.random.key(1)).astype(mx.bfloat16)
    b = mx.random.normal(a.shape, key=mx.random.key(2)).astype(mx.bfloat16)
    residual, normalized = add_rms(a, b, norm)
    expected = norm(a + b)
    mx.eval(residual, normalized, expected)
    assert mx.array_equal(residual, a + b).item()
    assert mx.array_equal(normalized, expected).item()


def test_logits_and_chunked_cache(models):
    ref, engine, _ = models
    assert len(check_forward(ref, engine)) == 12


@pytest.mark.parametrize("text", [
    "Explain the roofline model in two sentences.",
    "请用中文解释为什么单流 decode 受内存带宽限制。",
    "Write a Python function that computes the Fibonacci sequence.",
])
def test_real_prompt_greedy_sequence(models, text):
    ref, engine, tok = models
    prompt = tok.apply_chat_template([{"role": "user", "content": text}],
                                    tokenize=True, add_generation_prompt=True,
                                    enable_thinking=False)
    expected = [t for t, _ in generate_step(mx.array(prompt), ref,
                                           max_tokens=33, prefill_step_size=16)]
    mx.synchronize(generation_stream)
    actual = list(engine.generate(prompt, max_tokens=33, prefill_step_size=16))
    mx.synchronize(generation_stream)
    assert actual == expected


def test_zero_length_generation_and_input_checks(models):
    _, engine, _ = models
    assert list(engine.generate([42], max_tokens=0)) == []
    for kwargs in (dict(prompt=[]), dict(prompt=[-1]),
                   dict(prompt=[engine.args.vocab_size]),
                   dict(prompt=[42], max_tokens=-1),
                   dict(prompt=[42], prefill_step_size=0)):
        with pytest.raises(ValueError):
            list(engine.generate(**kwargs))


def test_cache_consistency_and_eos(models):
    _, engine, _ = models
    cache = engine.new_cache()
    cache[0].offset = 1
    with pytest.raises(ValueError):
        engine.forward(mx.array([[42]]), cache)
    first = list(engine.generate([42], max_tokens=1))[0]
    mx.synchronize(generation_stream)
    assert list(engine.generate([42], max_tokens=8, eos_ids=[first])) == [first]
    mx.synchronize(generation_stream)


def test_prefill_does_not_retain_unused_graph(models, monkeypatch):
    _, engine, _ = models
    original = engine.forward
    refs = []

    class HiddenGraph:
        def __init__(self, array):
            self.array = array

    def checked_forward(*args, **kwargs):
        # A prior unused output must be released BEFORE the next cache update.
        assert all(ref() is None for ref in refs)
        output = original(*args, **kwargs)
        if kwargs.get("logits", True) is False:
            holder = HiddenGraph(output)
            refs.append(weakref.ref(holder))
            return holder
        return output

    monkeypatch.setattr(engine, "forward", checked_forward)
    assert len(list(engine.generate([42] * 33, max_tokens=1, prefill_step_size=16))) == 1
    mx.synchronize(generation_stream)
    assert len(refs) == 2


def test_cli_retains_streaming_detokenizer(models, monkeypatch, capsys):
    ref, engine, tok = models
    tokens = tok.encode("Hello world!")
    monkeypatch.setattr(engine_module, "load", lambda _: (ref, tok))
    monkeypatch.setattr(engine_module, "Engine", lambda *a, **kw: engine)
    monkeypatch.setattr(engine, "generate", lambda *a, **kw: iter(tokens))
    monkeypatch.setattr("sys.argv", ["engine.py", "--prompt", "Hello"])
    engine_module.main()
    assert capsys.readouterr().out.strip() == "Hello world!"
