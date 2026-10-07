"""Official vLLM + vllm-metal, offline batch-one API (no HTTP transport).

Prefix caching and speculative decoding are disabled. This comparator includes
vLLM scheduling/output processing; it is not a CUDA or kernel-only comparison.
"""
from __future__ import annotations
import itertools
import mlx.core as mx


class V0:
    def __init__(self, model_path):
        from vllm import EngineArgs, LLMEngine
        self.engine = LLMEngine.from_engine_args(EngineArgs(
            model=str(model_path), dtype="bfloat16", max_model_len=40960,
            max_num_seqs=1, enable_prefix_caching=False,
            gpu_memory_utilization=0.40,
            enable_chunked_prefill=True, max_num_batched_tokens=4096,
            enforce_eager=True, disable_log_stats=True,
        ), enable_multiprocessing=False)
        from vllm.platforms import current_platform
        platform_name = type(current_platform).__module__
        if not platform_name.startswith("vllm_metal"):
            raise RuntimeError(f"Expected Metal platform, got {platform_name}")
        self.ids = itertools.count()
        self.stream = None
        self.identity = dict(engine="vLLM", backend="vllm-metal", platform=platform_name, dtype="bfloat16",
                             prefix_caching=False, speculative_decoding=False,
                             max_num_seqs=1, prefill_chunk=4096, gpu_memory_utilization=0.40, transport="offline")

    def generate(self, prompt, max_tokens):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        params = SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True,
                                detokenize=False, output_kind=RequestOutputKind.DELTA)
        self.engine.add_request(str(next(self.ids)), {"prompt_token_ids": prompt}, params)
        count = 0
        while self.engine.has_unfinished_requests():
            for output in self.engine.step():
                ids = output.outputs[0].token_ids
                # A packet timestamp cannot be treated as individual token latency.
                if len(ids) > 1:
                    raise RuntimeError("vLLM batched multiple tokens; per-token timing is not comparable")
                for token in ids:
                    count += 1
                    yield int(token)
        if count != max_tokens:
            raise RuntimeError(f"vLLM returned {count}, expected {max_tokens} tokens")

    def synchronize(self):
        # In-process engine.step materializes each token and drains finished work.
        mx.synchronize()
