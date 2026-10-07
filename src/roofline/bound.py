"""Physical upper bound for a workload on a single engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from roofline.device import Device
from roofline.pebble import operator_traffic
from roofline.workload import TOKEN_BYTES, Workload


@dataclass(frozen=True)
class Bound:
    workload: Workload
    engine: str
    dtype: str
    resource_time_s: dict[str, float]  # time each resource alone would need
    memory_limit: float
    inputs: dict[str, dict]  # device numbers used, with provenance
    kind: str = "physical"
    traffic: dict | None = None  # DRAM traffic when it differs from workload.bytes

    @property
    def time_s(self) -> float:
        return max(self.resource_time_s.values())

    @property
    def bottleneck(self) -> str:
        return max(self.resource_time_s, key=self.resource_time_s.get)

    @property
    def slack(self) -> dict[str, float]:
        """How many times slower each resource could be before it binds.
        1.0 marks the bottleneck."""
        t = self.time_s
        return {r: (t / rt if rt > 0 else math.inf) for r, rt in self.resource_time_s.items()}

    @property
    def fits(self) -> bool:
        return self.workload.total_memory <= self.memory_limit

    @property
    def dram_bytes(self) -> float:
        return self.traffic["total"] if self.traffic else self.workload.total_bytes

    @property
    def tokens_per_s(self) -> float:
        return self.workload.tokens / self.time_s

    def to_dict(self) -> dict:
        w = self.workload
        out = {
            "bound": self.kind,
            "interpretation": "calibrated_resource_reference",
            "hardware_ceiling_certified": False,
            "computation_scope": w.computation_scope,
            "assumptions": ["Measured throughput is a calibration, not a certified hardware maximum.",
                            "Traffic assumes explicit KV storage; alternative state representations are not exhausted.",
                            "Resource maximum does not establish an attainable schedule.",
                            "Exp count assumes standard softmax; it is not a universal algorithmic lower bound."],
            "phase": w.phase,
            "batch": w.batch,
            "engine": self.engine,
            "dtype": self.dtype,
            "time_s": self.time_s,
            "tokens_per_s": self.tokens_per_s,
            "bottleneck": self.bottleneck,
            "resource_time_s": self.resource_time_s,
            "slack": self.slack,
            "arithmetic_intensity": w.total_flops / self.dram_bytes,
            "flops": w.flops,
            "bytes": self.traffic["bytes"] if self.traffic else w.bytes,
            "memory": {
                **w.memory,
                "total": w.total_memory,
                "limit": self.memory_limit,
                "fits": self.fits,
            },
            "inputs": self.inputs,
        }
        if self.traffic:
            if "gemms" in self.traffic:
                out["gemms"] = self.traffic["gemms"]
            if "resident_input_credit" in self.traffic:
                out["resident_input_credit"] = self.traffic["resident_input_credit"]
            if self.traffic.get("attention"):
                out["attention"] = self.traffic["attention"]
        if w.phase == "decode":
            out["per_sequence_tokens_per_s"] = 1 / self.time_s
        else:
            out["ttft_s"] = self.time_s
        return out


def physical_bound(workload: Workload, device: Device, engine: str = "gpu", dtype: str = "bf16", *, initial_resident_bytes: float | None = None) -> Bound:
    """Calibrated resource reference; a lower bound only if rates are ceilings.

    Credit initially resident inputs (never compulsory output writes). The
    default credits all modeled storage inside DRAM; pass 0 for a cold start.
    """
    e = device.engine(engine)
    peak = e.peak_flops(dtype)
    bandwidth = device.bandwidth(engine)
    residency_overridden = initial_resident_bytes is not None
    if initial_resident_bytes is None:
        initial_resident_bytes = (sum(l.capacity.effective * l.instances for l in device.hierarchy)
                                  if device.hierarchy else device.on_chip.effective if device.on_chip else 0.0)
    if not math.isfinite(initial_resident_bytes) or initial_resident_bytes < 0:
        raise ValueError("initial_resident_bytes must be finite and nonnegative")
    readable = sum(v for k, v in workload.bytes.items() if k != "kv_write") - workload.batch * TOKEN_BYTES
    credit = min(initial_resident_bytes, max(0.0, readable))
    dram_bytes = workload.total_bytes - credit
    resources = {
        "compute": workload.total_flops / peak.effective,
        "dram_bandwidth": dram_bytes / bandwidth,
    }
    inputs = {
        f"{engine}.compute.{dtype}": peak.describe(),
        f"{engine}.bandwidth": e.bandwidth.describe(),
        "memory.total_bandwidth": device.total_bandwidth.describe(),
        f"{engine}.usable_memory": e.usable_memory.describe(),
    }
    inputs["initial_resident_bytes"] = {
        "value": initial_resident_bytes,
        "origin": "override" if residency_overridden else "modeled_capacity",
        "confidence": "conditional",
        "capacity_assumptions": ([] if residency_overridden else [
            {"name": l.name, "instances": l.instances, **l.capacity.describe()} for l in device.hierarchy]),
    }
    # Special functions run on their own units: a separate resource in the max.
    for name, count in workload.special.items():
        if name in e.special:
            resources[name] = count / e.special[name].effective
            inputs[f"{engine}.special.{name}"] = e.special[name].describe()
    return Bound(
        workload=workload,
        engine=engine,
        dtype=dtype,
        resource_time_s=resources,
        memory_limit=e.usable_memory.effective,
        inputs=inputs,
        traffic={"total": dram_bytes, "bytes": {**workload.bytes, "resident_input_credit": -credit},
                 "resident_input_credit": credit},
    )


def operator_bound(
    workload: Workload,
    device: Device,
    engine: str = "gpu",
    dtype: str = "bf16",
    on_chip_bytes: float | None = None,
) -> Bound:
    """Physical bound with DRAM traffic from pebble.operator_traffic: finite
    on-chip memory and an operator boundary at every GEMM (layer 2).

    `on_chip_bytes` overrides the device spec, so the constraint can be relaxed.
    """
    b = physical_bound(workload, device, engine, dtype)
    inputs = dict(b.inputs)
    if on_chip_bytes is None:
        if device.on_chip is None:
            raise ValueError(f"device {device.name!r} has no memory.on_chip; pass on_chip_bytes")
        on_chip_bytes = device.on_chip.effective
        inputs["memory.on_chip"] = device.on_chip.describe()
    else:
        inputs["memory.on_chip"] = {"value": on_chip_bytes, "origin": "override", "confidence": "n/a"}
    traffic = operator_traffic(workload, on_chip_bytes)
    credit = b.traffic["resident_input_credit"]
    traffic["total"] -= credit
    traffic["bytes"]["resident_input_credit"] = -credit
    traffic["resident_input_credit"] = credit
    return Bound(
        workload=workload,
        engine=engine,
        dtype=dtype,
        resource_time_s={**b.resource_time_s, "dram_bandwidth": traffic["total"] / device.bandwidth(engine)},
        memory_limit=b.memory_limit,
        inputs=inputs,
        kind="operator",
        traffic=traffic,
    )


def ttft(model, device: Device, prompt_len: int, engine: str = "gpu") -> float:
    """Time to first token: prefill of the prompt, LM head and sampling of the last position."""
    from roofline.workload import prefill

    return physical_bound(prefill(model, 1, prompt_len), device, engine, model.weight_dtype).time_s


def request_time(model, device: Device, prompt_len: int, gen: int, engine: str = "gpu") -> dict:
    """T_request(L, N) = TTFT(L) + sum_{i=1}^{N-1} T_decode(L + i) (design notes 5.4).

    Decode step i processes generated token i with L + i - 1 tokens cached,
    attending to L + i positions."""
    from roofline.workload import decode

    if gen < 1:
        raise ValueError("gen must be >= 1")
    t_prefill = ttft(model, device, prompt_len, engine)
    steps = [physical_bound(decode(model, 1, prompt_len + i - 1), device, engine, model.weight_dtype).time_s
             for i in range(1, gen)]
    total = t_prefill + sum(steps)
    return {
        "interpretation": "calibrated_resource_reference", "hardware_ceiling_certified": False,
        "computation_scope": "useful_product_graph",
        "prompt_len": prompt_len, "gen": gen,
        "ttft_s": t_prefill, "decode_s": sum(steps), "total_s": total,
        "prefill_share": t_prefill / total,
        "decode_first_s": steps[0] if steps else 0.0, "decode_last_s": steps[-1] if steps else 0.0,
    }
