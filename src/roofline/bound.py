"""Physical upper bound for a workload on a single engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from roofline.device import Device
from roofline.pebble import operator_traffic
from roofline.workload import Workload


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
            out["gemms"] = self.traffic["gemms"]
        if w.phase == "decode":
            out["per_sequence_tokens_per_s"] = 1 / self.time_s
        else:
            out["ttft_s"] = self.time_s
        return out


def physical_bound(workload: Workload, device: Device, engine: str = "gpu", dtype: str = "bf16") -> Bound:
    """T >= max(FLOPs / peak, bytes / bandwidth), plus a capacity check."""
    e = device.engine(engine)
    peak = e.peak_flops(dtype)
    bandwidth = device.bandwidth(engine)
    return Bound(
        workload=workload,
        engine=engine,
        dtype=dtype,
        resource_time_s={
            "compute": workload.total_flops / peak.effective,
            "dram_bandwidth": workload.total_bytes / bandwidth,
        },
        memory_limit=e.usable_memory.effective,
        inputs={
            f"{engine}.compute.{dtype}": peak.describe(),
            f"{engine}.bandwidth": e.bandwidth.describe(),
            "memory.total_bandwidth": device.total_bandwidth.describe(),
            f"{engine}.usable_memory": e.usable_memory.describe(),
        },
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
