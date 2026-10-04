"""Physical upper bound for a workload on a single engine."""

from __future__ import annotations

import math
from dataclasses import dataclass

from roofline.device import Device
from roofline.workload import Workload


@dataclass(frozen=True)
class Bound:
    workload: Workload
    engine: str
    dtype: str
    resource_time_s: dict[str, float]  # time each resource alone would need
    memory_limit: float
    inputs: dict[str, dict]  # device numbers used, with provenance

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
    def tokens_per_s(self) -> float:
        return self.workload.tokens / self.time_s

    def to_dict(self) -> dict:
        w = self.workload
        out = {
            "phase": w.phase,
            "batch": w.batch,
            "engine": self.engine,
            "dtype": self.dtype,
            "time_s": self.time_s,
            "tokens_per_s": self.tokens_per_s,
            "bottleneck": self.bottleneck,
            "resource_time_s": self.resource_time_s,
            "slack": self.slack,
            "arithmetic_intensity": w.arithmetic_intensity,
            "flops": w.flops,
            "bytes": w.bytes,
            "memory": {
                **w.memory,
                "total": w.total_memory,
                "limit": self.memory_limit,
                "fits": self.fits,
            },
            "inputs": self.inputs,
        }
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
