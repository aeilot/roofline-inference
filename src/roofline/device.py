"""Device spec: per-engine compute, bandwidth and memory, each with provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Quantity:
    """A device number plus where it came from.

    `measured` overrides `value` once a benchmark has filled it in.
    """

    value: float
    source: str
    confidence: str
    measured: float | None = None
    note: str = ""

    @property
    def effective(self) -> float:
        return self.measured if self.measured is not None else self.value

    @property
    def origin(self) -> str:
        return "measured" if self.measured is not None else self.source

    def describe(self) -> dict:
        return {
            "value": self.effective,
            "origin": self.origin,
            "confidence": self.confidence,
            "note": self.note,
            "certified_ceiling": False,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Quantity:
        return cls(
            value=float(data["value"]),
            source=data["source"],
            confidence=data["confidence"],
            measured=None if data.get("measured") is None else float(data["measured"]),
            note=data.get("note", ""),
        )


@dataclass(frozen=True)
class Engine:
    name: str
    compute: dict[str, Quantity]  # peak FLOP/s by dtype
    bandwidth: Quantity  # bytes/s this engine can reach on its own
    usable_memory: Quantity  # bytes this engine can address
    special: dict[str, Quantity] = field(default_factory=dict)  # ops/s of special functions (exp)

    def peak_flops(self, dtype: str) -> Quantity:
        if dtype not in self.compute:
            raise KeyError(f"engine {self.name!r} has no compute entry for {dtype!r}")
        return self.compute[dtype]


@dataclass(frozen=True)
class Level:
    """One level of on-chip storage, from the cores outward (design notes 6.8).

    `capacity` is bytes per instance (an upper estimate); `instances` is how
    many private copies exist (10 per-core memories, or 1 shared). `bandwidth`
    is the aggregate bytes/s across the boundary between this level and the
    next slower one; `dram` marks the boundary to main memory.
    """

    name: str
    instances: int
    capacity: Quantity
    bandwidth: Quantity
    dram: bool = False


@dataclass(frozen=True)
class Device:
    name: str
    total_bandwidth: Quantity
    capacity: Quantity
    engines: dict[str, Engine]
    on_chip: Quantity | None = None  # bytes of on-chip storage between DRAM and the engines
    hierarchy: tuple[Level, ...] = ()  # fastest first

    def engine(self, name: str) -> Engine:
        if name not in self.engines:
            raise KeyError(f"device {self.name!r} has no engine {name!r}")
        return self.engines[name]

    def bandwidth(self, engine: str) -> float:
        """Bandwidth available to a single engine running alone."""
        return min(self.engine(engine).bandwidth.effective, self.total_bandwidth.effective)


def load_device(path: str | Path) -> Device:
    data = yaml.safe_load(Path(path).read_text())
    engines = {
        name: Engine(
            name=name,
            compute={dt: Quantity.from_dict(q) for dt, q in e["compute"].items()},
            bandwidth=Quantity.from_dict(e["bandwidth"]),
            usable_memory=Quantity.from_dict(e["usable_memory"]),
            special={k: Quantity.from_dict(q) for k, q in e.get("special", {}).items()},
        )
        for name, e in data["engines"].items()
    }
    memory = {k: Quantity.from_dict(v) for k, v in data["memory"].items()}
    hierarchy = []
    for lv in data.get("hierarchy", []):
        cap, bw = lv["capacity"], lv["bandwidth"]
        # A string refers to memory.<name> (capacity) or engines.<name>.bandwidth (DRAM).
        capacity = memory[cap] if isinstance(cap, str) else Quantity.from_dict(cap)
        dram = isinstance(bw, str)
        bandwidth = engines[bw].bandwidth if dram else Quantity.from_dict(bw)
        hierarchy.append(Level(lv["name"], int(lv["instances"]), capacity, bandwidth, dram))
    return Device(
        name=data["name"],
        total_bandwidth=memory["total_bandwidth"],
        capacity=memory["capacity"],
        engines=engines,
        on_chip=memory.get("on_chip"),
        hierarchy=tuple(hierarchy),
    )
