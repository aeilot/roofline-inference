"""Device spec: per-engine compute, bandwidth and memory, each with provenance."""

from __future__ import annotations

from dataclasses import dataclass
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
            "confidence": "high" if self.measured is not None else self.confidence,
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

    def peak_flops(self, dtype: str) -> Quantity:
        if dtype not in self.compute:
            raise KeyError(f"engine {self.name!r} has no compute entry for {dtype!r}")
        return self.compute[dtype]


@dataclass(frozen=True)
class Device:
    name: str
    total_bandwidth: Quantity
    capacity: Quantity
    engines: dict[str, Engine]

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
        )
        for name, e in data["engines"].items()
    }
    return Device(
        name=data["name"],
        total_bandwidth=Quantity.from_dict(data["memory"]["total_bandwidth"]),
        capacity=Quantity.from_dict(data["memory"]["capacity"]),
        engines=engines,
    )
