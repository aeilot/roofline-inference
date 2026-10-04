"""Implementation-independent upper bounds on LLM inference performance."""

from roofline.bound import Bound, physical_bound
from roofline.device import Device, load_device
from roofline.model import ModelSpec, load_model
from roofline.workload import Workload, decode, prefill

__all__ = [
    "Bound",
    "Device",
    "ModelSpec",
    "Workload",
    "decode",
    "load_device",
    "load_model",
    "physical_bound",
    "prefill",
]
