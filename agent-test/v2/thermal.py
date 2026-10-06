"""Thermal gating: ProcessInfo.thermalState + a fixed GPU throughput probe."""
import subprocess
import time
from pathlib import Path

import mlx.core as mx

_BIN = Path(__file__).with_name("thermstate")
_A = None


def thermal_state() -> int:
    """0 nominal, 1 fair, 2 serious, 3 critical (NSProcessInfo.thermalState)."""
    return int(subprocess.run([str(_BIN)], capture_output=True, text=True).stdout.strip())


def probe_ms() -> float:
    """~10 ms fixed BF16 matmul workload; lower is a faster (cooler) GPU state."""
    global _A
    if _A is None:
        _A = mx.random.normal((2048, 2048)).astype(mx.bfloat16)
        mx.eval(_A)
    x = _A
    mx.eval(x @ x)
    s = time.perf_counter()
    for _ in range(2):
        x = (x @ _A) * 0.01
    mx.eval(x)
    return (time.perf_counter() - s) * 1000


def wait_cool(reference_ms=None, tol=0.05, idle=2.0, timeout=600):
    """Sleep until thermal state is nominal and the probe is within tol of reference."""
    start = time.time()
    while True:
        time.sleep(idle)
        st = thermal_state()
        p = min(probe_ms(), probe_ms())
        if st == 0 and (reference_ms is None or p <= reference_ms * (1 + tol)):
            return dict(thermal_state=st, probe_ms=p, waited_s=time.time() - start)
        if time.time() - start > timeout:
            return dict(thermal_state=st, probe_ms=p, waited_s=time.time() - start, timeout=True)
        time.sleep(10)
