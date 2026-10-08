#!/usr/bin/env python3
"""Thermal / power / sleep state for benchmark gating, on macOS and Linux (+NVIDIA).

Use in a worker around every timed request:

    from platform_state import snapshot, slept_between, ready
    before = snapshot()
    ... timed request ...
    after = snapshot()
    if not (before["on_ac"] and after["on_ac"]):   exclude("not_on_ac_power")
    elif slept_between(before, after):              exclude("system_slept_during_measurement")

and before a request: loop until `ready(snapshot())` is True (or a timeout -> exclude "gate_timeout").

Why each signal exists (lessons from the M4 study):
- thermal: sustained load throttled 4k TTFT from 1.9 s to 3.1 s; `pmset -g therm` recorded nothing, so
  macOS uses NSProcessInfo.thermalState (coarse: 0 nominal .. 3 critical).
- power: on battery the performance policy changes, and a closed lid forces sleep.
- sleep: a smoke run with the lid closed on battery ran 3x slow while thermalState stayed 0; only the
  kernel sleep/wake timestamps showed it.

Run directly to print a snapshot:  python platform_state.py
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

_SWIFT = 'import Foundation\nprint(ProcessInfo.processInfo.thermalState.rawValue)\n'
_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "roofline-platform-state"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ---------------- macOS ----------------

def _mac_thermal() -> int | None:
    try:  # PyObjC, if installed
        from Foundation import NSProcessInfo  # type: ignore
        return int(NSProcessInfo.processInfo().thermalState())
    except Exception:
        pass
    binary = _CACHE / "thermstate"
    if not binary.exists() and shutil.which("swiftc"):
        _CACHE.mkdir(parents=True, exist_ok=True)
        src = _CACHE / "thermstate.swift"
        src.write_text(_SWIFT)
        _run(["swiftc", "-O", str(src), "-o", str(binary)])
    out = _run([str(binary)]) if binary.exists() else ""
    return int(out) if out.isdigit() else None


def _mac() -> dict:
    batt = _run(["pmset", "-g", "batt"])
    times = _run(["sysctl", "-n", "kern.sleeptime", "kern.waketime"])
    lid = _run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"])
    return dict(platform="macos", thermal_state=_mac_thermal(), on_ac="AC Power" in batt,
                lid_closed=('"AppleClamshellState" = Yes' in lid), sleep_marker=times)


# ---------------- Linux ----------------

def _linux_on_ac() -> bool:
    supplies = list(Path("/sys/class/power_supply").glob("*/online"))
    if not supplies:
        return True  # desktop / server without a battery
    return any(p.read_text().strip() == "1" for p in supplies)


def _linux_sleep_marker() -> str:
    # CLOCK_BOOTTIME includes suspend, CLOCK_MONOTONIC does not; their difference grows across a suspend.
    diff = time.clock_gettime(time.CLOCK_BOOTTIME) - time.clock_gettime(time.CLOCK_MONOTONIC)
    return f"{diff:.3f}"


def _nvidia() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    q = "index,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,clocks_throttle_reasons.active"
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 6:
            gpus.append(dict(index=f[0], temp_c=f[1], sm_clock=f[2], sm_clock_max=f[3], power_w=f[4],
                             throttle_reasons=f[5]))
    return gpus


def _linux() -> dict:
    gpus = _nvidia()
    # Map to the same coarse scale: 0 if no active throttle reason other than idle (0x1), else 1.
    thermal = None
    if gpus:
        active = [int(g["throttle_reasons"], 16) if g["throttle_reasons"].startswith("0x") else 0 for g in gpus]
        thermal = 0 if all((a & ~0x1) == 0 for a in active) else 1
    return dict(platform="linux", thermal_state=thermal, on_ac=_linux_on_ac(), lid_closed=None,
                sleep_marker=_linux_sleep_marker(), gpus=gpus)


# ---------------- API ----------------

def snapshot() -> dict:
    s = _mac() if platform.system() == "Darwin" else _linux()
    s["time"] = time.time()
    return s


def slept_between(before: dict, after: dict, tol_s: float = 0.5) -> bool:
    """True if the system slept between two snapshots."""
    if before["platform"] == "macos":
        return before["sleep_marker"] != after["sleep_marker"]
    return float(after["sleep_marker"]) - float(before["sleep_marker"]) > tol_s


def ready(s: dict) -> bool:
    """Gate: nominal thermal state (when known), on AC power, lid not closed.
    (Clamshell use with an external display does not sleep; relax the lid check for that setup.)"""
    return (s["thermal_state"] in (0, None)) and s["on_ac"] and not s.get("lid_closed")


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=1))
