"""Recompute reference times under the corrected scope, without new timings."""
from dataclasses import replace
import json
from pathlib import Path

from roofline import decode, load_device, load_model, physical_bound, prefill

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = load_model(ROOT / "configs/models/qwen3_0.6b.yaml")
    device = load_device(ROOT / "configs/hardware/m4_mba_24g.yaml")
    cases = []
    for phase, lengths in (("decode", (512, 4096, 32768)),
                           ("prefill", (16, 64, 256, 1024, 4096, 16384, 32768))):
        for length in lengths:
            w = decode(model, 1, length) if phase == "decode" else prefill(model, 1, length)
            old_w = w if phase == "decode" else prefill(model, 1, length, prune_final=False)
            current = physical_bound(w, device)
            old = physical_bound(old_w, device, initial_resident_bytes=0)
            # Hypothetical ceilings for sensitivity; these are not measurements.
            sensitivity = {}
            for factor in (1.0, 1.1):
                gpu = device.engine("gpu")
                gpu = replace(gpu, compute={**gpu.compute, "bf16": replace(gpu.compute["bf16"], measured=gpu.compute["bf16"].effective * factor)},
                              bandwidth=replace(gpu.bandwidth, measured=120e9))
                alt = replace(device, engines={"gpu": gpu},
                              total_bandwidth=replace(device.total_bandwidth, measured=120e9))
                sensitivity[f"bandwidth_120GBs_compute_x{factor}"] = physical_bound(w, alt).time_s
            cases.append({"phase": phase, "length": length, "historical_full_graph_cold_s": old.time_s,
                          "corrected_reference_s": current.time_s, "reference": current.to_dict(),
                          "hypothetical_rate_sensitivity_s": sensitivity})
    out = {"analysis": "review_correction_20261007", "new_measurements": False,
           "interpretation": "calibrated references, not certified hardware limits or attained speeds",
           "cases": cases}
    path = ROOT / "results/figures/review_corrections.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for r in cases:
        print(f"{r['phase']:7} {r['length']:5}: historical {r['historical_full_graph_cold_s']:.6f} s; corrected {r['corrected_reference_s']:.6f} s")


if __name__ == "__main__":
    main()
