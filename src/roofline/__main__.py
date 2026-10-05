"""CLI: python -m roofline --model configs/models/qwen3_0.6b.yaml decode --batch 1 --context 1024"""

from __future__ import annotations

import argparse
import json

from roofline import decode, load_device, load_model, operator_bound, physical_bound, prefill


def main() -> None:
    p = argparse.ArgumentParser(prog="roofline")
    p.add_argument("--device", default="configs/hardware/m4_mba_24g.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--engine", default="gpu")
    p.add_argument("--bound", choices=["physical", "operator"], default="physical",
                   help="operator: finite on-chip memory, operator boundary at every GEMM")
    p.add_argument("--on-chip-mib", type=float, default=None,
                   help="override memory.on_chip for --bound operator")
    sub = p.add_subparsers(dest="phase", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--batch", type=int, default=1)
    d.add_argument("--context", type=int, default=0)
    f = sub.add_parser("prefill")
    f.add_argument("--batch", type=int, default=1)
    f.add_argument("--prompt-len", type=int, required=True)
    args = p.parse_args()

    model = load_model(args.model)
    device = load_device(args.device)
    if args.phase == "decode":
        w = decode(model, args.batch, args.context)
    else:
        w = prefill(model, args.batch, args.prompt_len)
    if args.bound == "physical":
        b = physical_bound(w, device, engine=args.engine, dtype=model.weight_dtype)
    else:
        on_chip = None if args.on_chip_mib is None else args.on_chip_mib * 2**20
        b = operator_bound(w, device, engine=args.engine, dtype=model.weight_dtype, on_chip_bytes=on_chip)
    print(json.dumps(b.to_dict(), indent=2))


if __name__ == "__main__":
    main()
