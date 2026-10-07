"""CLI: python -m roofline --model configs/models/qwen3_0.6b.yaml decode --batch 1 --context 1024"""

from __future__ import annotations

import argparse
import json

from roofline import decode, load_device, load_model, operator_bound, physical_bound, prefill
from roofline.bound import request_time
from roofline.graph import decode_graph, prefill_graph
from roofline.iobound import io_bound
from roofline.kvpolicy import parse_policy


def main() -> None:
    p = argparse.ArgumentParser(prog="roofline")
    p.add_argument("--device", default="configs/hardware/m4_mba_24g.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--engine", default="gpu")
    p.add_argument("--bound", choices=["physical", "operator", "hierarchy"], default="physical",
                   help="operator: finite on-chip memory, operator boundary at every GEMM; "
                        "hierarchy: multi-level, multi-processor bounds on the graph (fused and operator)")
    p.add_argument("--on-chip-mib", type=float, default=None,
                   help="override memory.on_chip for --bound operator")
    sub = p.add_subparsers(dest="phase", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--batch", type=int, default=1)
    d.add_argument("--context", type=int, default=0)
    d.add_argument("--kv", default=None,
                   help="KV policy (design notes 6.13): full | window:W[:sink] | budget:K[:V] | "
                        "retrieval:K[:page]; default is level A (full, model dtype)")
    d.add_argument("--kv-bits", type=float, default=None, help="KV precision in bits per element")
    f = sub.add_parser("prefill")
    f.add_argument("--batch", type=int, default=1)
    f.add_argument("--prompt-len", type=int, required=True)
    r = sub.add_parser("request", help="TTFT(L) + decode of N tokens, physical bound")
    r.add_argument("--prompt-len", type=int, required=True)
    r.add_argument("--gen", type=int, required=True)
    args = p.parse_args()

    model = load_model(args.model)
    device = load_device(args.device)
    if args.phase == "request":
        print(json.dumps(request_time(model, device, args.prompt_len, args.gen, args.engine), indent=2))
        return
    if args.bound == "hierarchy":
        if args.phase == "decode":
            g = decode_graph(model, args.batch, args.context)
        else:
            g = prefill_graph(model, args.batch, args.prompt_len)
        print(json.dumps(io_bound(g, device, engine=args.engine, dtype=model.weight_dtype).to_dict(), indent=2))
        return
    if args.phase == "decode":
        kv = parse_policy(args.kv, args.kv_bits) if args.kv or args.kv_bits else None
        w = decode(model, args.batch, args.context, kv)
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
