"""Token usage and time split of each navigation-pilot agent run, from its Codex session rollout.

Per run (runs/<id>/rollout.jsonl):
  tokens      cumulative input / cached input / output / reasoning output / total at the last
              token_count event (the rollout is written incrementally, so hard-stopped runs count too)
  requests    number of model requests (token_count events)
  tool calls  number of tool calls and their summed wall time (call -> matching output, by call_id)
  model time  wall time not spent inside tool calls (model thinking + streaming + overhead)
  reasoning   number of reasoning-summary items
  machine     minutes on battery / lid closed / user active / thermal not nominal (run.json, from
              run_agent.py's machine log)
Caveats: requests = token_count events (approximate); tool time can be low when a long command is
polled in pieces; model time includes generation, streaming and network, not only thinking.
Writes runs/usage.json and prints a table.

    python agent-test/nav/nav_usage.py [P1 B1 ...]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def usage(run: str) -> dict | None:
    f = HERE / "runs" / run / "rollout.jsonl"
    if not f.exists():
        return None
    calls, tool_time, n_calls, n_req, n_reason = {}, 0.0, 0, 0, 0
    first = last = None
    total = {}
    for line in f.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("timestamp")
        if not t:
            continue
        t = ts(t)
        first = t if first is None else first
        last = t
        p = e.get("payload") or {}
        kind = p.get("type", "")
        if e.get("type") == "event_msg" and kind == "token_count" and p.get("info"):
            n_req += 1
            total = p["info"].get("total_token_usage", total)
        elif e.get("type") == "response_item":
            if kind == "reasoning":
                n_reason += 1
            elif kind.endswith("_call") and p.get("call_id"):
                calls[p["call_id"]] = t
                n_calls += 1
            elif kind.endswith("_call_output") and p.get("call_id") in calls:
                tool_time += t - calls.pop(p["call_id"])
    wall = (last - first) if first is not None else 0.0
    rj = HERE / "runs" / run / "run.json"
    machine = json.loads(rj.read_text()).get("machine") if rj.exists() else None
    return dict(run=run, machine=machine, wall_s=wall, requests=n_req, tool_calls=n_calls, tool_time_s=tool_time,
                model_time_s=wall - tool_time, reasoning_summaries=n_reason,
                input_tokens=total.get("input_tokens", 0), cached_input_tokens=total.get("cached_input_tokens", 0),
                output_tokens=total.get("output_tokens", 0),
                reasoning_output_tokens=total.get("reasoning_output_tokens", 0),
                total_tokens=total.get("total_tokens", 0))


def main():
    runs = sys.argv[1:] or ["P1", "B1", "B2", "P2"]
    rows = [u for r in runs if (u := usage(r))]
    (HERE / "runs" / "usage.json").write_text(json.dumps(rows, indent=1))
    print("| run | wall h | requests | tool calls | tool time | model time | input tok (cached) | output tok | reasoning tok |")
    print("|---|---|---|---|---|---|---|---|---|")
    for u in rows:
        print(f"| {u['run']} | {u['wall_s'] / 3600:.2f} | {u['requests']} | {u['tool_calls']} | "
              f"{u['tool_time_s'] / max(u['wall_s'], 1):.0%} | {u['model_time_s'] / max(u['wall_s'], 1):.0%} | "
              f"{u['input_tokens']:,} ({u['cached_input_tokens']:,}) | {u['output_tokens']:,} | {u['reasoning_output_tokens']:,} |")


if __name__ == "__main__":
    main()
