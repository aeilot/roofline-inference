"""v3 unattended runner (PLAN.md). One command per session; safe to interrupt and re-run.

  .venv/bin/python agent-test/v3/run.py --session 1     # day 1
  .venv/bin/python agent-test/v3/run.py --session 2     # another day
  .venv/bin/python agent-test/v3/run.py --smoke         # ~30 min pipeline check, separate output dir

Per session: preflight (AC power, freeze check, keep-awake) -> probe calibration ->
correctness gate -> main/idle -> main/warm -> carryover -> ablation ->
divergence judging -> analysis (report.md, summary.json). Every timed request
runs in its own worker process (worker.py). Completed units are skipped on re-run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "v2"
ROOT = HERE.parents[1]
PY = sys.executable
SEED = 20261006

# Scenarios: id -> (prompt length, tokens generated)
SCEN = {"ttft16": (16, 8), "ttft64": (64, 8), "ttft256": (256, 8), "ttft1024": (1024, 8),
        "ctx512": (512, 129), "ctx4096": (4096, 129), "ctx16384": (16384, 129), "ctx32768": (32768, 129),
        "req256x256": (256, 256), "req1024x512": (1024, 512), "req4096x128": (4096, 128)}
MAIN_ENGINES = ["E0", "E1", "E2"]
ABLATION_ENGINES = ["E2", "E2-noskinny", "E2-nogemv", "E2-nogqa", "E2-noprefuse"]
ABLATION_SCENARIOS = ["ttft64", "ttft1024", "ctx512", "ctx4096"]
CARRYOVER_SCENARIOS = ["ttft64", "ctx4096"]
# pairs = blocks in total over both sessions. gate: "unit" = thermal gate before every
# timed request; "group" = once before each (block, scenario) group, whose engines then
# run back to back in random order (2 s idle each). See PLAN.md section 11.
PROFILES = {
    "full": dict(pairs={"ttft16": 40, "ttft64": 40, "ttft256": 30, "ttft1024": 12, "ctx512": 12, "ctx4096": 12,
                        "ctx16384": 6, "ctx32768": 6, "req256x256": 12, "req1024x512": 12, "req4096x128": 12},
                 regimes=["idle", "warm"], gate="unit", ablation_blocks=8, carryover_blocks=12),
    "quick": dict(pairs={"ttft16": 24, "ttft64": 24, "ttft256": 16, "ttft1024": 8, "ctx512": 8, "ctx4096": 8,
                         "ctx32768": 6, "req256x256": 8, "req4096x128": 8},
                  regimes=["idle"], gate="group", ablation_blocks=4, carryover_blocks=8),
}
FROZEN = [V2 / f for f in ["engine.py", "kernels.py", "attention.py", "thermal.py", "check.py",
                           "test_engine.py", "skinny_table.json"]] + sorted(HERE.glob("*.py"))
# text-prompt corpus: if it changed between sessions, the same prompt id would mean different content
FROZEN += [ROOT / f for f in ["docs/report.md", "README.md", "docs/design-notes.md", "agent-test/v2/README.md"]]


def session_of(block):
    return 1 if block % 4 in (0, 3) else 2       # each session gets both prompt kinds


def log(out, msg):
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(out / "log.txt", "a") as f:
        f.write(line + "\n")


def schedule(session, smoke, profile):
    """Deterministic ordered list of units for one session. `gate` marks units preceded by a thermal gate."""
    P = PROFILES[profile]
    units = []

    def unit(phase, regime, block, sid, engine, role="", prev=""):
        L, gen = SCEN[sid]
        kind = "random" if block % 2 == 0 else "text"
        uid = f"{phase}-{regime}-b{block:02d}-{sid}-{engine}" + (f"-{role}{prev}" if role else "")
        return dict(id=uid, phase=phase, regime=regime, block=block, scenario=sid, engine=engine,
                    role=role, prev=prev, kind=kind, length=L, variant=block, gen=gen, session=session, gate=True)

    def group(us):
        if P["gate"] == "group":
            for u in us[1:]:
                u["gate"] = False
        return us

    def in_session(b):
        return smoke or session_of(b) == session

    for regime in P["regimes"]:
        for b in range(1 if smoke else max(P["pairs"].values())):
            if not in_session(b):
                continue
            rng = random.Random(f"{SEED}-main-{regime}-{b}")
            sids = [s for s, n in P["pairs"].items() if b < (1 if smoke else n)]
            rng.shuffle(sids)
            for sid in sids:
                engines = MAIN_ENGINES[:]
                rng.shuffle(engines)
                units += group([unit("main", regime, b, sid, e) for e in engines])
    for b in range(1 if smoke else P["carryover_blocks"]):
        if not in_session(b):
            continue
        rng = random.Random(f"{SEED}-carry-{b}")
        for sid in CARRYOVER_SCENARIOS:
            seqs = ["E0", "E2"]
            rng.shuffle(seqs)
            for prev in seqs:     # previous process, then the measured E2 probe
                units += group([unit("carry", "idle", b, sid, prev, "prev", prev),
                                unit("carry", "idle", b, sid, "E2", "probe", prev)])
    for b in range(1 if smoke else P["ablation_blocks"]):
        if not in_session(b):
            continue
        rng = random.Random(f"{SEED}-abl-{b}")
        sids = ABLATION_SCENARIOS[:]
        rng.shuffle(sids)
        for sid in sids:
            engines = ABLATION_ENGINES[:]
            rng.shuffle(engines)
            units += group([unit("abl", "idle", b, sid, e) for e in engines])
    return units


def file_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in FROZEN}


def preflight(out, session, args):
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    on_ac = "AC Power" in batt
    if not on_ac and not args.allow_battery:
        sys.exit("Not on AC power: performance policy differs on battery. Plug in, or pass --allow-battery.")
    thermstate = V2 / "thermstate"
    if not thermstate.exists():
        subprocess.run(["swiftc", "-O", str(V2 / "therm.swift"), "-o", str(thermstate)], check=True)
    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    import mlx.core as mx
    import mlx_lm
    manifest = dict(session=session, started=datetime.now().isoformat(), git_head=git("rev-parse", "HEAD"),
                    git_dirty=bool(git("status", "--porcelain", "--", "agent-test/v2", "agent-test/v3")),
                    files=file_hashes(), mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
                    macos=platform.mac_ver()[0], device=mx.device_info(), on_ac_power=on_ac,
                    python=sys.version, seed=SEED)
    others = sorted(out.glob("manifest-s*.json"))
    for other in others:
        prev = json.loads(other.read_text())
        if prev["session"] != session and prev["files"] != manifest["files"] and not args.force_unfrozen:
            sys.exit(f"Code differs from {other.name}; v3 requires frozen code across sessions "
                     "(pass --force-unfrozen to override; the report will flag it).")
    mf = out / f"manifest-s{session}.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        manifest["started"] = old["started"]
        manifest["resumed"] = old.get("resumed", []) + [datetime.now().isoformat()]
        if old["files"] != manifest["files"] and not args.force_unfrozen:
            sys.exit("Code changed since this session started (see manifest).")
        manifest["probe_ref_ms"] = old.get("probe_ref_ms")
    return manifest, mf


def calibrate_probe():
    code = ("import sys; sys.path.insert(0, %r); import thermal, mlx.core as mx; "
            "[thermal.probe_ms() for _ in range(5)]; print(min(thermal.probe_ms() for _ in range(20)))" % str(V2))
    subprocess.run([PY, "-c", "import sys; sys.path.insert(0, %r); import thermal; "
                    "print(thermal.wait_cool(idle=2, timeout=900))" % str(V2)], check=True)
    return float(subprocess.run([PY, "-c", code], capture_output=True, text=True, check=True).stdout.strip())


def run_unit(u, out, probe_ref):
    dst = out / "runs" / f"s{u['session']}-{u['id']}.json"
    if dst.exists():
        return "skip"
    cmd = [PY, str(HERE / "worker.py"), "--engine", u["engine"], "--kind", u["kind"], "--length", str(u["length"]),
           "--variant", str(u["variant"]), "--gen", str(u["gen"]), "--regime", u["regime"],
           "--probe-ref", str(probe_ref), "--out", str(dst)] + ([] if u["gate"] else ["--no-gate"])
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        err = r.stderr[-2000:] if r.returncode else None
    except subprocess.TimeoutExpired:
        err = "worker timeout (1800 s)"
    if err or not dst.exists():
        dst.write_text(json.dumps(dict(excluded="worker_crash", error=err or "no output", **u)))
        return "crash"
    rec = json.loads(dst.read_text())
    rec["unit"] = u
    rec["wall_s"] = time.time() - t0
    dst.write_text(json.dumps(rec))
    return rec.get("excluded", "ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", type=int, choices=[1, 2])
    g.add_argument("--smoke", action="store_true")
    ap.add_argument("--profile", choices=list(PROFILES), default="quick",
                    help="quick (default, ~2-3 h per session) or full (PLAN.md sections 5-6, ~10 h per session)")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--force-unfrozen", action="store_true")
    ap.add_argument("--skip-gate", action="store_true", help="smoke testing only")
    args = ap.parse_args()
    session = 1 if args.smoke else args.session
    out = HERE / (f"results-smoke-{args.profile}" if args.smoke else f"results-{args.profile}")
    (out / "runs").mkdir(parents=True, exist_ok=True)

    manifest, mf = preflight(out, session, args)
    manifest["profile"] = args.profile
    (out / "plan.json").write_text(json.dumps(dict(profile=args.profile, **PROFILES[args.profile]), indent=1))
    caffeinate = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
    try:
        if not manifest.get("probe_ref_ms"):
            log(out, "calibrating GPU probe (waits for nominal thermal state)")
            manifest["probe_ref_ms"] = calibrate_probe()
        mf.write_text(json.dumps(manifest, indent=1))
        log(out, f"session {session}: probe reference {manifest['probe_ref_ms']:.2f} ms, git {manifest['git_head'][:8]}"
                 f"{' (dirty)' if manifest['git_dirty'] else ''}")

        gate_file = out / f"gate-s{session}.json"
        earlier = sorted(out.glob("gate-s*.json"))
        if args.profile == "quick" and earlier and not gate_file.exists():
            gate_file = earlier[0]      # code is frozen across sessions (checked above): one gate suffices
            log(out, f"reusing {gate_file.name} (quick profile, frozen code)")
        if not gate_file.exists() and not args.skip_gate:
            log(out, "correctness gate (unit tests + FP32 reference for all engines)")
            subprocess.run([PY, str(HERE / "gate.py"), "--out", str(gate_file)], check=True)
        if gate_file.exists():
            gate = json.loads(gate_file.read_text())
            log(out, f"gate: {'PASS' if gate['all_pass'] else 'FAIL'} "
                     f"{ {k: v['pass'] for k, v in gate['engines'].items()} }")
            if not gate["engines"].get("E2", {}).get("pass") or gate["unit_tests"]["returncode"]:
                sys.exit("E2 failed the correctness gate; timing results would be meaningless. Stopping.")

        units = schedule(session, args.smoke, args.profile)
        done = sum((out / "runs" / f"s{session}-{u['id']}.json").exists() for u in units)
        log(out, f"{len(units)} units in this session, {done} already done")
        t_start, n_new = time.time(), 0
        for i, u in enumerate(units, 1):
            status = run_unit(u, out, manifest["probe_ref_ms"])
            if status == "skip":
                continue
            n_new += 1
            rate = (time.time() - t_start) / n_new
            remaining = sum(not (out / "runs" / f"s{session}-{x['id']}.json").exists() for x in units[i:])
            log(out, f"[{i}/{len(units)}] {u['id']}: {status}  (ETA {remaining * rate / 3600:.1f} h)")

        log(out, "judging generated-sequence divergences against FP32")
        subprocess.run([PY, str(HERE / "divergence.py"), "--results", str(out)], check=True)
        log(out, "analysis")
        subprocess.run([PY, str(HERE / "analyze.py"), "--results", str(out)], check=True)
        manifest["finished"] = datetime.now().isoformat()
        mf.write_text(json.dumps(manifest, indent=1))
        log(out, f"session {session} done -> {out / 'report.md'}")
    finally:
        caffeinate.terminate()


if __name__ == "__main__":
    main()
