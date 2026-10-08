"""v3 unattended runner (PLAN.md). One command per session; safe to interrupt and re-run.

  .venv/bin/python agent-test/v3/run.py --session 1     # day 1
  .venv/bin/python agent-test/v3/run.py --session 2     # another day
  .venv/bin/python agent-test/v3/run.py --smoke         # ~30 min pipeline check, separate output dir

Per session: preflight (AC power, freeze check, keep-awake) -> probe calibration ->
correctness gate -> main/idle -> main/warm -> carryover -> ablation ->
divergence judging -> corrected analysis (report-corrected.md, summary-corrected.json). Every timed request
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
    # ~2 h for both sessions back to back: only the main question (E2 vs E0 under process
    # isolation) plus the two-way carryover check; no E1 attribution, no ablations.
    "simple": dict(pairs={"ttft16": 12, "ttft64": 12, "ttft256": 8, "ttft1024": 6, "ctx512": 6, "ctx4096": 6,
                          "ctx32768": 6, "req256x256": 6},
                   regimes=["idle"], gate="group", ablation_blocks=0, carryover_blocks=6,
                   engines=["E0", "E2"], gate_engines=["E0", "E2"]),
}
# Prospective three-engine validation; keep historical profiles unchanged.
PROFILES["confirm"] = dict(
    pairs={s: 24 for s in ("ttft16", "ttft64", "ttft256", "ttft1024", "ctx512",
                          "ctx4096", "ctx32768", "req256x256", "req4096x128")},
    regimes=["idle"], gate="unit", ablation_blocks=0, carryover_blocks=0,
    engines=["E0", "E2", "V0"], gate_engines=["E0", "E2"],
    analysis="prospective_sign_median_v2", comparator="official vLLM + vllm-metal",
)
FROZEN = [V2 / f for f in ["engine.py", "kernels.py", "attention.py", "thermal.py", "check.py",
                           "test_engine.py", "skinny_table.json"]] + sorted(HERE.glob("*.py"))
# text-prompt corpus snapshot: if it changed between sessions, the same prompt id would mean different content
FROZEN += sorted((HERE / "corpus").iterdir())
FROZEN += [HERE / "CONFIRM.md", HERE / "environment-vllm-metal.json", HERE / "run_both.sh"]

# Navigation pilot evaluation (agent-test/nav/PILOT.md): E0 vs blinded agent-built engines named in the
# JSON file given by NAV_CANDIDATES. Their files are frozen like the rest of the code under test.
_NAV = json.loads(Path(os.environ["NAV_CANDIDATES"]).read_text()) if os.environ.get("NAV_CANDIDATES") else {}
if _NAV:
    PROFILES["nav"] = dict(
        pairs={s: 8 for s in ("ttft16", "ttft64", "ttft256", "ttft1024", "ctx512", "ctx4096",
                              "ctx32768", "req256x256")},
        regimes=["idle"], gate="group", ablation_blocks=0, carryover_blocks=0,
        engines=["E0"] + sorted(_NAV), gate_engines=["E0"] + sorted(_NAV), analysis="nav_pilot")
    for _d in _NAV.values():
        FROZEN += sorted(Path(_d).rglob("*.py"))


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

    def unit(phase, regime, block, sid, engine, role="", prev="", target=""):
        L, gen = SCEN[sid]
        kind = "random" if block % 2 == 0 else "text"
        uid = f"{phase}-{regime}-b{block:02d}-{sid}-{engine}" + (f"-{role}{prev}-for{target}" if role else "")
        return dict(id=uid, phase=phase, regime=regime, block=block, scenario=sid, engine=engine,
                    role=role, prev=prev, target=target, kind=kind, length=L, variant=block, gen=gen,
                    session=session, gate=True, warm_full=profile == "confirm")

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
                engines = list(P.get("engines", MAIN_ENGINES))
                rng.shuffle(engines)
                units += group([unit("main", regime, b, sid, e) for e in engines])
    for b in range(min(1, P["carryover_blocks"]) if smoke else P["carryover_blocks"]):
        if not in_session(b):
            continue
        rng = random.Random(f"{SEED}-carry-{b}")
        for sid in CARRYOVER_SCENARIOS:
            # both directions: is E2 affected by a preceding E0, and E0 by a preceding E2?
            seqs = [(prev, target) for target in ("E2", "E0") for prev in ("E0", "E2")]
            rng.shuffle(seqs)
            for prev, target in seqs:     # previous process, then the measured probe
                units += group([unit("carry", "idle", b, sid, prev, "prev", prev, target),
                                unit("carry", "idle", b, sid, target, "probe", prev, target)])
    for b in range(min(1, P["ablation_blocks"]) if smoke else P["ablation_blocks"]):
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
                    git_dirty=bool(git("status", "--porcelain", "--untracked-files=no", "--", "agent-test/v2", "agent-test/v3")),
                    files=file_hashes(), mlx=mx.__version__, mlx_lm=mlx_lm.__version__,
                    macos=platform.mac_ver()[0], device=mx.device_info(), on_ac_power=on_ac,
                    python=sys.version, seed=SEED)
    from huggingface_hub import snapshot_download
    from importlib.metadata import version
    snapshot = Path(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))
    manifest["environment"] = dict(model_snapshot=snapshot.name,
        weights={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(snapshot.glob("*.safetensors"))},
        plan=PROFILES[args.profile],
        packages={k: version(k) for k in ("mlx", "mlx-lm", "numpy", "transformers")})
    if "V0" in PROFILES[args.profile].get("engines", []):
        vp = os.environ.get("VLLM_PYTHON", str(ROOT / ".venv-vllm-metal/bin/python"))
        code = "import json; from importlib.metadata import version; print(json.dumps({k:version(k) for k in ('vllm','vllm-metal','mlx','mlx-lm','torch','transformers')}))"
        manifest["environment"]["vllm_packages"] = json.loads(subprocess.check_output([vp, "-c", code], text=True))
    import importlib.metadata as metadata
    names = ["mlx", "mlx-lm", "transformers", "numpy"]
    if args.profile == "confirm":
        names += ["vllm", "vllm-metal", "torch"]
    manifest["environment"]["package_origins"] = {name: metadata.distribution(name).read_text("direct_url.json") for name in names}
    manifest["environment"]["model_metadata"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.iterdir()) if p.suffix == ".json"}
    others = sorted(out.glob("manifest-s*.json"))
    for other in others:
        prev = json.loads(other.read_text())
        if prev["session"] != session and (prev["files"] != manifest["files"] or prev.get("environment") != manifest["environment"]) and not args.force_unfrozen:
            sys.exit(f"Code differs from {other.name}; v3 requires frozen code across sessions "
                     "(pass --force-unfrozen to override; the report will flag it).")
    mf = out / f"manifest-s{session}.json"
    if mf.exists():
        old = json.loads(mf.read_text())
        manifest["started"] = old["started"]
        manifest["resumed"] = old.get("resumed", []) + [datetime.now().isoformat()]
        if (old["files"] != manifest["files"] or old.get("environment") != manifest["environment"]) and not args.force_unfrozen:
            sys.exit("Code changed since this session started (see manifest).")
        manifest["probe_ref_ms"] = old.get("probe_ref_ms")
    return manifest, mf


def calibrate_probe():
    code = ("import sys; sys.path.insert(0, %r); import thermal, mlx.core as mx; "
            "[thermal.probe_ms() for _ in range(5)]; print(min(thermal.probe_ms() for _ in range(20)))" % str(V2))
    subprocess.run([PY, "-c", "import sys; sys.path.insert(0, %r); import thermal; "
                    "print(thermal.wait_cool(idle=2, timeout=900))" % str(V2)], check=True)
    return float(subprocess.run([PY, "-c", code], capture_output=True, text=True, check=True).stdout.strip())


def wait_for_ac(out):
    """Never time on battery: pause (logging every 10 min) until AC power returns."""
    t0 = None
    while "AC Power" not in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout:
        if t0 is None or time.time() - t0 > 600:
            log(out, "on battery: paused until AC power is connected")
            t0 = time.time()
        time.sleep(30)


def wait_for_pause_and_deadline(out):
    """Navigation pilot: honour ~/nav-runs/PAUSE (the user needs the machine) and NAV_STOP_AT
    (HHMM in the morning when the machine must be returned): stop launching units after it."""
    pause = Path.home() / "nav-runs" / "PAUSE"
    while pause.exists():
        log(out, "paused (~/nav-runs/PAUSE exists)")
        time.sleep(300)
    stop = os.environ.get("NAV_STOP_AT")
    if stop and int(datetime.now().strftime("%H%M")) >= int(stop) and datetime.now().hour < 12:
        log(out, f"NAV_STOP_AT {stop} reached: stopping; rerun the same command to resume")
        sys.exit(3)


def run_unit(u, out, probe_ref):
    dst = out / "runs" / f"s{u['session']}-{u['id']}.json"
    if dst.exists():
        return "skip"
    wait_for_pause_and_deadline(out)
    wait_for_ac(out)
    worker_py = PY
    cmd = [worker_py, str(HERE / "worker.py"), "--engine", u["engine"], "--kind", u["kind"], "--length", str(u["length"]),
           "--variant", str(u["variant"]), "--gen", str(u["gen"]), "--regime", u["regime"],
           "--probe-ref", str(probe_ref), "--out", str(dst)] + ([] if u["gate"] else ["--no-gate"]) + (["--warm-full"] if u.get("warm_full") else [])
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
                    help="quick (default, ~2-2.5 h per session), simple (~1 h per session, E0 vs E2 only), "
                         "full (PLAN.md), or confirm (three engines, 24 pairs; see CONFIRM.md)")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--force-unfrozen", action="store_true")
    ap.add_argument("--skip-gate", action="store_true", help="smoke testing only")
    ap.add_argument("--out", type=Path, help="Separate directory for a new experiment; resume only this directory")
    ap.add_argument("--dry-run", action="store_true", help="Print fixed schedule without GPU work")
    args = ap.parse_args()
    if args.skip_gate and not args.smoke:
        ap.error("--skip-gate is allowed only with --smoke")
    if args.profile == "confirm" and Path(sys.prefix).name != ".venv-vllm-metal":
        sys.exit("Use .venv-vllm-metal/bin/python for confirm: all three arms must share one environment.")
    session = 1 if args.smoke else args.session
    out = args.out or HERE / (f"results-smoke-{args.profile}" if args.smoke else f"results-{args.profile}")
    if args.dry_run:
        units = schedule(session, args.smoke, args.profile)
        print(json.dumps(dict(profile=args.profile, session=session, units=len(units),
                              engines=sorted({u["engine"] for u in units}), plan=PROFILES[args.profile]), indent=2))
        return
    if "V0" in PROFILES[args.profile].get("engines", []):
        vp = os.environ.get("VLLM_PYTHON", str(ROOT / ".venv-vllm-metal/bin/python"))
        subprocess.run([vp, "-c", "import vllm, vllm_metal, mlx_lm"], check=True)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    manifest, mf = preflight(out, session, args)
    manifest["profile"] = args.profile
    (out / "plan.json").write_text(json.dumps(dict(profile=args.profile, **PROFILES[args.profile]), indent=1))
    # Unattended runs must not depend on the network: use the local HF cache only.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
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
        if args.profile in ("quick", "simple") and earlier and not gate_file.exists():
            gate_file = earlier[0]      # code is frozen across sessions (checked above): one gate suffices
            log(out, f"reusing {gate_file.name} (quick profile, frozen code)")
        if not gate_file.exists() and not args.skip_gate:
            log(out, "correctness gate (unit tests + FP32 reference for all engines)")
            subprocess.run([PY, str(HERE / "gate.py"), "--out", str(gate_file)]
                           + (["--engines", *PROFILES[args.profile]["gate_engines"]]
                              if "gate_engines" in PROFILES[args.profile] else []), check=True)
        if gate_file.exists():
            gate = json.loads(gate_file.read_text())
            log(out, f"gate: {'PASS' if gate['all_pass'] else 'FAIL'} "
                     f"{ {k: v['pass'] for k, v in gate['engines'].items()} }")
            required = ("E0", "E2") if "E2" in PROFILES[args.profile].get("engines", ["E2"]) else ("E0",)
            if any(not gate["engines"].get(e, {}).get("pass") for e in required) or gate["unit_tests"]["returncode"]:
                sys.exit("E0 or E2 failed the numerical gate; timing results would be meaningless. Stopping.")

        if args.profile == "confirm" and not args.skip_gate:
            vg = out / f"vllm-gate-s{session}.json"
            if not vg.exists():
                subprocess.run([PY, str(HERE / "vllm_gate.py"), "--out", str(vg),
                                "--probe-ref", str(manifest["probe_ref_ms"]), "--reference-gate", str(gate_file)], check=True)
            if not json.loads(vg.read_text())["pass"]:
                sys.exit("vLLM token/FP32 screening failed; inspect vllm-gate before timing.")
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

        # Pre-registered (CONFIRM.md): every excluded unit (gate timeout / worker crash) is
        # re-run exactly once at the end of the session; the first attempt is kept in
        # runs/retried/ and is not analysed. A unit excluded twice stays excluded.
        retry_dir = out / "runs" / "retried"
        for u in units:
            dst = out / "runs" / f"s{session}-{u['id']}.json"
            if dst.exists() and json.loads(dst.read_text()).get("excluded") and not (retry_dir / dst.name).exists():
                retry_dir.mkdir(exist_ok=True)
                dst.rename(retry_dir / dst.name)
                log(out, f"retry once: {u['id']} -> {run_unit(u, out, manifest['probe_ref_ms'])}")

        log(out, "judging generated-sequence divergences against FP32")
        subprocess.run([PY, str(HERE / "divergence.py"), "--results", str(out)], check=True)
        log(out, "analysis")
        subprocess.run([PY, str(HERE / "analyze.py"), "--results", str(out)], check=True)
        if args.profile == "nav":
            subprocess.run([PY, str(HERE.parent / "nav" / "nav_analyze.py"), "--results", str(out)], check=True)
        manifest["finished"] = datetime.now().isoformat()
        mf.write_text(json.dumps(manifest, indent=1))
        log(out, f"session {session} done -> {out / 'report-corrected.md'}")
    finally:
        caffeinate.terminate()


if __name__ == "__main__":
    main()
