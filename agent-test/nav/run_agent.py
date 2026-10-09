"""Run one navigation-experiment agent (Codex) in an isolated workspace.

Isolation (why each part exists):
- Workspace outside this repo (make_workspace.py): the agent sees only its arm's materials.
- A fresh per-run box ~/nav-runs/.sandbox/<run>/ with its own HOME, CODEX_HOME (only auth.json and a
  minimal config: no memories, skills, MCP servers or past sessions) and TMPDIR. Codex's session
  rollout is kept: per-request token usage (incl. reasoning tokens) with timestamps.
- macOS sandbox profile (sandbox-exec), allowlist for the home directory: nothing under the real
  HOME is readable except the run's workspace, its box, the shared Python env and the Hugging Face
  model cache (path metadata stays visible so realpath works, directory listings and contents do
  not). Previously a denylist left e.g. ~/.claude.json and the Claude / Codex app data readable.
  /private/tmp and the user's system temp dir are neither readable nor writable, so runs cannot leave
  files for each other there. Writes only to the workspace, the box and system caches.
  No network except the local allowlist proxy, which only tunnels to OpenAI hosts.
  Codex's own sandbox is disabled because it blocks the Metal GPU.
- Machine-state log (outside the sandbox, every 60 s): AC power, lid, thermal state, load and user
  activity (HID idle time), so runs disturbed by someone using or unplugging the machine are visible.
- Hard stop at the deadline; the engine at the agent's last git commit is what gets evaluated.

    python agent-test/nav/run_agent.py --arm P --run P1 --hours 2
    python agent-test/nav/run_agent.py --arm B --run T2 --test           # isolation self-test (Codex)
    python agent-test/nav/run_agent.py --arm P --run T2c --cmd "python tools/check.py"   # command in the same sandbox
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / ".venv-vllm-metal/bin/python"
HOME = Path.home()
NAV = HOME / "nav-runs"
BOXES = NAV / ".sandbox"
HF_HOME = HOME / ".cache/huggingface"
THERMSTATE = HOME / ".cache/roofline-platform-state/thermstate"
PORT = 18080
MODEL, EFFORT = "gpt-6.1-sol", "medium"
sys.path.insert(0, str(HERE / "kit/common/tools"))
from platform_state import snapshot  # noqa: E402

PROMPT = ("Read TASK.md in this directory and carry out the task described there. Work autonomously "
          "until you are done or the deadline in TASK.md is reached; there is no one to answer questions.")

TEST_PROMPT = f"""This is an environment self-test. Run each command below in order and report each output verbatim
(errors included; do not retry or work around failures).
1. {PY} tools/profile_ops.py --prefill 64 --decode 512
2. {PY} tools/platform_state.py
3. head -3 {ROOT}/README.md
4. ls {HOME}/.codex {HOME}/.claude; head -c 80 {HOME}/.claude.json
5. ls "{HOME}/Library/Application Support/Claude" {HOME}/Developer {HOME}/Documents
6. ls {NAV}; head -3 {NAV}/T0/TASK.md
7. echo x > /tmp/navtest.txt; ls /tmp | head -3
8. curl -sS -m 5 https://github.com | head -c 80
9. echo hello > hello.txt && git add hello.txt && git commit -m test && git log --oneline | head -2
10. {PY} -c "import tempfile, os; print(tempfile.gettempdir(), os.path.expanduser('~'))"
"""


def user_temp_dir() -> Path:
    t = subprocess.run(["getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True).stdout.strip()
    return Path(t).resolve()


def profile(ws: Path, box: Path) -> str:
    q = lambda p: f'(subpath "{p}")'  # noqa: E731
    readable = [ws, box, ROOT / ".venv-vllm-metal", HF_HOME]
    writable = [ws, box, Path("/private/var/folders")]
    hidden = [HOME, Path("/private/tmp"), user_temp_dir()]
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        f'(allow network-outbound (remote tcp "localhost:{PORT}"))',
        "(allow network* (local unix))",
        f"(deny file-read* {' '.join(q(p) for p in hidden)})",
        f"(allow file-read-metadata {q(HOME)})",          # stat / realpath only: no listing, no contents
        f"(allow file-read* {' '.join(q(p) for p in readable)})",
        f"(deny file-write* {' '.join(q(p) for p in hidden)})",
        f"(allow file-write* {' '.join(q(p) for p in writable)})",
        # the user's temp dir lies under /private/var/folders: keep it closed after the broad allow
        f"(deny file-read* file-write* {q(user_temp_dir())})",
    ])


def make_box(run: str) -> Path:
    """Fresh HOME / CODEX_HOME / TMPDIR for one run. An existing box (interrupted run) is kept aside."""
    box = BOXES / run
    if box.exists():
        box.rename(box.with_name(f"{run}.interrupted.{int(time.time())}"))
    codex = box / "codex"
    for d in (box / "home/.cache/roofline-platform-state", box / "home/.config", box / "tmp", codex):
        d.mkdir(parents=True)
    shutil.copy(HOME / ".codex/auth.json", codex / "auth.json")
    os.chmod(codex / "auth.json", 0o600)
    (codex / "config.toml").write_text(f'model = "{MODEL}"\nmodel_reasoning_effort = "{EFFORT}"\n'
                                       'model_reasoning_summary = "auto"\n')
    # /usr/bin/git is an xcrun shim that caches in the (closed) user temp dir and prints errors on
    # every call (self-test T2); put the real binary first on PATH instead.
    (box / "bin").mkdir()
    real_git = subprocess.run(["xcrun", "-f", "git"], capture_output=True, text=True).stdout.strip()
    if real_git:
        (box / "bin/git").symlink_to(real_git)
    # Codex runs commands with `zsh -lc`: /etc/zprofile's path_helper moves /usr/bin back in front,
    # so the real-git PATH entry must be restored in the login profile (pilot: the fix above had no
    # effect), and zsh's here-documents use TMPPREFIX (default /tmp/zsh), not TMPDIR.
    (box / "home/.zprofile").write_text(f'export PATH="{box / "bin"}:$PATH"\nexport TMPPREFIX="{box / "tmp"}/zsh"\n')
    if THERMSTATE.exists():   # thermal-state helper used by tools/platform_state.py (no compiler needed)
        shutil.copy(THERMSTATE, box / "home/.cache/roofline-platform-state/thermstate")
    return box


def ensure_proxy(log: Path):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            return
    subprocess.Popen([sys.executable, str(HERE / "allowlist_proxy.py"), "--port", str(PORT), "--log", str(log)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(1)


def hid_idle_s() -> float | None:
    out = subprocess.run(["ioreg", "-c", "IOHIDSystem", "-d", "4"], capture_output=True, text=True).stdout
    m = re.search(r'"HIDIdleTime" = (\d+)', out)
    return int(m.group(1)) / 1e9 if m else None


def machine_log(path: Path, stop: threading.Event, every=60):
    with open(path, "a") as f:
        while not stop.is_set():
            s = snapshot()
            f.write(json.dumps(dict(t=time.time(), on_ac=s["on_ac"], lid_closed=s.get("lid_closed"),
                                    thermal_state=s["thermal_state"], load1=os.getloadavg()[0],
                                    hid_idle_s=hid_idle_s())) + "\n")
            f.flush()
            stop.wait(every)


def machine_summary(path: Path, every=60) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    m = lambda c: round(sum(c(r) for r in rows) * every / 60)  # noqa: E731
    return dict(samples=len(rows), minutes_not_on_ac=m(lambda r: not r["on_ac"]),
                minutes_lid_closed=m(lambda r: bool(r["lid_closed"])),
                minutes_thermal_not_nominal=m(lambda r: (r["thermal_state"] or 0) > 0),
                minutes_user_active=m(lambda r: r["hid_idle_s"] is not None and r["hid_idle_s"] < 120))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["P", "B"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hours", type=float, default=2)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--cmd", help="run this shell command in the run's sandbox instead of Codex (self-test)")
    a = ap.parse_args()
    out = HERE / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    box = make_box(a.run)
    ensure_proxy(HERE / "runs" / "proxy.log")
    hours = 0.1 if a.test else a.hours
    ws = Path(subprocess.check_output([sys.executable, str(HERE / "make_workspace.py"), "--arm", a.arm,
                                       "--run", a.run, "--hours", str(hours)], text=True).strip())
    sb = out / "sandbox.sb"
    sb.write_text(profile(ws, box))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC", "OPENAI"))}
    env.update(HOME=str(box / "home"), CODEX_HOME=str(box / "codex"), TMPDIR=str(box / "tmp") + "/",
               TMPPREFIX=str(box / "tmp") + "/zsh",
               XDG_CACHE_HOME=str(box / "home/.cache"), XDG_CONFIG_HOME=str(box / "home/.config"),
               HF_HOME=str(HF_HOME), HF_HUB_OFFLINE="1", GIT_CONFIG_NOSYSTEM="1",
               HTTPS_PROXY=f"http://127.0.0.1:{PORT}", HTTP_PROXY=f"http://127.0.0.1:{PORT}",
               ALL_PROXY=f"http://127.0.0.1:{PORT}", NO_PROXY="localhost,127.0.0.1")
    env["PATH"] = f"{box / 'bin'}:{PY.parent}:{env.get('PATH', '')}"   # real git; `python` = the shared env
    if a.cmd:
        inner = ["/bin/zsh", "-c", a.cmd]
    else:
        inner = ["codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
                 "--skip-git-repo-check", "-m", MODEL, "-c", f'model_reasoning_effort="{EFFORT}"',
                 "-c", 'model_reasoning_summary="auto"',
                 "-C", str(ws), "--json", "-o", str(ws / ".last_message.txt"),   # the sandbox cannot write to this repo
                 TEST_PROMPT if a.test else PROMPT]
    cmd = ["sandbox-exec", "-f", str(sb), *inner]
    stop = threading.Event()
    mon = threading.Thread(target=machine_log, args=(out / "machine.jsonl", stop), daemon=True)
    mon.start()
    t0 = time.time()
    deadline = t0 + hours * 3600 + 300          # 5-minute grace to finish a commit
    with open(out / ("cmd.log" if a.cmd else "transcript.jsonl"), "w") as tr, open(out / "stderr.log", "w") as er:
        p = subprocess.Popen(cmd, cwd=ws, env=env, stdin=subprocess.DEVNULL, stdout=tr, stderr=er,
                             start_new_session=True)
        while p.poll() is None and time.time() < deadline:
            time.sleep(10)
        stopped = p.poll() is None
        if stopped:
            os.killpg(p.pid, signal.SIGTERM)
            time.sleep(10)
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    stop.set()
    mon.join(timeout=5)
    # snapshot of the evaluated engine: last commit only
    snap = out / "engine"
    shutil.rmtree(snap, ignore_errors=True)
    snap.mkdir()
    tar = subprocess.run(["git", "archive", "HEAD", "engine"], cwd=ws, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "--strip-components", "1", "-C", str(snap)], input=tar, check=True)
    # session rollout: timestamped model requests with token usage, tool calls and reasoning summaries
    for i, f in enumerate(sorted((box / "codex/sessions").rglob("*.jsonl"))):
        shutil.copy(f, out / f"rollout{'' if i == 0 else i}.jsonl")
    for f in ("NOTES.md", ".last_message.txt"):
        if (ws / f).exists():
            shutil.copy(ws / f, out / f)
    log = subprocess.run(["git", "log", "--format=%h %ad %s", "--date=iso"], cwd=ws, capture_output=True, text=True).stdout
    (out / "git_log.txt").write_text(log)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True, text=True).stdout
    (out / "uncommitted.txt").write_text(status)    # work left uncommitted at the stop is not evaluated
    machine = machine_summary(out / "machine.jsonl")
    (out / "run.json").write_text(json.dumps(dict(arm=a.arm, run=a.run, workspace=str(ws), started=t0,
                                                   ended=time.time(), hours=hours, hard_stopped=stopped,
                                                   exit_code=p.returncode, model=MODEL, effort=EFFORT,
                                                   test=bool(a.test or a.cmd), machine=machine), indent=1))
    print(f"{a.run}: exit {p.returncode}, hard_stopped={stopped}, commits={len(log.splitlines())}, machine={machine}")


if __name__ == "__main__":
    main()
