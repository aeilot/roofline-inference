"""Run one navigation-experiment agent (Codex) in an isolated workspace.

Isolation (why each part exists):
- Workspace outside this repo (make_workspace.py): the agent sees only its arm's materials.
- A fresh CODEX_HOME per run holding only auth.json and a minimal config: no memories, skills,
  plugins, MCP servers or past sessions (~/.codex/sessions contains transcripts about this project).
  Its session rollout is kept: per-request token usage (incl. reasoning tokens) with timestamps.
- macOS sandbox profile (sandbox-exec): no reads of this repo (except the shared Python env),
  ~/.claude or ~/.codex; writes only to the workspace, the fresh CODEX_HOME, caches and temp;
  no network except the local allowlist proxy, which only tunnels to OpenAI hosts.
  Codex's own sandbox is disabled because it blocks the Metal GPU.
- Hard stop at the deadline; the engine at the agent's last git commit is what gets evaluated.

    python agent-test/nav/run_agent.py --arm P --run P1 --hours 2.5
    python agent-test/nav/run_agent.py --arm P --run T0 --test      # 5-minute isolation self-test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PY = ROOT / ".venv-vllm-metal/bin/python"
HOME = Path.home()
NAV = HOME / "nav-runs"
CODEX_HOMES = NAV / ".codex-homes"
PORT = 18080
MODEL, EFFORT = "gpt-6.1-sol", "medium"

PROMPT = ("Read TASK.md in this directory and carry out the task described there. Work autonomously "
          "until you are done or the deadline in TASK.md is reached; there is no one to answer questions.")

TEST_PROMPT = f"""This is an environment self-test. Run each command below in order and report each output verbatim.
1. {PY} tools/profile_ops.py --prefill 64 --decode 512
2. head -3 {ROOT}/README.md
3. ls ~/.codex ~/.claude
4. curl -sS -m 5 https://github.com | head -c 80
5. echo hello > hello.txt && git add hello.txt && git commit -m test && git log --oneline | head -2
6. ls ~/nav-runs; head -3 ~/nav-runs/T0/TASK.md; ls ~/nav-runs/.codex-homes
"""


def profile(ws: Path, home: Path) -> str:
    # NAV is denied too: other runs' workspaces and session rollouts must stay invisible.
    deny_read = [ROOT, HOME / ".claude", HOME / ".codex", NAV]
    writable = [ws, home, HOME / ".cache", HOME / "Library/Caches", Path("/private/tmp"),
                Path("/private/var/folders")]
    q = lambda p: f'(subpath "{p}")'  # noqa: E731
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        f'(allow network-outbound (remote tcp "localhost:{PORT}"))',
        "(allow network* (local unix))",
        f"(deny file-read* {' '.join(q(p) for p in deny_read)})",
        f"(allow file-read-metadata {q(ROOT)} {q(NAV)})",
        f"(allow file-read* {q(ws)} {q(home)})",
        f"(allow file-read* {q(ROOT / '.venv-vllm-metal')})",
        f"(deny file-write* {q(HOME)})",
        f"(allow file-write* {' '.join(q(p) for p in writable)})",
    ])


def make_codex_home(run: str) -> Path:
    home = CODEX_HOMES / run
    if home.exists():
        shutil.rmtree(home)
    home.mkdir(parents=True)
    shutil.copy(HOME / ".codex/auth.json", home / "auth.json")
    os.chmod(home / "auth.json", 0o600)
    (home / "config.toml").write_text(f'model = "{MODEL}"\nmodel_reasoning_effort = "{EFFORT}"\n'
                                      'model_reasoning_summary = "auto"\n')
    return home


def ensure_proxy(log: Path):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            return
    subprocess.Popen([sys.executable, str(HERE / "allowlist_proxy.py"), "--port", str(PORT), "--log", str(log)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["P", "B"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--hours", type=float, default=2.5)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    out = HERE / "runs" / a.run
    out.mkdir(parents=True, exist_ok=True)
    home = make_codex_home(a.run)
    ensure_proxy(HERE / "runs" / "proxy.log")
    hours = 0.1 if a.test else a.hours
    ws = Path(subprocess.check_output([sys.executable, str(HERE / "make_workspace.py"), "--arm", a.arm,
                                       "--run", a.run, "--hours", str(hours)], text=True).strip())
    sb = out / "sandbox.sb"
    sb.write_text(profile(ws, home))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "ANTHROPIC", "OPENAI"))}
    env.update(CODEX_HOME=str(home), HF_HUB_OFFLINE="1",
               HTTPS_PROXY=f"http://127.0.0.1:{PORT}", HTTP_PROXY=f"http://127.0.0.1:{PORT}",
               ALL_PROXY=f"http://127.0.0.1:{PORT}", NO_PROXY="localhost,127.0.0.1")
    cmd = ["sandbox-exec", "-f", str(sb), "codex", "exec", "--dangerously-bypass-approvals-and-sandbox",
           "--skip-git-repo-check", "-m", MODEL, "-c", f'model_reasoning_effort="{EFFORT}"',
           "-c", 'model_reasoning_summary="auto"',
           "-C", str(ws), "--json", "-o", str(ws / ".last_message.txt"),   # the sandbox cannot write to this repo
           TEST_PROMPT if a.test else PROMPT]
    t0 = time.time()
    deadline = t0 + hours * 3600 + 300          # 5-minute grace to finish a commit
    with open(out / "transcript.jsonl", "w") as tr, open(out / "stderr.log", "w") as er:
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
    # snapshot of the evaluated engine: last commit only
    snap = out / "engine"
    shutil.rmtree(snap, ignore_errors=True)
    snap.mkdir()
    tar = subprocess.run(["git", "archive", "HEAD", "engine"], cwd=ws, capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "--strip-components", "1", "-C", str(snap)], input=tar, check=True)
    # session rollout: timestamped model requests with token usage, tool calls and reasoning summaries
    for i, f in enumerate(sorted((home / "sessions").rglob("*.jsonl"))):
        shutil.copy(f, out / f"rollout{'' if i == 0 else i}.jsonl")
    for f in ("NOTES.md", ".last_message.txt"):
        if (ws / f).exists():
            shutil.copy(ws / f, out / f)
    log = subprocess.run(["git", "log", "--format=%h %ad %s", "--date=iso"], cwd=ws, capture_output=True, text=True).stdout
    (out / "git_log.txt").write_text(log)
    (out / "run.json").write_text(json.dumps(dict(arm=a.arm, run=a.run, workspace=str(ws), started=t0,
                                                   ended=time.time(), hours=hours, hard_stopped=stopped,
                                                   exit_code=p.returncode, model=MODEL, effort=EFFORT), indent=1))
    print(f"{a.run}: exit {p.returncode}, hard_stopped={stopped}, commits={len(log.splitlines())}")


if __name__ == "__main__":
    main()
