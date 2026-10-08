#!/bin/zsh
# Wait for the four pilot runs, blind their engines, evaluate them against mlx-lm with the v3 harness.
set -uo pipefail
ROOT=${0:A:h:h:h}
cd "$ROOT"
LOG=agent-test/nav/runs/pilot.log
until grep -q "all four runs done" "$LOG" 2>/dev/null; do sleep 120; done
caffeinate -dimsu -w $$ &
print -r -- "[$(date '+%m-%d %H:%M:%S')] evaluation: blinding" | tee -a "$LOG"
.venv/bin/python agent-test/nav/make_eval.py P1 B1 B2 P2 2>&1 | tee -a "$LOG" || exit 1
sleep 600   # cool down after the last agent run
NAV_CANDIDATES=$ROOT/agent-test/nav/runs/candidates.json BENCH_PYTHON=.venv-vllm-metal/bin/python \
BENCH_OUT=agent-test/v3/results-nav-pilot GAP_HOURS=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 \
  ./agent-test/v3/run_both.sh --profile nav 2>&1 | tee -a "$LOG"
print -r -- "[$(date '+%m-%d %H:%M:%S')] evaluation done -> agent-test/v3/results-nav-pilot/nav_report.md" | tee -a "$LOG"
