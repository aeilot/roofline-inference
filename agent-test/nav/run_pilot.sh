#!/bin/zsh
# Navigation pilot: four sequential Codex runs (P1, B1, B2, P2), 2 h each, then the blinded evaluation.
# Pause between runs: touch ~/nav-runs/PAUSE (remove to continue). No run starts after LAST_START.
# Resumable: a run with an existing runs/<id>/run.json is skipped.
set -uo pipefail
ROOT=${0:A:h:h:h}
cd "$ROOT"
caffeinate -dimsu -w $$ &
LOG=agent-test/nav/runs/pilot.log
HOURS=${HOURS:-2}
LAST_START=${LAST_START:-0230}     # HHMM next morning; later runs are skipped so the evaluation fits
mkdir -p agent-test/nav/runs
say_log() { print -r -- "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
for spec in P:P1 B:B1 B:B2 P:P2; do
  arm=${spec%%:*}; run=${spec##*:}
  if [[ -f agent-test/nav/runs/$run/run.json ]]; then say_log "$run already done, skipping"; continue; fi
  if [[ -d ~/nav-runs/$run ]]; then say_log "$run: stale workspace from an interrupted run; moving aside"; mv ~/nav-runs/$run ~/nav-runs/$run.interrupted.$(date +%s); fi
  while [[ -f ~/nav-runs/PAUSE ]]; do say_log "paused (~/nav-runs/PAUSE exists)"; sleep 300; done
  now=$(date +%H%M)
  if [[ $now -ge $LAST_START && $now -lt 1200 ]]; then say_log "past LAST_START $LAST_START: not starting $run"; break; fi
  say_log "start $run (arm $arm)"
  .venv/bin/python agent-test/nav/run_agent.py --arm "$arm" --run "$run" --hours "$HOURS" 2>&1 | tee -a "$LOG"
  say_log "end $run"
  sleep 300   # let the machine cool between runs
done
say_log "agent runs finished"
./agent-test/nav/run_eval.sh
