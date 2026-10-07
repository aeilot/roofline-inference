#!/bin/zsh
# Run v3 session 1, wait until GAP_HOURS after it finished, then run session 2.
# Safe to re-run after an interruption: run.py resumes finished units, and the wait is
# measured from session 1's recorded finish time, not from when this script restarted.
#
#   agent-test/v3/run_both.sh                 # quick profile, 2 h gap
#   GAP_HOURS=3 agent-test/v3/run_both.sh --profile full
set -euo pipefail

ROOT=${0:A:h:h:h}
cd "$ROOT"
PY=.venv/bin/python
export HF_HUB_OFFLINE=1     # no network dependency during the unattended run (model is cached)
GAP_HOURS=${GAP_HOURS:-2}
PROFILE=quick
[[ "${1:-}" == "--profile" ]] && PROFILE=$2
OUT=agent-test/v3/results-$PROFILE
mkdir -p "$OUT"
LOG=$OUT/run_both.log

# keep the Mac awake for the whole sequence, including the wait
caffeinate -dimsu -w $$ &

say_log() { print -r -- "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say_log "session 1 (profile $PROFILE)"
$PY agent-test/v3/run.py --session 1 --profile "$PROFILE"

finished=$($PY -c "import json,sys; print(json.load(open(sys.argv[1])).get('finished',''))" "$OUT/manifest-s1.json")
if [[ -z "$finished" ]]; then
  say_log "session 1 did not record a finish time; stopping"
  exit 1
fi
wait_s=$($PY -c "
import datetime, sys
f = datetime.datetime.fromisoformat(sys.argv[1])
left = (f + datetime.timedelta(hours=float(sys.argv[2])) - datetime.datetime.now()).total_seconds()
print(max(0, int(left)))" "$finished" "$GAP_HOURS")
say_log "session 1 finished at $finished; waiting ${wait_s}s (${GAP_HOURS} h gap)"
sleep "$wait_s"

say_log "session 2 (profile $PROFILE)"
$PY agent-test/v3/run.py --session 2 --profile "$PROFILE"
say_log "all done -> $OUT/report.md"
