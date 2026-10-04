#!/usr/bin/env bash
# usage: stop_workers.sh [--now]: by default: each worker exits after its current job (STOP file).
# --now: also kill the workers by their recorded PIDs (a running build is lost).
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/env.sh"; touch $WORK/STOP
if [ "${1:-}" = "--now" ] && [ -f $WORK/workers.pid ]; then
  while read g pid; do kill $pid 2>/dev/null && echo "killed worker gpu=$g pid=$pid"; done < $WORK/workers.pid; rm -f $WORK/workers.pid
fi
