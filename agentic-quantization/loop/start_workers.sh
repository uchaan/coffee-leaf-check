#!/usr/bin/env bash
# usage: start_workers.sh GPU [GPU ...]: start one worker per GPU in the background; PIDs in $WORK/workers.pid
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/env.sh"; mkdir -p $WORK/logs $WORK/gguf/ours; rm -f $WORK/STOP
# keep only live workers in workers.pid (entries of exited workers would show as dead and could be reused PIDs)
P=$WORK/workers.pid; if [ -f $P ]; then while read g pid; do kill -0 $pid 2>/dev/null && echo "$g $pid"; done < $P > $P.tmp; mv $P.tmp $P; fi
for g in "$@"; do
  nohup bash $HERE/worker.sh $g > $WORK/logs/worker-gpu$g.log 2>&1 &
  echo "$g $!" >> $WORK/workers.pid; echo "WORKER gpu=$g pid=$! started $(date -Is)" >> $WORK/logs/events.log
done
