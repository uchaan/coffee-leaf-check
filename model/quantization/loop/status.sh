#!/usr/bin/env bash
# usage: status.sh [N]: queue, workers, GPUs and the last events, in one screen
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); source "$REPO/env.sh"
echo "queue: $(grep -c . $WORK/queue.txt 2>/dev/null || echo 0) jobs"; head -n 5 $WORK/queue.txt 2>/dev/null | cut -d'|' -f1 | sed 's/^/  next: /'
[ -f $WORK/workers.pid ] && while read g pid; do kill -0 $pid 2>/dev/null && s=alive || s=dead; echo "worker gpu=$g pid=$pid $s"; done < $WORK/workers.pid
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "--- last events"; tail -n ${1:-12} $WORK/logs/events.log 2>/dev/null
