#!/usr/bin/env bash
# usage: wait_event.sh [PATTERN] [TIMEOUT_S]: block until events.log has new lines (matching PATTERN, default: DEV|FAIL),
# print every new line since the last call and exit. The agent runs this in the background and is woken when it exits.
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/env.sh"; E=$WORK/logs/events.log; C=$WORK/.events.cursor
PAT=${1:-'^(DEV|FAIL)'}; T=${2:-7200}; touch $E; [ -f $C ] || echo 0 > $C; start=$(date +%s)
while :; do
  n=$(cat $C); t=$(( $(wc -l < $E) )); new=$(sed -n "$((n + 1)),${t}p" $E)   # one snapshot: lines appended meanwhile stay unread
  if echo "$new" | grep -qE "$PAT"; then echo "$new"; echo $t > $C; exit 0; fi
  [ $(( $(date +%s) - start )) -ge $T ] && { echo "TIMEOUT after ${T}s; new lines so far:"; echo "$new"; exit 2; }
  sleep 20
done
