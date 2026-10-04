#!/usr/bin/env bash
# usage: enqueue.sh TAG TEMPLATE NSEQ [EXTRA]: append one build job for the workers (TAG names the output file).
# TEMPLATE is stored as an absolute path; relative paths in EXTRA resolve against the repository root (build.sh runs there).
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); source "$REPO/env.sh"
[ -f "$2" ] || { echo "template not found: $2" >&2; exit 1; }; TPL=$(readlink -f "$2")
( flock 9; echo "$1|$TPL|$3|${4:-}" >> $WORK/queue.txt ) 9>$WORK/.queue.lock
echo "QUEUED $1 template=$(basename $2) $(date -Is)" >> $WORK/logs/events.log
