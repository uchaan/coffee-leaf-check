#!/usr/bin/env bash
# usage: worker.sh GPU: one per GPU (start_workers.sh). Pops jobs from $WORK/queue.txt and runs each one end to end:
#   build (gptq_iq.py) -> byte check against the template -> text KLD (check.sh) -> BRACOL dev run (devrun.sh)
# Every step appends one line to $WORK/logs/events.log; the agent reads those lines (wait_event.sh) and decides what to queue next.
# Queue line: TAG|TEMPLATE|NSEQ|EXTRA   (EXTRA = more gptq_iq.py args, e.g. the --calib-images set)
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); source "$REPO/env.sh"; GPU=$1; PORT=$((9700 + GPU)); cd $WORK; touch queue.txt
while :; do
  [ -f STOP ] && { echo "WORKER gpu=$GPU stopped (STOP file) $(date -Is)" >> logs/events.log; exit 0; }
  until [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) -lt ${GPU_FREE_MB:-30000} ]; do sleep 15; done
  job=$( ( flock 9; j=$(head -n 1 queue.txt); [ -n "$j" ] && sed -i 1d queue.txt; echo "$j" ) 9>.queue.lock )
  [ -z "$job" ] && { sleep 30; continue; }
  IFS='|' read tag tpl nseq extra <<< "$job"; out=$WORK/gguf/ours/$tag.gguf
  model=$(case $tag in (*0.8B*) echo Qwen3.5-0.8B;; (*) echo Qwen3.5-2B;; esac)
  bash $HERE/build.sh $GPU "$tag" "$tpl" "$CALIB_TEXT" "$nseq" "$extra"
  if [ "$(stat -c %s $out 2>/dev/null)" != "$(stat -Lc %s $tpl)" ]; then
    echo "FAIL $tag build output missing or bytes != template (see logs/build-$tag.log)" >> logs/events.log; continue; fi
  bash $HERE/check.sh $GPU $out
  bash $HERE/devrun.sh $GPU $PORT $tag $out $model
done
