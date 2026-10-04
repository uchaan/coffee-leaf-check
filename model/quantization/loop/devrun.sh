#!/usr/bin/env bash
# usage: devrun.sh GPU PORT TAG LM [MODEL]  -> harness run on BRACOL dev with the frozen prompt; ref = the BF16 dev run of the same host
set -uo pipefail; HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); source "$REPO/env.sh"
GPU=$1; PORT=$2; TAG=$3; LM=$(readlink -f $4); MODEL=${5:-Qwen3.5-2B}; mkdir -p $WORK/logs
REF=$HN04B_RUNS/bf16-$( [ $MODEL = Qwen3.5-2B ] && echo 2B || echo 0.8B )-dev/images.csv   # made once in setup (see the skill)
cd "$REPO"
$PY eval/harness.py run --model $MODEL --lm $LM --mmproj $WORK/gguf/mmproj-$MODEL-Q8_0.gguf --prompt protocol/prompt_v1.txt --split dev --gpu $GPU --port $PORT --tag $TAG --ref $REF > $WORK/logs/dev-$TAG.log 2>&1
echo "DEV $(tail -n 1 $WORK/logs/dev-$TAG.log)" >> $WORK/logs/events.log
