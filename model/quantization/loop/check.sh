#!/usr/bin/env bash
# usage: check.sh GPU FILE  -> text KLD vs BF16 on wikitext-2 test, first 40 chunks, ctx 512; appends to $WORK/textkld.csv
# The BF16 base is made once per model: llama-perplexity -m BF16.gguf -f $WIKITEXT -c 512 --chunks 40 --kl-divergence-base $WORK/kld-base-MODEL-wt2-512x40.kld
set -uo pipefail; HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); source "$REPO/env.sh"
GPU=$1; F=$2; BIN=$LLAMA_CPP_DIR/build/bin; export LD_LIBRARY_PATH=$BIN; mkdir -p $WORK/logs
T=$(basename $F .gguf); M=$(case $T in (*0.8B*) echo Qwen3.5-0.8B;; (*) echo Qwen3.5-2B;; esac)
CUDA_VISIBLE_DEVICES=$GPU $BIN/llama-perplexity -m $F -f $WIKITEXT -c 512 --chunks 40 -ngl 99 --kl-divergence-base $WORK/kld-base-$M-wt2-512x40.kld --kl-divergence > $WORK/logs/kld-$T.log 2>&1
kld=$(grep -m1 -E '^Mean +KLD' $WORK/logs/kld-$T.log | awk '{print $3}'); top=$(grep -m1 -E '^Same top p' $WORK/logs/kld-$T.log | awk '{print $4}')
( flock 9; echo "$T,$(stat -c %s $F),$kld,$top" >> $WORK/textkld.csv ) 9>$WORK/.textkld.lock
echo "CHECK $T kld=$kld same_top1=$top" >> $WORK/logs/events.log
