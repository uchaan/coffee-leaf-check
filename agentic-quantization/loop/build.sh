#!/usr/bin/env bash
# usage: build.sh GPU TAG TEMPLATE CALIB NSEQ [EXTRA gptq_iq args]  -> $WORK/gguf/ours/TAG.gguf (GPTQ-IQ, seqlen 2048, mb 8, percdamp 0.01)
# Runs from agentic-quantization/, so relative paths in EXTRA (manifest.csv, protocol/...) resolve there.
set -uo pipefail; HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/env.sh"
GPU=$1; TAG=$2; TPL=$(readlink -f "$3"); CAL=$(readlink -f "$4"); NSEQ=$5; TOOLS=$HERE/../quantizer
export PYTHONPATH=$LLAMA_CPP_DIR/gguf-py:$TOOLS LD_LIBRARY_PATH=$LLAMA_CPP_DIR/build/bin PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$(case $TAG in (*0.8B*) echo $WORK/models/Qwen3.5-0.8B;; (*) echo $WORK/models/Qwen3.5-2B;; esac); mkdir -p $WORK/gguf/ours $WORK/logs; t=$(date +%s)
cd "$HERE/.."
CUDA_VISIBLE_DEVICES=$GPU $PY $TOOLS/gptq_iq.py --hf $M --template $TPL --out $WORK/gguf/ours/$TAG.gguf --calib $CAL --seqlen 2048 --nseq $NSEQ --mb 8 --device cuda:0 --threads 16 ${6:-} > $WORK/logs/build-$TAG.log 2>&1
echo "BUILD $TAG exit=$? $(( $(date +%s)-t ))s bytes=$(stat -c %s $WORK/gguf/ours/$TAG.gguf 2>/dev/null) template=$(stat -Lc %s $TPL)" >> $WORK/logs/events.log
