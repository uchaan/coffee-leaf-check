#!/usr/bin/env bash
# usage: eval_lora.sh GPU PORT TAG BASE.gguf LORA.gguf MMPROJ.gguf [SPLIT dev|test|all] [MANIFEST DATA_ROOT]
# Harness run of a base GGUF + LoRA adapter (llama-server --lora), same prompt and scoring as the quantization benchmark.
# Default: BRACOL (data/manifest.csv at the repository root, DATA_ROOT=$BRACOL_ROOT).
# JMuBEN held-out: MANIFEST=data/jmuben_grouped_split.csv DATA_ROOT=$WORK/jmuben (test = 240 originals).
# Paths may be relative to the caller's directory. Needs env.sh at the repository root (copy of env.example.sh);
# runs are written under $HN04B_RUNS/<TAG>.
set -uo pipefail; HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); source "$REPO/env.sh"
GPU=$1; PORT=$2; TAG=$3; BASE=$(readlink -f "$4"); LORA=$(readlink -f "$5"); MMPROJ=$(readlink -f "$6"); SPLIT=${7:-all}
MAN=$(readlink -f "${8:-$REPO/data/manifest.csv}"); ROOT=$(readlink -f "${9:-$BRACOL_ROOT}")
export HN04B_SERVER_EXTRA="--lora $LORA"; cd "$REPO"
$PY eval/harness.py run --model Qwen3.5-2B --lm "$BASE" --mmproj "$MMPROJ" --prompt protocol/prompt_v1.txt \
    --manifest "$MAN" --data-root "$ROOT" --split "$SPLIT" --gpu "$GPU" --port "$PORT" --tag "$TAG"
