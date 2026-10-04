#!/data/data/com.termux/files/usr/bin/bash
# Start the offline Coffee Leaf Check on the phone (Termux).
#   one-time:  pkg upgrade && pkg install llama-cpp jq
#   run:       bash run_phone.sh   then open http://127.0.0.1:<port>/ in Chrome
# Model, mmproj, LoRA, context, threads and port come from config.json, so a model swap
# is a config edit and a restart. Check flags against `llama-server --help` on the phone.
set -euo pipefail
cd "$(dirname "$0")"

cfg() { jq -r "$1 // empty" config.json; }
MODEL=$(cfg .model_file); MMPROJ=$(cfg .mmproj_file); LORA=$(cfg .lora_file)
PORT=$(cfg .port); CTX=$(cfg .ctx); THREADS=$(cfg .threads)

for f in "$MODEL" "$MMPROJ" ${LORA:+"$LORA"}; do
  [ -f "$f" ] || { echo "missing: $f (edit config.json)"; exit 1; }
done

ARGS=(-m "$MODEL" --mmproj "$MMPROJ" --jinja -c "$CTX" --host 127.0.0.1 --port "$PORT" --path "$PWD")
[ -n "$LORA" ] && ARGS+=(--lora "$LORA")
[ -n "$THREADS" ] && [ "$THREADS" != 0 ] && ARGS+=(-t "$THREADS")
echo "llama-server ${ARGS[*]}"
exec llama-server "${ARGS[@]}"
