# Coffee Leaf Check: agent guide

Offline coffee-leaf disease check: Qwen3.5-2B quantized to a 2.66 bpw GGUF by an agentic loop, plus a coffee LoRA and a
fine-tuned vision projector (package 1.01 GB), served by llama.cpp to a static web page on Android. Overview:
[`README.md`](README.md); every published number: [`RESULTS.md`](RESULTS.md).

## Layout

| Path | What |
|---|---|
| `data/` | manifests (SHA-256, split), label map, JMuBEN split, image preparation |
| `protocol/` | frozen prompt and chat templates |
| `eval/` | harness, queues, collection; scores in `eval/results/` |
| `model/quantization/` | agentic quantization loop (its own agent guide and Claude Code skill) |
| `model/fine-tuning/` | LoRA, export, translator, CNN / YOLO baselines |
| `app/` | phone / desktop client |

Each folder has a README (humans) and an `AGENTS.md` (you).

## How to run

- Run every command from the repository root: `cp env.example.sh env.sh` once, then `source env.sh`.
- Python 3.11: `pip install -r requirements.txt` (data, eval, quantization) and `model/fine-tuning/requirements.txt`.
- Model files, datasets and llama.cpp live outside git under `$WORK` and `$LLAMA_CPP_DIR`.

## Invariants

- `protocol/prompt_v1.txt` is frozen; `app/prompt.txt` must stay byte-identical to it.
- BRACOL test images are never used for training, calibration or selection: dev for every choice, test read once.
- Never commit model weights (`*.gguf`, `*.safetensors`), images or `env.sh`. The package files are not in the repo.
- Numbers in docs come from `RESULTS.md` and `eval/results/`; do not edit a number without a measurement behind it.
- `model/quantization/example-run/logs/` are historical records: do not rewrite them.
- Docs: English, short, tables over prose.
