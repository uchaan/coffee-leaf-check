# model/fine-tuning/: agent guide

LoRA on the quantized base, export to GGUF, pt-BR translator, CNN / YOLO baselines. Commands, data files and the
build-output-to-published-name map: [`README.md`](README.md).

## Run (from the repository root, after `source env.sh`)

```bash
pip install -r model/fine-tuning/requirements.txt
python model/fine-tuning/train_lora.py --gguf $WORK/gguf/ours/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf --out $WORK/ft/coffee ...
python model/fine-tuning/export_lora.py $WORK/ft/coffee/epoch3 $WORK/ft/coffee-e3
model/fine-tuning/scripts/eval_lora.sh GPU PORT TAG BASE LORA MMPROJ test     # runs eval/harness.py from the root
```

Inputs: `data/manifest.csv`, `data/jmuben_*.csv`, `data/bracol_paper_sized_split.csv`, `protocol/prompt_v1.txt`,
images from `data/prep_images.py`.

## Rules

- Train on BRACOL dev (337 train + 82 validation) and the 559 JMuBEN training originals only. Never on BRACOL test or
  the 240 JMuBEN held-out originals.
- The base must be the shipped quantized file (SHA-256 `cb88d42f…`); the adapter is trained against those weights.
- Use the frozen prompt `protocol/prompt_v1.txt`; do not pass another prompt for released runs.
- Never commit adapters, GGUFs, checkpoints, images or the translator folder. Numbers in docs come from `RESULTS.md`.
