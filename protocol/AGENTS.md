# protocol/: agent guide

The prompt and chat templates that every stage shares: evaluation (`eval/harness.py`), ours-task calibration,
LoRA training (`model/fine-tuning/train_lora.py`) and the app. Details and hashes: [`README.md`](README.md).

## Rules

- `prompt_v1.txt` is **frozen** (SHA-256 `15f21d3e…`). Do not edit it, reformat it or change its line endings.
- `app/prompt.txt` must stay byte-identical to `prompt_v1.txt`. Check: `cmp protocol/prompt_v1.txt app/prompt.txt`.
- `prompt_v0.txt` and `prompt_v1_rejected_2026-10-04.txt` are records; keep them unchanged.
- `chat_templates/*.jinja` are the official Qwen templates; the harness picks `chat_templates/<model>.jinja` by default.
- A new prompt is the owner's decision, measured on BRACOL dev only, saved under a new file name; runs made with a
  different prompt are not comparable with `eval/results/`.
