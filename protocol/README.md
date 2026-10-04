# Protocol

| File | What |
|---|---|
| `prompt_v1.txt` | the frozen benchmark and calibration prompt (`### system` / `### user` / `### grammar`); SHA-256 `15f21d3e…` |
| `prompt_v0.txt` | the original name of the same text: byte-identical to `prompt_v1.txt` (same SHA-256) |
| `prompt_v1_rejected_2026-10-04.txt` | a rejected candidate (SHA-256 `db415bb4…`), kept as a record: BF16 Qwen3.5-2B dev F1 0.3230 vs 0.6527 (4090) |
| `chat_templates/Qwen3.5-*.jinja` | the official chat templates of `Qwen/Qwen3.5-2B` and `Qwen/Qwen3.5-0.8B` (identical, SHA-256 `273d8e0e…`) |

Why v1 holds the v0 text: the owner measured the candidate on BRACOL dev, rejected it and kept v0 under the frozen name
([`../model/quantization/example-run/logs/a100-quantization-LOG.md`](../model/quantization/example-run/logs/a100-quantization-LOG.md), "Frozen prompt = v0"). Runs made with the rejected text are void.
The app uses the same text ([`../app/prompt.txt`](../app/prompt.txt), byte-identical; keep it so).

Users: `eval/harness.py` (`--prompt protocol/prompt_v1.txt`; chat template defaults to
`protocol/chat_templates/<model>.jinja`), the ours-task calibration (`--prompt-file protocol/prompt_v1.txt`),
`model/fine-tuning/train_lora.py` and `explain_demo.py`, and the app.
