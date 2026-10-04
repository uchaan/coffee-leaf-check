# model/: agent guide

The two stages that make the model package. Overview: [`README.md`](README.md).

| Folder | Stage | Output |
|---|---|---|
| `quantization/` | 1. agentic GGUF quantization loop | base LM `Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf` (row `ours-task` in `quantization/selection/final.csv`) |
| `fine-tuning/` | 2. LoRA on that exact base | LoRA adapter GGUF and fine-tuned Q8_0 vision projector |

- Each folder has its own `AGENTS.md`; read it before working there. The quantization agent is launched in
  `model/quantization/` (its skill is in `.claude/` there).
- All scripts run from the repository root with `env.sh` sourced.
- Weights are built under `$WORK` and never committed. Numbers come from `RESULTS.md` and `eval/results/`.
- BRACOL test images are never used for training or selection.
