# Model

The model package is built in two stages, in this order. Both run from the repository root with `env.sh` sourced, and
both write their files under `$WORK` (weights are not in the repository).

| Stage | Folder | What it does | Produces |
|---|---|---|---|
| 1. Quantization | [`quantization/`](quantization/) | an agent (Claude Code) plans, builds, checks and selects GGUF files for byte targets | base LM `Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf`, 2.66 bpw, published as `Qwen3.5-2B-coffee-base-Q2.gguf`; every built file in [`quantization/selection/`](quantization/selection/) |
| 2. Fine-tuning | [`fine-tuning/`](fine-tuning/) | LoRA trained against the exact stage-1 weights, vision LoRA merged into the projector | LoRA adapter `Qwen3.5-2B-ours-coffee-lora-Q8_0.gguf`, vision projector `mmproj-Qwen3.5-2B-ours-coffee-Q8_0.gguf`, optional pt-BR translator |

Stage 1 and the evaluation share [`../data/`](../data/), [`../protocol/`](../protocol/) and [`../eval/`](../eval/).
Sizes and scores of the final package: [`../RESULTS.md`](../RESULTS.md).
