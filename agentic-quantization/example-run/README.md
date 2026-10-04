# Example run: Hack-Nation 04B, 2026-10-04 (KST)

The run that produced the files in `../selection/` and the numbers in `../results/`. Two Claude Code sessions on two
servers, one owner giving direction:

| Session | Server | Role |
|---|---|---|
| build (coordinator) | 7× A100 80 GB | planned templates, ran the build / check / dev loop for Qwen3.5-2B and most 0.8B targets, wrote the selection rule |
| benchmark | RTX 4090 server | ran every file (ours and public) through `../benchmark/harness.py`, collected `../results/`; later ran the 0.8B low-end loop itself |

| File | What |
|---|---|
| [`logs/a100-quantization-LOG.md`](logs/a100-quantization-LOG.md) | the build session's log: setup, findings, each change of plan with its dev numbers, the selection rule |
| [`logs/4090-benchmark-LOG.md`](logs/4090-benchmark-LOG.md) | the benchmark session's log: reference runs, prompt decision, every measurement step |
| [`prompts.md`](prompts.md) | the kind of requests that started and steered the sessions |

What the logs show about the loop:
- The owner set goals and made the calls reserved for them (prompt choice, targets, the extra fine-tuning track);
  between those, the sessions ran iterations on their own: 112 successful builds on the A100 alone. The 61 quantized
  files handed to the benchmark side (32 ours-general, 29 ours-task) are listed in `../selection/files.csv`, with the BF16
  and mmproj references and the fine-tuned package.
- Each change of direction came from dev numbers read in the loop: text-only calibration → image + text calibration;
  IQ2_XXS bodies → UD-IQ2_M bodies with a smaller tied embedding (`-v2` templates), then a sweep of tied-embedding types.
- The selection rule was written before any test number (`../selection/final.csv`); a test line seen early is recorded as a deviation.

The automation in `../loop/` is a cleaned-up, path-independent version of the scripts used in this run
(per-GPU build workers, an automatic check / dev runner and an event wait).

`../quantizer/gptq_iq.py` differs from the copy used in the run (SHA-256 `b6601b44…` in `../selection/files.csv`) only in
comments, log messages, an unused import and one default value that `main()` overwrites from the model config.
The superseded text-only builds (rows citing `856f10f5`) used an earlier version that is not included.
