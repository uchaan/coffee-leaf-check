---
name: gguf-quant-loop
description: Run the autonomous GGUF quantization loop for a small vision-language model (Qwen3.5-2B / 0.8B here) with llama.cpp: set up, plan per-tensor types for byte targets, queue GPTQ-IQ builds, wait for the automatic check and dev run of each build, decide the next builds from the results, then select on dev, read test once and report against public GGUF files at matched bytes. Use when asked to quantize a model to GGUF, beat a public GGUF at a size, build files for a byte budget, continue or resume the loop, or read build / dev results.
---

# gguf-quant-loop

The owner states a goal once; you run the iterations yourself. One iteration = plan → pre-register → queue → wait →
read → decide. The workers do the slow part (build, byte check, text KLD, dev run) without you; your job is to
keep the queue full of the *right* next builds and to stop when the evidence says so.

Read `model/quantization/AGENTS.md` for the ground rules. Every command runs from the repository root
(`cd "$(git rev-parse --show-toplevel)"`, or `cd ../..` from `model/quantization/`) after `source env.sh`.
`loop/`, `quantizer/`, `selection/`, `LOG.md` and `README.md` below are in `model/quantization/`.
Commands: `references/commands.md`. What earlier runs learned and how to read the numbers: `references/lessons.md`.

## 0. Goal → targets (once, at the start)

Turn the request into a table in `LOG.md`: model, byte targets (e.g. "LM + Q8_0 mmproj < 1 GB" → LM < 638,481,344 B
for 2B), the public file(s) to compare against at each target, the deadline, the GPUs you may use.
If any of these is missing and you cannot infer it from the request, ask once, then proceed.

## 1. Setup (once; skip what exists)

1. `env.sh` from `env.example.sh` (repository root); llama.cpp built at the pinned commit; HF model downloaded.
2. BF16 GGUF + Q8_0 mmproj (`convert_hf_to_gguf.py`, `--mmproj`); record bytes and SHA-256 in `LOG.md`.
3. Public files of the model (Unsloth at least) in `$WORK/unsloth/`; they are both the templates and the opponents.
4. BF16 dev reference: `eval/harness.py run --model M --lm BF16.gguf --mmproj ... --prompt protocol/prompt_v1.txt
   --split dev --tag bf16-2B-dev` (the dev runs score agreement against it).
5. Text-KLD base: `llama-perplexity ... --kl-divergence-base $WORK/kld-base-M-wt2-512x40.kld` (see `loop/check.sh`).
6. Calibration: `$CALIB_TEXT` (FineWeb-Edu + C4), generic images (COCO) with a generic prompt for ours-general,
   BRACOL **dev** images (512 px, JPEG q90, by file name in `$WORK/bracol/dev512`; `data/prep_images.py bracol`)
   for ours-task. The example run's text and COCO assets are not published (`README.md`, "Reproduce").
7. `loop/start_workers.sh <gpus>`; then `loop/wait_event.sh` once to set the cursor.

## 2. The iteration

**Plan.** For each open target: a template = types from a neighbouring Unsloth file, a tied-embedding type, tensors
moved up/down to hit the budget (`plan_alloc.py` → `retype_template.py`). Try 2–3 embedding types per target in
the first round; later rounds change *one* thing relative to the best so far.

**Pre-register.** Append to `LOG.md` before queuing: what each build tests, and the decision rule
("if ours-general dev F1 at 637 MB < Unsloth's at 768 MB, try …; else …").

**Queue.** `loop/enqueue.sh TAG TEMPLATE 128 "EXTRA"`; one job per (template, variant). Variants:
- ours-general (`general2` / `general3` in the tag): `--tied-embd-gptq --calib-images <generic manifest> --image-dir <coco512> --prompt-file <generic prompt>`
- ours-task: `--tied-embd-gptq --calib-images data/manifest.csv --image-dir $WORK/bracol/dev512 --prompt-file protocol/prompt_v1.txt`
Relative paths in EXTRA resolve against the repository root (`loop/build.sh` runs there).
Queue enough that every GPU has a next job; order by what decides the most.

**Wait.** Run `loop/wait_event.sh` **in the background** (Claude Code: `run_in_background: true`) and end your turn's
active work; you are re-invoked when it exits with new `DEV` / `FAIL` lines. Do not poll in a foreground loop.
While waiting you may plan the next round, but do not queue builds that depend on results you have not read.

**Read.** For each `DEV` line: forced macro-F1, letter mass, agreement with BF16, letter KLD. For `CHECK`: text KLD.
For `FAIL`: open `$WORK/logs/build-TAG.log`, fix, re-queue (count it in the log). `loop/status.sh` for queue and GPUs.

**Decide** (write the decision in `LOG.md` with the numbers):
- a target is *done* when its best ours-general file beats the public comparator on dev by more than noise, or when
  two rounds in a row brought no gain → mark it and stop building for it;
- otherwise plan the next round for it (back to Plan): change the factor the evidence points at (see lessons);
- when every target is done or the deadline is close, go to Selection. Never let the queue run dry while targets
  are open.

## 3. Selection (before any test number)

Write the rule in `LOG.md` first (here: per target, ours-general = highest dev F1 among ours-general files, ties →
lower letter KLD; ours-task = the ours-task file on the same template). Fill `model/quantization/selection/final.csv` with bytes and
SHA-256 (`model/quantization/quantizer/fileinfo.py`). Only then run test.

## 4. Test and report

Stop the build workers (`loop/stop_workers.sh`). Put the picks, every other built file, BF16 and the Q8_0 mmproj under
one folder at their `selection/files.csv` paths (`<model>/<name>.gguf`), then `eval/make_jobs.py` → one
`eval/gpu_queue.py` per GPU (`--split all` against the BF16 rows) → `eval/collect.py` → `eval/results/summary.md`
(paired bootstrap intervals at matched bytes). Commands: `references/commands.md`. `collect.py` reads only runs tagged by
the queues, so do not run test with `harness.py` directly. Report: conclusion first, one table, the wins and the losses.

## Resuming

Read the tail of `LOG.md` and `loop/status.sh`. Files already built are in `$WORK/gguf/ours/`; dev runs in
`$HN04B_RUNS/TAG/`. Re-queue only what is missing; restart workers; continue from the last decision.
