# 4090 server: benchmark log, Hack-Nation 04B

> Editor's note (added after the run): the fine-tuning code is now in `../../../fine-tuning/` and the final numbers are in
> `../../../../RESULTS.md`. Per-image rows were not published (`../../../../eval/results/README.md`).
> Paths below are the run's layout: the harness folder is now `eval/`, the scores `eval/results/`, the manifests `data/` (`public_gguf_log.csv` is in `eval/`), the prompt `protocol/`, `loop/env.example.sh` is the root `env.example.sh`, and the rest `model/quantization/`.

Times KST. Container `hn04b-llama` (nvidia/cuda 12.1 devel), physical GPUs 6 and 7 (`--gpu 0` / `--gpu 1`).

## Setup (2026-10-04)
- llama.cpp `9a7570587ce908b0073a0458877205b80627f393`, CUDA 12.1, sm_89 (same commit as the A100 server).
- Harness `benchmark/harness.py`. Probability mode `pre50` (see `benchmark/README.md`).
- BRACOL: 1,685 images (code 5 excluded), 419 dev / 1,266 test; `data/class_counts.txt`.
- Canonical mmproj (A100): 2B `3761d22e…`, 0.8B `aae6e712…`; BF16 LMs 2B `624e245b…`, 0.8B `3fdaa108…` (all verified).

## Step 2: BF16 reference, prompt v0 (finished about 04:05 KST)
Go/no-go line per model (BRACOL test n=1,266; τ from BRACOL dev n=419):
- **Qwen3.5-2B: GO.** forced_macro_f1 0.5815 [0.5547, 0.6063]; per-class F1 A 0.821, B 0.777, C 0.365, D 0.568, E 0.377;
  dev τ 0.6386 reaches 90% (dev precision 0.902, dev coverage 0.291); test at τ: answered_acc 0.893, coverage 0.287.
  coarse_macro_f1 0.8176.
- **Qwen3.5-0.8B: under 0.40, reported but deprioritised.** forced_macro_f1 0.3536 [0.3401, 0.3655]; per-class F1 A 0.948,
  B 0.595, C 0.217, D 0.008, E 0.000; dev τ: 90% not reached, best precision 0.398 at τ 0.3127 (dev coverage 0.816).
  coarse_macro_f1 0.7970.
- Only one model is under 0.40, so the five-letter label set stays.
- Neither model ever puts its argmax on F with prompt v0 (test share 0.0).
- A to F are single tokens for both models (ids 32 to 37).
- Per-image rows and metrics: kept on the benchmark server (not included).

## Step 3a (VOID): candidate prompt v1 (SHA-256 db415bb4…), BF16 reference with it (finished 04:32 KST)
These runs used the candidate text that the owner rejected; they are void and are not reported.
- **Qwen3.5-2B BF16, v1:** test forced_macro_f1 0.3023 [0.2864, 0.3174]; per-class F1 A 0.000, B 0.525, C 0.279, D 0.707,
  E 0.000; dev 0.3230; dev τ 0.7706 reaches 90% at dev coverage 0.029; test answered_acc 0.919 at coverage 0.029.
  coarse_macro_f1 0.4405.
- **Qwen3.5-0.8B BF16, v1:** test 0.1526 [0.1395, 0.1660]; per-class F1 A 0.000, B 0.005, C 0.187, D 0.571, E 0.000; dev 0.1576;
  τ does not reach 90% on dev.
- **v1 is worse than v0 on BRACOL dev for both models** (2B dev 0.6527 → 0.3230; 0.8B 0.3627 → 0.1576). With v1 the 2B BF16
  never answers A or E on dev: healthy 68 → B 29 / C 38 / D 1; leaf miner 96 → C 41 / D 54 / B 1. Letter mass 0.999, same code
  path as v0. Flagged to the A100 session and the owner at 04:34 KST; the choice is theirs and must rest on dev numbers only.
- Until they decide: GPU 6 runs every 2B file with the frozen v1 (protocol). GPU 7 runs the same 2B files with v0 as a hedge
  (0.8B paused, 2B first). CPU proxy runs with v1.

## CPU proxy settings (04:40 KST)
- First CPU-proxy runs used llama-server defaults; RSS grew per request (RAM prompt cache, context checkpoints).
  All CPU measurements were deleted and restarted with `--cache-ram 0 --ctx-checkpoints 0` (benchmark/README.md).
  GPU runs keep the server defaults (no prompt reuse: `cache_prompt` false), so no GPU number changes.

## Step 3b: owner decision, frozen prompt = v0 text (04:3x KST)
- `protocol/prompt_v1.txt` now holds the v0 text byte for byte (SHA-256 15f21d3e…); the candidate is kept as
  `protocol/prompt_v1_rejected_2026-10-04.txt`. Decided on BRACOL dev only.
- All runs with the rejected text moved to `runs/void_rejected_prompt_v1/` (4090 local), the v0 hedge runs to
  `runs/superseded_hedge_prompt_v0/`. Everything is rerun with `protocol/prompt_v1.txt` from 04:42 KST.
- Both GPUs share one job list (2B first, then 0.8B). The same file gave bit-identical probabilities on physical
  GPU 6 and GPU 7 (BF16 2B, 1,685 of 1,685 rows), and across two harness revisions.

## 05:00–05:30 KST
- 2B: BF16, six ours-general (text-only calibration), five Unsloth, bartowski IQ2_M, mradermacher Q2_K and ten i1 files done.
- Six GPU workers since 05:21 (three per GPU, GPU utilisation ~25% with one). Determinism under that load: rerun of
  Unsloth UD-IQ2_M 2B identical in 1,685 of 1,685 rows.
- Several public 2B files put most first-token mass outside A to F (mradermacher i1-IQ2_M: "To" 0.62, "Here" 0.17 on
  dev images without grammar), so their forced answers come from a small residual. Real file behaviour, not the harness.
- JMuBEN (optional): archives verified against Mendeley hashes. Byte-identical copies are frequent (Healthy: 18,984
  files, 63 distinct images); the manifest keeps distinct images only. Runs only when no BRACOL job is waiting.

## 06:11–06:12 KST: numpy changed for about 30 s
- Installing pandas/datasets for the general-benchmark replication replaced numpy 2.2.6 with 1.26.4 in the container
  (21:11:4x UTC); restored to 2.2.6 at 21:12:12 UTC (`pip check` clean). Only one run started inside that window
  (0.8B mradermacher i1-IQ4_XS, 21:12:08); it was stopped and rerun under numpy 2.2.6. Runs already running keep the
  numpy they imported at start. All final numbers are rescored by `benchmark/collect.py` under one environment.

## 10:00–10:15 KST: host memory, server cache flags
- With about 14 llama-server processes at once (BRACOL, JMuBEN, general benchmarks), each server's default host-RAM
  prompt cache (8 GiB) filled host memory (120 of 125 GB used, swap full; the container held 107 GB). One JMuBEN run
  died (server disconnected) and requests slowed. The JMuBEN workers were stopped at once, which freed 54 GB.
- Tested before changing anything: `--cache-ram 0 --ctx-checkpoints 0` on GPU runs changes the probabilities (0.8B Unsloth
  Q4_K_M: 0 of 1,685 rows identical, max |Δp| 0.138, argmax equal in 1,449), so context checkpoints alter how the
  hybrid model's prompt is processed. `--cache-ram 0` alone gives bit-identical rows (1,685 of 1,685).
- Four BRACOL runs had started in the four minutes when the first variant was live (2B general2-UD-IQ3_XXS,
  task-967M-v2b, general3-967M-v2b; 0.8B general-1GBpkg-Q8_0); they were deleted and rerun with the defaults.
- From 10:12 KST every GPU worker runs with `--cache-ram 0` only (env `HN04B_SERVER_EXTRA`), four BRACOL and two JMuBEN
  workers. The CPU proxy keeps `--cache-ram 0 --ctx-checkpoints 0` (its probabilities are not used).

## 10:30–11:00 KST: 0.8B low-end search moved to the 4090 (A100 coordination, owner decision)
- Targets 338 / 310 / 290 MB, registered rule (ours-general chosen on BRACOL dev F1, ours-task follows the template).
  Separation note: this host also judges, and had seen test numbers of earlier 0.8B low-end ours files (all collapsed to one
  answer, as on dev). Candidates are screened on dev only (`runs/screen/`, never test or JMuBEN) until the freeze.
- Build container `hn04b-build` (pytorch 2.10.0+cu126 image with its forward-compat libcuda removed; transformers 5.5.4)
  on GPUs 3, 4, 5, 6; A100 `gptq_iq.py` b6601b44ea30 unchanged; calibration tarball 381413a0… verified.
- Grid change, with the reason: the first screens of the A100 grid D points with a large tied embedding (338 MB e Q5_K,
  IQ4_XS; Q4_K) left the body at UD-IQ2_XXS types and collapsed (dev F1 0.056–0.088, letter mass 0.03–0.79). With a small
  embedding and UP = UD-IQ3_XXS the planner stops 17–48 MB under the cap, so six "u" templates use UP = Unsloth
  UD-Q3_K_XL instead (338 e IQ3_XXS / Q2_K, 310 e IQ3_XXS / Q2_K, 290 e Q2_K / IQ2_XXS), each about 0.3 MB under its cap.

## 11:10–11:55 KST: 0.8B low end frozen, final collect
- Registered rule on 39 dev-screened candidates (`selection/lowend_0.8B_dev_screen.md`): 338 MB general3 e Q4_K (dev 0.1424),
  310 MB general3 e IQ3_XXS "u" (dev 0.2116), 290 MB general2 e Q3_K (dev 0.2831); ours-task built on the same templates.
  Unsloth 0.8B UD-IQ2_XXS (the only same-size opponent, 338 MB) on this harness: dev 0.0559, every answer A, letter_kld 15.94.
- The six files went to the A100 over shared storage (SHA-256 checked on both ends).
- A100 note for the report: 2B 768 MB eQ5_K (exactly Unsloth UD-IQ2_XXS's types) dev 0.331 (general3) vs Unsloth 0.309, so
  at identical types the quantiser alone adds little; the 768 MB gain comes from the allocation.
- `results/` written from every run (163 runs, 489 rows): BRACOL dev and test, JMuBEN test, text KLD, CPU proxy, all with
  llama.cpp 9a7570587 and the frozen prompt (SHA-256 15f21d3e…).

## 11:55–12:25 KST: fine-tuned track (owner's change of scope; the brief said "No fine-tuning")
- 2B quantised base + LoRA adapter served with `llama-server --lora`; evaluated with the unchanged harness
  (`HN04B_SERVER_EXTRA`), results kept apart from the quantisation tables (now in `../../../../RESULTS.md`).
- The A100 trains the adapter on BRACOL dev only (337 train / 82 early-stopping images); BRACOL test is untouched.
- 12:2x: owner target 95% BRACOL test accuracy; on request the 4090 put every distinct JMuBEN image (3,756, preprocessed,
  manifest with an in_eval column) on shared storage as training data. JMuBEN is therefore not a clean test set for fine-tuned
  files; the quantisation tables keep their JMuBEN numbers.
