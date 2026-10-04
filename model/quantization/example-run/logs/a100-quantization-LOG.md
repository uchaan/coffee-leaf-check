# A100 server: quantization log, Hack-Nation 04B

> Editor's note (added after the run): the owner's quotes are translated from Korean. Model files and calibration assets
> were kept in a private Hugging Face repo during the run and are not published. The fine-tuning code is now in
> `../../../fine-tuning/` and the final numbers are in `../../../../RESULTS.md`.
> Paths below are the run's layout: the harness folder is now `eval/`, the scores `eval/results/`, the manifests `data/` (`public_gguf_log.csv` is in `eval/`), the prompt `protocol/`, `loop/env.example.sh` is the root `env.example.sh`, and the rest `model/quantization/`.

Model files went to a private model repo (not public). Every file: a line in `../../selection/files.csv`.

## Setup (2026-10-04)
- llama.cpp: `9a7570587ce908b0073a0458877205b80627f393` (2026-09-07, "convert : write explicit recurrent_layers for Qwen3-Next / Qwen3.5"),
  CUDA build, arch 80 (A100). The 4090 server should build the same commit.
- Pipeline: GPTQ-IQ (`quantizer/gptq_iq.py`, with layer / lm_head offload). It loads `Qwen3_5ForConditionalGeneration`, the
  class of both Qwen3.5-2B and Qwen3.5-0.8B. The linear-attention head layout (key heads, value heads, head dim) is read from the model
  config (both small models: 16 / 16 / 128).
- Base models: `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-0.8B` (architectures `Qwen3_5ForConditionalGeneration`, tied embeddings, 24 layers =
  18 linear-attention + 6 full-attention, vocab 248,320, 1 MTP layer).

## Base files (uploaded 2026-10-04 05:1x KST to the private repo)
| file (repo path) | bytes | sha256 |
|---|---|---|
| Qwen3.5-2B/Qwen3.5-2B-BF16.gguf | 3,897,387,936 | 624e245bb747fe072b1917116c583d84d9961ff9d159bd8ba56e1cadeb19156a |
| **Qwen3.5-2B/mmproj-Qwen3.5-2B-Q8_0.gguf (canonical)** | 361,518,656 | **3761d22e4ea8d5fc047289a6d9672cc22537dd7dd1a36ed6791a218a891e26ab** |
| Qwen3.5-2B/mmproj-Qwen3.5-2B-F16.gguf | 668,227,136 | 044a0ea136cca70711ae16e23b24d754b44eab6f2462d187aee4d7c7a9503d36 |
| Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf | 1,557,662,624 | 3fdaa1085c1bda18d78335e247151875790c4397cee1a1594b9f7c0339f3e85e |
| **Qwen3.5-0.8B/mmproj-Qwen3.5-0.8B-Q8_0.gguf (canonical)** | 113,564,384 | **aae6e712f47e44d4f18d6c2f4eb0007ea61f19dd72fd4ef27dff2aaa1b1a81bb** |
| Qwen3.5-0.8B/mmproj-Qwen3.5-0.8B-F16.gguf | 204,987,104 | 1dc1351c82e41b48edb55fd6ddfa7ca60fb5a16b3d5abf3ce7054880dd022847 |
Notes for the 4090 server: our BF16 conversion includes the MTP layer (blk.24, 335 tensors; Unsloth's GGUFs have 320, no MTP); MTP is
not used by normal decoding. Sanity check on the A100: 2B BF16 + the canonical Q8_0 mmproj answer "green and orange" for a synthetic
green image with an orange disc (llama-mtmd-cli). `llama-server` at this commit has `--chat-template-kwargs` and `--reasoning-budget`;
`llama-mtmd-cli` has neither. 1 GB package budgets: 2B LM < 638,481,344 B; 0.8B LM < 886,435,616 B (with the canonical Q8_0 mmproj).

## Frozen prompt = v0 (owner decision, 2026-10-04 04:3x KST)
The owner's candidate v1 (agronomist wording; kept as `protocol/prompt_v1_rejected_2026-10-04.txt`) was frozen and measured
on BRACOL **dev** only: BF16 Qwen3.5-2B forced_macro_f1 v0 0.6462 (A100) / 0.6527 (4090) vs v1 0.3238 / 0.3230; the model never answers A
or E with v1; 0.8B 0.36 → 0.16 (4090). Owner's choice: "revert to v0" → `protocol/prompt_v1.txt` now holds the v0
text byte for byte (frozen benchmark prompt). Runs made with the rejected text are void for the benchmark. ours-task calibrates with this
file, so the ours-task builds made with the rejected text are discarded and rebuilt.

## Findings and selection rule (2026-10-04 05:5x KST, before any test number is seen by the A100)
- Text-only calibration (FineWeb-Edu + C4) gives much lower text KLD than Unsloth at the same bytes (2B: 0.419 vs 0.711 at 768 MB, 0.197
  vs 0.281 at 860 MB, 0.115 vs 0.159 at 932 MB, 0.164 vs 0.243 at 967 MB; 0.8B: 0.598 vs 1.177 at 338 MB) but **fails on the task**
  (BRACOL dev F1 0.06–0.30 vs Unsloth 0.31–0.42; 11–20 % of the first-token mass outside A–F): with no image tokens in H, GPTQ's error
  compensation lands in directions the image tokens use.
- Fix: image + text calibration (`gptq_iq.py --calib-images`; images through the HF vision tower, merged into the LM inputs; the Conv3d
  patch embed is computed as the equivalent matmul because HF's bf16 Conv3d falls back to slow_conv_dilated3d, 3.5 s per image).
  ours-task (419 BRACOL dev images + the frozen prompt) at 860 MB: dev F1 0.587 (in-sample), letter_kld 0.070 (Unsloth 0.418 / 0.345).
- **ours-general is now text + 400 generic COCO val2017 images** (first 400 by file name, generic captioning prompt; no BRACOL data):
  files `*-ours-general2-*` (variant ours-general in files.csv). The text-only `*-ours-general-*` files stay in files.csv as superseded.
- IQ2_XXS-body files collapse on the task even with task calibration (637/700/768 MB: F1 0.08–0.33) → new allocations `-v2`: Unsloth
  UD-IQ2_M body with a smaller tied embedding (637 MB: IQ2_XXS embedding, 7 body tensors moved down; 700 MB: Q2_K embedding + 40 tensors
  raised toward UD-IQ3_XXS; 768 MB: Q3_K embedding + 48 tensors raised), built in both variants.
- **Selection rule:** for each (model, target) the allocation is chosen on the BRACOL dev numbers of the **ours-general** files (no BRACOL
  data in their calibration, so not in-sample); ours-task uses the same allocation. `selection/final.csv` lists one ours-general and one
  ours-task file per (model, target) and is written before any test number is seen.
- General quality beyond BRACOL (owner's request): `benchmark/general_bench.py`: MMStar (1,500), AI2D test (1,000 by SHA-256), MMLU-Redux 2.0
  (5,330 items with error_type ok), one-letter multiple choice scored like the harness (pre50 letter probabilities, thinking off).

## final.csv selection rule, made precise (2026-10-04 06:5x KST, before any test number and before the v2b results)
For each (model, target): **ours-general** = among every ours-general-labelled file at that target (text-only `ours-general`,
`ours-general2` COCO captioning, `ours-general3` COCO + generic scene multiple choice; any allocation), the highest BRACOL **dev**
forced_macro_f1 on the A100 harness (frozen prompt; ties: lower letter_kld). **ours-task** = the ours-task file built on the same template
(allocation) as the chosen ours-general file. Files not chosen stay in files.csv and results.csv as evaluated variants.
General benchmarks (MMStar / AI2D / MMLU-Redux, `benchmark/general_bench.py`) and text KLD are reported for the chosen files and their
matched Unsloth files; they do not enter the selection.

## Low-end search (2026-10-04 10:2x KST, registered before any of its builds finished)
- Scope (owner, 10:0x): UD-IQ2_XXS size and below only: 2B 768 / 700 / 637 MB (1 GB package), 0.8B 338 / 310 / 290 MB. Owner, 10:1x:
  screen with a benchmark → the screen is BRACOL **dev** on real builds (forced_macro_f1, letter_kld as the low-noise second reading),
  i.e. the registered selection metric itself. The A100 builds on cards 0–6.
- Split (owner, 10:2x: the 4090 now has six cards and should build and experiment too): A100 = 2B low end; 4090 = 0.8B low end end to
  end (build, dev, selection by the rule below, upload, files.csv / final.csv rows), with the pipeline in `quantizer/` and the
  calibration assets in the private repo (`calib/hn04b-calib-assets.tar.gz`, sha256 381413a0…).
- Why the grid sweeps the tied embedding: Unsloth keeps token_embd at Q5_K at the low end (2B UD-IQ2_XXS: embedding 349.6 MB + body
  407.7 MB; 0.8B: 174.8 + 152.4 MB). Our low-end files cut the tied embedding, which is also the output head that scores A–F: 2B 768 MB
  with a Q3_K embedding reaches dev F1 0.51 (general2) / 0.61 (task), 700 MB with an IQ2_XXS embedding 0.20 / 0.37.
- 2B grid: `plan_alloc.py` BASE UD-IQ2_M, UP UD-IQ3_XXS, DOWN UD-IQ2_XXS, token_embd = 637 MB {Q2_K, IQ3_XXS, Q3_K}; 700 MB {IQ3_XXS,
  Q3_K, IQ4_XS} (Q4_K plans to 704.7 MB at the DOWN limit: dropped); 768 MB {IQ4_XS, Q4_K, Q5_K}. `768M-eQ5_K` has exactly the tensor
  types of Unsloth UD-IQ2_XXS (0 of 320 tensors differ, same bytes): our quantisation against theirs at identical types. Each template ×
  {general2, general3, task}, nseq 128. Selection: the 06:5x rule, unchanged.
- Deviation: at 10:12 a tail of `logs/events.log` showed the FINAL (test) lines of three files at low-end targets (2B ours-general3-1GBpkg-v3,
  ours-general3-700M-v3, ours-general2-768M-v2) before final.csv was frozen for them. The selection stays the mechanical dev rule; the
  A100 reads the event log with FINAL lines filtered out until the freeze.

## Fine-tuning track (owner's change of scope, 2026-10-04 11:3x–14:0x KST)
- Owner (this session, translated): "there are no hackathon rules as such, I set them, so this can be flexible" → option 2
  "fine-tuning after quantization"; then "get the best model on general tasks by quantization (one that clearly beats Unsloth),
  then a high BRACOL score by fine-tuning", "fix on the 2B model", "let's go for 95%", base choice "go with (A)"
  (task-1GBpkg-eQ2_K, package with adapter 1,011.9 MB, 1.2 % over the 1 GB package rule the owner set), and "can we find a
  combination that scores high on the benchmark at 450 MB?".
- Method (A100 only; the fine-tuning code is now in `../../../fine-tuning/`): the LM linears + tied embedding of OUR GGUF are dequantised into the HF model (inverse of
  gptq_iq.py's layout; check: BF16 HF vs llama-server letter probabilities |Δ| ≤ 0.05), the base stays frozen, a LoRA adapter is trained
  (PEFT; rank 8/16 on q/k/v/o, gate/up/down, in_proj_qkv/z, out_proj; loss = CE over the six letters at the answer position, frozen
  prompt); optional: LoRA on the vision blocks + merger trained jointly and merged into the canonical Q8_0 mmproj (
  re-quantised Q8_0, same bytes). LM adapters are exported with llama.cpp `convert_lora_to_gguf.py` (Q8_0, 12.6 MB at rank 8) and served
  with `llama-server --lora` (harness unchanged; a wrapper adds the flag).
- Data: BRACOL **dev** only: train 337 (int(sha256,16) % 5 != 0), epoch choice on the other 82; `--all-dev` runs train on all 419 with a
  fixed epoch count. BRACOL **test** is never used for training, epoch choice or configuration choice. Some runs add JMuBEN distinct
  images (4090 dedup, 3,756; subsets of 1,284 / 2,184); for those adapters JMuBEN numbers are training-set numbers.
- Results so far (BRACOL test n=1,266, A100; 4090 reproduces within 0.01): 637 MB general3 + r8 0.860 acc / 0.816 F1; 637 MB task + r8
  0.866 / 0.839; 625 MB + r8 0.877 / 0.852 (package 999.3 MB); 450 MB + r8 0.887 / 0.864 (823.7 MB); Unsloth UD-IQ2_XXS + same r8
  recipe 0.859 / 0.837 (1,142.4 MB); 450 MB + r16 + aug + JMuBEN-2184 0.912 / 0.887; 450 MB + vision LoRA + aug + JMuBEN-1284 0.908 /
  0.879. BF16 zero-shot 0.609 / 0.584. Weakest class: C (brown eye spot) F1 ≈ 0.6–0.68.
- 4090: adapters raise the general benchmarks (format compliance: one-letter answers), and do not transfer to
  JMuBEN field photos when JMuBEN is not in training (g3 +0.058, task −0.013 F1).
- Bases: the 12.6 MB-smaller 625 MB rebuild lost general accuracy (MMStar 0.316 / AI2D 0.305 / MMLU-R 0.353 vs 0.439 / 0.574 / 0.340),
  so the owner chose (A). 450 / 525 MB bases are below Unsloth 768 MB on general benchmarks without an adapter → a 450 MB
  allocation/calibration search is running (IQ1_M floor template, two priority orders, task/general2/general3).
