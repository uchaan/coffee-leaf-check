# Benchmark harness

One harness, one llama.cpp build, the same inputs for every GGUF. Protocol: this file (pinned facts, data, metrics).
Run every command from the repository root; manifests are in [`../data/`](../data/), the prompt and chat templates in
[`../protocol/`](../protocol/).

## Run one file

```bash
source env.sh                                               # from the repository root
# dev metrics for one file (prompt tuning, the loop's dev check)
python eval/harness.py run --model Qwen3.5-2B --lm X.gguf --mmproj mmproj-Qwen3.5-2B-Q8_0.gguf \
    --prompt protocol/prompt_v1.txt --split dev --gpu 1
# full run: dev (tau) + test, scored against the BF16 reference rows
python eval/harness.py run ... --split all --ref $HN04B_RUNS/<bf16 tag>/images.csv
python eval/harness.py cpu ...                         # CPU proxy, -ngl 0, 4 threads, 50 test images
python eval/harness.py textkld --bf16 BF16.gguf --lm X.gguf --gpu 0
python eval/harness.py score --run-dir <run> --ref <bf16 images.csv>   # rescore only
```

Env (`env.example.sh` at the repository root): `LLAMA_CPP_DIR` (llama.cpp checkout with `build/bin`), `BRACOL_ROOT` (folder with
`dataset.csv` and `images/`), `HN04B_CACHE` (preprocessed 512 px JPEGs), `HN04B_RUNS` (output root), `WIKITEXT`,
`HN04B_SERVER_EXTRA` (extra `llama-server` flags). Python packages: `requirements.txt` (root). `--gpu N` sets
`CUDA_VISIBLE_DEVICES`.

Outputs per run (`<runs>/<tag>/`): `images.csv` (path, image sha256, split, label, `p_A`..`p_F`,
argmax over A to F, argmax over A to E, raw letter mass, sampled letter), `metrics.json` (per split),
`meta.json` (file sizes and SHA-256s, llama.cpp commit, harness commit, prompt and template SHA-256,
probability mode, s/img, bpw), `server.log`.

## Full benchmark (every file, as in `eval/results/`)

```bash
python eval/fetch_public.py $HN04B_PUBLIC eval/public_gguf_log.csv     # public GGUFs (<= 1 GB) + log
# ours: every model/quantization/selection/files.csv `file` path (<model>/<name>.gguf) under one folder, here $WORK/eval, including
#       <model>/<model>-BF16.gguf and <model>/mmproj-<model>-Q8_0.gguf (or fetch_ours.py from a model repo);
#       make_jobs.py turns only the bf16 / ours-general / ours-task rows into jobs
python eval/make_jobs.py --a100-dir $WORK/eval --public-dir $HN04B_PUBLIC --out $WORK/jobs.csv
python eval/gpu_queue.py --models Qwen3.5-2B,Qwen3.5-0.8B --gpu 0 --port 8090 \
    --prompt protocol/prompt_v1.txt --jobs $WORK/jobs.csv --a100-dir $WORK/eval          # one per GPU; add
                                                                                         # --dataset jmuben for JMuBEN
python eval/cpu_queue.py --model Qwen3.5-2B --cores 0-3 --port 8190 --prompt protocol/prompt_v1.txt \
    --jobs $WORK/jobs.csv --mmproj $WORK/eval/Qwen3.5-2B/mmproj-Qwen3.5-2B-Q8_0.gguf     # optional CPU proxy
touch $HN04B_RUNS/STOP                                      # queues exit when nothing runnable is left
python eval/collect.py --prompt prompt_v1 --jobs $WORK/jobs.csv --out eval/results/
```

Run tags are `<model>__<source>__<file stem>__<prompt stem>`; `collect.py` only reads runs named that way, so use the
queues (or pass the same `--tag` to `harness.py run`).

| Script | What |
|---|---|
| `harness.py` | one file: run, CPU proxy, text KLD, rescore |
| `queue_common.py` | run-tag naming and env defaults shared by the queues |
| `fetch_public.py`, `rebuild_public_log.py` | download the public GGUFs; rebuild `public_gguf_log.csv` from a download folder |
| `public_gguf_log.csv` | public GGUFs: repo, revision, SHA-256; `local_path` relative to `$HN04B_PUBLIC` |
| `fetch_ours.py` | poll `files.csv`, download each listed file from a model repo, check SHA-256, rebuild the jobs list |
| `make_jobs.py` | jobs list: BF16, ours (`model/quantization/selection/files.csv`), public (`public_gguf_log.csv`) |
| `gpu_queue.py`, `cpu_queue.py` | shared-list workers: GPU (`run --split all` + text KLD), CPU proxy |
| `collect.py` | every run of one prompt -> `eval/results/` (csv, charts, `summary.md`) |
| `pairs.py` | quick side-by-side of ours vs the Unsloth file of the same name |
| `general_bench.py`, `general_bench_sys.py`, `general_collect.py` | MMStar / AI2D / MMLU-Redux, the same with a one-letter system line (control), summary |

## Pinned facts

- llama.cpp `9a7570587ce908b0073a0458877205b80627f393` (same commit as the A100 server), CUDA 12.1, sm_89.
- Thinking off: request field `chat_template_kwargs: {"enable_thinking": false}` (the official template
  then emits an empty think block), server `--reasoning-format none`, `--chat-template-file` = the
  official template of `Qwen/<model>` (both models ship the same file, SHA-256 `273d8e0e…`).
- Request: `max_tokens` 1, `temperature` 1.0, `top_k` 0, `top_p` 1.0, `min_p` 0, no penalties, seed 0,
  grammar from the prompt file, `cache_prompt` false, one image per request.
- **Probability mode `pre50` (default).** At this commit `llama-server` samples with the grammar applied
  lazily (`grammar_first=false`): the grammar mask is applied only when the unconstrained sample is not a
  letter. So `post_sampling_probs` returns the unmasked distribution whenever the sample is a letter (the
  six letters then sum to about 0.9996, not 1), and the masked one otherwise. The harness therefore
  requests `n_probs` 50 without `post_sampling_probs` (raw softmax at temperature 1), keeps A to F,
  counts a missing letter as 0 and renormalises: this equals the grammar-masked distribution and does not
  depend on the sampled token. On 8 dev images the two modes agree to 6 digits after renormalisation.
  The phone app should read the letters the same way and renormalise.
- A to F are single tokens for both models (`/tokenize` check, stored in `meta.json`).
- Several GPU workers (`eval/gpu_queue.py`, up to three per GPU) share one job list. Results do not depend on
  which GPU or how many workers: the same file gave bit-identical probabilities on two different GPUs
  (BF16 2B, 1,685 of 1,685 rows) and when rerun under six concurrent workers (Unsloth UD-IQ2_M 2B, 1,685 of 1,685).
- `eval/collect.py` rescores every run from its per-image file, so every number in `eval/results/` comes from one
  code version.
- CPU proxy (`harness.py cpu`): `-ngl 0`, 4 threads (`-t 4 -tb 4`, process pinned to 4 cores with `taskset`),
  mmproj on CPU, first 50 BRACOL test images by SHA-256 after one untimed warm-up image, peak RSS from
  `/usr/bin/time -v`. The server runs with `--cache-ram 0 --ctx-checkpoints 0`: with the defaults (8 GiB RAM
  prompt cache, 32 context checkpoints per slot) RSS grows with every request (1GBpkg file: 1,460 MB vs
  2,056 MB after 10 images, 4,614 MB after 50), which a phone app answering one image at a time does not do.
  It is a proxy measured on an x86 server, not a phone.

## Data: BRACOL

Krohling, R. A., Esgario, J. G. M., Ventura, J. A. (2019). *BRACOL – A Brazilian Arabica Coffee Leaf
images dataset to identification and quantification of coffee diseases and pests*. Mendeley Data, v1,
doi:10.17632/yy2k5y8mxg.1. CC BY 4.0. Leaf-level set (`leaf/dataset.csv`, `leaf/images/`).

- The Mendeley zip (`BRACOL_coffee_leaf_ images_datasets.zip`, SHA-256 `25a2fc97…` = the hash Mendeley
  publishes) is truncated: no central directory, it stops inside `images/688.jpg`. 1,401 of 1,747 leaf
  images and `dataset.csv` could be recovered from it.
- Full copy used: HF dataset `luisangelico/bracol` (revision `66178a06febde553e2c9d6f4d90dfc462e018268`,
  origin stated as the authors' repo `esgario/lara2018`). Its `dataset.csv` and all 1,401 recoverable
  Mendeley images are byte-identical to the Mendeley copy.
- Labels: `predominant_stress` codes from the authors' code (`lara2018/classification/results.py`):
  0 Healthy → A, 1 Leaf miner → E, 2 Rust → B, 3 Phoma (brown leaf spot) → D, 4 Cercospora → C.
  Code 5 (62 images, no single predominant stress) is excluded, as in the authors' own classification
  split. `data/label_map.csv`, `data/class_counts.txt`.
- Split: per class, sorted by SHA-256 of file bytes, first floor(25%) dev, rest test
  (`data/make_manifest.py`). 419 dev, 1,266 test.
- One byte-identical pair (`images/813.jpg`, `images/1022.jpg`, both B, both test) is kept as two rows.

```bash
hf download luisangelico/bracol --repo-type dataset --revision 66178a06febde553e2c9d6f4d90dfc462e018268 \
    --local-dir $BRACOL_ROOT                              # dataset.csv + images/, the paths of data/manifest.csv
```
`data/manifest.csv` records each image's SHA-256; the harness caches the 512 px copy under that name (`$HN04B_CACHE/<sha256>.jpg`).

## Data: JMuBEN (optional second test set)

JMuBEN and JMuBEN2 (Jepkoech et al. 2021, Mendeley Data doi:10.17632/t2r6rszp5c.1 and doi:10.17632/tgv3zb82nd.1, CC BY 4.0).
Extract both archives into `$JMUBEN_ROOT`; `data/make_manifest_jmuben.py` documents the test-set rule (`data/manifest_jmuben.csv`,
2,184 images) and `gpu_queue.py --dataset jmuben` runs it.

## Metric definitions (as implemented in `harness.py`)

- Per image: six probabilities P(A)..P(F), renormalised over the letters the prompt's grammar allows.
- `forced_acc`, `forced_macro_f1`: prediction = argmax over A to E only (F ignored); macro-F1 over the five
  classes, a class with no true positive scores F1 0.
- `coarse_macro_f1`: the same forced predictions and labels mapped to A / B / other (C, D, E), macro-F1 over three.
- `agree_bf16`: share of images whose argmax over A to F equals the BF16 file's argmax (same prompt, same image).
- `letter_kld`: mean over images of KL(P_BF16 ‖ P_file) on the six letters; both clipped at 1e-10 and renormalised.
- `tau`: from BRACOL dev of the same file. Candidates: 0 and every dev top-letter probability where the top
  letter is not F. Answered = top letter not F and its probability ≥ τ. τ = the smallest candidate whose
  answered dev cases are ≥ 90% correct; if none reaches 90%, the candidate with the highest precision
  (`tau_reached_90` false in `metrics.json`). If the file never answers A to E on dev, τ is empty and
  coverage is 0.
- `answered_acc`, `coverage`: on test at that τ. `notsure_share` = 1 − coverage (the app shows "not sure" for
  every unanswered image). `model_F_share` (in `metrics.json`) = share whose top letter is F.
- `f1_ci_low`, `f1_ci_high`: 2.5th and 97.5th percentiles of forced macro-F1 over 1,000 resamples of the
  test images with replacement (numpy `default_rng(0)`, one (1000, n) index draw). Paired differences use the
  same resample indices for both files; an interval that includes 0 is reported as no difference.
- `text_kld_mean`, `text_same_top1`: `llama-perplexity --kl-divergence`, wikitext-2 raw test, first 40 chunks
  of 512 tokens, base logits from the BF16 file on this server.
- Sizes: `lm_bytes`, `mmproj_bytes`, `total_bytes` are file sizes; `bpw` = tensor bytes × 8 / tensor elements
  of the language-model file (llama.cpp's definition).
