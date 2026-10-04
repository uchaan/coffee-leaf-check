# Fine-tuning

LoRA on top of the quantized base (`Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf`, released as `Qwen3.5-2B-coffee-base-Q2.gguf`),
so the adapter is trained against the exact weights that ship. The base GGUF is dequantized into the HF model and frozen;
LoRA (rank 8) is trained on the LM linears and the vision blocks; loss = cross-entropy over the six answer letters with the
frozen prompt.

| File | What |
|---|---|
| `train_lora.py` | train the adapter (LM + optional vision LoRA) on BRACOL dev + extra images |
| `export_lora.py` | LM LoRA → GGUF adapter (Q8_0); vision LoRA merged into the Q8_0 mmproj |
| `scripts/eval_lora.sh` | benchmark harness with `llama-server --lora` |
| `explain_demo.py` | one-sentence explanation per class, adapter off vs on (needs a running `llama-server`) |
| `translate_pt.py` | optional offline EN → pt-BR translation of the explanation; holds the fixed pt-BR diagnosis cards |
| `requirements.txt` | Python packages for every script here |
| `data/prep_images.py` | 512 px images from the raw BRACOL and JMuBEN files (the inputs of every script) |
| `data/make_jmuben_split.py` | rebuilds the JMuBEN grouped split |
| `baselines/cnn_baseline.py`, `yolo_cls.py`, `make_yolo_dirs.py` | ResNet50 / YOLO11 baselines on the same splits |
| `baselines/eval_cnn.py` | the trained baselines on the 240 JMuBEN held-out originals and BRACOL test |
| `baselines/check_jmuben_overlap.py` | nearest-neighbour check of the JMuBEN held-out photos against the training photos |

| Data file | Rows | What |
|---|---:|---|
| `data/jmuben_distinct_manifest.csv` | 3,756 | distinct JMuBEN images (`source_path` = file in the archives; `in_eval` = in the 2,184-image set of `../agentic-quantization/manifest_jmuben.csv`) |
| `data/jmuben_grouped_split.csv` | 799 | one image per original photo: 559 `dev` (training) / 240 `test` |
| `data/jmuben_train_originals.csv` | 559 | the training originals (`--extra` of `train_lora.py`) |
| `data/manifest_jmuben_heldout240.csv` | 240 | the same 240 test originals with archive paths, for harness runs on the raw JMuBEN folder (like `manifest_jmuben.csv`) |
| `data/bracol_paper_sized_split.csv` | 1,685 | BRACOL re-split to the paper's size: 1,179 dev / 506 test (`cnn_baseline.py --split s2`, `explain_demo.py`) |

## Setup

```bash
cp ../agentic-quantization/loop/env.example.sh ../agentic-quantization/loop/env.sh   # edit paths
source ../agentic-quantization/loop/env.sh       # WORK, LLAMA_CPP_DIR, PY, BRACOL_ROOT, HN04B_CACHE, HN04B_RUNS
pip install -r requirements.txt
hf download Qwen/Qwen3.5-2B --local-dir $WORK/models/Qwen3.5-2B
python data/prep_images.py bracol                    # $HN04B_CACHE/<sha256>.jpg + $WORK/bracol/dev512/<file name>
python data/prep_images.py jmuben $JMUBEN_ROOT       # $WORK/jmuben/images/<sha256>.jpg ($JMUBEN_ROOT = extracted JMuBEN + JMuBEN2)
```
- The base GGUF is built by `../agentic-quantization` (row `ours-task` in `selection/final.csv`, SHA-256 `cb88d42f…`) or
  obtained as the published `Qwen3.5-2B-coffee-base-Q2.gguf` (weights on request, see the root README); the commands below expect it at
  `$WORK/gguf/ours/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf`. The canonical vision projector
  `$WORK/gguf/mmproj-Qwen3.5-2B-Q8_0.gguf` (361,518,656 B, SHA-256 `3761d22e…`) is listed in
  `../agentic-quantization/example-run/logs/a100-quantization-LOG.md`.
- Training and the baselines need one CUDA GPU.

## Reproduce the released adapter

```bash
python train_lora.py --gguf $WORK/gguf/ours/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf --out $WORK/ft/coffee \
    --rank 8 --epochs 3 --aug --vision-lora --dev-repeat 2 \
    --extra data/jmuben_train_originals.csv --extra-dir $WORK/jmuben
python export_lora.py $WORK/ft/coffee/epoch3 $WORK/ft/coffee-e3      # -> coffee-e3-lm-q8.gguf, coffee-e3-mmproj-Q8_0.gguf
```
- Train set: 337 BRACOL dev images (×2) + 559 JMuBEN originals = 1,233; the other 82 dev images are the validation set.
- The released adapter is **epoch 3**, picked among three candidates by test score; validation would have picked
  epoch 1 (87.5 % / F1 0.840 on BRACOL test, 91.7 % / F1 0.905 on JMuBEN).
- No BRACOL test or JMuBEN test image is used for training.
- `export_lora.py` uses `gguf-py` and `convert_lora_to_gguf.py` from `$LLAMA_CPP_DIR` (llama.cpp `9a75705`).

| Build output | Name in `RESULTS.md` and `app/config.json` | Published name (root README) |
|---|---|---|
| `Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf` (base) | same | `Qwen3.5-2B-coffee-base-Q2.gguf` |
| `coffee-e3-lm-q8.gguf` | `Qwen3.5-2B-ours-coffee-lora-Q8_0.gguf` | `Qwen3.5-2B-coffee-lora-Q8_0.gguf` |
| `coffee-e3-mmproj-Q8_0.gguf` | `mmproj-Qwen3.5-2B-ours-coffee-Q8_0.gguf` | `mmproj-Qwen3.5-2B-coffee-Q8_0.gguf` |

## Evaluate

```bash
B=$WORK/gguf/ours/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf; A=$WORK/ft/coffee-e3-lm-q8.gguf; M=$WORK/ft/coffee-e3-mmproj-Q8_0.gguf
scripts/eval_lora.sh 0 9800 coffee-bracol $B $A $M test                                            # BRACOL test, 1,266
scripts/eval_lora.sh 0 9800 coffee-jmuben $B $A $M test data/jmuben_grouped_split.csv $WORK/jmuben  # JMuBEN, 240 originals
```
Arguments: GPU, port, run tag, base, adapter, mmproj, split, then optionally manifest and data root. Runs land in `$HN04B_RUNS/<tag>`.

## Explanation demo

```bash
$LLAMA_CPP_DIR/build/bin/llama-server -m $B --mmproj $M --lora $A --jinja -c 8192 --port 9899 &
python explain_demo.py --port 9899
```
Prints the letter and one sentence for the first BRACOL test image of each class, with the adapter off and on.

## JMuBEN split

```bash
python data/make_jmuben_split.py $WORK/jmuben      # images/<sha256>.jpg from data/prep_images.py
python baselines/check_jmuben_overlap.py
```
JMuBEN ships each photo as rotated/flipped copies; the script groups the 3,756 distinct images into 799 original photos
(rotation/flip-invariant 16x16 hash) and splits originals 70/30 per class, so no photo has copies on both sides.

## Baselines

```bash
python baselines/make_yolo_dirs.py hack && python baselines/make_yolo_dirs.py hackjm
python baselines/yolo_cls.py hack yolo11m-cls 384
python baselines/yolo_cls.py hackjm yolo11m-cls 384
python baselines/yolo_cls.py hackjm yolo11n-cls 224
python baselines/cnn_baseline.py --split hack   --out $WORK/ft/cnn-hack
python baselines/cnn_baseline.py --split hackjm --out $WORK/ft/cnn-hackjm
python baselines/cnn_baseline.py --split s2     --out $WORK/ft/cnn-s2     # paper-sized split (91.7 % in RESULTS.md)
python baselines/eval_cnn.py                                              # needs the five runs above except s2
```
| Split | Train | Val | Test |
|---|---|---|---|
| `hack` | 337 BRACOL dev | 82 BRACOL dev | BRACOL test 1,266 |
| `hackjm` | `hack` + 559 JMuBEN originals | 82 BRACOL dev | BRACOL test 1,266 |
| `s2` | 1,179 paper-sized dev minus val | `int(sha256,16) % 10 == 0` of it | 506 |

`RESULTS.md` section 4 also reports BRACOL-only YOLO11n/YOLO11s and the BRACOL-only CNNs on the 2,184 JMuBEN images
(`../agentic-quantization/manifest_jmuben.csv`); that evaluation has no committed script, and the image size of the
YOLO11n/YOLO11s BRACOL-only runs is not recorded here.

## Translator

```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ROMANCE --quantization int8 \
    --output_dir translate-en-ptBR-opus-mt-ct2-int8
# copy source.spm and target.spm from the Hugging Face model into translate-en-ptBR-opus-mt-ct2-int8/, then:
python translate_pt.py translate-en-ptBR-opus-mt-ct2-int8
```
82.3 MB on disk, ~0.2 s per sentence on a laptop CPU. Disease names are replaced by "the diagnosis" before
translation; the fixed Portuguese cards in `translate_pt.py` carry the names (source of the pt-BR name and look in `../app/cards.json`).
