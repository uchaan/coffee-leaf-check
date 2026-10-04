# Fine-tuning

LoRA on top of the quantized base (`Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf`, released as `Qwen3.5-2B-coffee-base-Q2.gguf`),
so the adapter is trained against the exact weights that ship. The base GGUF is dequantized into the HF model and frozen;
LoRA (rank 8) is trained on the LM linears and the vision blocks; loss = cross-entropy over the six answer letters with the
frozen prompt ([`../../protocol/prompt_v1.txt`](../../protocol/prompt_v1.txt)).

Every command below runs from the repository root. Data files and image preparation live in [`../../data/`](../../data/).

| File | What |
|---|---|
| `train_lora.py` | train the adapter (LM + optional vision LoRA) on BRACOL dev + extra images |
| `export_lora.py` | LM LoRA → GGUF adapter (Q8_0); vision LoRA merged into the Q8_0 mmproj |
| `scripts/eval_lora.sh` | benchmark harness with `llama-server --lora` |
| `explain_demo.py` | one-sentence explanation per class, adapter off vs on (needs a running `llama-server`) |
| `translate_pt.py` | optional offline EN → pt-BR translation of the explanation; holds the fixed pt-BR diagnosis cards |
| `requirements.txt` | Python packages for every script here |
| `baselines/cnn_baseline.py`, `yolo_cls.py`, `make_yolo_dirs.py` | ResNet50 / YOLO11 baselines on the same splits |
| `baselines/eval_cnn.py` | the trained baselines on the 240 JMuBEN held-out originals and BRACOL test |
| `baselines/check_jmuben_overlap.py` | nearest-neighbour check of the JMuBEN held-out photos against the training photos |

Inputs from [`../../data/`](../../data/README.md): `manifest.csv` (BRACOL dev for training, test for evaluation),
`jmuben_train_originals.csv` (`--extra`), `jmuben_grouped_split.csv` (JMuBEN held-out 240), `bracol_paper_sized_split.csv`
(`s2` baseline split), and `prep_images.py` / `make_jmuben_split.py`.

## Setup

```bash
cp env.example.sh env.sh                           # once, at the repository root; edit paths
source env.sh                                      # WORK, LLAMA_CPP_DIR, PY, BRACOL_ROOT, HN04B_CACHE, HN04B_RUNS
pip install -r model/fine-tuning/requirements.txt
hf download Qwen/Qwen3.5-2B --local-dir $WORK/models/Qwen3.5-2B
python data/prep_images.py bracol                    # $HN04B_CACHE/<sha256>.jpg + $WORK/bracol/dev512/<file name>
python data/prep_images.py jmuben $JMUBEN_ROOT       # $WORK/jmuben/images/<sha256>.jpg ($JMUBEN_ROOT = extracted JMuBEN + JMuBEN2)
```
- The base GGUF is built by [`../quantization/`](../quantization/) (row `ours-task` in `model/quantization/selection/final.csv`, SHA-256 `cb88d42f…`) or
  obtained as the published `Qwen3.5-2B-coffee-base-Q2.gguf` (weights on request, see the root README); the commands below expect it at
  `$WORK/gguf/ours/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf`. The canonical vision projector
  `$WORK/gguf/mmproj-Qwen3.5-2B-Q8_0.gguf` (361,518,656 B, SHA-256 `3761d22e…`) is listed in
  [`../quantization/example-run/logs/a100-quantization-LOG.md`](../quantization/example-run/logs/a100-quantization-LOG.md).
- Training and the baselines need one CUDA GPU.

## Reproduce the released adapter

```bash
python model/fine-tuning/train_lora.py --gguf $WORK/gguf/ours/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf --out $WORK/ft/coffee \
    --rank 8 --epochs 3 --aug --vision-lora --dev-repeat 2 \
    --extra data/jmuben_train_originals.csv --extra-dir $WORK/jmuben
python model/fine-tuning/export_lora.py $WORK/ft/coffee/epoch3 $WORK/ft/coffee-e3      # -> coffee-e3-lm-q8.gguf, coffee-e3-mmproj-Q8_0.gguf
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
model/fine-tuning/scripts/eval_lora.sh 0 9800 coffee-bracol $B $A $M test                                            # BRACOL test, 1,266
model/fine-tuning/scripts/eval_lora.sh 0 9800 coffee-jmuben $B $A $M test data/jmuben_grouped_split.csv $WORK/jmuben  # JMuBEN, 240 originals
```
Arguments: GPU, port, run tag, base, adapter, mmproj, split, then optionally manifest and data root. Runs land in `$HN04B_RUNS/<tag>`.

## Explanation demo

```bash
$LLAMA_CPP_DIR/build/bin/llama-server -m $B --mmproj $M --lora $A --jinja -c 8192 --port 9899 &
python model/fine-tuning/explain_demo.py --port 9899
```
Prints the letter and one sentence for the first BRACOL test image of each class, with the adapter off and on.

## JMuBEN split

```bash
python data/make_jmuben_split.py $WORK/jmuben      # images/<sha256>.jpg from data/prep_images.py
python model/fine-tuning/baselines/check_jmuben_overlap.py
```
JMuBEN ships each photo as rotated/flipped copies; the script groups the 3,756 distinct images into 799 original photos
(rotation/flip-invariant 16x16 hash) and splits originals 70/30 per class, so no photo has copies on both sides.

## Baselines

```bash
python model/fine-tuning/baselines/make_yolo_dirs.py hack && python model/fine-tuning/baselines/make_yolo_dirs.py hackjm
python model/fine-tuning/baselines/yolo_cls.py hack yolo11m-cls 384
python model/fine-tuning/baselines/yolo_cls.py hackjm yolo11m-cls 384
python model/fine-tuning/baselines/yolo_cls.py hackjm yolo11n-cls 224
python model/fine-tuning/baselines/cnn_baseline.py --split hack   --out $WORK/ft/cnn-hack
python model/fine-tuning/baselines/cnn_baseline.py --split hackjm --out $WORK/ft/cnn-hackjm
python model/fine-tuning/baselines/cnn_baseline.py --split s2     --out $WORK/ft/cnn-s2     # paper-sized split (91.7 % in RESULTS.md)
python model/fine-tuning/baselines/eval_cnn.py                                              # needs the five runs above except s2
```
| Split | Train | Val | Test |
|---|---|---|---|
| `hack` | 337 BRACOL dev | 82 BRACOL dev | BRACOL test 1,266 |
| `hackjm` | `hack` + 559 JMuBEN originals | 82 BRACOL dev | BRACOL test 1,266 |
| `s2` | 1,179 paper-sized dev minus val | `int(sha256,16) % 10 == 0` of it | 506 |

`RESULTS.md` section 4 also reports BRACOL-only YOLO11n/YOLO11s and the BRACOL-only CNNs on the 2,184 JMuBEN images
(`data/manifest_jmuben.csv`); that evaluation has no committed script, and the image size of the
YOLO11n/YOLO11s BRACOL-only runs is not recorded here.

## Translator

```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ROMANCE --quantization int8 \
    --output_dir translate-en-ptBR-opus-mt-ct2-int8
# copy source.spm and target.spm from the Hugging Face model into translate-en-ptBR-opus-mt-ct2-int8/, then:
python model/fine-tuning/translate_pt.py translate-en-ptBR-opus-mt-ct2-int8
```
82.3 MB on disk, ~0.2 s per sentence on a laptop CPU. Disease names are replaced by "the diagnosis" before
translation; the fixed Portuguese cards in `translate_pt.py` carry the names (source of the pt-BR name and look in [`../../app/cards.json`](../../app/cards.json)).
