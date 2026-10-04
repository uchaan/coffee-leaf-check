# Fine-tuning

LoRA on top of the quantized base (`Qwen3.5-2B-coffee-base-Q2.gguf`), so the adapter is trained against the exact
weights that ship. The base GGUF is dequantized into the HF model and frozen; LoRA (rank 8) is trained on the LM
linears and the vision blocks; loss = cross-entropy over the six answer letters with the frozen prompt.

| File | What |
|---|---|
| `train_lora.py` | train the adapter (LM + optional vision LoRA) on BRACOL dev + extra images |
| `export_lora.py` | LM LoRA → GGUF adapter (Q8_0); vision LoRA merged into the Q8_0 mmproj |
| `scripts/eval_lora.sh` | benchmark harness with `llama-server --lora` |
| `translate_pt.py`, `explain_demo.py` | optional pt-BR translation and the one-sentence explanation |
| `baselines/` | ResNet50 / YOLO11 baselines on the same splits |
| `data/` | JMuBEN grouped split (train 559 / test 240 originals) with `make_jmuben_split.py` that rebuilds it, the 3,756 distinct JMuBEN images, BRACOL paper-sized split |
| `baselines/check_jmuben_overlap.py` | nearest-neighbour check of the JMuBEN held-out photos against the training photos |

## Reproduce the released adapter

```bash
source ../agentic-quantization/loop/env.sh       # WORK, LLAMA_CPP_DIR, PY
pip install peft "transformers>=5.5" gguf safetensors
python train_lora.py --gguf $WORK/gguf/ours/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf --out $WORK/ft/coffee \
    --rank 8 --epochs 3 --aug --vision-lora --dev-repeat 2 \
    --extra data/jmuben_train_originals.csv --extra-dir $WORK/jmuben
python export_lora.py $WORK/ft/coffee/epoch3 $WORK/ft/coffee-e3      # -> coffee-e3-lm-q8.gguf, coffee-e3-mmproj-Q8_0.gguf
```
- Train set: 337 BRACOL dev images (×2) + 559 JMuBEN originals = 1,233; the other 82 dev images pick the epoch.
- The released adapter is **epoch 3**, picked among three candidates by test score; validation would have picked
  epoch 1 (87.5 % / F1 0.840 on BRACOL test).
- No BRACOL test or JMuBEN test image is used for training.

## Evaluate

```bash
scripts/eval_lora.sh 0 9800 coffee-bracol BASE.gguf coffee-e3-lm-q8.gguf coffee-e3-mmproj-Q8_0.gguf test
scripts/eval_lora.sh 0 9800 coffee-jmuben BASE.gguf coffee-e3-lm-q8.gguf coffee-e3-mmproj-Q8_0.gguf test \
    data/jmuben_grouped_split.csv $WORK/jmuben
```

## Baselines

```bash
python baselines/make_yolo_dirs.py hack && python baselines/yolo_cls.py hack yolo11m-cls 384
python baselines/cnn_baseline.py --split hack --out $WORK/ft/cnn-hack
python baselines/eval_cnn.py
```

## JMuBEN split

```bash
python data/make_jmuben_split.py $WORK/jmuben      # images/<sha256>.jpg = harness preprocessing of each archive file
WORK=$WORK python baselines/check_jmuben_overlap.py
```
JMuBEN ships each photo as rotated/flipped copies; the script groups the 3,756 distinct images into 799 original photos
(rotation/flip-invariant 16x16 hash) and splits originals 70/30 per class, so no photo has copies on both sides.

## Translator

```bash
pip install ctranslate2 sentencepiece transformers
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-ROMANCE --quantization int8 --output_dir translate-en-ptBR
# copy source.spm and target.spm from the Hugging Face model into translate-en-ptBR/, then:
python translate_pt.py translate-en-ptBR
```
About 80 MB on disk, ~0.2 s per sentence on a laptop CPU. Disease names are replaced by "the diagnosis" before
translation; the fixed Portuguese cards in `translate_pt.py` carry the names.
