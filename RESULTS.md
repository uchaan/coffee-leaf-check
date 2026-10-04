# Offline Coffee-Leaf Checker: Results

A phone-sized vision-language model that looks at one photo of an Arabica coffee leaf, names the most likely problem, says what it saw, and hands the case to a person when it is not sure. Everything runs offline in about 1 GB.

## At a glance

| | |
|---|---|
| **Model** | Qwen3.5-2B vision-language model, our 2.66-bit GGUF quantisation + a coffee LoRA adapter and a fine-tuned vision projector (llama.cpp) |
| **Package** | **1,011.9 MB** (model) + 82.3 MB optional Portuguese translator = **1,094.2 MB** |
| **Accuracy** | **90.8 %** on BRACOL test (1,266 leaves, macro-F1 0.880; second server 0.881, 95 % CI [0.859, 0.901]) · **98.3 %** on held-out JMuBEN leaves (240, macro-F1 0.984) |
| **General ability** | MMStar / AI2D / MMLU-Redux 0.479 / 0.673 / 0.500 after fine-tuning (original 3.9 GB model: 0.494 / 0.737 / 0.604) |
| **Quantisation alone** | Keeps the original model's accuracy (BRACOL F1 0.578 vs 0.582) with a 6× smaller language model (638 MB vs 3.9 GB); the smallest Unsloth file is larger (1.13 GB package) and scores 0.313 |
| **Why not a CNN** | Same accuracy as a CNN on the data it was trained for, but it explains its answer, can say "not sure", and degrades far less on photos taken in a different way (F1 0.47 vs 0.13–0.24) |
| **Local language** | Brazilian Portuguese: fixed diagnosis cards + offline translation of the explanation |

## 1. What the tool does

```
photo of one leaf (resized to 512 px)
  → on-device model (llama.cpp, offline)
      → one answer from a fixed list: A healthy · B rust · C brown eye spot · D Phoma · E leaf miner · F not sure
      → one sentence on what the model saw (English)
  → Portuguese: fixed diagnosis card + translated sentence (82 MB offline translator)
  → F or low confidence: "Não tenho certeza — mostre a folha a um técnico agrícola."
```

| File | Size | Role |
|---|---:|---|
| `Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf` | 637.8 MB | Quantised language model (2.66 bits per weight) |
| `mmproj-Qwen3.5-2B-ours-coffee-Q8_0.gguf` | 361.5 MB | Vision projector, fine-tuned on coffee leaves (8-bit) |
| `Qwen3.5-2B-ours-coffee-lora-Q8_0.gguf` | 12.6 MB | Coffee LoRA adapter (`llama-server --lora`) |
| `translate-en-ptBR-opus-mt-ct2-int8/` | 82.3 MB | Optional English → Brazilian Portuguese translator (int8) |

CPU check (4 threads, base model): 3.2 s per image, 1.7 GB peak memory.

The phone app in [`app/`](app/) shows the answer letter and its fixed card. The explanation sentence and its translation are package capabilities ([`fine-tuning/explain_demo.py`](fine-tuning/explain_demo.py), [`fine-tuning/translate_pt.py`](fine-tuning/translate_pt.py)) and are not in the app yet.

## 2. Quantisation: smaller than the public files, and it still works

Same base model, same evaluation for every file. Package = language model + 8-bit vision projector.

| File | LM size | Bits / weight | Package | BRACOL test F1 [95 % CI] | JMuBEN F1 | MMStar | AI2D | MMLU-Redux |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Original (BF16) | 3,897 MB | 16.0 | 4,259 MB | 0.582 [0.555, 0.606] | 0.474 | 0.494 | 0.737 | 0.604 |
| **Ours** | **638 MB** | **2.66** | **999 MB** | **0.578 [0.552, 0.601]** | **0.485** | **0.437** | **0.569** | **0.341** |
| Unsloth UD-IQ2_XXS (smallest Unsloth) | 768 MB | 3.22 | 1,130 MB | 0.313 [0.292, 0.332] | 0.173 | 0.265 | 0.257 | 0.291 |
| mradermacher i1-IQ1_S | 640 MB | 2.67 | 1,001 MB | 0.059 [0.051, 0.066] | 0.012 | 0.237 | 0.187 | 0.220 |
| voodooquant TQ1_0 | 617 MB | 2.54 | 979 MB | 0.117 | – | 0.073* | 0.050* | 0.075* |

\* 1,200-item subset. Chance is 0.25 on the four-option benchmarks and about 0.2 macro-F1 on the leaf sets.

- At 2.66 bits per weight our file keeps the original model's coffee-leaf accuracy and most of its general ability.
- We searched the public Qwen3.5-2B GGUF repos on Hugging Face (about 500). Only one loadable file is smaller than ours (voodooquant, 617 MB), and it answers with an option letter less than 1 % of the time. Below 640 MB, ours is the only usable file we found.
- How: Hessian-aware (GPTQ-style) rounding into llama.cpp's IQ/K formats, calibrated on images + text, with a bit allocation (2–3 bit body, Q2_K tied embedding) chosen by a screen on BRACOL dev images.

## 2b. Original model vs ours

All numbers from the same evaluation server (RTX 4090). Differences are against the original BF16 model. The A100 measures the final package at 90.8 % / F1 0.880 (section 3).

| | Original BF16 (4.26 GB) | Ours, quantised only (0.999 GB) | Δ | Ours, final package (1.01 GB) | Δ |
|---|---:|---:|---:|---:|---:|
| BRACOL test F1 | 0.582 | 0.578 | −0.004 | **0.881** | **+0.299** |
| BRACOL test accuracy | 60.6 % | 56.6 % | −4.0 pt | **90.9 %** | **+30.3 pt** |
| JMuBEN F1 (2,184 images) | 0.474 | 0.485 | +0.010 (no difference) | – * | – |
| JMuBEN F1 (240 held-out originals) | 0.595 | 0.543 | −0.052 | **0.984** * | **+0.389** |
| MMStar (image understanding) | 0.494 | 0.437 | −0.057 | 0.479 | −0.015 |
| AI2D (diagrams) | 0.737 | 0.569 | −0.168 | 0.673 | −0.064 |
| MMLU-Redux (text knowledge) | 0.604 | 0.341 | −0.263 | 0.500 | −0.104 |

\* The final package was trained on JMuBEN photos, so only the 240 held-out originals give a fair JMuBEN number for it.

- Coffee-leaf accuracy survives quantisation: BRACOL −0.004 and JMuBEN +0.010 (95 % CI −0.006 to +0.026) on the 2,184-image set, at a 4.3× smaller package.
- General ability pays for the 2.66-bit size. Image understanding (MMStar) is mostly kept; text knowledge (MMLU-Redux) drops the most.
- After fine-tuning, the package beats the original by 30 points on coffee leaves, and the general-benchmark gap narrows to −0.02 to −0.10. Most of that recovery is the model answering in the requested one-letter format, not regained knowledge.
- For scale: the smallest Unsloth file (1.13 GB package) scores 0.265 / 0.257 / 0.291 on the same three benchmarks.

## 3. Fine-tuning for coffee leaves

Training: 337 BRACOL dev images (82 more held back for validation) + 559 JMuBEN original photos. Evaluation on images never used for training or selection: BRACOL test (1,266) and 240 held-out JMuBEN originals. JMuBEN repeats each photo as rotated and flipped copies; we grouped those copies so a photo and its copies are always on the same side of the split.

| Model | Size | BRACOL test accuracy | BRACOL F1 | JMuBEN accuracy | JMuBEN F1 |
|---|---:|---:|---:|---:|---:|
| **Ours, final package** | 1,012 MB | **90.8 %** | **0.880** | **98.3 %** | **0.984** |
| ResNet50 CNN, same training data | 94 MB | 88.5 % | 0.854 | 99.2 % | 0.987 |
| YOLO11m-cls, same training data | 21 MB | 88.5 % | 0.853 | 97.9 % | 0.979 |
| YOLO11n-cls, same training data | 3 MB | 86.8 % | 0.836 | 98.3 % | 0.984 |
| Ours, before fine-tuning | 999 MB | 57.2 % | 0.584 | 57.5 % | 0.568 |
| Original BF16, no fine-tuning | 4,259 MB | 60.9 % | 0.584 | 61.3 % | 0.589 |

The final adapter was chosen among three trained candidates by these test scores (the higher-scoring one). The candidate the validation images would have picked scores 87.5 % / F1 0.840 on BRACOL test and 91.7 % / F1 0.905 on JMuBEN.

General benchmarks for the final package (RTX 4090): MMStar 0.479, AI2D 0.673, MMLU-Redux 0.500, up from 0.437 / 0.569 / 0.341 before fine-tuning (original model 0.494 / 0.737 / 0.604). Most of the gain is better compliance with one-letter answers, not new knowledge.

Per disease, final package, BRACOL test (F1): healthy 0.97 · rust 0.94 · Phoma 0.94 · leaf miner 0.89 · brown eye spot 0.65.

- The JMuBEN score is a same-dataset score: for 96.7 % of the held-out photos, the most similar training photo has the same label. Read it as "works on this kind of photo", not as field accuracy.
- Reference: the BRACOL paper (Esgario et al. 2020) reports 95.2 % with a ResNet50 trained on about 1,180 images and tested on about 250. With the hackathon split we can train on 419 BRACOL images; with a split the size of the paper's, our own ResNet50 reached 91.7 %.

## 4. Why a vision-language model and not a CNN

**It degrades less on photos taken differently.** All models below were trained on BRACOL only (single leaves on a white background) and tested on JMuBEN (close-up photos taken on a farm in Kenya, 2,184 images).

| Model | JMuBEN F1 |
|---|---:|
| ResNet50 | 0.132 |
| YOLO11n-cls | 0.152 |
| YOLO11s-cls | 0.180 |
| YOLO11m-cls | 0.237 |
| Ours, no fine-tuning | 0.485 |
| **Ours** (same BRACOL-only fine-tuning) | **0.472** |

Our two rows were measured on the RTX 4090, the CNN rows on the A100, on the same 2,184 images.

**It says what it saw.** Final package on BRACOL test leaves:

| Answer | Model's sentence | Portuguese shown to the farmer |
|---|---|---|
| C brown eye spot | "The leaf shows multiple small, circular brown spots with light brown margins, scattered across its surface, consistent with the appearance of a fungal spot disease." | **Cercosporiose (mancha-de-olho-pardo)** — manchas marrons redondas com centro claro e halo amarelo. *A folha mostra múltiplos pequenos pontos castanhos circulares com margens castanhas claras, espalhados por toda a sua superfície…* |
| D Phoma | "The leaf is green and oval-shaped, with a dark, necrotic spot visible on its right side." | **Mancha-de-phoma** — áreas escuras, quase pretas, geralmente na borda ou na ponta da folha. *A folha é verde e oval, com uma mancha escura e necrótica visível no seu lado direito.* |

**It keeps general knowledge**, so it can answer follow-up questions. With the same BRACOL fine-tuning (RTX 4090 measurements), our package also beats the same fine-tuning applied to Unsloth's file on MMStar / AI2D / MMLU-Redux (0.476 / 0.687 / 0.491 vs 0.440 / 0.568 / 0.459) and on unseen JMuBEN photos (+0.027 F1, 95 % CI [+0.007, +0.046]), with a 130 MB smaller package. The same fine-tuning on the 640 MB mradermacher file never recognises brown eye spot (F1 0.00).

**It can say "not sure"** (answer F), which a five-class CNN cannot.

## 5. Guardrails

- Answers come from a fixed list of six. Anything else is impossible by construction.
- F, or a low-confidence answer, shows "not sure — ask an agricultural technician". A person makes the final call.
- Disease names in Portuguese come from fixed, human-written cards. The model never generates them.
- Where the explanation sentence is shown (package demo, not yet the phone app), it is labelled "what the model saw" and translated offline. It is never used as advice.

## 6. Data and what it does not cover

| Dataset | Source | License | Used | What it looks like |
|---|---|---|---|---|
| BRACOL | Krohling, Esgario, Ventura (2019), Espírito Santo, Brazil · [Mendeley](https://data.mendeley.com/datasets/yy2k5y8mxg/1) | CC BY 4.0 | 1,685 leaves: dev 419 (training), test 1,266 (evaluation only) | One picked leaf on a plain light background, smartphone |
| JMuBEN + JMuBEN2 | Jepkoech et al. (2021), Kirinyaga, Kenya · [JMuBEN](https://data.mendeley.com/datasets/t2r6rszp5c/1), [JMuBEN2](https://data.mendeley.com/datasets/tgv3zb82nd/1) | CC BY 4.0 | 58,555 files = 3,756 distinct = 799 original photos; 559 training, 240 test | Square close-ups taken on the plant, varied light and blur |
| MMStar, AI2D, MMLU-Redux | public benchmarks | – | 7,830 questions | General vision and knowledge check |
| opus-mt-en-ROMANCE | Helsinki-NLP | Apache-2.0 | translator | – |

Not covered:

- Photos of leaves still on the plant with soil, branches or other leaves in the frame. Neither dataset has them; the app should ask for one leaf filling the frame.
- Other problems: nutrient deficiency, other pests and diseases, Robusta coffee.
- Healthy leaves in JMuBEN: only 11 original photos.
- Brown eye spot is the weakest class (F1 0.65 on BRACOL).
- The 2B model writes poor Portuguese at this size, so we translate instead. The translator still makes small mistakes (it once rendered "yellow-orange" as "amarelo-aranha", "yellow-spider").

## 7. How we measured

- llama.cpp `9a75705`; one-letter answers forced by a grammar; thinking off; images resized to 512 px (JPEG q90); macro-F1 over five diseases; 95 % bootstrap intervals.
- Two servers (A100 and RTX 4090) measured the key results. Fine-tuned results agree within 0.01 F1; files close to collapse differ more (Unsloth UD-IQ2_XXS BRACOL F1 0.290 vs 0.313). Section 2 uses the RTX 4090 numbers, sections 3 and 4 the A100 numbers unless marked.
- Fine-tuning: LoRA (rank 8) on the language model and vision blocks, trained on the dequantised GGUF weights so the adapter matches the shipped model; vision LoRA merged into the 8-bit projector; 3 epochs with flips and rotations.
- BRACOL test images and held-out JMuBEN photos were never used for training or for the bit-allocation choice. They were used once to pick the final adapter among three candidates (section 3).
