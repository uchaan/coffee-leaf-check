# Coffee Leaf Check

Offline coffee-leaf disease check on a budget Android phone. A small VLM (Qwen3.5-2B, llama.cpp) reads one leaf
photo and answers A healthy · B rust · C cercospora · D phoma · E leaf miner · F not sure.

Hack-Nation 7th Global AI Hackathon — World Bank challenge 04B.

The model was quantized by an **agentic pipeline**: give Claude Code one goal, and it plans, builds, evaluates and
iterates on its own until every size target is met. → [`agentic-quantization/`](agentic-quantization/)

| Path | What |
|---|---|
| [`agentic-quantization/`](agentic-quantization/) | agentic GGUF quantization pipeline, benchmark, results |
| [`fine-tuning/`](fine-tuning/) | LoRA on the quantized base, CNN / YOLO baselines |
| [`app/`](app/) | Android app (to be added) |

## Model

[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) quantized to 2.66 bpw GGUF, plus a coffee LoRA adapter and a fine-tuned
vision projector. Package **1.01 GB** (1.09 GB with the optional pt-BR translator).

| File | Size | bpw |
|---|---:|---:|
| `Qwen3.5-2B-coffee-base-Q2.gguf` (language model) | 0.638 GB | 2.66 |
| `mmproj-Qwen3.5-2B-coffee-Q8_0.gguf` (vision projector) | 0.362 GB | 8.73 |
| `Qwen3.5-2B-coffee-lora-Q8_0.gguf` (LoRA adapter) | 0.013 GB | 12.89 |

```
llama-server -m Qwen3.5-2B-coffee-base-Q2.gguf --mmproj mmproj-Qwen3.5-2B-coffee-Q8_0.gguf \
  --lora Qwen3.5-2B-coffee-lora-Q8_0.gguf --jinja -c 8192
```

## Results

Full report, with confidence intervals, per-disease scores and what the data does not cover: [`RESULTS.md`](RESULTS.md).

### Original vs ours

Same evaluation server (RTX 4090); Δ against the original BF16 model.

| | Original BF16 | Ours, quantized only | Δ | **Ours, final package** | Δ |
|---|---:|---:|---:|---:|---:|
| Package size | 4.26 GB | 1.00 GB | ÷4.3 | **1.01 GB** | ÷4.2 |
| LM bpw | 16.0 | 2.66 | | 2.66 | |
| BRACOL test F1 | 0.582 | 0.578 | −0.004 | **0.881** | **+0.299** |
| BRACOL test accuracy | 60.6 % | 56.6 % | −4.0 pt | **90.9 %** | **+30.3 pt** |
| JMuBEN F1, 2,184 images | 0.474 | 0.485 | +0.010 | –¹ | |
| JMuBEN F1, 240 held-out originals | 0.595 | 0.543 | −0.052 | **0.984**¹ | **+0.389** |
| MMStar (image understanding) | 0.494 | 0.437 | −0.057 | 0.479 | −0.015 |
| AI2D (diagrams) | 0.737 | 0.569 | −0.168 | 0.673 | −0.064 |
| MMLU-Redux (text knowledge) | 0.604 | 0.341 | −0.263 | 0.500 | −0.104 |

¹ The final package was trained on JMuBEN photos; only the 240 held-out originals are a fair number for it
(in-domain: read it as same-dataset performance).

- Quantization keeps the coffee-leaf accuracy at a 4.3× smaller package; general ability pays for 2.66 bpw.
- Fine-tuning adds 30 points on coffee leaves and narrows the general gap (mostly better one-letter compliance).
- The released adapter was picked among three by test score; the validation pick scores 87.5 % / F1 0.840.

### Against public quantized files (no fine-tuning)

| File | LM size | LM bpw | Package | BRACOL F1 | MMStar | AI2D | MMLU-Redux |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ours** | 0.64 GB | 2.66 | 1.00 GB | **0.578** | **0.437** | **0.569** | **0.341** |
| Unsloth UD-IQ2_XXS (smallest Unsloth) | 0.77 GB | 3.22 | 1.13 GB | 0.313 | 0.265 | 0.257 | 0.291 |
| mradermacher i1-IQ1_S | 0.64 GB | 2.67 | 1.00 GB | 0.059 | 0.237 | 0.187 | 0.220 |

### Against CNNs (same training data, A100)

| Model | Size | BRACOL test acc / F1 | JMuBEN held-out acc / F1 | JMuBEN F1, trained on BRACOL only |
|---|---:|---|---|---:|
| **Ours, final package** | 1.01 GB | 90.8 % / 0.880 | 98.3 % / 0.984 | **0.472** |
| ResNet50 | 0.09 GB | 88.5 % / 0.854 | 99.2 % / 0.987 | 0.132 |
| YOLO11m-cls | 0.02 GB | 88.5 % / 0.853 | 97.9 % / 0.979 | 0.237 |

Similar accuracy in-domain; on a different photo domain the VLM degrades far less, and it can explain its answer and
say "not sure".

Training data: BRACOL dev (337 train, 82 for epoch choice) + 559 JMuBEN originals; no test image used for training.

## License and credits

Code in this repository: [MIT](LICENSE). Model weights, datasets and the translator keep their own licenses, below.


- Base model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), Apache 2.0.
- BRACOL: Krohling, Esgario, Ventura (2019), Mendeley Data, [doi:10.17632/yy2k5y8mxg.1](https://doi.org/10.17632/yy2k5y8mxg.1), CC BY 4.0.
- JMuBEN / JMuBEN2: Jepkoech et al. (2021), Mendeley Data, [doi:10.17632/t2r6rszp5c.1](https://doi.org/10.17632/t2r6rszp5c.1) /
  [doi:10.17632/tgv3zb82nd.1](https://doi.org/10.17632/tgv3zb82nd.1), CC BY 4.0.
- Translator: [Helsinki-NLP/opus-mt-en-ROMANCE](https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE), Apache 2.0.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT.

Images are not included; [`manifest.csv`](agentic-quantization/manifest.csv) lists them with SHA-256 and split.
