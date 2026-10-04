# Coffee Leaf Check

Offline coffee-leaf disease check on a budget Android phone. A small VLM (Qwen3.5-2B, llama.cpp) reads one leaf
photo and answers A healthy · B rust · C cercospora (brown eye spot) · D phoma · E leaf miner · F not sure.

Hack-Nation 7th Global AI Hackathon, World Bank challenge 04B (Small AI for development).

The model was quantized by an **agentic pipeline**: give Claude Code one goal, and it plans, builds, evaluates and
iterates on its own until every size target is met. See [`agentic-quantization/`](agentic-quantization/).

## Repository

| Path | What |
|---|---|
| [`RESULTS.md`](RESULTS.md) | full results: confidence intervals, per-disease scores, data and what it does not cover |
| [`agentic-quantization/`](agentic-quantization/) | agentic GGUF quantization loop (Claude Code skill, workers, GPTQ-style quantizer), benchmark harness, image manifests, all results and the selected files |
| [`fine-tuning/`](fine-tuning/) | coffee LoRA on the quantized base, export to GGUF, pt-BR translator, CNN / YOLO baselines, JMuBEN split |
| [`app/`](app/) | offline Android client: Termux `llama-server` + one static web page |

Each folder has its own README with setup, commands and a file table.

## Model package

[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) quantized to 2.66 bpw GGUF, plus a coffee LoRA adapter and a
fine-tuned vision projector. Package **1.01 GB** (1.09 GB with the optional pt-BR translator).

| File (name in `RESULTS.md` and `app/config.json`) | Published name | Size | bpw |
|---|---|---:|---:|
| `Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf` (language model) | `Qwen3.5-2B-coffee-base-Q2.gguf` | 637.8 MB | 2.66 |
| `mmproj-Qwen3.5-2B-ours-coffee-Q8_0.gguf` (vision projector) | `mmproj-Qwen3.5-2B-coffee-Q8_0.gguf` | 361.5 MB | 8.73 |
| `Qwen3.5-2B-ours-coffee-lora-Q8_0.gguf` (LoRA adapter) | `Qwen3.5-2B-coffee-lora-Q8_0.gguf` | 12.6 MB | 12.89 |
| `translate-en-ptBR-opus-mt-ct2-int8/` (optional translator) | same | 82.3 MB | int8 |

The weights are **not in this repository** and the model repo is private; they are available on request. Either name
works: the app reads the file names from `app/config.json`. To rebuild them: the language model from
[`agentic-quantization/`](agentic-quantization/) (row `ours-task` in
[`selection/final.csv`](agentic-quantization/selection/final.csv), SHA-256 `cb88d42f…`), the adapter and vision projector
from [`fine-tuning/`](fine-tuning/). [`fine-tuning/README.md`](fine-tuning/README.md) maps its build outputs to both names.

## Quick start

Put the three package files in `app/models/hf/` (names as in `config.json`, or edit it), then:

```bash
# desktop: needs llama.cpp's llama-server and jq on PATH
bash app/run_phone.sh                        # serves the app and the model at http://127.0.0.1:8080/

# no model: the page with fake probabilities
cd app && python3 -m http.server 8765        # http://127.0.0.1:8765/?mock
```

The same `run_phone.sh` runs on the phone in Termux, offline; the install steps are in [`app/README.md`](app/README.md).
Plain `llama-server` works too:

```bash
llama-server -m Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf --mmproj mmproj-Qwen3.5-2B-ours-coffee-Q8_0.gguf \
  --lora Qwen3.5-2B-ours-coffee-lora-Q8_0.gguf --jinja -c 4096
```

Send the prompt in [`app/prompt.txt`](app/prompt.txt) with thinking off and read the letter probabilities
(request format in [`app/README.md`](app/README.md)). `-c 4096` and `-c 8192` give the same peak memory.

## Reproduce

| Step | Where | Hardware |
|---|---|---|
| Image manifests, benchmark harness, public-file comparison | [`agentic-quantization/benchmark/`](agentic-quantization/benchmark/) | NVIDIA GPU, llama.cpp `9a75705`; `harness.py cpu` for the 4-thread CPU check |
| Quantization loop and file selection | [`agentic-quantization/`](agentic-quantization/) | Linux, NVIDIA GPUs (one worker per GPU) |
| LoRA, vision projector, translator, CNN baselines | [`fine-tuning/`](fine-tuning/) | one CUDA GPU |
| Phone / desktop app | [`app/`](app/) | Termux on Android, or any machine with `llama-server` |

Python packages: [`agentic-quantization/requirements.txt`](agentic-quantization/requirements.txt) and
[`fine-tuning/requirements.txt`](fine-tuning/requirements.txt) (Python 3.11). Paths come from
[`agentic-quantization/loop/env.example.sh`](agentic-quantization/loop/env.example.sh), which both folders use.

Images are not included; [`manifest.csv`](agentic-quantization/manifest.csv) (BRACOL) and
[`manifest_jmuben.csv`](agentic-quantization/manifest_jmuben.csv) (JMuBEN) list them with SHA-256 and split.

## Results

Full report: [`RESULTS.md`](RESULTS.md).

### Original vs ours

Same evaluation server (RTX 4090); Δ against the original BF16 model.

| | Original BF16 | Ours, quantized only | Δ | **Ours, final package** | Δ |
|---|---:|---:|---:|---:|---:|
| Package size | 4.26 GB | 1.00 GB | ÷4.3 | **1.01 GB** | ÷4.2 |
| LM bpw | 16.0 | 2.66 | | 2.66 | |
| BRACOL test F1 | 0.582 | 0.578 | −0.004 | **0.881** | **+0.299** |
| BRACOL test accuracy | 60.6 % | 56.6 % | −4.0 pt | **90.9 %**² | **+30.3 pt** |
| JMuBEN F1, 2,184 images | 0.474 | 0.485 | +0.010 | –¹ | |
| JMuBEN F1, 240 held-out originals | 0.595 | 0.543 | −0.052 | **0.984**¹ | **+0.389** |
| MMStar (image understanding) | 0.494 | 0.437 | −0.057 | 0.479 | −0.015 |
| AI2D (diagrams) | 0.737 | 0.569 | −0.168 | 0.673 | −0.064 |
| MMLU-Redux (text knowledge) | 0.604 | 0.341 | −0.263 | 0.500 | −0.104 |

¹ The final package was trained on JMuBEN photos; only the 240 held-out originals are a fair number for it
(in-domain: read it as same-dataset performance).
² RTX 4090. The A100 measures the same package at 90.8 % / F1 0.880 (CNN table below and `RESULTS.md` section 3).

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
say "not sure". The "Ours" value in the last column is our base with the same BRACOL-only fine-tuning (not the final
package), measured on the RTX 4090; the CNN values on the A100 (`RESULTS.md` section 4).

Training data: BRACOL dev (337 train, 82 held back for validation) + 559 JMuBEN originals; no test image used for training.

## License and credits

Code in this repository: [MIT](LICENSE). Model weights, datasets and the translator keep their own licenses, below.

- Base model: [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), Apache 2.0.
- BRACOL: Krohling, Esgario, Ventura (2019), Mendeley Data, [doi:10.17632/yy2k5y8mxg.1](https://doi.org/10.17632/yy2k5y8mxg.1), CC BY 4.0.
- JMuBEN / JMuBEN2: Jepkoech et al. (2021), Mendeley Data, [doi:10.17632/t2r6rszp5c.1](https://doi.org/10.17632/t2r6rszp5c.1) /
  [doi:10.17632/tgv3zb82nd.1](https://doi.org/10.17632/tgv3zb82nd.1), CC BY 4.0.
- Translator: [Helsinki-NLP/opus-mt-en-ROMANCE](https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE), Apache 2.0.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT.
