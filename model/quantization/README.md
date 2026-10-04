# Agentic GGUF quantization

Give an AI agent (Claude Code) one goal; it runs the quantization loop by itself until every size target has a file,
then selects, tests and reports. Built on stock [llama.cpp](https://github.com/ggml-org/llama.cpp) (`9a7570587`);
outputs are ordinary GGUF files.

The agent is launched in this folder (its rules and skill live here); every script runs from the repository root,
where `env.sh`, [`../../data/`](../../data/), [`../../protocol/`](../../protocol/) and [`../../eval/`](../../eval/) are.

## Prerequisites

Linux with NVIDIA GPUs (the loop uses GNU `stat` / `sed`, `flock`, `nvidia-smi`; the quantizer loads `libggml-base.so`).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout 9a7570587ce908b0073a0458877205b80627f393
cmake -B build -DGGML_CUDA=ON && cmake --build build -j && cd ..
pip install -r requirements.txt                              # repository root; Python 3.11; gguf comes from llama.cpp/gguf-py
```

## Start

```bash
cp env.example.sh env.sh             # repository root; set paths
cd model/quantization && claude      # the agent runs scripts from the repository root
```
> Quantize Qwen3.5-2B to fit LM + Q8_0 mmproj under 1 GB, and beat the public GGUFs on BRACOL dev. Use GPUs 0–3.

The agent does the one-time setup itself (skill `gguf-quant-loop`, step 1): BF16 GGUF and Q8_0 mmproj
(`convert_hf_to_gguf.py`), the BF16 dev reference run, the text-KLD base, then `loop/start_workers.sh`.
More requests: [`example-run/prompts.md`](example-run/prompts.md).

## The loop

```mermaid
flowchart LR
    G([Goal]) --> P
    subgraph A["Agent"]
        P[Plan types<br/>per byte target] --> R[Pre-register<br/>in LOG.md] --> Q[Enqueue builds]
        D{Decide}
    end
    subgraph W["Workers, one per GPU"]
        B[Build<br/>gptq_iq.py] --> C[Byte check<br/>+ text KLD] --> V[Dev eval<br/>BRACOL dev]
    end
    Q --> B
    V -- events.log --> D
    D -- next builds --> P
    D -- all targets done --> S[Select on dev] --> T[Test once<br/>+ report]
```

## Layout

| Path | What |
|---|---|
| `AGENTS.md` (imported by `CLAUDE.md`) | agent rules |
| `.claude/skills/gguf-quant-loop/` | loop procedure, commands, lessons |
| `loop/` | workers, queue, event wait, status; settings from `env.sh` at the repository root ([`../../env.example.sh`](../../env.example.sh)) |
| `quantizer/gptq_iq.py` | GPTQ with llama.cpp rounding and image + text calibration |
| `quantizer/plan_alloc.py`, `retype_template.py` | per-tensor type plan for a byte budget, template GGUF |
| `quantizer/fileinfo.py`, `ggml_quant.py` | bytes, bpw, SHA-256 of a GGUF; ctypes binding to `libggml-base.so` |
| `quantizer/gguf_transplant.py` | optional: copy named tensors between two builds; not used for any file in `selection/` |
| `selection/` | every built file and the dev-selected picks ([`selection/README.md`](selection/README.md)) |
| `example-run/` | logs of the real run (2026-10-04) |

Used from elsewhere in the repository:

| Path | What |
|---|---|
| [`../../eval/`](../../eval/) | evaluation harness, queues, collection ([`../../eval/README.md`](../../eval/README.md)) |
| [`../../eval/results/`](../../eval/results/) | quantization scores ([`../../eval/results/README.md`](../../eval/results/README.md)) |
| [`../../protocol/`](../../protocol/) | frozen prompt and chat templates ([`../../protocol/README.md`](../../protocol/README.md)) |
| [`../../data/`](../../data/) | BRACOL split (`manifest.csv`, 419 dev / 1,266 test), JMuBEN test set, label map ([`../../data/README.md`](../../data/README.md)) |

## Method

1. **Types:** start from an Unsloth GGUF, move tensors up/down to fit the byte budget.
2. **Values:** GPTQ with llama.cpp's own rounding (`ggml_quantize_chunk`), error carried across blocks.
3. **Calibration:** text + images through the vision tower: generic COCO (*ours-general*) or BRACOL dev (*ours-task*).
4. **Selection:** on dev, by a rule written before any test run.

`llama-quantize --imatrix` can produce the same types and bytes; it lacks the error correction and image calibration.
`eval/results/summary.md` compares ours with the public files at matched bytes, which measures allocation, calibration and
rounding together. At identical types (2B, Unsloth UD-IQ2_XXS types, 768 MB) our quantizer with generic calibration
(`general3`) reached A100 dev F1 0.331 against Unsloth's 0.309, so the quantizer alone adds little; the gain at 768 MB comes
from the allocation ([`example-run/logs/4090-benchmark-LOG.md`](example-run/logs/4090-benchmark-LOG.md)).

The released base `Qwen3.5-2B-coffee-base-Q2.gguf` is `Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf` in `selection/` (SHA-256 `cb88d42f…`).

## Reproduce

The code, manifests, prompt and file hashes are here; model files are not (`.gitignore`). Inputs:

| Input | Source |
|---|---|
| BRACOL images (`$BRACOL_ROOT`) | Hugging Face `luisangelico/bracol` at a pinned revision ([`../../eval/README.md`](../../eval/README.md)) |
| BRACOL dev images by file name (`$WORK/bracol/dev512`, for ours-task) | `python data/prep_images.py bracol` |
| Public GGUFs (`$HN04B_PUBLIC`) | `eval/fetch_public.py`; revisions and SHA-256 in `eval/public_gguf_log.csv` |
| Text calibration (`$CALIB_TEXT`) | FineWeb-Edu + C4, 128 × 2048 tokens (256 × 2048 for the superseded text-only `general` builds); the run's text file is not published |
| COCO calibration (ours-general) | first 400 COCO val2017 images by file name, 512 px JPEG q90, with a generic captioning (`general2`) or scene multiple-choice (`general3`) prompt; the run's manifest and prompts are not published |
| Text-KLD check (`$WIKITEXT`) | wikitext-2 raw, test split |

The unpublished calibration assets were a private archive during the run (`example-run/logs/a100-quantization-LOG.md`).
A rebuild with your own text and COCO files follows the same procedure but will not give byte-identical GGUFs.
Testing every file as in `eval/results/`: [`../../eval/README.md`](../../eval/README.md), "Full benchmark".
