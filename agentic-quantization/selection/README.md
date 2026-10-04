# Selection

| File | What |
|---|---|
| `files.csv` | every file handed to the benchmark: 61 quantized (32 ours-general, 29 ours-task), BF16 and canonical Q8_0 mmproj per model, and the fine-tuned package (adapter, vision projector, translator); bytes, bpw, SHA-256, calibration, recipe |
| `final.csv` | the dev-selected picks, one ours-general and one ours-task file per (model, target), written before any test number |
| `lowend_0.8B_dev_screen.md` | the 4090 dev screen and selection for the 0.8B 338 / 310 / 290 MB targets |

File names: `<model>-ours-<calibration>-<target>.gguf`.

| Infix | Calibration |
|---|---|
| `general` | text only, FineWeb-Edu + C4 (superseded: fails on the task) |
| `general2` | text + 400 COCO val2017 images, generic captioning prompt |
| `general3` | text + 400 COCO val2017 images, generic scene multiple-choice prompt |
| `task` | text + 419 BRACOL dev images with the frozen prompt (`../protocol/prompt_v1.txt`); dev numbers are in-sample |

`general`, `general2` and `general3` files all have variant `ours-general`. The `<target>` part is either a public quant
name (`UD-IQ2_M`, `Q4_K_M`, ...: that Unsloth file's tensor types at the same bytes) or a size (`700M`, `1GBpkg` = LM + Q8_0
mmproj under 1 GB) with a planned allocation: `-v2` / `-v2b` / `-v3` are later allocation rounds, `-e<TYPE>` is the
tied-embedding type, a trailing `u` means the UD-Q3_K_XL file was the UP limit of `plan_alloc.py`.

Notes:
- Rows citing `gptq_iq.py sha 856f10f5` are the superseded text-only builds; that code version is not included.
  The other builds used `b6601b44` (see `../example-run/README.md`).
- `dev_*_a100` columns in `final.csv`: A100 harness, except the 0.8B 338 / 310 / 290 MB rows, which come from the 4090
  dev screen (`lowend_0.8B_dev_screen.md`).
- The released base `Qwen3.5-2B-coffee-base-Q2.gguf` is `Qwen3.5-2B/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf` (637,772,032 B, SHA-256 `cb88d42f…`).
