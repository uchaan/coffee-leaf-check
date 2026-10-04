# Data

Image lists, labels and splits for every stage (evaluation, calibration, fine-tuning, baselines). The images themselves
are not in the repository: each row names the file in the original dataset and its SHA-256, and the scripts check that
hash. Dataset sources and the split rules: [`../eval/README.md`](../eval/README.md) ("Data: BRACOL", "Data: JMuBEN").

Rule: BRACOL **test** images are never used for training, calibration or selection. Choices use dev; test is read once.

## Manifests and splits

Columns: `path` (relative to the dataset root), `sha256` (of the image file), `split`, `label_letter` (A to E).

| File | Rows | Split | File SHA-256 | Used by |
|---|---:|---|---|---|
| `manifest.csv` | 1,685 | BRACOL: 419 `dev` / 1,266 `test` | `2adcdeb1…` | harness (default `--manifest`), ours-task calibration, `train_lora.py`, baselines |
| `manifest_jmuben.csv` | 2,184 | JMuBEN: all `test` | `064accce…` | `eval/gpu_queue.py --dataset jmuben` |
| `jmuben_distinct_manifest.csv` | 3,756 | none; `in_eval` = 1 for the 2,184 rows of `manifest_jmuben.csv`, `source_path` = file in the archives | `5abd0225…` | `prep_images.py jmuben`, `make_jmuben_split.py` |
| `jmuben_grouped_split.csv` | 799 | one image per original photo: 559 `dev` (training) / 240 `test` | `1e5c9d74…` | `eval_lora.sh` (JMuBEN held-out), baselines |
| `jmuben_train_originals.csv` | 559 | the `dev` rows above, training only | `fe0588f7…` | `train_lora.py --extra`, `make_yolo_dirs.py hackjm` |
| `manifest_jmuben_heldout240.csv` | 240 | the 240 held-out originals with archive paths, all `test` | `e9a1607f…` | harness runs on the raw JMuBEN folder |
| `bracol_paper_sized_split.csv` | 1,685 | BRACOL re-split to the paper's size: 1,179 `dev` / 506 `test` | `db6dd4f9…` | `cnn_baseline.py --split s2`, `explain_demo.py` |

## Labels and counts

| File | What |
|---|---|
| `label_map.csv` | BRACOL `predominant_stress` code → letter: 0 Healthy A, 2 Rust B, 4 Cercospora C, 3 Phoma D, 1 Leaf miner E; code 5 excluded |
| `class_counts.txt` | BRACOL per class: total, dev, test (62 code-5 images excluded) |
| `jmuben_class_counts.txt` | JMuBEN per class: files, distinct images, undecodable, used |

## Scripts (run from the repository root, after `source env.sh`)

| Script | What | Command |
|---|---|---|
| `make_manifest.py` | rebuild `manifest.csv` from the BRACOL leaf set | `python data/make_manifest.py --bracol-root $BRACOL_ROOT` |
| `make_manifest_jmuben.py` | rebuild `manifest_jmuben.csv` (first 600 distinct images per class by SHA-256) | `python data/make_manifest_jmuben.py --root $JMUBEN_ROOT --out data/manifest_jmuben.csv` |
| `prep_images.py` | 512 px JPEG q90 copies with the harness preprocessing (imports `eval/harness.py`) | `python data/prep_images.py bracol` / `python data/prep_images.py jmuben $JMUBEN_ROOT` |
| `make_jmuben_split.py` | rebuild `jmuben_grouped_split.csv` and `jmuben_train_originals.csv` (rotation/flip-invariant grouping) | `python data/make_jmuben_split.py $WORK/jmuben` |

`prep_images.py bracol` writes `$HN04B_CACHE/<sha256>.jpg` (all 1,685) and `$WORK/bracol/dev512/<file name>` (the 419 dev
images, for calibration and training); `jmuben` writes `$WORK/jmuben/images/<sha256>.jpg`. The manifest and split scripts write
into this folder (`make_manifest_jmuben.py` via `--out`), so a rebuild shows up in `git diff`.
