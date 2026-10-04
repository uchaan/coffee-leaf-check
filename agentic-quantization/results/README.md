# Results (frozen prompt `protocol/prompt_v1.txt`)

Written by `benchmark/collect.py` (RTX 4090 harness, llama.cpp `9a7570587`). 163 files: 2 BF16, 61 ours, 100 public;
each on BRACOL dev, BRACOL test and JMuBEN test (489 rows in `results.csv`).

| File | What |
|---|---|
| `summary.md` | matched-size differences, under-1 GB package table, confusion matrix, JMuBEN table, checks |
| `results.csv` | one row per model x source x file x dataset x split (metrics: `benchmark/README.md`) |
| `f1_<model>.png` | forced macro-F1 on BRACOL test vs language-model bytes, 95% bootstrap intervals, 1 GB package line |
| `kld_<model>.png` | letter KLD vs BF16 on BRACOL test vs language-model bytes (log scale) |
| `low_letter_mass.csv` | files that put on average under 90% of the first-token probability on A to F |

Every number is recomputed by `benchmark/collect.py` from the per-image files. Metric definitions: `benchmark/README.md`.

Not included here:
- `per_image/` (per-image probabilities and run metadata that `collect.py` also writes): left out for size.
- General benchmarks (MMStar, AI2D, MMLU-Redux) quoted in `../../RESULTS.md`: measured with `benchmark/general_bench.py`
  and summarised with `benchmark/general_collect.py`; their output (`results/general/`) is not in this repository.
- Fine-tuned package numbers: `../../RESULTS.md` (code in `../../fine-tuning/`).
