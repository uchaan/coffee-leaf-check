# data/: agent guide

Image lists, labels and splits used by every stage. Images are not here; rows name the dataset file and its SHA-256.
File table, row counts and commands: [`README.md`](README.md).

## Run (from the repository root, after `source env.sh`)

```bash
python data/prep_images.py bracol                       # 512 px images for harness, calibration, training
python data/prep_images.py jmuben $JMUBEN_ROOT
python data/make_manifest.py --bracol-root $BRACOL_ROOT # rebuilds data/manifest.csv
python data/make_jmuben_split.py $WORK/jmuben           # rebuilds the JMuBEN grouped split
```

## Rules

- Do not edit the CSVs by hand. A rebuilt manifest must match the committed one (`git diff` empty); a change of
  split invalidates every number in `RESULTS.md` and `eval/results/`.
- BRACOL `test` rows (1,266) are never used for training, calibration or selection; `dev` (419) is for every choice.
- JMuBEN held-out originals (240 `test` rows of `jmuben_grouped_split.csv`) are never used for training.
- Never commit images (`images/` is ignored) or anything from `$WORK`.
- `prep_images.py` imports `preprocess` from `eval/harness.py`; keep the preprocessing in one place.
