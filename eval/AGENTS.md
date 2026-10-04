# eval/: agent guide

Benchmark harness, job queues and collection; one llama.cpp build (`9a7570587`) and the same inputs for every GGUF.
Protocol, pinned facts, metric definitions and full commands: [`README.md`](README.md). Scores: `results/`.

## Run (from the repository root, after `source env.sh`)

```bash
python eval/harness.py run --model Qwen3.5-2B --lm X.gguf --mmproj M.gguf --prompt protocol/prompt_v1.txt --split dev --gpu 0
python eval/make_jobs.py --a100-dir $WORK/eval --out $WORK/jobs.csv      # then one eval/gpu_queue.py per GPU
python eval/collect.py --prompt prompt_v1 --jobs $WORK/jobs.csv --out eval/results/
```

Defaults: `--manifest data/manifest.csv`, chat template `protocol/chat_templates/<model>.jinja`, file lists
`model/quantization/selection/files.csv` and `final.csv`, public log `eval/public_gguf_log.csv`.

## Rules

- Use `--split dev` for any choice. Run test only after `model/quantization/selection/final.csv` is written, through
  the queues (`collect.py` reads only queue-tagged runs).
- Do not change the request settings, probability mode or metrics in `harness.py`: every number in `results/` and
  `RESULTS.md` depends on them. A change means rerunning everything.
- `results/` is written only by `collect.py`; do not edit its CSVs or `summary.md` by hand (paths in prose excepted).
- Never commit GGUFs, images, run folders (`$HN04B_RUNS`) or `env.sh`.
