# Commands

All paths are relative to the repository root; run from there after `source env.sh`.

## Loop control

| Command | What |
|---|---|
| `model/quantization/loop/start_workers.sh 0 1 2 3` | one worker per GPU (PIDs in `$WORK/workers.pid`) |
| `model/quantization/loop/enqueue.sh TAG TEMPLATE NSEQ "EXTRA"` | add a build job; TAG names `$WORK/gguf/ours/TAG.gguf` (put `0.8B` in the tag for the 0.8B model) |
| `model/quantization/loop/wait_event.sh [PATTERN] [TIMEOUT_S]` | block until new events match (default `^(DEV\|FAIL)`), print all new lines, exit; run in the background |
| `model/quantization/loop/status.sh [N]` | queue head, worker liveness, GPU memory, last N events |
| `model/quantization/loop/stop_workers.sh` / `--now` | stop after the current job / kill now by PID |

Event lines in `$WORK/logs/events.log`:
```
QUEUED <tag> template=<file> <time>
BUILD <tag> exit=<code> <seconds>s bytes=<out> template=<bytes>
FAIL <tag> build output missing or bytes != template
CHECK <tag> kld=<mean KLD vs BF16> same_top1=<%>
DEV [<tag>] dev n=419 forced_acc=… forced_macro_f1=… coarse_macro_f1=… agree_bf16=… letter_kld=… per_class=A:…,E:…
```

## Templates

```bash
export PYTHONPATH=$LLAMA_CPP_DIR/gguf-py:$PWD/model/quantization/quantizer
# BASE EMB BUDGET UP DOWN OUT: BASE's types, token_embd=EMB, then tensors toward UP (if under budget) or DOWN (if over)
python model/quantization/quantizer/plan_alloc.py $WORK/unsloth/Qwen3.5-2B-UD-IQ2_M.gguf Q2_K 638481344 \
    $WORK/unsloth/Qwen3.5-2B-UD-IQ3_XXS.gguf $WORK/unsloth/Qwen3.5-2B-UD-IQ2_XXS.gguf $WORK/plan-637-eQ2_K.txt
python model/quantization/quantizer/retype_template.py $WORK/unsloth/Qwen3.5-2B-UD-IQ2_M.gguf $WORK/tpl-637-eQ2_K.gguf $WORK/plan-637-eQ2_K.txt
```
`plan_alloc.py` prints the planned bytes; check them against the budget before queuing. It cannot go below DOWN's
types. Move order (up): full-attention q/k/v/o, ffn_down, ffn_gate, ffn_up, linear-attention qkv/gate/out.

## Build arguments (EXTRA)

| Variant | EXTRA |
|---|---|
| ours-general, `general2` | `--tied-embd-gptq --calib-images $WORK/coco/generic_manifest.csv --image-dir $WORK/coco/gen512 --prompt-file <generic captioning prompt>` |
| ours-general, `general3` | `--tied-embd-gptq --calib-images $WORK/coco/generic_manifest.csv --image-dir $WORK/coco/gen512 --prompt-file $WORK/coco/prompt_generic_mcq.txt` |
| ours-task | `--tied-embd-gptq --calib-images data/manifest.csv --image-dir $WORK/bracol/dev512 --prompt-file protocol/prompt_v1.txt` |

Relative paths resolve against the repository root (`loop/build.sh` runs there). The COCO files are the example run's
private assets (`README.md`, "Reproduce"): a manifest in the `data/manifest.csv` format with every row `split=dev`, the
512 px JPEGs by file name, and prompt files in the protocol format. Make your own the same way.

`gptq_iq.py` reads only `split == dev` rows of the manifest and opens each image by its file name in `--image-dir`. NSEQ 128 × seqlen 2048 of `$CALIB_TEXT` is the default.
A100: 2B 13–15 min and ~10 GB per build, 0.8B 5–11 min. Add `--offload-layers` on small GPUs.

## Files and numbers

```bash
python model/quantization/quantizer/fileinfo.py FILE.gguf          # name,bytes,bpw,tensors,sha256
python $LLAMA_CPP_DIR/gguf-py/gguf/scripts/gguf_dump.py FILE.gguf | head -40   # tensor types
```

## Test and collect (after selection only)

```bash
# EVAL = one folder with every model/quantization/selection/files.csv path: <model>/<name>.gguf, <model>/<model>-BF16.gguf,
#        <model>/mmproj-<model>-Q8_0.gguf (links to $WORK/gguf/ours/TAG.gguf etc. are fine);
#        make_jobs.py turns only the bf16 / ours-general / ours-task rows into jobs
python eval/make_jobs.py --a100-dir $EVAL --public-dir $HN04B_PUBLIC --out $WORK/jobs.csv
python eval/gpu_queue.py --models Qwen3.5-2B --gpu 0 --port 8090 --prompt protocol/prompt_v1.txt \
    --jobs $WORK/jobs.csv --a100-dir $EVAL                  # one per GPU; BF16 runs first, the rest wait for it
touch $HN04B_RUNS/STOP                                      # queues exit when nothing runnable is left
python eval/collect.py --prompt prompt_v1 --jobs $WORK/jobs.csv --out eval/results/
```
`collect.py` reads only runs tagged `<model>__<source>__<file stem>__<prompt stem>`, which the queues write. Details,
the CPU proxy and JMuBEN: `eval/README.md`.
