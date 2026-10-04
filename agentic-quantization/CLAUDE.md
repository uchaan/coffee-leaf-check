# Agentic GGUF quantization: instructions for the agent

You run an autonomous quantization loop. The owner gives you a goal (model, byte targets, the public files to beat);
you plan the builds, queue them, wait for the results, decide the next builds from those results, and repeat until
every target has a pick or the goal is shown to be out of reach. Then you select on dev, read test once and report.
You do not wait for the owner between iterations; you stop for the owner only at the points listed under "Ask first".

**Use the `gguf-quant-loop` skill** (`.claude/skills/gguf-quant-loop/`): it has the step-by-step loop, the commands
and what earlier runs learned. `README.md` explains the method; `example-run/logs/` is a complete real run.

## The loop in one screen

```
setup (once)      loop/env.sh, llama.cpp at the pinned commit, BF16 + Q8_0 mmproj, BF16 dev reference run
start workers     loop/start_workers.sh 0 1 2 3          one worker per GPU: build -> byte check -> text KLD -> dev run
┌─> plan          quantizer/plan_alloc.py + retype_template.py -> template GGUF per (target, variant)
│   pre-register  LOG.md: what you build, why, and which dev result makes you do what next
│   queue         loop/enqueue.sh TAG TEMPLATE NSEQ "EXTRA"
│   wait          loop/wait_event.sh   (run it in the background; you are woken when a DEV or FAIL line arrives)
│   read          the DEV / CHECK / FAIL lines; loop/status.sh for the whole picture
└── decide        next builds (back to plan), or the target is done
select            write the rule, fill selection/final.csv from dev numbers only
test + report     benchmark/make_jobs.py -> benchmark/gpu_queue.py (one per GPU) -> benchmark/collect.py -> results/
```

## Ground rules

1. **Write it down before you measure it.** Hypotheses, reading rules and the selection rule go into `LOG.md` with a
   timestamp from `date`, before the run that tests them. A later change of rule is recorded as a deviation.
2. **Dev for every choice, test once.** Prompt, allocation, calibration and selection use BRACOL **dev** only.
   Test numbers are read after `selection/final.csv` is written. If you see a test number early, log it as a deviation.
3. **Same bytes, same setup.** Compare files at matched bytes with the same llama.cpp commit, harness, prompt, mmproj
   and images, with paired bootstrap intervals. Anything else is not a result.
4. **Verify, don't assume.** Output bytes = template bytes (the worker checks), types as planned, text KLD sane. A flag
   you passed is not proof it was applied.
5. **Exact numbers.** Bytes from `stat`, times from `date`, estimates labelled as estimates.
6. **Keep every GPU busy** while work remains; the queue should never run dry while you think.
7. **Stop processes by PID** (`loop/stop_workers.sh`), never by a pattern match that can hit other jobs or your shell.
8. **Model files never go into git.** Upload only final picks, and only when the owner says so.

## Ask first (the owner's calls)

Prompt wording, adding or dropping targets, a new track (e.g. fine-tuning), using more GPUs than allowed, uploads, and
anything after the deadline. Bring a recommendation with the dev numbers behind it.

## Reporting

Conclusion first, then one table: what was measured, on which split, against which file, at what bytes, with the
interval. Report dead ends as plainly as wins.

## Files you own

| File | What |
|---|---|
| `LOG.md` | your work log: append-only, one timestamped section per decision or finding (create it at setup) |
| `$WORK/queue.txt` | build queue (`TAG\|TEMPLATE\|NSEQ\|EXTRA`), filled with `loop/enqueue.sh` |
| `$WORK/logs/events.log` | one line per QUEUED / BUILD / CHECK / DEV / FAIL, written by the loop scripts |
| `selection/files.csv`, `selection/final.csv` | every built file (bytes, bpw, SHA-256, recipe); the dev-selected picks |
