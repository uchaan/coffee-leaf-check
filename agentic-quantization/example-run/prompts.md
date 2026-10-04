# Example requests

The standing rules are in `../CLAUDE.md` and the procedure is in the `gguf-quant-loop` skill, so one request is
enough to start the loop; later requests only steer it.

## Start the loop

> Quantize Qwen3.5-2B for the coffee-leaf task. Targets: LM + Q8_0 mmproj under 1 GB, 700 MB, and Unsloth
> UD-IQ2_XXS's size. Compare against every public GGUF at those sizes on BRACOL dev. Use GPUs 0–3, deadline 13:00.
> Do the setup, start the workers and run the loop until every target has a pick; then select, test and report.

## Steer while it runs

> The 0.8B model matters too: add its 338 MB target to the loop, after the 2B targets.

> Text KLD is better than Unsloth but dev F1 is worse. Find out why before queuing anything else: design the
> cheapest experiment that separates the explanations.

> Stop at 12:45. Write the selection rule now and fill final.csv from dev.

## Check in

> Status? Conclusion first, one table per target: best ours-general, best ours-task, the public file, dev F1.

## Resume in a new session

> Resume the loop: read the tail of LOG.md and loop/status.sh, re-queue what is missing and continue.

## Two servers

> You coordinate both sessions. The benchmark server runs every file through benchmark/harness.py and runs the
> 0.8B low-end loop itself; you run the 2B targets. Move model files through shared storage, never git.
