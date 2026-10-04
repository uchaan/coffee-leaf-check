# Lessons from the 2026-10-04 run (Qwen3.5-2B / 0.8B, BRACOL)

Each item is a measured result from `model/quantization/example-run/logs/`. Treat them as strong priors for this model family and task,
and as hypotheses for anything else.

## Reading the numbers

- **Text KLD is not task quality for a VLM.** Text-only calibration beat Unsloth on text KLD at every size
  (2B 768 MB: 0.419 vs 0.711) yet lost on the task (dev F1 0.06–0.30 vs 0.31–0.42). Use dev F1 to decide; text KLD
  only catches broken builds.
- **Watch the letter mass.** A file that puts 11–20 % of the first-token probability outside A–F is broken for the task
  even if its F1 looks close. `eval/results/low_letter_mass.csv` lists such files.
- **Agreement with BF16 and letter KLD** separate "different but right" from "degraded": prefer the file closer to BF16
  when F1 ties.
- Dev has 419 images; treat differences under ~0.03 F1 as noise (a rough estimate, not measured). Paired intervals on test decide claims, not dev.

## What moved the result

1. **Image + text calibration** (`--calib-images`): images go through the HF vision tower and are merged into the LM
   inputs, so the Hessians see image tokens. This turned task-failing files into the best files. (HF's bf16 Conv3d
   patch embed is slow; `gptq_iq.py` computes it as the equivalent matmul.)
2. **Body type matters more than the embedding at the low end.** IQ2_XXS bodies collapsed on the task even with task
   calibration (637–768 MB: F1 0.08–0.33). Keeping the UD-IQ2_M body and shrinking the tied embedding
   (IQ2_XXS / Q2_K / Q3_K) recovered it (`-v2` templates).
3. **Tied embedding as output head** (`--tied-embd-gptq`): quantize `token_embd` with H from the final hidden states,
   since it is also the output layer.
4. **ours-task vs ours-general.** Task images in calibration gave the largest gain at the smallest sizes (2B 637 MB:
   test F1 0.58 vs 0.35). Keep both variants: ours-general shows the method works without task data and is the
   honest basis for selecting the allocation.

## Where it did not win

- Above ~0.93 GB (2B) public files were equal or better: with enough bits the rounding method matters less than the
  type choice Unsloth already made. Do not spend the queue there unless the goal requires it.
- 0.8B at most sizes: the model itself is weak on the task (BF16 F1 0.35); differences are mostly noise.

## Process mistakes worth avoiding

- A candidate prompt was frozen before it was measured on dev; it halved F1 and every build calibrated with it had to
  be redone. Measure prompt candidates on dev with BF16 first.
- Test lines were visible in a tailed log before selection; it was logged as a deviation. Filter test lines out of
  anything you read before `final.csv` exists.
- Uploading every candidate to the model hub wasted time and quota; upload only final picks.
