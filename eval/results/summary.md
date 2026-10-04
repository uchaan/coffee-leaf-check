# Hack-Nation 04B benchmark summary (prompt_v1)

BRACOL test (1,266 images), forced five-class macro-F1 over A to E. Every file: same llama.cpp commit,
harness, Q8_0 mmproj, prompt and images. Intervals: 95% bootstrap (1,000 resamples, seed 0).
A paired difference whose interval includes 0 is reported as no difference.

## Qwen3.5-0.8B

BF16 reference: forced_macro_f1 0.3536 [0.3401, 0.3655], coarse_macro_f1 0.7970, LM 1,557,662,624 B.

Ours files evaluated for this model: 32. Comparisons use the 26 files named in model/quantization/selection/final.csv, chosen by the registered dev rule before their test numbers were read (A100: Qwen3.5-2B and the 0.8B targets above 338 MB; 4090: the 0.8B 338 / 310 / 290 MB targets, see model/quantization/selection/lowend_0.8B_dev_screen.md).

Note: ours-task files were calibrated on the 419 BRACOL dev images (with the frozen prompt). Their dev
numbers and their tau (chosen on dev) are therefore in-sample; the test numbers below are not.

### Matched sizes: ours minus best public at the same or larger size

Comparator: the public file with the highest forced_macro_f1 among public files whose LM bytes are
at least ours and at most 10% larger. If there is none, the smallest public file larger than ours.

| ours | ours LM bytes | public comparator | public LM bytes | Δ forced_macro_f1 [paired 95% CI] | verdict | agree_bf16 ours / public (Δ) | letter_kld ours / public (Δ) |
|---|---:|---|---:|---|---|---|---|
| ours-general general2-290M-eQ3_K | 289,534,208 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ1_M | 299,403,840 | +0.1859 [+0.1678, +0.2031] | ours higher | 0.2694 / 0.2085 (+0.0608) | 0.41061 / 3.31955 (-2.90894) |
| ours-task task-290M-eQ3_K | 289,534,208 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ1_M | 299,403,840 | -0.0340 [-0.0476, -0.0217] | ours lower | 0.1754 / 0.2085 (-0.0332) | 1.99882 / 3.31955 (-1.32073) |
| ours-general general3-310M-eIQ3_XXSu | 309,698,816 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ2_S | 335,793,216 | +0.1286 [+0.1055, +0.1519] | ours higher | 0.2622 / 0.1754 (+0.0869) | 0.27626 / 12.39008 (-12.11382) |
| ours-task task-310M-eIQ3_XXSu | 309,698,816 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ2_S | 335,793,216 | -0.0206 [-0.0311, -0.0116] | ours lower | 0.1761 / 0.1754 (+0.0008) | 0.27082 / 12.39008 (-12.11925) |
| ours-general general3-338M-eQ4_K | 337,789,184 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ2_M | 350,508,096 | +0.0732 [+0.0586, +0.0875] | ours higher | 0.3436 / 0.0008 (+0.3428) | 0.08849 / 1.00483 (-0.91633) |
| ours-task task-338M-eQ4_K | 337,789,184 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ2_M | 350,508,096 | +0.1691 [+0.1475, +0.1916] | ours higher | 0.2591 / 0.0008 (+0.2583) | 0.11038 / 1.00483 (-0.89445) |
| ours-task task-UD-IQ2_M | 371,933,440 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-Q2_K_S | 388,142,656 | +0.0441 [+0.0208, +0.0655] | ours higher | 0.3870 / 0.1675 (+0.2196) | 0.15549 / 10.00028 (-9.84479) |
| ours-general general2-UD-IQ2_M | 371,933,440 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-Q2_K_S | 388,142,656 | -0.0085 [-0.0247, +0.0067] | no difference | 0.2227 / 0.1675 (+0.0553) | 0.13722 / 10.00028 (-9.86306) |
| ours-task task-UD-IQ3_XXS | 398,237,952 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ3_XS | 428,261,440 | -0.1809 [-0.2031, -0.1555] | ours lower | 0.6043 / 0.3017 (+0.3025) | 0.06736 / 0.13919 (-0.07184) |
| ours-general general2-UD-IQ3_XXS | 398,237,952 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ3_XS | 428,261,440 | -0.1363 [-0.1553, -0.1164] | ours lower | 0.3831 / 0.3017 (+0.0814) | 0.08139 / 0.13919 (-0.05781) |
| ours-general general2-UD-Q2_K_XL | 417,718,528 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ3_XS | 428,261,440 | -0.2277 [-0.2447, -0.2092] | ours lower | 0.6430 / 0.3017 (+0.3412) | 0.03482 / 0.13919 (-0.10438) |
| ours-task task-UD-Q2_K_XL | 417,718,528 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ3_XS | 428,261,440 | -0.0616 [-0.0814, -0.0395] | ours lower | 0.7899 / 0.3017 (+0.4882) | 0.06679 / 0.13919 (-0.07241) |
| ours-general general2-Q3_K_S | 440,750,336 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-Q3_K_M | 464,781,376 | -0.2566 [-0.2815, -0.2307] | ours lower | 0.7030 / 0.1769 (+0.5261) | 0.18050 / 0.32711 (-0.14661) |
| ours-task task-Q3_K_S | 440,750,336 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-Q3_K_M | 464,781,376 | +0.0520 [+0.0253, +0.0810] | ours higher | 0.8499 / 0.1769 (+0.6730) | 0.02777 / 0.32711 (-0.29934) |
| ours-general general2-Q3_K_M | 470,167,808 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ4_NL | 501,846,592 | -0.1880 [-0.2098, -0.1656] | ours lower | 0.7852 / 0.7299 (+0.0553) | 0.03178 / 0.03858 (-0.00680) |
| ours-task task-Q3_K_M | 470,167,808 | other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ4_NL | 501,846,592 | -0.1312 [-0.1575, -0.1065] | ours lower | 0.2757 / 0.7299 (-0.4542) | 0.15421 / 0.03858 (+0.11563) |
| ours-general general2-Q4_K_S | 508,104,960 | other:bartowski/Qwen_Qwen3.5-0.8B-GGUF IQ4_XS | 523,058,272 | -0.1050 [-0.1283, -0.0811] | ours lower | 0.7520 / 0.8436 (-0.0916) | 0.04576 / 0.06613 (-0.02037) |
| ours-task task-Q4_K_S | 508,104,960 | other:bartowski/Qwen_Qwen3.5-0.8B-GGUF IQ4_XS | 523,058,272 | -0.0713 [-0.0933, -0.0482] | ours lower | 0.8878 / 0.8436 (+0.0442) | 0.03154 / 0.06613 (-0.03458) |
| ours-general general2-Q4_K_M | 532,517,120 | other:mradermacher/Qwen3.5-0.8B-GGUF Q5_K_S | 563,654,976 | -0.0692 [-0.0819, -0.0564] | ours lower | 0.6524 / 0.8112 (-0.1588) | 0.02973 / 0.02065 (+0.00909) |
| ours-task task-Q4_K_M | 532,517,120 | other:mradermacher/Qwen3.5-0.8B-GGUF Q5_K_S | 563,654,976 | -0.0473 [-0.0579, -0.0346] | ours lower | 0.8120 / 0.8112 (+0.0008) | 0.01687 / 0.02065 (-0.00378) |
| ours-general general2-Q5_K_M | 590,057,728 | unsloth UD-Q5_K_XL | 606,585,088 | -0.0431 [-0.0557, -0.0305] | ours lower | 0.9028 / 0.9439 (-0.0411) | 0.00949 / 0.00986 (-0.00037) |
| ours-task task-Q5_K_M | 590,057,728 | unsloth UD-Q5_K_XL | 606,585,088 | -0.0087 [-0.0196, +0.0018] | no difference | 0.9431 / 0.9439 (-0.0008) | 0.00589 / 0.00986 (-0.00397) |
| ours-general general2-Q6_K | 639,029,504 | other:bartowski/Qwen_Qwen3.5-0.8B-GGUF Q5_K_M | 646,093,920 | +0.0100 [+0.0003, +0.0200] | ours higher | 0.9676 / 0.9581 (+0.0095) | 0.00239 / 0.00926 (-0.00687) |
| ours-task task-Q6_K | 639,029,504 | other:bartowski/Qwen_Qwen3.5-0.8B-GGUF Q5_K_M | 646,093,920 | +0.0021 [-0.0082, +0.0127] | no difference | 0.9573 / 0.9581 (-0.0008) | 0.00254 / 0.00926 (-0.00672) |
| ours-task task-1GBpkg-Q8_0 | 811,843,840 | official Q8_0 | 833,592,096 | +0.0108 [+0.0039, +0.0176] | ours higher | 0.9573 / 0.9613 (-0.0039) | 0.00279 / 0.00394 (-0.00115) |
| ours-general general-1GBpkg-Q8_0 | 811,843,840 | official Q8_0 | 833,592,096 | +0.0293 [+0.0203, +0.0376] | ours higher | 0.9613 / 0.9613 (+0.0000) | 0.00188 / 0.00394 (-0.00206) |

### Against Unsloth: ours minus the Unsloth file of the same bytes, or the smallest larger one

| ours | ours LM bytes | Unsloth file | Unsloth LM bytes | Δ forced_macro_f1 [paired 95% CI] | verdict | ours / Unsloth forced_macro_f1 |
|---|---:|---|---:|---|---|---|
| ours-general general2-290M-eQ3_K | 289,534,208 | UD-IQ2_XXS | 338,227,456 | +0.2199 [+0.2070, +0.2315] | ours higher | 0.2754 / 0.0555 |
| ours-task task-290M-eQ3_K | 289,534,208 | UD-IQ2_XXS | 338,227,456 | +0.0000 [+0.0000, +0.0000] | no difference | 0.0555 / 0.0555 |
| ours-general general3-310M-eIQ3_XXSu | 309,698,816 | UD-IQ2_XXS | 338,227,456 | +0.1492 [+0.1284, +0.1691] | ours higher | 0.2047 / 0.0555 |
| ours-task task-310M-eIQ3_XXSu | 309,698,816 | UD-IQ2_XXS | 338,227,456 | +0.0000 [+0.0000, +0.0001] | no difference | 0.0555 / 0.0555 |
| ours-general general3-338M-eQ4_K | 337,789,184 | UD-IQ2_XXS | 338,227,456 | +0.0861 [+0.0709, +0.1009] | ours higher | 0.1416 / 0.0555 |
| ours-task task-338M-eQ4_K | 337,789,184 | UD-IQ2_XXS | 338,227,456 | +0.1819 [+0.1631, +0.2038] | ours higher | 0.2374 / 0.0555 |
| ours-general general2-UD-IQ2_M | 371,933,440 | UD-IQ2_M | 371,933,440 | +0.0657 [+0.0511, +0.0807] | ours higher | 0.1341 / 0.0684 |
| ours-task task-UD-IQ2_M | 371,933,440 | UD-IQ2_M | 371,933,440 | +0.1182 [+0.0972, +0.1386] | ours higher | 0.1866 / 0.0684 |
| ours-general general2-UD-IQ3_XXS | 398,237,952 | UD-IQ3_XXS | 398,237,952 | +0.2385 [+0.2220, +0.2538] | ours higher | 0.2940 / 0.0555 |
| ours-task task-UD-IQ3_XXS | 398,237,952 | UD-IQ3_XXS | 398,237,952 | +0.1939 [+0.1720, +0.2166] | ours higher | 0.2495 / 0.0555 |
| ours-general general2-UD-Q2_K_XL | 417,718,528 | UD-Q2_K_XL | 417,718,528 | +0.1704 [+0.1610, +0.1804] | ours higher | 0.2026 / 0.0322 |
| ours-task task-UD-Q2_K_XL | 417,718,528 | UD-Q2_K_XL | 417,718,528 | +0.3365 [+0.3225, +0.3496] | ours higher | 0.3688 / 0.0322 |
| ours-general general2-Q3_K_S | 440,750,336 | Q3_K_S | 440,750,336 | -0.0501 [-0.0707, -0.0320] | ours lower | 0.0564 / 0.1065 |
| ours-task task-Q3_K_S | 440,750,336 | Q3_K_S | 440,750,336 | +0.2585 [+0.2362, +0.2775] | ours higher | 0.3650 / 0.1065 |
| ours-general general2-Q3_K_M | 470,167,808 | Q3_K_M | 470,167,808 | +0.0066 [-0.0203, +0.0306] | no difference | 0.2199 / 0.2133 |
| ours-task task-Q3_K_M | 470,167,808 | Q3_K_M | 470,167,808 | +0.0634 [+0.0350, +0.0919] | ours higher | 0.2767 / 0.2133 |
| ours-general general2-Q4_K_S | 508,104,960 | Q4_K_S | 508,104,960 | -0.0180 [-0.0303, -0.0070] | ours lower | 0.2686 / 0.2867 |
| ours-task task-Q4_K_S | 508,104,960 | Q4_K_S | 508,104,960 | +0.0157 [+0.0032, +0.0283] | ours higher | 0.3023 / 0.2867 |
| ours-general general2-Q4_K_M | 532,517,120 | Q4_K_M | 532,517,120 | -0.0318 [-0.0442, -0.0181] | ours lower | 0.2638 / 0.2956 |
| ours-task task-Q4_K_M | 532,517,120 | Q4_K_M | 532,517,120 | -0.0099 [-0.0228, +0.0016] | no difference | 0.2858 / 0.2956 |
| ours-general general2-Q5_K_M | 590,057,728 | Q5_K_M | 590,057,728 | -0.0160 [-0.0291, -0.0021] | ours lower | 0.3032 / 0.3192 |
| ours-task task-Q5_K_M | 590,057,728 | Q5_K_M | 590,057,728 | +0.0184 [+0.0047, +0.0311] | ours higher | 0.3375 / 0.3192 |
| ours-general general2-Q6_K | 639,029,504 | Q6_K | 639,029,504 | +0.0104 [+0.0033, +0.0192] | ours higher | 0.3519 / 0.3414 |
| ours-task task-Q6_K | 639,029,504 | Q6_K | 639,029,504 | +0.0026 [-0.0036, +0.0091] | no difference | 0.3440 / 0.3414 |
| ours-general general-1GBpkg-Q8_0 | 811,843,840 | Q8_0 | 811,843,840 | +0.0301 [+0.0207, +0.0388] | ours higher | 0.3650 / 0.3349 |
| ours-task task-1GBpkg-Q8_0 | 811,843,840 | Q8_0 | 811,843,840 | +0.0116 [+0.0039, +0.0193] | ours higher | 0.3465 / 0.3349 |

### Under-1 GB package (LM + Q8_0 mmproj 113,564,384 B < 1,000,000,000 B)

| source | best file | LM bytes | total bytes | forced_macro_f1 [95% CI] | coarse_macro_f1 | agree_bf16 | letter_kld | tau | answered_acc / coverage | CPU s/img (proxy) | peak RSS MB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|
| ours-general | general-1GBpkg-Q8_0 | 811,843,840 | 925,408,224 | 0.3650 [0.3514, 0.3782] | 0.8186 | 0.9613 | 0.00188 | 0.6288 | 0.0000 / 0.0024 | 1.02 | 2251 |
| ours-task | task-UD-Q2_K_XL | 417,718,528 | 531,282,912 | 0.3688 [0.3529, 0.3836] | 0.8190 | 0.7899 | 0.06679 | 0.4417 | 0.0000 / 0.0008 | 1.45 | 1321 |
| official | Q8_0 | 833,592,096 | 947,156,480 | 0.3357 [0.3209, 0.3491] | 0.7651 | 0.9613 | 0.00394 | 0.3329 | 0.3702 / 0.7062 | 0.96 | 2242 |
| other:bartowski/Qwen_Qwen3.5-0.8B-GGUF | IQ4_XS | 523,058,272 | 636,622,656 | 0.3737 [0.3506, 0.3968] | 0.7052 | 0.8436 | 0.06613 | 0.2247 | 0.3656 / 0.9376 | 0.93 | 1564 |
| other:lmstudio-community/Qwen3.5-0.8B-GGUF | Q6_K | 629,743,584 | 743,307,968 | 0.3365 [0.3209, 0.3520] | 0.7512 | 0.8799 | 0.00858 | 0.3679 | 0.4833 / 0.2844 | 1.33 | 1801 |
| other:mradermacher/Qwen3.5-0.8B-GGUF | IQ4_XS | 488,114,496 | 601,678,880 | 0.3599 [0.3367, 0.3817] | 0.6650 | 0.5190 | 0.06616 | 0.3601 | 0.7325 / 0.1801 | 1.02 | 1534 |
| other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-IQ3_XS | 428,261,440 | 541,825,824 | 0.4303 [0.4145, 0.4450] | 0.6417 | 0.3017 | 0.13919 | 0.3107 | 0.8847 / 0.3562 | 1.46 | 1247 |
| unsloth | UD-Q5_K_XL | 606,585,088 | 720,149,472 | 0.3463 [0.3302, 0.3638] | 0.7538 | 0.9439 | 0.00986 | 0.5514 | 0.0000 / 0.0008 | 1.08 | 1764 |

Best ours package: ours-task task-UD-Q2_K_XL. Paired difference against each public package:

- vs official Q8_0: +0.0331 [+0.0163, +0.0501], ours higher; total bytes 531,282,912 vs 947,156,480.
- vs other:bartowski/Qwen_Qwen3.5-0.8B-GGUF IQ4_XS: -0.0049 [-0.0295, +0.0214], no difference; total bytes 531,282,912 vs 636,622,656.
- vs other:lmstudio-community/Qwen3.5-0.8B-GGUF Q6_K: +0.0323 [+0.0144, +0.0504], ours higher; total bytes 531,282,912 vs 743,307,968.
- vs other:mradermacher/Qwen3.5-0.8B-GGUF IQ4_XS: +0.0089 [-0.0123, +0.0317], no difference; total bytes 531,282,912 vs 601,678,880.
- vs other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ3_XS: -0.0616 [-0.0814, -0.0395], ours lower; total bytes 531,282,912 vs 541,825,824.
- vs unsloth UD-Q5_K_XL: +0.0225 [+0.0038, +0.0423], ours higher; total bytes 531,282,912 vs 720,149,472.

### Confusion matrix of the best under-1 GB package (other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ3_XS), BRACOL test

Rows: true label. Columns: forced prediction (A to E).

| true \ pred | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| A healthy | 200 | 4 | 0 | 0 | 0 |
| B rust | 3 | 395 | 0 | 1 | 0 |
| C cercospora | 2 | 107 | 0 | 2 | 0 |
| D phoma | 1 | 143 | 0 | 114 | 3 |
| E leaf miner | 3 | 264 | 4 | 19 | 1 |

Per-class F1: A 0.969, B 0.602, C 0.000, D 0.574, E 0.007.

## Qwen3.5-2B

BF16 reference: forced_macro_f1 0.5815 [0.5547, 0.6063], coarse_macro_f1 0.8176, LM 3,897,387,936 B.

Ours files evaluated for this model: 29. Comparisons use the 12 files named in model/quantization/selection/final.csv, chosen by the registered dev rule before their test numbers were read (A100: Qwen3.5-2B and the 0.8B targets above 338 MB; 4090: the 0.8B 338 / 310 / 290 MB targets, see model/quantization/selection/lowend_0.8B_dev_screen.md).

Note: ours-task files were calibrated on the 419 BRACOL dev images (with the frozen prompt). Their dev
numbers and their tau (chosen on dev) are therefore in-sample; the test numbers below are not.

### Matched sizes: ours minus best public at the same or larger size

Comparator: the public file with the highest forced_macro_f1 among public files whose LM bytes are
at least ours and at most 10% larger. If there is none, the smallest public file larger than ours.

| ours | ours LM bytes | public comparator | public LM bytes | Δ forced_macro_f1 [paired 95% CI] | verdict | agree_bf16 ours / public (Δ) | letter_kld ours / public (Δ) |
|---|---:|---|---:|---|---|---|---|
| ours-general general3-1GBpkg-eQ2_K | 637,772,032 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-IQ1_S | 639,796,768 | +0.2913 [+0.2647, +0.3161] | ours higher | 0.5806 / 0.2322 (+0.3483) | 0.49778 / 12.56831 (-12.07053) |
| ours-task task-1GBpkg-eQ2_K | 637,772,032 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-IQ1_S | 639,796,768 | +0.5190 [+0.4935, +0.5442] | ours higher | 0.8057 / 0.2322 (+0.5735) | 0.17013 / 12.56831 (-12.39819) |
| ours-general general3-700M-eIQ3_XXS | 699,662,592 | unsloth UD-IQ2_XXS | 768,270,592 | +0.1630 [+0.1386, +0.1878] | ours higher | 0.4976 / 0.0000 (+0.4976) | 0.40074 / 0.99904 (-0.59830) |
| ours-task task-700M-eIQ3_XXS | 699,662,592 | unsloth UD-IQ2_XXS | 768,270,592 | +0.1533 [+0.1257, +0.1809] | ours higher | 0.7038 / 0.0000 (+0.7038) | 0.09958 / 0.99904 (-0.89946) |
| ours-task task-768M-v2 | 767,965,440 | unsloth UD-IQ2_XXS | 768,270,592 | +0.2566 [+0.2292, +0.2841] | ours higher | 0.8942 / 0.0000 (+0.8942) | 0.09461 / 0.99904 (-0.90443) |
| ours-general general2-768M-v2 | 767,965,440 | unsloth UD-IQ2_XXS | 768,270,592 | +0.1807 [+0.1533, +0.2065] | ours higher | 0.7536 / 0.0000 (+0.7536) | 0.20841 / 0.99904 (-0.79064) |
| ours-task task-UD-IQ2_M | 859,857,152 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-Q2_K | 915,458,592 | +0.0506 [+0.0220, +0.0811] | ours higher | 0.8942 / 0.6461 (+0.2480) | 0.06816 / 0.28845 (-0.22028) |
| ours-general general3-UD-IQ2_M | 859,857,152 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-Q2_K | 915,458,592 | -0.1417 [-0.1689, -0.1140] | ours lower | 0.4937 / 0.6461 (-0.1524) | 0.38618 / 0.28845 (+0.09774) |
| ours-task task-UD-IQ3_XXS | 931,823,872 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-IQ3_XS | 997,121,568 | -0.0184 [-0.0475, +0.0113] | no difference | 0.8325 / 0.4992 (+0.3333) | 0.10357 / 0.53798 (-0.43441) |
| ours-general general2-UD-IQ3_XXS | 931,823,872 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-IQ3_XS | 997,121,568 | -0.0639 [-0.0942, -0.0359] | ours lower | 0.7804 / 0.4992 (+0.2812) | 0.10441 / 0.53798 (-0.43357) |
| ours-task task-967M-v2b | 966,172,928 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-IQ3_XS | 997,121,568 | -0.3073 [-0.3368, -0.2780] | ours lower | 0.5032 / 0.4992 (+0.0039) | 0.30276 / 0.53798 (-0.23522) |
| ours-general general3-967M-v2b | 966,172,928 | other:mradermacher/Qwen3.5-2B-i1-GGUF i1-IQ3_XS | 997,121,568 | -0.2099 [-0.2414, -0.1798] | ours lower | 0.5750 / 0.4992 (+0.0758) | 0.18597 / 0.53798 (-0.35201) |

### Against Unsloth: ours minus the Unsloth file of the same bytes, or the smallest larger one

| ours | ours LM bytes | Unsloth file | Unsloth LM bytes | Δ forced_macro_f1 [paired 95% CI] | verdict | ours / Unsloth forced_macro_f1 |
|---|---:|---|---:|---|---|---|
| ours-general general3-1GBpkg-eQ2_K | 637,772,032 | UD-IQ2_XXS | 768,270,592 | +0.0369 [+0.0107, +0.0637] | ours higher | 0.3498 / 0.3129 |
| ours-task task-1GBpkg-eQ2_K | 637,772,032 | UD-IQ2_XXS | 768,270,592 | +0.2646 [+0.2386, +0.2905] | ours higher | 0.5776 / 0.3129 |
| ours-general general3-700M-eIQ3_XXS | 699,662,592 | UD-IQ2_XXS | 768,270,592 | +0.1630 [+0.1386, +0.1878] | ours higher | 0.4760 / 0.3129 |
| ours-task task-700M-eIQ3_XXS | 699,662,592 | UD-IQ2_XXS | 768,270,592 | +0.1533 [+0.1257, +0.1809] | ours higher | 0.4662 / 0.3129 |
| ours-general general2-768M-v2 | 767,965,440 | UD-IQ2_XXS | 768,270,592 | +0.1807 [+0.1533, +0.2065] | ours higher | 0.4936 / 0.3129 |
| ours-task task-768M-v2 | 767,965,440 | UD-IQ2_XXS | 768,270,592 | +0.2566 [+0.2292, +0.2841] | ours higher | 0.5696 / 0.3129 |
| ours-general general3-UD-IQ2_M | 859,857,152 | UD-IQ2_M | 859,857,152 | -0.0472 [-0.0737, -0.0209] | ours lower | 0.3653 / 0.4125 |
| ours-task task-UD-IQ2_M | 859,857,152 | UD-IQ2_M | 859,857,152 | +0.1451 [+0.1205, +0.1707] | ours higher | 0.5576 / 0.4125 |
| ours-general general2-UD-IQ3_XXS | 931,823,872 | UD-IQ3_XXS | 931,823,872 | +0.1836 [+0.1573, +0.2087] | ours higher | 0.5026 / 0.3190 |
| ours-task task-UD-IQ3_XXS | 931,823,872 | UD-IQ3_XXS | 931,823,872 | +0.2291 [+0.2018, +0.2519] | ours higher | 0.5481 / 0.3190 |
| ours-general general3-967M-v2b | 966,172,928 | UD-Q2_K_XL | 966,533,376 | +0.0542 [+0.0308, +0.0772] | ours higher | 0.3565 / 0.3024 |
| ours-task task-967M-v2b | 966,172,928 | UD-Q2_K_XL | 966,533,376 | -0.0432 [-0.0658, -0.0217] | ours lower | 0.2592 / 0.3024 |

### Under-1 GB package (LM + Q8_0 mmproj 361,518,656 B < 1,000,000,000 B)

| source | best file | LM bytes | total bytes | forced_macro_f1 [95% CI] | coarse_macro_f1 | agree_bf16 | letter_kld | tau | answered_acc / coverage | CPU s/img (proxy) | peak RSS MB |
|---|---|---:|---:|---|---|---|---|---|---|---|---|
| ours-general | general3-1GBpkg-eQ2_K | 637,772,032 | 999,290,688 | 0.3498 [0.3245, 0.3742] | 0.5121 | 0.5806 | 0.49778 | 0.8424 | 0.6512 / 0.0340 | 3.69 | 1670 |
| ours-task | task-1GBpkg-eQ2_K | 637,772,032 | 999,290,688 | 0.5776 [0.5521, 0.6006] | 0.7899 | 0.8057 | 0.17013 | 0.9128 | 0.7797 / 0.0466 | 3.19 | 1675 |
| unsloth (smallest; does not fit) | UD-IQ2_XXS | 768,270,592 | 1,129,789,248 | 0.3129 [0.2923, 0.3323] | 0.5265 | 0.0000 | 0.99904 | – | – / 0.0000 | 3.08 | 1994 |

Best ours package: ours-task task-1GBpkg-eQ2_K. Paired difference against each public package:

- vs unsloth UD-IQ2_XXS (smallest Unsloth file; its package is over 1 GB): +0.2646 [+0.2386, +0.2905], ours higher; total bytes 999,290,688 vs 1,129,789,248.

### Confusion matrix of the best under-1 GB package (ours-task task-1GBpkg-eQ2_K), BRACOL test

Rows: true label. Columns: forced prediction (A to E).

| true \ pred | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| A healthy | 204 | 0 | 0 | 0 | 0 |
| B rust | 9 | 174 | 196 | 11 | 9 |
| C cercospora | 10 | 1 | 87 | 13 | 0 |
| D phoma | 7 | 0 | 19 | 140 | 95 |
| E leaf miner | 8 | 0 | 123 | 48 | 112 |

Per-class F1: A 0.923, B 0.606, C 0.325, D 0.592, E 0.442.

## Optional extra test set: JMuBEN + JMuBEN2 (field photos, Kenya)

Distinct images only (the archives repeat files byte for byte), first 600 per class by SHA-256:
A 62, B 600, C 322, D 600, E 600 (2,184; one damaged JPEG dropped). Same harness, prompt and mmproj; tau from each file's BRACOL dev run.

| model | source | file | LM bytes | forced_macro_f1 [95% CI] | minus BF16 [paired 95% CI] | coarse_macro_f1 | agree_bf16 | letter_kld | answered_acc / coverage |
|---|---|---|---:|---|---|---|---|---|---|
| Qwen3.5-0.8B | bf16 | BF16 | 1,557,662,624 | 0.3395 [0.3202, 0.3559] |  | 0.7256 | 1.0000 | 0.00000 | 0.3737 / 0.7596 |
| Qwen3.5-0.8B | ours-general | general2-290M-eQ3_K | 289,534,208 | 0.1162 [0.0992, 0.1338] | -0.2232 [-0.2434, -0.2024] | 0.1937 | 0.4418 | 0.41462 | 0.1641 / 0.4547 |
| Qwen3.5-0.8B | ours-task | task-290M-eQ3_K | 289,534,208 | 0.0163 [0.0124, 0.0212] | -0.3231 [-0.3392, -0.3058] | 0.0272 | 0.0412 | 7.82112 | 0.0643 / 0.1424 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-IQ1_M | 299,403,840 | 0.0309 [0.0232, 0.0379] | -0.3085 [-0.3248, -0.2908] | 0.0572 | 0.0353 | 5.76885 | 0.0086 / 0.1603 |
| Qwen3.5-0.8B | ours-general | general3-310M-eIQ3_XXSu | 309,698,816 | 0.0199 [0.0150, 0.0250] | -0.3196 [-0.3372, -0.3011] | 0.0561 | 0.1900 | 0.49956 | 0.0087 / 0.2097 |
| Qwen3.5-0.8B | ours-task | task-310M-eIQ3_XXSu | 309,698,816 | 0.0111 [0.0084, 0.0138] | -0.3284 [-0.3437, -0.3117] | 0.0197 | 0.0375 | 0.41667 | 0.0291 / 0.1886 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-IQ2_S | 335,793,216 | 0.0350 [0.0290, 0.0412] | -0.3044 [-0.3215, -0.2877] | 0.1168 | 0.0023 | 13.43319 | 0.0000 / 0.0064 |
| Qwen3.5-0.8B | ours-general | general3-338M-eQ4_K | 337,789,184 | 0.0938 [0.0886, 0.0984] | -0.2456 [-0.2617, -0.2285] | 0.2144 | 0.5293 | 0.10595 | 0.5073 / 0.2193 |
| Qwen3.5-0.8B | ours-task | task-338M-eQ4_K | 337,789,184 | 0.0980 [0.0875, 0.1089] | -0.2415 [-0.2590, -0.2230] | 0.2297 | 0.5069 | 0.13065 | 0.3498 / 0.5760 |
| Qwen3.5-0.8B | unsloth | UD-IQ2_XXS | 338,227,456 | 0.0110 [0.0084, 0.0138] | -0.3284 [-0.3438, -0.3117] | 0.0184 | 0.0375 | 16.50092 | 0.0284 / 1.0000 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-IQ2_M | 350,508,096 | 0.1108 [0.1045, 0.1168] | -0.2286 [-0.2470, -0.2084] | 0.2584 | 0.0046 | 0.91038 | 0.0000 / 0.0023 |
| Qwen3.5-0.8B | ours-task | task-UD-IQ2_M | 371,933,440 | 0.1336 [0.1217, 0.1452] | -0.2058 [-0.2218, -0.1885] | 0.2995 | 0.4799 | 0.11967 | 0.4375 / 0.0073 |
| Qwen3.5-0.8B | ours-general | general2-UD-IQ2_M | 371,933,440 | 0.0312 [0.0246, 0.0378] | -0.3082 [-0.3251, -0.2908] | 0.0521 | 0.0682 | 0.25455 | 0.0880 / 0.0989 |
| Qwen3.5-0.8B | unsloth | UD-IQ2_M | 371,933,440 | 0.0882 [0.0830, 0.0934] | -0.2513 [-0.2696, -0.2317] | 0.2770 | 0.0009 | 0.48025 | 0.0000 / 0.0261 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-Q2_K_S | 388,142,656 | 0.0919 [0.0844, 0.0995] | -0.2476 [-0.2632, -0.2305] | 0.1532 | 0.2592 | 9.19235 | 0.5841 / 0.0517 |
| Qwen3.5-0.8B | ours-task | task-UD-IQ3_XXS | 398,237,952 | 0.1327 [0.1230, 0.1418] | -0.2067 [-0.2220, -0.1879] | 0.3093 | 0.6511 | 0.08204 | 0.4730 / 0.5517 |
| Qwen3.5-0.8B | ours-general | general2-UD-IQ3_XXS | 398,237,952 | 0.2660 [0.2475, 0.2816] | -0.0734 [-0.0935, -0.0533] | 0.4556 | 0.5169 | 0.07195 | 0.3898 / 0.7248 |
| Qwen3.5-0.8B | unsloth | UD-IQ3_XXS | 398,237,952 | 0.0929 [0.0846, 0.1012] | -0.2466 [-0.2632, -0.2288] | 0.1548 | 0.1690 | 0.27665 | – / 0.0000 |
| Qwen3.5-0.8B | ours-general | general2-UD-Q2_K_XL | 417,718,528 | 0.1351 [0.1258, 0.1438] | -0.2044 [-0.2204, -0.1859] | 0.3688 | 0.6859 | 0.03652 | – / 0.0000 |
| Qwen3.5-0.8B | ours-task | task-UD-Q2_K_XL | 417,718,528 | 0.3263 [0.3021, 0.3474] | -0.0131 [-0.0387, +0.0109] | 0.6306 | 0.7262 | 0.08343 | 0.3200 / 0.0114 |
| Qwen3.5-0.8B | unsloth | UD-Q2_K_XL | 417,718,528 | 0.0638 [0.0568, 0.0711] | -0.2757 [-0.2939, -0.2563] | 0.2936 | 0.4739 | 0.21610 | 0.2014 / 0.5206 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-IQ3_XS | 428,261,440 | 0.2447 [0.2258, 0.2638] | -0.0948 [-0.1153, -0.0742] | 0.3576 | 0.4332 | 0.13472 | 0.5212 / 0.3558 |
| Qwen3.5-0.8B | ours-general | general2-Q3_K_S | 440,750,336 | 0.0832 [0.0745, 0.0923] | -0.2563 [-0.2741, -0.2367] | 0.3408 | 0.6053 | 0.17895 | 0.0769 / 0.0238 |
| Qwen3.5-0.8B | ours-task | task-Q3_K_S | 440,750,336 | 0.3034 [0.2842, 0.3208] | -0.0360 [-0.0504, -0.0219] | 0.6016 | 0.7550 | 0.02480 | – / 0.0000 |
| Qwen3.5-0.8B | unsloth | Q3_K_S | 440,750,336 | 0.0840 [0.0740, 0.0934] | -0.2554 [-0.2723, -0.2366] | 0.2425 | 0.2592 | 0.17949 | – / 0.0000 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-Q3_K_M | 464,781,376 | 0.1786 [0.1597, 0.1986] | -0.1609 [-0.1817, -0.1414] | 0.3987 | 0.1163 | 0.30687 | 0.4815 / 0.0247 |
| Qwen3.5-0.8B | ours-general | general2-Q3_K_M | 470,167,808 | 0.2060 [0.1890, 0.2244] | -0.1335 [-0.1500, -0.1155] | 0.5230 | 0.7234 | 0.03759 | 0.1576 / 0.0929 |
| Qwen3.5-0.8B | ours-task | task-Q3_K_M | 470,167,808 | 0.1095 [0.0971, 0.1250] | -0.2299 [-0.2490, -0.2082] | 0.1797 | 0.4899 | 0.22298 | 0.3387 / 0.7935 |
| Qwen3.5-0.8B | unsloth | Q3_K_M | 470,167,808 | 0.2248 [0.1980, 0.2501] | -0.1147 [-0.1442, -0.0847] | 0.5060 | 0.0925 | 0.82339 | 0.2800 / 0.0114 |
| Qwen3.5-0.8B | unsloth | UD-Q3_K_XL | 492,216,576 | 0.0994 [0.0889, 0.1094] | -0.2400 [-0.2571, -0.2209] | 0.1716 | 0.0513 | 0.26126 | 0.1848 / 0.0421 |
| Qwen3.5-0.8B | unsloth | IQ4_XS | 492,605,696 | 0.1966 [0.1742, 0.2209] | -0.1429 [-0.1691, -0.1189] | 0.3882 | 0.1003 | 0.19847 | 0.5505 / 0.0499 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-i1-GGUF | i1-IQ4_NL | 501,846,592 | 0.3036 [0.2829, 0.3250] | -0.0358 [-0.0560, -0.0162] | 0.5913 | 0.7386 | 0.04131 | 0.4769 / 0.3168 |
| Qwen3.5-0.8B | unsloth | IQ4_NL | 506,859,776 | 0.1011 [0.0932, 0.1101] | -0.2383 [-0.2585, -0.2171] | 0.2773 | 0.0156 | 0.20744 | 0.5000 / 0.0037 |
| Qwen3.5-0.8B | unsloth | Q4_0 | 507,154,688 | 0.1023 [0.0924, 0.1131] | -0.2371 [-0.2554, -0.2168] | 0.3623 | 0.5316 | 0.20938 | 0.4286 / 0.0096 |
| Qwen3.5-0.8B | ours-general | general2-Q4_K_S | 508,104,960 | 0.1963 [0.1824, 0.2112] | -0.1432 [-0.1584, -0.1283] | 0.4742 | 0.7569 | 0.04559 | 0.6117 / 0.0472 |
| Qwen3.5-0.8B | ours-task | task-Q4_K_S | 508,104,960 | 0.2536 [0.2365, 0.2712] | -0.0859 [-0.1023, -0.0699] | 0.6001 | 0.8741 | 0.02985 | 0.4276 / 0.1328 |
| Qwen3.5-0.8B | unsloth | Q4_K_S | 508,104,960 | 0.2039 [0.1905, 0.2190] | -0.1355 [-0.1502, -0.1199] | 0.4786 | 0.7170 | 0.02716 | 0.4571 / 0.0321 |
| Qwen3.5-0.8B | other:bartowski/Qwen_Qwen3.5-0.8B-GGUF | IQ4_XS | 523,058,272 | 0.4227 [0.3996, 0.4423] | +0.0833 [+0.0629, +0.1052] | 0.7361 | 0.6488 | 0.09692 | 0.3596 / 0.8837 |
| Qwen3.5-0.8B | ours-general | general2-Q4_K_M | 532,517,120 | 0.1592 [0.1478, 0.1713] | -0.1803 [-0.1949, -0.1643] | 0.4040 | 0.6644 | 0.03418 | 0.5848 / 0.0783 |
| Qwen3.5-0.8B | ours-task | task-Q4_K_M | 532,517,120 | 0.2476 [0.2301, 0.2665] | -0.0919 [-0.1070, -0.0769] | 0.5519 | 0.8187 | 0.01622 | 0.6683 / 0.0925 |
| Qwen3.5-0.8B | unsloth | Q4_K_M | 532,517,120 | 0.2541 [0.2369, 0.2711] | -0.0853 [-0.1003, -0.0706] | 0.6018 | 0.8462 | 0.02069 | 0.5337 / 0.2239 |
| Qwen3.5-0.8B | unsloth | Q4_1 | 535,171,328 | 0.1787 [0.1648, 0.1924] | -0.1607 [-0.1761, -0.1438] | 0.3243 | 0.5412 | 0.07800 | 0.8529 / 0.0156 |
| Qwen3.5-0.8B | unsloth | UD-Q4_K_XL | 558,772,480 | 0.1996 [0.1873, 0.2138] | -0.1398 [-0.1537, -0.1249] | 0.5110 | 0.7587 | 0.02723 | 0.3189 / 0.0847 |
| Qwen3.5-0.8B | official | Q4_0 | 563,036,064 | 0.2256 [0.2115, 0.2400] | -0.1139 [-0.1283, -0.0989] | 0.5475 | 0.7500 | 0.05840 | 0.4118 / 0.0156 |
| Qwen3.5-0.8B | other:mradermacher/Qwen3.5-0.8B-GGUF | Q5_K_S | 563,654,976 | 0.2380 [0.2192, 0.2559] | -0.1014 [-0.1177, -0.0857] | 0.5213 | 0.7564 | 0.02512 | 0.6618 / 0.1571 |
| Qwen3.5-0.8B | unsloth | Q5_K_S | 568,889,600 | 0.2356 [0.2192, 0.2524] | -0.1039 [-0.1191, -0.0880] | 0.5494 | 0.8439 | 0.01704 | 0.5736 / 0.2147 |
| Qwen3.5-0.8B | ours-general | general2-Q5_K_M | 590,057,728 | 0.2856 [0.2659, 0.3042] | -0.0539 [-0.0696, -0.0387] | 0.6491 | 0.9084 | 0.00582 | 0.0000 / 0.0009 |
| Qwen3.5-0.8B | ours-task | task-Q5_K_M | 590,057,728 | 0.3108 [0.2906, 0.3290] | -0.0286 [-0.0405, -0.0179] | 0.6762 | 0.9217 | 0.00445 | 0.4343 / 0.6305 |
| Qwen3.5-0.8B | unsloth | Q5_K_M | 590,057,728 | 0.3217 [0.3016, 0.3407] | -0.0178 [-0.0331, -0.0020] | 0.7156 | 0.8636 | 0.01065 | 0.3515 / 0.8141 |
| Qwen3.5-0.8B | unsloth | UD-Q5_K_XL | 606,585,088 | 0.3019 [0.2816, 0.3213] | -0.0376 [-0.0539, -0.0209] | 0.6730 | 0.8961 | 0.00675 | 0.6667 / 0.0027 |
| Qwen3.5-0.8B | ours-general | general2-Q6_K | 639,029,504 | 0.3238 [0.3031, 0.3409] | -0.0157 [-0.0241, -0.0075] | 0.7018 | 0.9441 | 0.00342 | 0.5714 / 0.0032 |
| Qwen3.5-0.8B | ours-task | task-Q6_K | 639,029,504 | 0.3215 [0.3016, 0.3390] | -0.0180 [-0.0308, -0.0057] | 0.6952 | 0.9492 | 0.00228 | 0.4233 / 0.5733 |
| Qwen3.5-0.8B | unsloth | Q6_K | 639,029,504 | 0.3107 [0.2898, 0.3274] | -0.0288 [-0.0430, -0.0148] | 0.6783 | 0.9354 | 0.00278 | 0.3811 / 0.7569 |
| Qwen3.5-0.8B | other:bartowski/Qwen_Qwen3.5-0.8B-GGUF | Q5_K_M | 646,093,920 | 0.3403 [0.3170, 0.3583] | +0.0008 [-0.0100, +0.0113] | 0.7269 | 0.9135 | 0.01229 | 0.3651 / 0.8265 |
| Qwen3.5-0.8B | unsloth | UD-Q6_K_XL | 771,092,736 | 0.3093 [0.2891, 0.3269] | -0.0301 [-0.0444, -0.0159] | 0.6898 | 0.9313 | 0.00294 | 0.4670 / 0.2637 |
| Qwen3.5-0.8B | ours-task | task-1GBpkg-Q8_0 | 811,843,840 | 0.3221 [0.3015, 0.3396] | -0.0174 [-0.0298, -0.0046] | 0.6920 | 0.9464 | 0.00283 | 0.0000 / 0.0014 |
| Qwen3.5-0.8B | ours-general | general-1GBpkg-Q8_0 | 811,843,840 | 0.3075 [0.2866, 0.3252] | -0.0320 [-0.0460, -0.0192] | 0.6723 | 0.9332 | 0.00210 | 0.0000 / 0.0018 |
| Qwen3.5-0.8B | unsloth | Q8_0 | 811,843,840 | 0.3100 [0.2894, 0.3272] | -0.0295 [-0.0427, -0.0153] | 0.6813 | 0.9441 | 0.00391 | 0.3648 / 0.8109 |
| Qwen3.5-0.8B | official | Q8_0 | 833,592,096 | 0.3061 [0.2867, 0.3235] | -0.0333 [-0.0470, -0.0194] | 0.6785 | 0.9428 | 0.00414 | 0.4177 / 0.6039 |
| Qwen3.5-2B | bf16 | BF16 | 3,897,387,936 | 0.4744 [0.4421, 0.5033] |  | 0.6529 | 1.0000 | 0.00000 | 0.6591 / 0.3842 |
| Qwen3.5-2B | ours-general | general3-1GBpkg-eQ2_K | 637,772,032 | 0.3257 [0.2944, 0.3556] | -0.1488 [-0.1808, -0.1187] | 0.5329 | 0.6896 | 0.36218 | 0.6471 / 0.0545 |
| Qwen3.5-2B | ours-task | task-1GBpkg-eQ2_K | 637,772,032 | 0.4847 [0.4542, 0.5147] | +0.0102 [-0.0058, +0.0264] | 0.6971 | 0.7761 | 0.19927 | 0.6615 / 0.1488 |
| Qwen3.5-2B | other:mradermacher/Qwen3.5-2B-i1-GGUF | i1-IQ1_S | 639,796,768 | 0.0124 [0.0093, 0.0155] | -0.4621 [-0.4910, -0.4291] | 0.0193 | 0.0114 | 17.49632 | 0.0296 / 0.7431 |
| Qwen3.5-2B | ours-general | general3-700M-eIQ3_XXS | 699,662,592 | 0.4440 [0.4188, 0.4694] | -0.0304 [-0.0584, -0.0002] | 0.6551 | 0.5733 | 0.36752 | 0.5507 / 0.0316 |
| Qwen3.5-2B | ours-task | task-700M-eIQ3_XXS | 699,662,592 | 0.3357 [0.3153, 0.3562] | -0.1388 [-0.1687, -0.1045] | 0.5122 | 0.6932 | 0.15422 | 0.7101 / 0.0632 |
| Qwen3.5-2B | ours-task | task-768M-v2 | 767,965,440 | 0.5230 [0.4949, 0.5454] | +0.0486 [+0.0270, +0.0716] | 0.7393 | 0.9025 | 0.09533 | 0.7052 / 0.3961 |
| Qwen3.5-2B | ours-general | general2-768M-v2 | 767,965,440 | 0.4767 [0.4530, 0.4987] | +0.0022 [-0.0315, +0.0361] | 0.6578 | 0.6589 | 0.32071 | 0.7542 / 0.2198 |
| Qwen3.5-2B | unsloth | UD-IQ2_XXS | 768,270,592 | 0.1734 [0.1617, 0.1856] | -0.3010 [-0.3311, -0.2678] | 0.4179 | 0.0087 | 1.45138 | – / 0.0000 |
| Qwen3.5-2B | ours-task | task-UD-IQ2_M | 859,857,152 | 0.5043 [0.4824, 0.5274] | +0.0298 [+0.0012, +0.0588] | 0.7392 | 0.9038 | 0.09498 | 0.7703 / 0.1016 |
| Qwen3.5-2B | ours-general | general3-UD-IQ2_M | 859,857,152 | 0.1189 [0.1071, 0.1328] | -0.3556 [-0.3875, -0.3189] | 0.1984 | 0.4153 | 0.63839 | 0.3576 / 0.5275 |
| Qwen3.5-2B | unsloth | UD-IQ2_M | 859,857,152 | 0.2796 [0.2590, 0.3014] | -0.1948 [-0.2221, -0.1655] | 0.4197 | 0.5636 | 0.32941 | 0.7292 / 0.0220 |
| Qwen3.5-2B | other:mradermacher/Qwen3.5-2B-i1-GGUF | i1-Q2_K | 915,458,592 | 0.4089 [0.3870, 0.4317] | -0.0656 [-0.0987, -0.0318] | 0.6918 | 0.6277 | 0.31904 | 0.6623 / 0.1383 |
| Qwen3.5-2B | ours-task | task-UD-IQ3_XXS | 931,823,872 | 0.4214 [0.3868, 0.4507] | -0.0530 [-0.0720, -0.0341] | 0.6031 | 0.8420 | 0.11996 | 0.6261 / 0.5082 |
| Qwen3.5-2B | ours-general | general2-UD-IQ3_XXS | 931,823,872 | 0.5240 [0.5011, 0.5464] | +0.0495 [+0.0196, +0.0827] | 0.7308 | 0.7962 | 0.11735 | 0.6957 / 0.3837 |
| Qwen3.5-2B | unsloth | UD-IQ3_XXS | 931,823,872 | 0.2735 [0.2548, 0.2915] | -0.2010 [-0.2285, -0.1703] | 0.3956 | 0.5536 | 0.33502 | 0.6241 / 0.2509 |
| Qwen3.5-2B | ours-task | task-967M-v2b | 966,172,928 | 0.2320 [0.2109, 0.2526] | -0.2424 [-0.2775, -0.2040] | 0.4361 | 0.2143 | 0.77541 | 0.2544 / 0.0522 |
| Qwen3.5-2B | ours-general | general3-967M-v2b | 966,172,928 | 0.3465 [0.3232, 0.3712] | -0.1280 [-0.1608, -0.0978] | 0.6052 | 0.3915 | 0.51701 | 0.2966 / 0.0540 |
| Qwen3.5-2B | unsloth | UD-Q2_K_XL | 966,533,376 | 0.3279 [0.3057, 0.3495] | -0.1465 [-0.1822, -0.1103] | 0.4725 | 0.4386 | 0.53988 | 0.9474 / 0.0174 |
| Qwen3.5-2B | other:mradermacher/Qwen3.5-2B-i1-GGUF | i1-IQ3_XS | 997,121,568 | 0.4089 [0.3851, 0.4295] | -0.0656 [-0.0939, -0.0349] | 0.7431 | 0.6168 | 0.47974 | 0.7041 / 0.1223 |
| Qwen3.5-2B | unsloth | Q4_K_M | 1,280,835,840 | 0.4373 [0.4032, 0.4694] | -0.0372 [-0.0605, -0.0164] | 0.6082 | 0.8594 | 0.07789 | 0.6685 / 0.1644 |

Rows: BF16, every Unsloth and official file, the ours files used in the comparisons, and the public files that
appear as comparators in the BRACOL tables; every file is in results.csv (dataset JMuBEN).

## Checks

- Prompt SHA-256 over all runs: 15f21d3e54e0863233def4ba50d7d4566fe4889e6600b5d3babeffb77461e954. llama.cpp commit: 9a7570587ce908b0073a0458877205b80627f393.
- Every listed file loaded and completed.
- A to F are the same single tokens (ids 32 to 37) in every file.
- Qwen3.5-0.8B: 48 files put on average less than 90% of the first-token probability on A to F (the rest on other tokens, such as "To" or "Here"); their answers come from what the grammar leaves. Lowest: 0.001. List: low_letter_mass.csv.
- Qwen3.5-2B: 20 files put on average less than 90% of the first-token probability on A to F (the rest on other tokens, such as "To" or "Here"); their answers come from what the grammar leaves. Lowest: 0.001. List: low_letter_mass.csv.
- Qwen3.5-0.8B: quantized files above BF16 on test (paired interval above 0): ours-general general-1GBpkg-Q8_0 +0.0114 [+0.0041, +0.0184]; ours-task task-UD-Q2_K_XL +0.0152 [+0.0006, +0.0308]; other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ3_XS +0.0768 [+0.0572, +0.0966]; other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-IQ4_NL +0.0543 [+0.0359, +0.0732]; other:mradermacher/Qwen3.5-0.8B-i1-GGUF i1-Q4_K_S +0.0406 [+0.0215, +0.0596].
- Qwen3.5-2B: no quantized file is above BF16 on test with a paired interval above 0.

