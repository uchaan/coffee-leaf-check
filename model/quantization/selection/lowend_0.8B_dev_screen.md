# Qwen3.5-0.8B low end: dev screen and registered-rule selection (4090)

BRACOL dev (n=419), frozen prompt, 4090 harness (llama-server defaults + --cache-ram 0), ref = 4090 BF16 0.8B dev rows.
Dev only: no test or JMuBEN number of these candidates was computed before the freeze. Builds: A100 gptq_iq.py b6601b44ea30,
templates from plan_alloc (BASE Unsloth UD-IQ2_M, DOWN UD-IQ2_XXS, UP UD-IQ3_XXS, or UD-Q3_K_XL for the "u" templates).
BF16 0.8B dev forced_macro_f1 0.3627; one fixed answer A for every image gives 0.0559.

| target | template | variant | candidate | source | dev forced_macro_f1 | dev letter_kld | dev agree_bf16 |
|---|---|---|---|---|---:|---:|---:|
| 290 | 4090:290M-eQ3_K | general2 | 0.8B-290M-eQ3_K-general2 | 4090 build | 0.2831 | 0.4056 | 0.294 |
| 290 | 4090:290M-eIQ3_XXS | general3 | 0.8B-290M-eIQ3_XXS-general3 | 4090 build | 0.2309 | 0.4604 | 0.289 |
| 290 | 4090:290M-eQ2_K | general2 | 0.8B-290M-eQ2_K-general2 | 4090 build | 0.1907 | 0.4548 | 0.258 |
| 290 | 4090:290M-eIQ2_XXSu | general3 | 0.8B-290M-eIQ2_XXSu-general3 | 4090 build | 0.1865 | 0.1906 | 0.652 |
| 290 | a100:290M-v2 | general3 | Qwen3.5-0.8B-ours-general3-290M-v2 | A100 candidate | 0.1094 | 0.2931 | 0.370 |
| 290 | 4090:290M-eIQ3_XXS | general2 | 0.8B-290M-eIQ3_XXS-general2 | 4090 build | 0.1052 | 0.5223 | 0.150 |
| 290 | 4090:290M-eIQ2_XXSu | general2 | 0.8B-290M-eIQ2_XXSu-general2 | 4090 build | 0.0958 | 0.5007 | 0.148 |
| 290 | 4090:290M-eQ2_Ku | general3 | 0.8B-290M-eQ2_Ku-general3 | 4090 build | 0.0958 | 1.1292 | 0.148 |
| 290 | 4090:290M-eQ2_Ku | general2 | 0.8B-290M-eQ2_Ku-general2 | 4090 build | 0.0958 | 1.2307 | 0.148 |
| 290 | handoff:290M | general | Qwen3.5-0.8B-ours-general-290M | files.csv | 0.0559 | 0.6512 | 0.169 |
| 290 | 4090:290M-eQ2_K | general3 | 0.8B-290M-eQ2_K-general3 | 4090 build | 0.0559 | 0.7788 | 0.169 |
| 290 | 4090:290M-eQ3_K | general3 | 0.8B-290M-eQ3_K-general3 | 4090 build | 0.0545 | 9.0239 | 0.198 |
| 290 | a100:290M-v2 | general2 | Qwen3.5-0.8B-ours-general2-290M-v2 | A100 candidate | 0.0316 | 1.3777 | 0.678 |
| 290 | handoff:290M | task | Qwen3.5-0.8B-ours-task-290M | files.csv | 0.0877 | 0.1707 | 0.186 |
| 290 | a100:290M-v2 | task | Qwen3.5-0.8B-ours-task-290M-v2 | A100 candidate | 0.0559 | 13.7391 | 0.169 |
| 310 | 4090:310M-eIQ3_XXSu | general3 | 0.8B-310M-eIQ3_XXSu-general3 | 4090 build | 0.2116 | 0.2844 | 0.272 |
| 310 | 4090:310M-eQ4_K | general3 | 0.8B-310M-eQ4_K-general3 | 4090 build | 0.2010 | 11.9313 | 0.191 |
| 310 | a100:310M-v2 | general3 | Qwen3.5-0.8B-ours-general3-310M-v2 | A100 candidate | 0.1850 | 0.4696 | 0.158 |
| 310 | 4090:310M-eQ4_K | general2 | 0.8B-310M-eQ4_K-general2 | 4090 build | 0.1388 | 1.8704 | 0.000 |
| 310 | 4090:310M-eIQ4_XS | general3 | 0.8B-310M-eIQ4_XS-general3 | 4090 build | 0.1185 | 0.8444 | 0.169 |
| 310 | 4090:310M-eIQ3_XXSu | general2 | 0.8B-310M-eIQ3_XXSu-general2 | 4090 build | 0.1063 | 0.0999 | 0.148 |
| 310 | 4090:310M-eQ2_Ku | general3 | 0.8B-310M-eQ2_Ku-general3 | 4090 build | 0.1018 | 0.3231 | 0.031 |
| 310 | handoff:310M | general | Qwen3.5-0.8B-ours-general-310M | files.csv | 0.1009 | 8.8388 | 0.208 |
| 310 | 4090:310M-eIQ3_XXS | general3 | 0.8B-310M-eIQ3_XXS-general3 | 4090 build | 0.0688 | 0.3869 | 0.203 |
| 310 | 4090:310M-eIQ3_XXS | general2 | 0.8B-310M-eIQ3_XXS-general2 | 4090 build | 0.0662 | 0.1985 | 0.184 |
| 310 | a100:310M-v2 | general2 | Qwen3.5-0.8B-ours-general2-310M-v2 | A100 candidate | 0.0559 | 0.4899 | 0.169 |
| 310 | 4090:310M-eQ2_Ku | general2 | 0.8B-310M-eQ2_Ku-general2 | 4090 build | 0.0559 | 0.8218 | 0.169 |
| 310 | 4090:310M-eIQ4_XS | general2 | 0.8B-310M-eIQ4_XS-general2 | 4090 build | 0.0559 | 8.0125 | 0.012 |
| 310 | handoff:310M | task | Qwen3.5-0.8B-ours-task-310M | files.csv | 0.0559 | 0.8753 | 0.169 |
| 310 | a100:310M-v2 | task | Qwen3.5-0.8B-ours-task-310M-v2 | A100 candidate | 0.0519 | 0.6381 | 0.155 |
| 338 | 4090:338M-eQ4_K | general3 | 0.8B-338M-eQ4_K-general3 | 4090 build | 0.1424 | 0.0895 | 0.341 |
| 338 | a100:338M-v2 | general2 | Qwen3.5-0.8B-ours-general2-338M-v2 | A100 candidate | 0.1421 | 0.1366 | 0.193 |
| 338 | 4090:338M-eIQ3_XXSu | general3 | 0.8B-338M-eIQ3_XXSu-general3 | 4090 build | 0.1180 | 0.1499 | 0.212 |
| 338 | a100:338M-v2 | general3 | Qwen3.5-0.8B-ours-general3-338M-v2 | A100 candidate | 0.1117 | 0.1227 | 0.305 |
| 338 | 4090:338M-eQ2_Ku | general3 | 0.8B-338M-eQ2_Ku-general3 | 4090 build | 0.1095 | 0.1585 | 0.573 |
| 338 | 4090:338M-eQ2_Ku | general2 | 0.8B-338M-eQ2_Ku-general2 | 4090 build | 0.0958 | 0.4451 | 0.148 |
| 338 | 4090:338M-eQ4_K | general2 | 0.8B-338M-eQ4_K-general2 | 4090 build | 0.0881 | 0.1580 | 0.685 |
| 338 | handoff:UD-IQ2_XXS | general | Qwen3.5-0.8B-ours-general-UD-IQ2_XXS | files.csv | 0.0805 | 12.0104 | 0.203 |
| 338 | 4090:338M-eQ5_K | general3 | 0.8B-338M-eQ5_K-general3 | 4090 build | 0.0601 | 0.8511 | 0.067 |
| 338 | 4090:338M-eIQ4_XS | general3 | 0.8B-338M-eIQ4_XS-general3 | 4090 build | 0.0559 | 0.4585 | 0.169 |
| 338 | 4090:338M-eIQ4_XS | general2 | 0.8B-338M-eIQ4_XS-general2 | 4090 build | 0.0559 | 3.2079 | 0.169 |
| 338 | 4090:338M-eQ5_K | general2 | 0.8B-338M-eQ5_K-general2 | 4090 build | 0.0559 | 15.5900 | 0.169 |
| 338 | 4090:338M-eIQ3_XXSu | general2 | 0.8B-338M-eIQ3_XXSu-general2 | 4090 build | 0.0481 | 0.0773 | 0.644 |
| 338 | handoff:UD-IQ2_XXS | task | Qwen3.5-0.8B-ours-task-UD-IQ2_XXS | files.csv | 0.0708 | 0.3128 | 0.267 |
| 338 | a100:338M-v2 | task | Qwen3.5-0.8B-ours-task-338M-v2 | A100 candidate | 0.0432 | 0.1616 | 0.683 |

- 338 MB: ours-general = 0.8B-338M-eQ4_K-general3 (dev F1 0.1424, letter_kld 0.0895); ours-task = to build on template 4090:338M-eQ4_K.
- 310 MB: ours-general = 0.8B-310M-eIQ3_XXSu-general3 (dev F1 0.2116, letter_kld 0.2844); ours-task = to build on template 4090:310M-eIQ3_XXSu.
- 290 MB: ours-general = 0.8B-290M-eQ3_K-general2 (dev F1 0.2831, letter_kld 0.4056); ours-task = to build on template 4090:290M-eQ3_K.

Build log (bytes = template bytes for every build):

```
2026-10-04 01:44:36 BUILD 0.8B-338M-eIQ4_XS-general2 OK rc=0 250s bytes=337920256 template=337920256
2026-10-04 01:44:40 BUILD 0.8B-338M-eQ5_K-general2 OK rc=0 254s bytes=338227456 template=338227456
2026-10-04 01:44:53 BUILD 0.8B-338M-eQ4_K-general2 OK rc=0 267s bytes=337789184 template=337789184
2026-10-04 01:45:15 BUILD 0.8B-310M-eQ4_K-general2 OK rc=0 289s bytes=309604608 template=309604608
2026-10-04 01:48:57 BUILD 0.8B-310M-eIQ4_XS-general2 OK rc=0 261s bytes=309625088 template=309625088
2026-10-04 01:49:20 BUILD 0.8B-310M-eIQ3_XXS-general2 OK rc=0 280s bytes=309693696 template=309693696
2026-10-04 01:49:49 BUILD 0.8B-290M-eQ3_K-general2 OK rc=0 296s bytes=289534208 template=289534208
2026-10-04 01:50:27 BUILD 0.8B-290M-eIQ3_XXS-general2 OK rc=0 312s bytes=289345792 template=289345792
2026-10-04 01:53:15 BUILD 0.8B-290M-eQ2_K-general2 OK rc=0 258s bytes=289694976 template=289694976
2026-10-04 01:53:53 BUILD 0.8B-338M-eQ5_K-general3 OK rc=0 273s bytes=338227456 template=338227456
2026-10-04 01:54:41 BUILD 0.8B-338M-eQ4_K-general3 OK rc=0 292s bytes=337789184 template=337789184
2026-10-04 01:55:38 BUILD 0.8B-338M-eIQ4_XS-general3 OK rc=0 311s bytes=337920256 template=337920256
2026-10-04 01:57:48 BUILD 0.8B-310M-eQ4_K-general3 OK rc=0 273s bytes=309604608 template=309604608
2026-10-04 01:58:29 BUILD 0.8B-310M-eIQ4_XS-general3 OK rc=0 276s bytes=309625088 template=309625088
2026-10-04 01:59:14 BUILD 0.8B-338M-eIQ3_XXSu-general2 OK rc=0 273s bytes=337916160 template=337916160
2026-10-04 01:59:17 BUILD 0.8B-338M-eQ2_Ku-general2 OK rc=0 244s bytes=337920256 template=337920256
2026-10-04 01:59:47 BUILD 0.8B-310M-eIQ3_XXSu-general2 OK rc=0 274s bytes=309698816 template=309698816
2026-10-04 02:00:23 BUILD 0.8B-310M-eQ2_Ku-general2 OK rc=0 285s bytes=309688576 template=309688576
2026-10-04 02:02:02 BUILD 0.8B-290M-eQ2_Ku-general2 OK rc=0 254s bytes=289688832 template=289688832
2026-10-04 02:02:49 BUILD 0.8B-290M-eIQ2_XXSu-general2 OK rc=0 260s bytes=289699072 template=289699072
2026-10-04 02:03:13 BUILD 0.8B-338M-eQ2_Ku-general3 OK rc=0 236s bytes=337920256 template=337920256
2026-10-04 02:03:41 BUILD 0.8B-338M-eIQ3_XXSu-general3 OK rc=0 267s bytes=337916160 template=337916160
2026-10-04 02:04:13 BUILD 0.8B-310M-eIQ3_XXSu-general3 OK rc=0 266s bytes=309698816 template=309698816
2026-10-04 02:04:24 BUILD 0.8B-310M-eQ2_Ku-general3 OK rc=0 241s bytes=309688576 template=309688576
2026-10-04 02:06:06 BUILD 0.8B-290M-eQ2_Ku-general3 OK rc=0 244s bytes=289688832 template=289688832
2026-10-04 02:07:01 BUILD 0.8B-290M-eIQ2_XXSu-general3 OK rc=0 252s bytes=289699072 template=289699072
2026-10-04 02:08:09 BUILD 0.8B-310M-eIQ3_XXS-general3 OK rc=0 296s bytes=309693696 template=309693696
2026-10-04 02:08:23 BUILD 0.8B-290M-eQ3_K-general3 OK rc=0 282s bytes=289534208 template=289534208
2026-10-04 02:08:52 BUILD 0.8B-290M-eQ2_K-general3 OK rc=0 268s bytes=289694976 template=289694976
2026-10-04 02:09:05 BUILD 0.8B-290M-eIQ3_XXS-general3 OK rc=0 292s bytes=289345792 template=289345792
```
