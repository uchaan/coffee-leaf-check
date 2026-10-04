# App

Offline Android app for the coffee-leaf checker: llama.cpp `llama-server` in Termux, and one
static web page it serves (`--path`), so page and model share an origin and everything works in
airplane mode. No external scripts, fonts or styles.

| File | What |
|---|---|
| `index.html`, `app.js` | camera → resize → request → letter probabilities → decision → answer card |
| `config.json` | model, mmproj, LoRA, prompt path, tau, context, threads, languages. A model swap edits only this |
| `cards.json` | the six answer cards (A–E + not sure). pt-BR and en are offered; sw is an unchecked draft. The model never writes card text |
| `prompt.txt` | the package prompt, a copy of [`agentic-quantization/protocol/prompt_v1.txt`](../agentic-quantization/protocol/prompt_v1.txt) |
| `run_phone.sh` | launcher (Termux or desktop); reads paths, `-c`, `-t` and port from `config.json` |
| `tools/make_voice.py` | optional: records each card once with ElevenLabs into `audio/<lang>/<card>.mp3` |
| `manifest.json`, `icon-*.png` | "Add to Home screen" → opens full screen without the address bar |

`models/` and `audio/` are generated locally and are not in git.

## Model package

The weights are not in this repository. `run_phone.sh` expects the three package files from the
[RESULTS.md package table](../RESULTS.md#1-what-the-tool-does) flat in `app/models/hf/`:

    app/models/hf/Qwen3.5-2B-ours-task-1GBpkg-eQ2_K.gguf      # 637.8 MB
    app/models/hf/mmproj-Qwen3.5-2B-ours-coffee-Q8_0.gguf     # 361.5 MB
    app/models/hf/Qwen3.5-2B-ours-coffee-lora-Q8_0.gguf       # 12.6 MB

To rebuild them: the quantized model comes from [`agentic-quantization/`](../agentic-quantization/),
the LoRA and vision projector from [`fine-tuning/`](../fine-tuning/) (`export_lora.py`). If you have
the package as a zip: `unzip <package>.zip -x '__MACOSX/*' '*.DS_Store' -d app/models/`.
Other file names work too: edit `model_file`, `mmproj_file`, `lora_file` in `config.json`.

## Run

Desktop (needs llama.cpp's `llama-server` and `jq` on `PATH`):

    bash app/run_phone.sh                          # → http://127.0.0.1:8080/
    cd app && python3 -m http.server 8765          # no model: http://127.0.0.1:8765/?mock

Phone (Termux, from F-Droid or GitHub, not the Play Store):

    pkg upgrade            # required: without it llama-server fails to link (__hash_memory missing)
    pkg install llama-cpp jq git
    git clone https://github.com/uchaan/coffee-leaf-check && cp -r coffee-leaf-check/app ~/app
    # copy the three package files into ~/app/models/hf/ (termux-setup-storage gives access to Downloads)
    termux-wake-lock       # and set Termux battery use to Unrestricted, or Android kills the server
    cd ~/app && bash run_phone.sh

Then open `http://127.0.0.1:8080/` in Chrome, ⋮ → Add to Home screen.

## Config

| Key | Default | What |
|---|---|---|
| `model_file`, `mmproj_file`, `lora_file` | `models/hf/…` | package files, relative to `app/`; `lora_file` may be empty |
| `prompt_file` | `prompt.txt` | package prompt |
| `tau` | 0.5 | below this, "not sure". Fixed default, not tuned on dev for the final package |
| `very_likely` | 0.8 | at or above this, "very likely" instead of "likely" |
| `languages` | `["pt", "en"]` | offered languages, first is the default; add `"sw"` for the draft |
| `ctx`, `threads`, `port` | 4096, 0, 8080 | `llama-server -c`, `-t` (0 = llama.cpp default), `--port` |
| `server` | `""` | base URL of llama-server; empty = same origin |
| `mock` | false | fake probabilities, no server (also `?mock` in the URL) |
| `leaf_gate.enabled` | true | leaf check before the diagnosis (below) |
| `leaf_gate.system` | null | null = the package system text |
| `leaf_gate.user`, `leaf_gate.grammar` | yes/no question, `root ::= [AB]` | |
| `leaf_gate.lora_scale` | unset | optional LoRA scale for the leaf check; unset = adapter as loaded |

## Request (matches the package protocol)

- Prompt from `prompt.txt` (`### system`, `### user`, `### grammar`), word for word.
- Longest side 512 px, JPEG quality 0.9, base64 data URL; image before the user text.
- `POST /v1/chat/completions`, `max_tokens 1`, temperature 1.0, top_k 0, top_p 1.0, min_p 0,
  `logprobs`/`top_logprobs 6`, `n_probs 6`, `post_sampling_probs true`,
  `chat_template_kwargs: {enable_thinking: false}`.
- Answer = letter with the highest returned probability; the sampled token is ignored.
- Not sure = top letter F, or its probability < tau → card "not sure", photo saved to the queue.
- Result shows "provável" / "muito provável" (≥ `very_likely`), not a percentage.

Findings from running it (llama.cpp 0.5.0 / b11146, early test model Unsloth Qwen3.5-0.8B Q4_K_M):

- **Thinking is ON by default** in the Qwen3.5 template; `enable_thinking: false` makes it pre-fill
  an empty `<think></think>`. The benchmark harness must send it too.
- **`top_probs` are not restricted to the grammar**: they include `<think>`, `G`, `**`, empty
  tokens. With 6 slots a low letter can drop out (reads as 0), and the letters sum to < 1. The top
  letter is unaffected; tau is applied to the raw probability. Harness and app must agree on this.
- The sampled token often differs from the top letter, so reading probabilities matters.
- `" A"` and `"a"` also appear in the top 6; the app counts only the exact letters.

## Leaf check before the diagnosis

The fine-tuned model never picks F: on photos that are not leaves it gives a confident disease
(a building → C 0.94, a cartoon house → D 0.85), far above tau, so "not sure" never fires. That
fails the brief's "not sure, ask a person" rule the first time someone points the camera at
something else.

So the app sends one extra request first, then the package diagnosis unchanged:
same model, adapter as loaded, the package system text, the image, and
`Is this a close-up photo of a plant leaf? A Yes, a plant leaf fills most of the photo. B No.`
with grammar `root ::= [AB]`. If P(A) ≤ P(B): "Isto não parece uma folha", retake, not queued.

Mac, 20 BRACOL test images (4 per class, SHA-checked against manifest.csv) + 1 field photo of rust on
the plant: P(A) ≥ 0.88 for every leaf; building 0.005, cartoon 0.14. Small sample, not yet run over
the whole test split or a set of non-leaf photos. The server does not reuse the encoded image
between the two requests (cache stops before the image), so it costs a second image pass.
`leaf_gate.enabled: false` turns it off.

## Measured

| Where | Model | Per photo | Peak RSS |
|---|---|---|---|
| Mac M4 Pro (Metal) | early test: Unsloth 0.8B Q4_K_M + F16 mmproj | ~0.7 s | – |
| Android emulator, **3 GB RAM, 4 cores**, airplane mode | same | 2.9–4.7 s | **1.04 GB** (1.2 GB free) |
| Mac M4 Pro (Metal) | **final package** (eQ2_K + LoRA + Q8_0 mmproj) | ~0.65 s incl. leaf check | – |
| Android emulator, 3 GB, 4 cores, airplane mode | **final package**, `-c 8192` and `-c 4096` | ~6 s leaf check + ~8 s diagnosis | **1.66 GB** both (0.9 GB left) |

Not measured yet on a real phone.

Final package on the Mac, BRACOL test images: 18/20 top letters correct (both misses C, read as B
and D); letters sum to 0.998–1.000. In the 3 GB emulator with Chrome open, Android's low-memory
killer once killed Chrome's renderer and other apps while llama-server held 1.65 GB; with nothing
else open it did not. `-c 4096` does not lower the peak (one photo + prompt is ~300 tokens).

Sample, `C_1574.jpg` (test, truth C): A 0.00, B 0.01, C 0.89, D 0.01, E 0.08, F 0.00.

The emulator shows **fit in 3 GB**, not budget-phone speed: its cores are the host Mac's.

## Voice

Every possible answer is one of six fixed cards, so each can be recorded once with ElevenLabs and
played offline. The app never calls ElevenLabs. The clips are not in git; to make them:

    cd app
    python3 tools/make_voice.py --dry-run                        # text that will be spoken
    ELEVENLABS_API_KEY=... python3 tools/make_voice.py           # model eleven_v3, pt and en → audio/

Fallback, all offline: no clip → the phone's own text-to-speech (`pt-BR`/`en-US`) → if the phone
has no voice for the language, a short message. Never an English voice reading Portuguese.

## Not done

- Sending the queue (stub).
- The optional one-sentence explanation + Portuguese translator (package capability, see
  `fine-tuning/explain_demo.py` and `translate_pt.py`); the app sends `max_tokens 1` only.
- Native-speaker check of the pt-BR "do today" / "call" lines (name + description are the package CARDS)
  and of the sw draft.
- A tau tuned for the final package.
