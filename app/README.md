# App

Offline Android app for the coffee-leaf checker: llama.cpp `llama-server` in Termux, and one
static web page it serves (`--path`), so page and model share an origin and everything works in
airplane mode. No external scripts, fonts or styles.

| File | What |
|---|---|
| `index.html`, `app.js` | camera → resize → request → letter probabilities → decision → answer card |
| `config.json` | model, mmproj, LoRA, prompt path, tau, context, threads, languages. A model swap edits only this |
| `cards.json` | the six answer cards (A–E + not sure) in pt-BR, en, sw. The model never writes card text |
| `run_phone.sh` | Termux launcher; reads paths, `-c`, `-t` and port from `config.json` |
| `tools/make_voice.py` | records each card once with ElevenLabs into `audio/<lang>/<card>.mp3` |
| `manifest.json`, `icon-*.png` | "Add to Home screen" → opens full screen without the address bar |

## Run

    # model package: unzip hf.zip into app/models/hf/ (flat), check SHA-256 against the package table
    unzip hf.zip -x '__MACOSX/*' '*.DS_Store' -d app/models/

    # desktop, any machine with llama.cpp:   bash app/run_phone.sh  →  http://127.0.0.1:8080/
    # desktop, no model:                     python3 -m http.server 8765 (in app/) → http://127.0.0.1:8765/?mock

Phone (Termux, from F-Droid or GitHub, not the Play Store):

    pkg upgrade            # required: without it llama-server fails to link (__hash_memory missing)
    pkg install llama-cpp jq
    termux-wake-lock       # and set Termux battery use to Unrestricted, or Android kills the server
    cd ~/app && bash run_phone.sh

Then open `http://127.0.0.1:8080/` in Chrome, ⋮ → Add to Home screen.

## Request (matches the package protocol)

- Prompt from `prompt.txt` (`### system`, `### user`, `### grammar`), word for word.
- Longest side 512 px, JPEG quality 0.9, base64 data URL; image before the user text.
- `POST /v1/chat/completions`, `max_tokens 1`, temperature 1.0, top_k 0, top_p 1.0, min_p 0,
  `logprobs`/`top_logprobs 6`, `n_probs 6`, `post_sampling_probs true`,
  `chat_template_kwargs: {enable_thinking: false}`.
- Answer = letter with the highest returned probability; the sampled token is ignored.
- Not sure = top letter F, or its probability < tau → card "not sure", photo saved to the queue.
- Result shows "provável" / "muito provável" (≥ `very_likely`, default 0.8), not a percentage.

Findings from running it (llama.cpp 0.5.0 / b11146, placeholder Unsloth 0.8B):

- **Thinking is ON by default** in the Qwen3.5 template; `enable_thinking: false` makes it pre-fill
  an empty `<think></think>`. The benchmark harness must send it too.
- **`top_probs` are not restricted to the grammar**: they include `<think>`, `G`, `**`, empty
  tokens. With 6 slots a low letter can drop out (reads as 0), and the letters sum to < 1. The top
  letter is unaffected; tau is applied to the raw probability. Harness and app must agree on this.
- The sampled token often differs from the top letter, so reading probabilities matters.

## Leaf check before the diagnosis (added, please review)

The fine-tuned model never picks F: on photos that are not leaves it gives a confident disease
(a building → C 0.94, a cartoon house → D 0.85), far above tau, so "not sure" never fires. That
fails the brief's pass/fail "not sure, ask a person" rule the first time a judge points the camera at
something else.

So the app sends one extra request first, then the package diagnosis unchanged:
same model, adapter as loaded, the package system text, the image, and
`Is this a close-up photo of a plant leaf? A Yes, a plant leaf fills most of the photo. B No.`
with grammar `root ::= [AB]`. If P(A) ≤ P(B): "Isto não parece uma folha", retake, not queued.

Mac, 20 BRACOL test images (4 per class, SHA-checked against manifest.csv) + 1 field photo of rust on
the plant: P(A) ≥ 0.88 for every leaf; building 0.005, cartoon 0.14. Small sample; it would be worth
running over the whole test split plus a set of non-leaf photos in the harness. The server does not
reuse the encoded image between the two requests (cache stops before the image), so it costs a
second image pass. `config.json` → `leaf_gate.enabled: false` turns it off.

## Measured

| Where | Model | Per photo | Peak RSS |
|---|---|---|---|
| Mac M4 Pro (Metal) | placeholder 0.8B Q4_K_M + F16 mmproj | ~0.7 s | – |
| Android emulator, **3 GB RAM, 4 cores**, airplane mode | same | 2.9–4.7 s | **1.04 GB** (1.2 GB free) |
| Mac M4 Pro (Metal) | **final package** (eQ2_K + LoRA + Q8_0 mmproj) | ~0.65 s incl. leaf check | – |
| Android emulator, 3 GB, 4 cores, airplane mode | **final package**, `-c 8192` and `-c 4096` | ~6 s leaf check + ~8 s diagnosis | **1.66 GB** both (0.9 GB left) |
| Galaxy Z Flip5 (SM-F731U1, SD 8 Gen 2, 7 GB) | final package | to do | to do |

Final package on the Mac, BRACOL test images: 18/20 top letters correct (both misses C, read as B
and D); letters sum to 0.998–1.000. In the 3 GB emulator with Chrome open, Android's low-memory
killer once killed Chrome's renderer and other apps while llama-server held 1.65 GB; with nothing
else open it did not. `-c 4096` does not lower the peak (one photo + prompt is ~300 tokens).

Sample, `C_1574.jpg` (test, truth C): A 0.00, B 0.01, C 0.89, D 0.01, E 0.08, F 0.00.

The emulator shows **fit in 3 GB**, not budget-phone speed: its cores are the host Mac's.

## Voice

Every possible answer is one of six fixed cards, so each is recorded once with ElevenLabs
(`tools/make_voice.py`, model `eleven_v4`) and shipped as ~1 MB of MP3s. The app never calls
ElevenLabs. Fallback, all offline: no clip → the phone's own text-to-speech (`pt-BR`/`en-US`)
→ if the phone has no voice for the language, a short message. Never an English voice reading
Portuguese.

## Not done

- Sending the queue (stub).
- The optional one-sentence explanation + translator (package `translate/`).
- Native-speaker check of the pt-BR "do today" / "call" lines (name + description are the package CARDS).
- Exact-token parsing: `" A"` and `"a"` also appear in the top 6; the app counts only the exact letters.
