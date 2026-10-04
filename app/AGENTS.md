# app/: agent guide

Offline client: Termux `llama-server` plus one static page (`index.html`, `app.js`) served from this folder. Files,
config keys, request format and measurements: [`README.md`](README.md).

## Run

```bash
bash app/run_phone.sh                        # from the repository root; needs llama-server and jq, package in app/models/hf/
cd app && python3 -m http.server 8765        # no model: http://127.0.0.1:8765/?mock
```

## Rules

- `prompt.txt` must stay byte-identical to `protocol/prompt_v1.txt` (`cmp protocol/prompt_v1.txt app/prompt.txt`).
- The request must match the harness protocol (`eval/README.md`): thinking off, `max_tokens` 1, read letter
  probabilities and renormalise; the answer is the top letter, never the sampled token.
- A model swap edits only `config.json`. The model never writes card text; cards come from `cards.json`.
- No external scripts, fonts or styles: the page must work in airplane mode.
- Never commit `models/` (the package files) or `audio/`; both are ignored.
