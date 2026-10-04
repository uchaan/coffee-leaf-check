"""General benchmarks from the Qwen3.5 model card, scored the same way as the BRACOL harness (A100 server, 2026-10-04).

Benchmarks (all multiple choice, one letter, thinking off):
  mmstar      MMStar (Lin-Chen/MMStar @bc98d668), all 1,500 items, vision.
  ai2d        AI2D test (lmms-lab/ai2d @c83a9b96), 1,000 items: the first 1,000 by SHA-256 of (question + image bytes), vision.
  mmlu_redux  MMLU-Redux 2.0 (edinburgh-dawg/mmlu-redux-2.0 @372ea425), every item with error_type == "ok", text only.
Request: the harness's own body (max_tokens 1, temperature 1.0, top_k 0, top_p 1, min_p 0, no penalties, seed 0, cache_prompt false,
enable_thinking false, top_logprobs 50 without post-sampling probs = harness mode pre50), grammar root ::= [A-<last option>], answer =
argmax of the option-letter probabilities renormalised over the options. Images: the protocol preprocessing (RGB, longest side 512 px,
Lanczos, JPEG q90). Prompt: question, "A. ..." option lines, then "Answer with the option's letter from the given choices directly."
(MMStar's questions already carry their options). Same llama.cpp commit, chat template and canonical Q8_0 mmproj as the harness.

usage: general_bench.py prepare --data DIR --cache DIR
       general_bench.py run --model M --lm X.gguf --mmproj MMPROJ --bench mmstar,ai2d,mmlu_redux --cache DIR --out DIR [--ref DIR] --gpu N --port P
"""
import argparse, csv, glob, hashlib, io, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

INSTR = "Answer with the option's letter from the given choices directly."


def _jpeg512(raw, dst):
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB"); w, h = im.size; s = 512 / max(w, h)
    im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS).save(dst, "JPEG", quality=90)


def prepare(a):
    import pandas as pd, datasets
    os.makedirs(os.path.join(a.cache, "img"), exist_ok=True); items = []
    m = pd.read_parquet(os.path.join(a.data, "MMStar", "mmstar.parquet"))
    for _, r in m.iterrows():
        raw = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
        p = os.path.join(a.cache, "img", f"mmstar-{r['index']}.jpg"); os.path.exists(p) or _jpeg512(raw, p)
        items.append(dict(id=f"mmstar-{r['index']}", bench="mmstar", image=p, text=f"{r['question']}\n{INSTR}", letters="ABCD", answer=r["answer"].strip()))
    ai = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.data, "ai2d", "data", "*.parquet")))]).reset_index(drop=True)
    rows = []
    for i, r in ai.iterrows():
        raw = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
        rows.append((hashlib.sha256(r["question"].encode() + raw).hexdigest(), i, raw, r))
    for sha, i, raw, r in sorted(rows)[:1000]:
        opts = list(r["options"]); L = "ABCD"[: len(opts)]
        p = os.path.join(a.cache, "img", f"ai2d-{i}.jpg"); os.path.exists(p) or _jpeg512(raw, p)
        text = r["question"] + "\n" + "\n".join(f"{L[k]}. {o}" for k, o in enumerate(opts)) + "\n" + INSTR
        items.append(dict(id=f"ai2d-{i}", bench="ai2d", image=p, text=text, letters=L, answer=L[int(r["answer"])]))
    for sub in sorted(d for d in os.listdir(os.path.join(a.data, "mmlu-redux-2.0")) if not d.startswith(".") and os.path.isdir(os.path.join(a.data, "mmlu-redux-2.0", d))):
        ds = datasets.load_from_disk(os.path.join(a.data, "mmlu-redux-2.0", sub))
        for k, r in enumerate(ds):
            if r["error_type"] != "ok" or len(r["choices"]) != 4: continue
            text = (f"The following is a multiple choice question about {sub.replace('_', ' ')}.\n\n{r['question']}\n"
                    + "\n".join(f"{'ABCD'[j]}. {c}" for j, c in enumerate(r["choices"])) + "\n" + INSTR)
            items.append(dict(id=f"mmlu_redux-{sub}-{k}", bench="mmlu_redux", image=None, text=text, letters="ABCD", answer="ABCD"[int(r["answer"])]))
    with open(os.path.join(a.cache, "items.jsonl"), "w") as f:
        for it in items: f.write(json.dumps(it) + "\n")
    from collections import Counter
    print("items:", dict(Counter(it["bench"] for it in items)))


def body(it):
    content = [{"type": "text", "text": it["text"]}]
    if it["image"]:
        import base64
        content.insert(0, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(it["image"], "rb").read()).decode()}})
    prompt = {"system": "", "user": "", "grammar": f"root ::= [A-{it['letters'][-1]}]"}
    b = {"messages": [{"role": "user", "content": content}], "max_tokens": 1, "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0,
         "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
         "xtc_probability": 0.0, "seed": 0, "grammar": prompt["grammar"], "logprobs": True, "top_logprobs": 50,
         "post_sampling_probs": False, "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    return b


def run(a):
    items = [json.loads(l) for l in open(os.path.join(a.cache, "items.jsonl"))]
    benches = a.bench.split(","); items = [it for it in items if it["bench"] in benches]
    template = a.chat_template_file or os.path.join(H.ROOT, "protocol", "chat_templates", a.model + ".jinja")
    os.makedirs(a.out, exist_ok=True)
    srv = H.Server(a.lm, a.mmproj, template, a.port, a.gpu, 99, 4, os.path.join(a.out, "server.log"))
    rows = []; t0 = time.time()
    try:
        for n, it in enumerate(items):
            resp = srv.post("/v1/chat/completions", body(it))
            v, rawsum, other, _ = H.letter_probs(resp, "pre50", it["letters"])
            p = {L: float(v[H.LETTERS.index(L)]) for L in it["letters"]}
            pred = max(p, key=p.get) if sum(p.values()) > 0 else ""
            rows.append(dict(id=it["id"], bench=it["bench"], answer=it["answer"], pred=pred, letter_mass=round(rawsum, 6), **{f"p_{L}": p.get(L, 0.0) for L in "ABCD"}))
            if (n + 1) % 500 == 0: print(f"{n+1}/{len(items)} {(time.time()-t0)/(n+1):.3f} s/item", flush=True)
    finally:
        srv.stop()
    with open(os.path.join(a.out, "items.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    finish(a.out, a.ref, a.lm)


def finish(out, ref, lm=None):
    rows = list(csv.DictReader(open(os.path.join(out, "items.csv"))))
    refrows = {r["id"]: r for r in csv.DictReader(open(os.path.join(ref, "items.csv")))} if ref else {}
    met = {}
    for b in sorted({r["bench"] for r in rows}):
        R = [r for r in rows if r["bench"] == b]; acc = float(np.mean([r["pred"] == r["answer"] for r in R]))
        m = dict(n=len(R), acc=round(acc, 4), letter_mass=round(float(np.mean([float(r["letter_mass"]) for r in R])), 4))
        if refrows:
            kls, agree = [], []
            for r in R:
                q = refrows.get(r["id"]);
                if not q: continue
                P = np.array([float(q[f"p_{L}"]) for L in "ABCD"]); Q = np.array([float(r[f"p_{L}"]) for L in "ABCD"])
                kls.append(float(H.kl(P[None], Q[None])[0])); agree.append(q["pred"] == r["pred"])   # harness.kl takes (n, k) arrays
            m.update(letter_kld=round(float(np.mean(kls)), 5), agree_ref=round(float(np.mean(agree)), 4))
        met[b] = m
    meta = dict(lm=os.path.basename(lm) if lm else None, lm_bytes=os.path.getsize(lm) if lm else None, llama_cpp_commit=H.git_head(H.LLAMA_DIR),
                ref=ref, metrics=met)
    json.dump(meta, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    print(os.path.basename(out.rstrip("/")), json.dumps(met))


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prepare"); p.add_argument("--data", required=True); p.add_argument("--cache", required=True)
    p = sp.add_parser("run"); p.add_argument("--model", required=True); p.add_argument("--lm", required=True); p.add_argument("--mmproj", required=True)
    p.add_argument("--bench", default="mmstar,ai2d,mmlu_redux"); p.add_argument("--cache", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ref", default=None); p.add_argument("--gpu", default=None); p.add_argument("--port", type=int, default=8095)
    p.add_argument("--chat-template-file", default=None)
    p = sp.add_parser("score"); p.add_argument("--out", required=True); p.add_argument("--ref", default=None)
    a = ap.parse_args()
    {"prepare": prepare, "run": run, "score": lambda a: finish(a.out, a.ref)}[a.cmd](a)


if __name__ == "__main__":
    main()
