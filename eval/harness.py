#!/usr/bin/env python3
"""Hack-Nation 04B coffee-leaf benchmark harness (protocol: eval/README.md).

One language-model GGUF + one mmproj GGUF + one prompt file -> per-image letter
probabilities on a manifest split, through llama-server, plus the metrics row.

  # dev metrics for one file (what the A100 server and prompt tuning use)
  python eval/harness.py run --model Qwen3.5-2B --lm X.gguf --mmproj mmproj.gguf \
      --prompt protocol/prompt_v1.txt --split dev --gpu 1

  # full run (dev for tau + test), scored against the BF16 reference rows
  python eval/harness.py run ... --split all --ref runs/<bf16 tag>/images.csv

  # CPU proxy (-ngl 0, 4 threads, first 50 test images by SHA-256)
  python eval/harness.py cpu --model ... --lm ... --mmproj ... --prompt ...

  # text KLD vs BF16 (llama-perplexity, wikitext-2 test, 40 chunks, ctx 512)
  python eval/harness.py textkld --bf16 BF16.gguf --lm X.gguf

  # rescore an existing run (e.g. once the BF16 reference exists)
  python eval/harness.py score --run-dir runs/<tag> --ref runs/<bf16 tag>/images.csv

Environment (defaults as in env.example.sh): LLAMA_CPP_DIR (default ./llama.cpp; binaries in build/bin),
WORK (default ./work), BRACOL_ROOT (folder with dataset.csv and images/), HN04B_CACHE (preprocessed images),
HN04B_RUNS (output root), WIKITEXT (text-KLD text), HN04B_SERVER_EXTRA (extra llama-server flags for GPU runs).
"""
import argparse, base64, csv, hashlib, io, json, os, re, signal, subprocess, sys, time

import numpy as np
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repository root
LETTERS = "ABCDEF"
FORCED = "ABCDE"
COARSE = {"A": "A", "B": "B", "C": "X", "D": "X", "E": "X"}
LLAMA_DIR = os.environ.get("LLAMA_CPP_DIR", "llama.cpp")
WORK = os.environ.get("WORK", "work")  # defaults below follow env.example.sh
HARNESS_VERSION = "1"


# ---------------------------------------------------------------- inputs
def sha256_file(p, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_prompt(path):
    """Sections '### system', '### user', optional '### grammar'."""
    sec, cur = {}, None
    for line in open(path, encoding="utf-8").read().splitlines():
        m = re.match(r"^### (system|user|grammar)\s*$", line)
        if m:
            cur = m.group(1); sec[cur] = []
        elif cur:
            sec[cur].append(line)
    out = {k: "\n".join(v).strip("\n") for k, v in sec.items()}
    out.setdefault("system", "")
    out.setdefault("grammar", "root ::= [A-F]")
    if "user" not in out:
        sys.exit(f"{path}: needs a '### user' section")
    m = re.search(r"\[A-([A-F])\]", out["grammar"])
    out["letters"] = LETTERS[: LETTERS.index(m.group(1)) + 1] if m else LETTERS
    return out


def read_manifest(path, split):
    rows = list(csv.DictReader(open(path)))
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    return rows


def preprocess(src, sha, cache_dir):
    """Load, RGB, longest side 512 (Lanczos), JPEG q90. Cached by source SHA-256."""
    out = os.path.join(cache_dir, sha + ".jpg")
    if not os.path.exists(out):
        im = Image.open(src).convert("RGB")
        w, h = im.size
        s = 512 / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        tmp = f"{out}.{os.getpid()}.tmp"
        im.save(tmp, "JPEG", quality=90)
        os.replace(tmp, out)
    return out


# ---------------------------------------------------------------- server
class Server:
    def __init__(self, lm, mmproj, template, port, gpu, ngl, threads, log, cpu=False, extra=None):
        self.port, self.log = port, log
        b = os.path.join(LLAMA_DIR, "build", "bin", "llama-server")
        cmd = [b, "-m", lm, "--mmproj", mmproj, "--chat-template-file", template, "--jinja",
               "--reasoning-format", "none", "-c", "8192", "-np", "1", "--port", str(port),
               "--host", "127.0.0.1", "--seed", "0", "-ngl", str(ngl), "--no-webui"]
        env = dict(os.environ)
        if cpu:
            # no host-RAM prompt cache and no context checkpoints: they grow RSS per request and a phone app
            # running one image at a time does not keep them; GPU runs keep the server defaults
            cmd += ["-t", str(threads), "-tb", str(threads), "--no-mmproj-offload"] + (extra or [])
            env["CUDA_VISIBLE_DEVICES"] = ""
            cmd = ["/usr/bin/time", "-v"] + cmd
        else:
            # optional extra server flags for GPU runs (unset = llama-server defaults, as in every reported run)
            cmd += os.environ.get("HN04B_SERVER_EXTRA", "").split()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        self.cmd = cmd
        self.logf = open(log, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
        url = f"http://127.0.0.1:{port}/health"
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}); see {log}")
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError(f"llama-server not healthy after 600 s; see {log}")

    def post(self, path, body):
        r = requests.post(f"http://127.0.0.1:{self.port}{path}", json=body, timeout=600)
        r.raise_for_status()
        return r.json()

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.logf.close()


def gguf_bpw(path):
    """llama.cpp's BPW: sum of tensor bytes * 8 / sum of tensor elements (all tensors in the file)."""
    sys.path.insert(0, os.path.join(LLAMA_DIR, "gguf-py"))
    from gguf import GGUFReader
    r = GGUFReader(path)
    nb = sum(int(t.n_bytes) for t in r.tensors)
    ne = sum(int(t.n_elements) for t in r.tensors)
    return nb * 8 / ne


def log_facts(log):
    txt = open(log, errors="replace").read()
    bpw = re.search(r"file size\s*=\s*[\d.]+ \w+ \(([\d.]+) BPW\)", txt)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    return {"bpw": float(bpw.group(1)) if bpw else None,
            "peak_rss_mb": int(rss.group(1)) / 1024 if rss else None}


def build_request(prompt, img_path, prob_mode):
    b64 = base64.b64encode(open(img_path, "rb").read()).decode()
    msgs = []
    if prompt["system"]:
        msgs.append({"role": "system", "content": prompt["system"]})
    msgs.append({"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt["user"]}]})
    body = {"messages": msgs, "max_tokens": 1, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
            "min_p": 0.0, "typical_p": 1.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0,
            "seed": 0, "grammar": prompt["grammar"], "logprobs": True,
            "chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    if prob_mode == "post":
        body.update({"top_logprobs": 6, "post_sampling_probs": True})
    else:  # "pre50": raw softmax top-50, renormalised over the letters afterwards
        body.update({"top_logprobs": 50, "post_sampling_probs": False})
    return body


def letter_probs(resp, prob_mode, letters):
    tok = resp["choices"][0]["logprobs"]["content"][0]
    key = "top_probs" if prob_mode == "post" else "top_logprobs"
    raw = {}
    for t in tok[key]:
        p = t["prob"] if prob_mode == "post" else float(np.exp(t["logprob"]))
        s = t["token"]
        if s in LETTERS:
            raw[s] = raw.get(s, 0.0) + p
    other = sum((t["prob"] if prob_mode == "post" else float(np.exp(t["logprob"])))
                for t in tok[key] if t["token"] not in LETTERS)
    v = np.array([raw.get(L, 0.0) if L in letters else 0.0 for L in LETTERS])
    rawsum = float(v.sum())
    if rawsum > 0:
        v = v / rawsum
    content = resp["choices"][0]["message"].get("content") or ""
    return v, rawsum, other, content.strip()[-1:]  # content echoes the empty think block; keep the letter


# ---------------------------------------------------------------- metrics
def f1_per_class(y, p, classes):
    out = {}
    for c in classes:
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        out[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return out


def macro_f1(y, p, classes):
    return float(np.mean(list(f1_per_class(y, p, classes).values())))


def forced_pred(P):  # argmax over A..E
    return [FORCED[i] for i in np.argmax(P[:, :5], axis=1)]


def full_pred(P):
    return [LETTERS[i] for i in np.argmax(P, axis=1)]


def _macro_f1_rows(Y, Q, k=5):
    """Row-wise five-class macro-F1 for integer label matrices Y, Q of shape (B, n)."""
    out = np.zeros(Y.shape[0])
    for c in range(k):
        yc, qc = Y == c, Q == c
        tp = (yc & qc).sum(1); fp = (~yc & qc).sum(1); fn = (yc & ~qc).sum(1)
        den = 2 * tp + fp + fn
        out += np.where(tp > 0, 2 * tp / np.maximum(den, 1), 0.0)
    return out / k


def _codes(v):
    return np.array([FORCED.index(x) for x in v])


def bootstrap_f1(y, p, n=1000, seed=0):
    """95% percentile interval of forced macro-F1 over n resamples of the images (numpy Generator, seed 0)."""
    rng = np.random.default_rng(seed)
    y, p = _codes(y), _codes(p)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    vals = _macro_f1_rows(y[idx], p[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_diff(y, pa, pb, n=1000, seed=0):
    """F1(a) - F1(b) on the same images, with the 95% interval from the same resamples for both."""
    rng = np.random.default_rng(seed)
    y, pa, pb = _codes(y), _codes(pa), _codes(pb)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = _macro_f1_rows(y[idx], pa[idx]) - _macro_f1_rows(y[idx], pb[idx])
    full = _macro_f1_rows(y[None, :], pa[None, :])[0] - _macro_f1_rows(y[None, :], pb[None, :])[0]
    return float(full), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def choose_tau(y, P, target=0.9):
    """Smallest tau with precision of answered cases (top != F and p_top >= tau) >= target."""
    top = np.argmax(P, axis=1)
    ptop = P[np.arange(len(P)), top]
    ok = np.array([LETTERS[t] == l for t, l in zip(top, y)])
    nonF = top != 5
    cands = sorted(set([0.0] + [float(x) for x in ptop[nonF]]))
    best = (-1.0, None, 0)
    for t in cands:
        ans = nonF & (ptop >= t)
        if ans.sum() == 0:
            continue
        prec = float(ok[ans].mean())
        if prec >= target:
            return {"tau": t, "dev_precision": prec, "dev_coverage": float(ans.mean()), "reached": True}
        if prec > best[0]:
            best = (prec, t, float(ans.mean()))
    return {"tau": best[1], "dev_precision": best[0], "dev_coverage": best[2], "reached": False}


def answered_stats(y, P, tau):
    top = np.argmax(P, axis=1)
    ptop = P[np.arange(len(P)), top]
    ans = (top != 5) & (ptop >= tau)
    ok = np.array([LETTERS[t] == l for t, l in zip(top, y)])
    return {"answered_acc": float(ok[ans].mean()) if ans.any() else None,
            "coverage": float(ans.mean()), "notsure_share": float(1 - ans.mean()),
            "model_F_share": float((top == 5).mean())}


def kl(p, q, eps=1e-10):
    p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
    p = p / p.sum(1, keepdims=True); q = q / q.sum(1, keepdims=True)
    return np.sum(p * np.log(p / q), axis=1)


def load_images_csv(path):
    rows = list(csv.DictReader(open(path)))
    P = np.array([[float(r["p_" + L]) for L in LETTERS] for r in rows])
    return rows, P


def score(rows, P, ref=None):
    """rows: per-image dicts with split/label; P: n x 6. Returns metrics per split."""
    res = {}
    splits = sorted(set(r["split"] for r in rows))
    tau_info = None
    if "dev" in splits:
        di = [i for i, r in enumerate(rows) if r["split"] == "dev"]
        tau_info = choose_tau([rows[i]["label"] for i in di], P[di])
    refmap = None
    if ref is not None:
        rrows, RP = ref
        refmap = {r["path"]: RP[i] for i, r in enumerate(rrows)}
    for s in splits:
        ii = [i for i, r in enumerate(rows) if r["split"] == s]
        y = [rows[i]["label"] for i in ii]
        Ps = P[ii]
        fp = forced_pred(Ps)
        m = {"split": s, "n": len(ii),
             "forced_acc": float(np.mean([a == b for a, b in zip(y, fp)])),
             "forced_macro_f1": macro_f1(y, fp, FORCED),
             "per_class_f1": f1_per_class(y, fp, FORCED),
             "coarse_macro_f1": macro_f1([COARSE[a] for a in y], [COARSE[b] for b in fp], "ABX"),
             "confusion": {a: {b: sum(1 for u, v in zip(y, fp) if u == a and v == b) for b in FORCED}
                           for a in FORCED},
             "argmax_F_share": float(np.mean(np.argmax(Ps, 1) == 5))}
        if s == "test":
            m["f1_ci_low"], m["f1_ci_high"] = bootstrap_f1(y, fp)
        if tau_info:
            m.update(tau=tau_info["tau"], tau_reached_90=tau_info["reached"],
                     tau_dev_precision=tau_info["dev_precision"], tau_dev_coverage=tau_info["dev_coverage"])
            if tau_info["tau"] is not None:
                m.update(answered_stats(y, Ps, tau_info["tau"]))
            else:  # the model never answers A to E on dev: nothing is answered at any threshold
                m.update(answered_acc=None, coverage=0.0, notsure_share=1.0,
                         model_F_share=float(np.mean(np.argmax(Ps, 1) == 5)))
        if refmap is not None:
            R = np.array([refmap[rows[i]["path"]] for i in ii])
            m["agree_bf16"] = float(np.mean(np.argmax(R, 1) == np.argmax(Ps, 1)))
            m["letter_kld"] = float(np.mean(kl(R, Ps)))
        res[s] = m
    return res


# ---------------------------------------------------------------- commands
def git_head(d):
    try:
        return subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def single_token_check(srv, letters):
    out = {}
    for L in letters:
        r = srv.post("/tokenize", {"content": L, "with_pieces": True})
        out[L] = r["tokens"]
    bad = [L for L, t in out.items() if len(t) != 1]
    return out, bad


def cmd_run(a):
    prompt = read_prompt(a.prompt)
    template = a.chat_template_file or os.path.join(ROOT, "protocol", "chat_templates", a.model + ".jinja")
    rows_in = read_manifest(a.manifest, a.split)
    if a.limit:
        rows_in = rows_in[: a.limit]
    tag = a.tag or "__".join([a.model, os.path.basename(a.lm).replace(".gguf", ""),
                              os.path.basename(a.prompt).replace(".txt", ""), a.split])
    out = os.path.join(a.out_root, tag)
    os.makedirs(out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    meta = {"harness_version": HARNESS_VERSION, "harness_commit": git_head(ROOT),
            "llama_cpp_commit": git_head(LLAMA_DIR), "model": a.model,
            "lm_file": os.path.basename(a.lm), "lm_bytes": os.path.getsize(a.lm),
            "lm_sha256": sha256_file(a.lm), "mmproj_file": os.path.basename(a.mmproj),
            "mmproj_bytes": os.path.getsize(a.mmproj), "mmproj_sha256": sha256_file(a.mmproj),
            "prompt_file": os.path.basename(a.prompt), "prompt_sha256": sha256_file(a.prompt),
            "chat_template_sha256": sha256_file(template), "split": a.split,
            "prob_mode": a.prob_mode, "thinking": "off via chat_template_kwargs enable_thinking=false",
            "manifest_sha256": sha256_file(a.manifest), "started": time.strftime("%F %T %Z")}
    srv = Server(a.lm, a.mmproj, template, a.port, a.gpu, a.ngl, a.threads, os.path.join(out, "server.log"))
    try:
        tok, bad = single_token_check(srv, prompt["letters"])
        meta["letter_tokens"] = tok
        if bad:
            print(f"WARNING: letters not single tokens: {bad}", file=sys.stderr)
        rows, P, warn = [], [], 0
        t0 = time.time()
        for k, r in enumerate(rows_in):
            img = preprocess(os.path.join(a.data_root, r["path"]), r["sha256"], a.cache)
            resp = srv.post("/v1/chat/completions", build_request(prompt, img, a.prob_mode))
            v, rawsum, other, sampled = letter_probs(resp, a.prob_mode, prompt["letters"])
            if k < 5:
                print(f"check img {k}: letters={dict(zip(LETTERS, np.round(v, 4)))} "
                      f"raw_letter_mass={rawsum:.4f} non_letter_mass_in_top={other:.4g}", file=sys.stderr)
            if a.prob_mode == "post" and abs(rawsum - 1) > 0.01:
                warn += 1
            rows.append({"path": r["path"], "sha256": r["sha256"], "split": r["split"], "label": r["label_letter"],
                         **{"p_" + L: f"{x:.6g}" for L, x in zip(LETTERS, v)},
                         "argmax": LETTERS[int(np.argmax(v))], "forced_argmax": FORCED[int(np.argmax(v[:5]))],
                         "raw_letter_mass": f"{rawsum:.6g}", "sampled": sampled})
            P.append(v)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(rows_in)} {(time.time() - t0) / (k + 1):.3f} s/img", file=sys.stderr)
        meta["sec_per_img"] = (time.time() - t0) / max(1, len(rows_in))
        meta["post_mass_warnings"] = warn
    finally:
        srv.stop()
    meta.update(log_facts(os.path.join(out, "server.log")))
    meta["bpw"] = gguf_bpw(a.lm)
    with open(os.path.join(out, "images.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    finish(out, a.ref)


def finish(out, ref_path):
    rows, P = load_images_csv(os.path.join(out, "images.csv"))
    ref = load_images_csv(ref_path) if ref_path else None
    m = score(rows, P, ref)
    json.dump(m, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    for s, x in m.items():
        line = (f"[{os.path.basename(out)}] {s} n={x['n']} forced_acc={x['forced_acc']:.4f} "
                f"forced_macro_f1={x['forced_macro_f1']:.4f} coarse_macro_f1={x['coarse_macro_f1']:.4f}")
        if "f1_ci_low" in x:
            line += f" CI=[{x['f1_ci_low']:.4f},{x['f1_ci_high']:.4f}]"
        if x.get("tau") is not None:
            line += f" tau={x['tau']:.4f}(reached90={x['tau_reached_90']}) answered_acc={x.get('answered_acc')} coverage={x.get('coverage'):.4f}"
        if "agree_bf16" in x:
            line += f" agree_bf16={x['agree_bf16']:.4f} letter_kld={x['letter_kld']:.5f}"
        line += " per_class=" + ",".join(f"{k}:{v:.3f}" for k, v in x["per_class_f1"].items())
        print(line)


def cmd_score(a):
    finish(a.run_dir, a.ref)


def cmd_cpu(a):
    """CPU proxy: -ngl 0, 4 threads, first 50 test images by SHA-256, peak RSS via /usr/bin/time -v."""
    prompt = read_prompt(a.prompt)
    template = a.chat_template_file or os.path.join(ROOT, "protocol", "chat_templates", a.model + ".jinja")
    rows_in = sorted(read_manifest(a.manifest, "test"), key=lambda r: r["sha256"])[: a.n]
    tag = a.tag or "__".join([a.model, os.path.basename(a.lm).replace(".gguf", ""), "cpu4"])
    out = os.path.join(a.out_root, tag)
    os.makedirs(out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    imgs = [preprocess(os.path.join(a.data_root, r["path"]), r["sha256"], a.cache) for r in rows_in]
    extra = [] if a.server_cache else ["--cache-ram", "0", "--ctx-checkpoints", "0"]
    srv = Server(a.lm, a.mmproj, template, a.port, None, 0, a.threads, os.path.join(out, "server.log"), cpu=True,
                 extra=extra)
    times = []
    try:
        srv.post("/v1/chat/completions", build_request(prompt, imgs[0], a.prob_mode))  # warm-up, not timed
        for img in imgs:
            t = time.time()
            srv.post("/v1/chat/completions", build_request(prompt, img, a.prob_mode))
            times.append(time.time() - t)
    finally:
        srv.stop()
    facts = log_facts(os.path.join(out, "server.log"))
    res = {"model": a.model, "lm_file": os.path.basename(a.lm), "n": len(times), "threads": a.threads,
           "cpu4_sec_per_img": float(np.mean(times)), "cpu_peak_rss_mb": facts["peak_rss_mb"],
           "llama_cpp_commit": git_head(LLAMA_DIR), "server_extra": extra,
           "note": "proxy for a budget phone, x86 CPU"}
    json.dump(res, open(os.path.join(out, "cpu.json"), "w"), indent=1)
    print(json.dumps(res))


def cmd_textkld(a):
    """llama-perplexity --kl-divergence vs BF16, wikitext-2 test, first 40 chunks, ctx 512."""
    b = os.path.join(LLAMA_DIR, "build", "bin", "llama-perplexity")
    env = dict(os.environ)
    if a.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    base = a.base or (os.path.splitext(a.bf16)[0] + ".kld-base.bin")
    common = ["-f", a.text, "-c", "512", "--chunks", "40", "-ngl", "99", "--seed", "0"]
    lock = base + ".lock"
    if not os.path.exists(base):
        try:
            os.mkdir(lock)  # one writer per base file; other workers wait for it
        except FileExistsError:
            for _ in range(1800):
                if os.path.exists(base):
                    break
                time.sleep(1)
    if not os.path.exists(base):
        rb = subprocess.run([b, "-m", a.bf16, "--kl-divergence-base", base + ".tmp"] + common, env=env,
                            capture_output=True, text=True)
        if rb.returncode != 0:
            res = {"lm_file": os.path.basename(a.lm), "text_kld_mean": None, "text_same_top1": None,
                   "error": "base failed: " + (rb.stdout + rb.stderr)[-500:]}
            print(json.dumps(res))
            if a.out:
                json.dump(res, open(a.out, "w"), indent=1)
            return
        os.replace(base + ".tmp", base)
    r = subprocess.run([b, "-m", a.lm, "--kl-divergence-base", base, "--kl-divergence"] + common,
                       env=env, capture_output=True, text=True)
    txt = r.stdout + r.stderr
    if a.log:
        open(a.log, "w").write(txt)
    kld = re.search(r"Mean\s+KLD:\s+([-\d.]+)", txt)
    top = re.search(r"Same top p:\s+([\d.]+)", txt)
    res = {"lm_file": os.path.basename(a.lm), "text_kld_mean": float(kld.group(1)) if kld else None,
           "text_same_top1": float(top.group(1)) / 100 if top else None, "returncode": r.returncode}
    print(json.dumps(res))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", required=True, choices=["Qwen3.5-2B", "Qwen3.5-0.8B"])
        p.add_argument("--lm", required=True)
        p.add_argument("--mmproj", required=True)
        p.add_argument("--prompt", required=True)
        p.add_argument("--chat-template-file", default=None, help="default: protocol/chat_templates/<model>.jinja")
        p.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.csv"))
        p.add_argument("--data-root", default=os.environ.get("BRACOL_ROOT", os.path.join(WORK, "bracol", "raw")))
        p.add_argument("--cache", default=os.environ.get("HN04B_CACHE", os.path.join(WORK, "bracol", "cache512")))
        p.add_argument("--out-root", default=os.environ.get("HN04B_RUNS", os.path.join(WORK, "runs")))
        p.add_argument("--tag", default=None)
        p.add_argument("--port", type=int, default=8090)
        p.add_argument("--prob-mode", choices=["pre50", "post"], default="pre50",
                       help="pre50: n_probs 50 raw softmax renormalised over the letters (default, see README); "
                            "post: post_sampling_probs n_probs 6")
        p.add_argument("--threads", type=int, default=4)

    p = sub.add_parser("run"); common(p)
    p.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--gpu", default=None, help="physical GPU index for CUDA_VISIBLE_DEVICES")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ref", default=None, help="BF16 images.csv for agree_bf16 / letter_kld")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("cpu"); common(p)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--server-cache", action="store_true", help="keep llama-server's RAM prompt cache and checkpoints")
    p.set_defaults(fn=cmd_cpu)

    p = sub.add_parser("score")
    p.add_argument("--run-dir", required=True); p.add_argument("--ref", default=None)
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("textkld")
    p.add_argument("--bf16", required=True); p.add_argument("--lm", required=True)
    p.add_argument("--text", default=os.environ.get("WIKITEXT", os.path.join(WORK, "calib", "wikitext2_test.txt")))
    p.add_argument("--base", default=None); p.add_argument("--gpu", default=None)
    p.add_argument("--out", default=None); p.add_argument("--log", default=None)
    p.set_defaults(fn=cmd_textkld)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
