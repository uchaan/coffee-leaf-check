#!/usr/bin/env python3
"""Summarise eval/general_bench.py runs (4090 replication): accuracy with a 95% bootstrap interval per benchmark,
letter KLD and agreement against BF16, and paired differences for files of identical size (ours vs Unsloth).
  python eval/general_collect.py --runs $HN04B_RUNS/general --model Qwen3.5-2B --out eval/results/general
"""
import argparse, csv, glob, json, os

import numpy as np

BENCHES = ["mmstar", "ai2d", "mmlu_redux"]


def load(d):
    rows = list(csv.DictReader(open(os.path.join(d, "items.csv"))))
    meta = json.load(open(os.path.join(d, "metrics.json")))
    return rows, meta


def acc_ci(ok, n=1000, seed=0):
    ok = np.asarray(ok, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(ok), size=(n, len(ok)))
    v = ok[idx].mean(1)
    return float(ok.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def paired(oka, okb, n=1000, seed=0):
    oka, okb = np.asarray(oka, dtype=float), np.asarray(okb, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(oka), size=(n, len(oka)))
    d = oka[idx].mean(1) - okb[idx].mean(1)
    return float(oka.mean() - okb.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pair", action="append", default=[],
                    help="extra comparison OURS_STEM:PUBLIC_STEM (e.g. the 1 GB package file vs the smallest Unsloth file)")
    ap.add_argument("--final", default=None, help="handoff final.csv: keep only the ours files it names")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    keep = None
    if a.final:
        keep = {os.path.basename(r["file"]).replace(".gguf", "") for r in csv.DictReader(open(a.final))}
    runs = {}
    for d in sorted(glob.glob(os.path.join(a.runs, f"{a.model}__*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metrics.json")):
            stem = os.path.basename(d).split("__", 1)[1]
            if keep is not None and "-ours-" in stem and stem not in keep:
                continue
            runs[stem] = load(d)
    table, L = [], []
    for stem, (rows, meta) in runs.items():
        for b in BENCHES:
            R = sorted([r for r in rows if r["bench"] == b], key=lambda r: r["id"])
            acc, lo, hi = acc_ci([r["pred"] == r["answer"] for r in R])
            m = meta["metrics"].get(b, {})
            table.append({"file": stem, "lm_bytes": meta["lm_bytes"], "bench": b, "n": len(R), "acc": acc, "ci_low": lo,
                          "ci_high": hi, "letter_mass": m.get("letter_mass"), "letter_kld": m.get("letter_kld"),
                          "agree_bf16": m.get("agree_ref")})
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader()
        for r in sorted(table, key=lambda r: (r["bench"], r["lm_bytes"])):
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    L += [f"# General benchmarks, 4090 replication ({a.model})", "",
          "Script: `eval/general_bench.py` from the A100; items identical to the A100's",
          "(7,830 of 7,830). Same llama.cpp commit, chat template, canonical Q8_0 mmproj and request body as the BRACOL",
          "harness. Accuracy intervals: 95% bootstrap over items (1,000 resamples, seed 0).", ""]
    for b in BENCHES:
        L += [f"## {b}", "", "| file | LM bytes | n | accuracy [95% CI] | letter mass | letter KLD vs BF16 | agree with BF16 |",
              "|---|---:|---:|---|---|---|---|"]
        for r in sorted([r for r in table if r["bench"] == b], key=lambda r: -r["lm_bytes"]):
            kld = "–" if r["letter_kld"] is None else f"{r['letter_kld']:.4f}"
            ag = "–" if r["agree_bf16"] is None else f"{r['agree_bf16']:.4f}"
            L.append(f"| {r['file']} | {r['lm_bytes']:,} | {r['n']} | {r['acc']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | "
                     f"{r['letter_mass']} | {kld} | {ag} |")
        L.append("")
    # pairs: ours vs Unsloth with identical LM bytes
    pairs = []
    for so, (ro, mo) in runs.items():
        if "-ours-" not in so:
            continue
        for su, (ru, mu) in runs.items():
            if "-ours-" in su or su.endswith("BF16") or mu["lm_bytes"] != mo["lm_bytes"]:
                continue
            pairs.append((so, su, ro, ru, mo["lm_bytes"]))
    for spec in a.pair:
        so, su = spec.split(":")
        if so in runs and su in runs:
            pairs.append((so, su, runs[so][0], runs[su][0], runs[so][1]["lm_bytes"]))
    if pairs:
        L += ["## Paired differences (ours minus public), accuracy",
              "Pairs: identical LM bytes, plus any pair named with --pair (LM bytes column: ours).", "",
              "| ours | public | LM bytes | benchmark | Δ accuracy [paired 95% CI] | verdict |", "|---|---|---:|---|---|---|"]
        for so, su, ro, ru, nb in sorted(pairs, key=lambda x: x[4]):
            for b in BENCHES:
                A = {r["id"]: r for r in ro if r["bench"] == b}; B = {r["id"]: r for r in ru if r["bench"] == b}
                ids = sorted(set(A) & set(B))
                d, lo, hi = paired([A[i]["pred"] == A[i]["answer"] for i in ids], [B[i]["pred"] == B[i]["answer"] for i in ids])
                verdict = "no difference" if lo <= 0 <= hi else ("ours higher" if lo > 0 else "ours lower")
                L.append(f"| {so} | {su} | {nb:,} | {b} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {verdict} |")
        L.append("")
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print(len(runs), "runs,", len(pairs), "pairs ->", a.out)


if __name__ == "__main__":
    main()
