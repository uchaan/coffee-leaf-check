#!/usr/bin/env python3
"""Side-by-side dev (and test) numbers for each ours file and its size-matched public counterpart.
  python benchmark/pairs.py --prompt prompt_v1 --jobs $WORK/jobs.csv --model Qwen3.5-2B
"""
import argparse, csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from queue_common import RUNS, tag_of  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True); ap.add_argument("--jobs", required=True)
ap.add_argument("--model", required=True); ap.add_argument("--runs", default=RUNS)
a = ap.parse_args()
jobs = [j for j in csv.DictReader(open(a.jobs)) if j["model"] == a.model]


def m(j):
    p = os.path.join(a.runs, tag_of(j, a.prompt), "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def line(j):
    x = m(j)
    b = os.path.getsize(j["lm_path"])
    if not x:
        return f"{j['source']:<14} {j['quant_name']:<12} {b:>13,}  (not run yet)"
    d, t = x["dev"], x.get("test", {})
    return (f"{j['source']:<14} {j['quant_name']:<12} {b:>13,}  dev F1 {d['forced_macro_f1']:.4f} "
            f"kld {d.get('letter_kld', 0):.4f} agree {d.get('agree_bf16', 1):.3f} | test F1 {t.get('forced_macro_f1', float('nan')):.4f} "
            f"[{t.get('f1_ci_low', float('nan')):.4f},{t.get('f1_ci_high', float('nan')):.4f}] kld {t.get('letter_kld', 0):.4f} agree {t.get('agree_bf16', 1):.3f}")


for j in jobs:
    if j["source"] == "bf16":
        print(line(j))
for j in [j for j in jobs if j["source"].startswith("ours")]:
    print(line(j))
    q = j["quant_name"]
    for p in jobs:
        if p["source"] == "unsloth" and p["quant_name"] == q:
            print("   vs", line(p))
