#!/usr/bin/env python3
"""Collect every run of one prompt into results.csv, per-image files, charts and summary.md.

  python benchmark/collect.py --prompt prompt_v1 --jobs $WORK/jobs.csv --out results/
"""
import argparse, csv, json, os, shutil, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
from queue_common import RUNS, tag_of  # noqa: E402

COLS = ["model", "source", "quant_name", "lm_file", "lm_bytes", "mmproj_file", "mmproj_bytes", "total_bytes",
        "bpw", "calibration", "llama_cpp_commit", "dataset", "split", "n", "forced_acc", "forced_macro_f1",
        "f1_ci_low", "f1_ci_high", "coarse_macro_f1", "agree_bf16", "letter_kld", "tau", "answered_acc",
        "coverage", "notsure_share", "text_kld_mean", "text_same_top1", "cpu4_sec_per_img", "cpu_peak_rss_mb"]
PACKAGE = 1_000_000_000
GROUPS = ["ours-general", "ours-task", "unsloth", "official", "others"]
runs_j = {}
STYLE = {"ours-general": ("#1f77b4", "o"), "ours-task": ("#17becf", "s"), "unsloth": ("#d62728", "^"),
         "official": ("#2ca02c", "D"), "others": ("#7f7f7f", "v")}


RESULTS_README = """# Results (frozen prompt `protocol/prompt_v1.txt`)

| File | What |
|---|---|
| `summary.md` | matched-size differences, under-1 GB package table, confusion matrix, JMuBEN table, checks |
| `results.csv` | one row per model x source x file x dataset x split (metrics: `benchmark/README.md`); BRACOL dev and test, JMuBEN test |
| `f1_<model>.png` | forced macro-F1 on BRACOL test vs language-model bytes, 95% bootstrap intervals, 1 GB package line |
| `kld_<model>.png` | letter KLD vs BF16 on BRACOL test vs language-model bytes (log scale) |
| `per_image/<run>.csv.gz` | per image (gzip): path, image SHA-256, split, label, P(A)..P(F), argmax, forced argmax, letter mass, sampled letter |
| `per_image/<run>.meta.json` | file sizes and SHA-256, llama.cpp and harness commits, prompt and template SHA-256, probability mode |
| `low_letter_mass.csv` | files that put on average under 90% of the first-token probability on A to F |

Every number is recomputed by `benchmark/collect.py` from the per-image files. Metric definitions: `benchmark/README.md`.
`per_image/` can be large; leave it out when publishing and say so.
"""


def group(src):
    return "others" if src.startswith("other:") else src


def load_rows(a):
    jobs = list(csv.DictReader(open(a.jobs)))
    rows, runs = [], {}
    global runs_j
    refs, bf_lm = {}, {}
    for j in jobs:
        if j["source"] == "bf16":
            refs[j["model"]] = os.path.join(a.runs, tag_of(j, a.prompt), "images.csv")
            bf_lm[j["model"]] = j["lm_path"]
    odd = {"failed": [], "letter_tokens": [], "low_letter_mass": [], "prompt_sha": set(), "commit": set(), "stale": []}
    ours_sha = {}
    if a.files and os.path.exists(a.files):
        for r in csv.DictReader(open(a.files)):
            ours_sha[os.path.basename(r["file"])] = r["sha256"]
    odd["final"] = None
    if a.final and os.path.exists(a.final):
        odd["final"] = {os.path.basename(r["file"]) for r in csv.DictReader(open(a.final)) if r.get("file")}
    bf_tokens = {}
    for j in sorted(jobs, key=lambda j: j["source"] != "bf16"):
        tag = tag_of(j, a.prompt)
        d = os.path.join(a.runs, tag)
        if os.path.exists(os.path.join(d, "FAILED")):
            tail = open(os.path.join(d, "harness.out"), errors="replace").read()[-400:] if os.path.exists(os.path.join(d, "harness.out")) else ""
            odd["failed"].append((j["model"], j["source"], j["quant_name"], tail))
            continue
        if not os.path.exists(os.path.join(d, "images.csv")) or not os.path.exists(os.path.join(d, "meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        want = ours_sha.get(os.path.basename(j["lm_path"]))
        if want and meta.get("lm_sha256") != want:
            odd["stale"].append((j["model"], j["source"], j["quant_name"], meta.get("lm_sha256", "?")[:12], want[:12]))
            continue
        odd["prompt_sha"].add(meta["prompt_sha256"]); odd["commit"].add(meta["llama_cpp_commit"])
        if j["source"] == "bf16":
            bf_tokens[j["model"]] = meta.get("letter_tokens")
        elif meta.get("letter_tokens") != bf_tokens.get(j["model"]):
            odd["letter_tokens"].append((j["model"], j["source"], j["quant_name"], meta.get("letter_tokens")))
        irows, IP = H.load_images_csv(os.path.join(d, "images.csv"))
        mass = float(np.mean([float(r["raw_letter_mass"]) for r in irows]))
        if mass < 0.9:
            odd["low_letter_mass"].append((j["model"], j["source"], j["quant_name"], mass))
        ref = None
        if j["source"] != "bf16" and os.path.exists(refs.get(j["model"], "")):
            ref = H.load_images_csv(refs[j["model"]])
        met = H.score(irows, IP, ref)  # rescored here so every number comes from one code version
        stem = os.path.basename(j["lm_path"]).replace(".gguf", "")
        kj = os.path.join(a.runs, "textkld", f"{j['model']}__{stem}.json")
        kld = json.load(open(kj)) if os.path.exists(kj) else {}
        cj = os.path.join(a.runs, "cpu", tag + "__cpu4", "cpu.json")
        cpu = json.load(open(cj)) if os.path.exists(cj) else {}
        runs[(j["model"], j["source"], j["quant_name"])] = d
        # optional JMuBEN test run of the same file: tau comes from this file's BRACOL dev run
        jd = os.path.join(a.runs, tag_of(j, a.prompt + "__jmuben"))
        if "dev" in met and os.path.exists(os.path.join(jd, "images.csv")):
            jrows, JP = H.load_images_csv(os.path.join(jd, "images.csv"))
            jref_path = os.path.join(a.runs, tag_of({"model": j["model"], "source": "bf16", "lm_path": bf_lm.get(j["model"], "")},
                                                    a.prompt + "__jmuben"), "images.csv")
            jref = H.load_images_csv(jref_path) if j["source"] != "bf16" and os.path.exists(jref_path) else None
            jm = H.score(jrows, JP, jref)["test"]
            tau = met["dev"].get("tau")
            if tau is not None:
                jm.update(H.answered_stats([r["label"] for r in jrows], JP, tau))
            else:
                jm.update(answered_acc=None, coverage=0.0, notsure_share=1.0)
            jm["tau"] = tau
            met["jmuben"] = jm
            runs_j[(j["model"], j["source"], j["quant_name"])] = jd
        for split in ("test", "dev", "jmuben"):
            if split not in met:
                continue
            m = met[split]
            r = {"model": j["model"], "source": j["source"], "quant_name": j["quant_name"],
                 "lm_file": meta["lm_file"], "lm_bytes": meta["lm_bytes"], "mmproj_file": meta["mmproj_file"],
                 "mmproj_bytes": meta["mmproj_bytes"], "total_bytes": meta["lm_bytes"] + meta["mmproj_bytes"],
                 "bpw": meta.get("bpw"), "calibration": j.get("calibration", ""),
                 "llama_cpp_commit": meta["llama_cpp_commit"],
                 "dataset": "JMuBEN" if split == "jmuben" else "BRACOL", "split": "test" if split == "jmuben" else split}
            for k in ["n", "forced_acc", "forced_macro_f1", "f1_ci_low", "f1_ci_high", "coarse_macro_f1",
                      "agree_bf16", "letter_kld", "tau", "answered_acc", "coverage", "notsure_share"]:
                r[k] = m.get(k)
            if j["source"] == "bf16":
                r["agree_bf16"], r["letter_kld"] = 1.0, 0.0
                r["text_kld_mean"], r["text_same_top1"] = 0.0, 1.0
            else:
                r["text_kld_mean"], r["text_same_top1"] = kld.get("text_kld_mean"), kld.get("text_same_top1")
            r["cpu4_sec_per_img"], r["cpu_peak_rss_mb"] = cpu.get("cpu4_sec_per_img"), cpu.get("cpu_peak_rss_mb")
            r["_per_class"] = m.get("per_class_f1")
            r["_confusion"] = m.get("confusion")
            r["_tau_reached"] = m.get("tau_reached_90")
            r["_letter_mass"] = mass
            rows.append(r)
    return rows, runs, odd


def fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def test_preds(run_dir):
    rows, P = H.load_images_csv(os.path.join(run_dir, "images.csv"))
    ii = [i for i, r in enumerate(rows) if r["split"] == "test"]
    order = sorted(ii, key=lambda i: rows[i]["path"])
    y = [rows[i]["label"] for i in order]
    fp = H.forced_pred(P[order])
    return [rows[i]["path"] for i in order], y, fp


def paired(run_a, run_b):
    pa, y, fa = test_preds(run_a)
    pb, yb, fb = test_preds(run_b)
    assert pa == pb and y == yb
    return H.paired_bootstrap_diff(y, fa, fb)


def charts(test, out, final=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for model in sorted(set(r["model"] for r in test)):
        rs = [r for r in test if r["model"] == model]
        bf = [r for r in rs if r["source"] == "bf16"]
        mm = rs[0]["mmproj_bytes"]
        for metric, ylabel, fname in [("forced_macro_f1", "forced macro-F1, BRACOL test (95% bootstrap CI)", "f1"),
                                      ("letter_kld", "letter KLD vs BF16, BRACOL test (lower is better)", "kld")]:
            fig, ax = plt.subplots(figsize=(10, 6))
            extra = [r for r in rs if r["source"].startswith("ours") and final is not None
                     and r["lm_file"] not in final and r[metric] is not None]
            if extra:
                ax.plot([r["lm_bytes"] / 1e9 for r in extra], [r[metric] for r in extra], ls="none", marker="o",
                        mfc="none", mec="#9ecae1", ms=5, label="ours, other variants (not in final.csv)")
            for g in GROUPS:
                pts = sorted([r for r in rs if group(r["source"]) == g and r[metric] is not None
                              and (final is None or not g.startswith("ours") or r["lm_file"] in final)],
                             key=lambda r: r["lm_bytes"])
                if not pts:
                    continue
                x = [r["lm_bytes"] / 1e9 for r in pts]
                yv = [r[metric] for r in pts]
                c, mk = STYLE[g]
                # "others" mixes several repositories, so its points are not joined by a line
                ls = "none" if g == "others" else "-"
                lab = "others (bartowski, lmstudio-community, mradermacher; points only)" if g == "others" else g
                if metric == "forced_macro_f1":
                    err = [[r[metric] - r["f1_ci_low"] for r in pts], [r["f1_ci_high"] - r[metric] for r in pts]]
                    ax.errorbar(x, yv, yerr=err, color=c, marker=mk, ls=ls, lw=1.2, ms=5, capsize=2, label=lab, alpha=.9)
                else:
                    ax.plot(x, yv, color=c, marker=mk, ls=ls, lw=1.2, ms=5, label=lab, alpha=.9)
            if bf and metric == "forced_macro_f1":
                b = bf[0]
                ax.axhline(b[metric], color="k", ls=":", lw=1, label=f"BF16 {b[metric]:.3f} ({b['lm_bytes'] / 1e9:.2f} GB)")
                # best macro-F1 reachable by answering one fixed letter for every test image
                conf = b["_confusion"]
                n_c = {c: sum(conf[c].values()) for c in "ABCDE"}
                N = sum(n_c.values())
                cbest = max(n_c, key=lambda c: n_c[c])
                const = 2 * n_c[cbest] / (n_c[cbest] + N) / 5
                ax.axhline(const, color="#999999", ls="-.", lw=1, label=f"one fixed answer for every image (always {cbest}): {const:.3f}")
            ax.axvline((PACKAGE - mm) / 1e9, color="k", ls="--", lw=1,
                       label=f"1 GB package: LM < {(PACKAGE - mm) / 1e9:.3f} GB with the Q8_0 mmproj")
            ax.set_xlabel("language-model file size (GB, 1e9 bytes)")
            ax.set_ylabel(ylabel)
            if metric == "letter_kld":
                ax.set_yscale("log")
            ax.set_title(f"{model}: {metric} vs LM bytes (prompt and mmproj identical for every file)")
            ax.grid(alpha=.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"{fname}_{model}.png"), dpi=130)
            plt.close(fig)


def summary(test, runs, out, prompt, odd, jm=()):
    L = [f"# Hack-Nation 04B benchmark summary ({prompt})", "",
         "BRACOL test (1,266 images), forced five-class macro-F1 over A to E. Every file: same llama.cpp commit,",
         "harness, Q8_0 mmproj, prompt and images. Intervals: 95% bootstrap (1,000 resamples, seed 0).",
         "A paired difference whose interval includes 0 is reported as no difference.", ""]
    shown = set()
    for model in sorted(set(r["model"] for r in test)):
        rs = [r for r in test if r["model"] == model]
        mm = rs[0]["mmproj_bytes"]
        L += [f"## {model}", ""]
        bf = [r for r in rs if r["source"] == "bf16"]
        if bf:
            b = bf[0]
            L += [f"BF16 reference: forced_macro_f1 {fmt(b['forced_macro_f1'])} [{fmt(b['f1_ci_low'])}, {fmt(b['f1_ci_high'])}], "
                  f"coarse_macro_f1 {fmt(b['coarse_macro_f1'])}, LM {b['lm_bytes']:,} B.", ""]
        all_ours = [r for r in rs if r["source"].startswith("ours")]
        ours = [r for r in all_ours if odd.get("final") is None or r["lm_file"] in odd["final"]]
        L += [f"Ours files evaluated for this model: {len(all_ours)}. " + (
              f"Comparisons use the {len(ours)} files named in selection/final.csv, chosen by the registered dev rule before their test "
              f"numbers were read (A100: Qwen3.5-2B and the 0.8B targets above 338 MB; 4090: the 0.8B 338 / 310 / 290 MB targets, "
              f"see selection/lowend_0.8B_dev_screen.md)."
              if odd.get("final") is not None else "No selection/final.csv yet: comparisons use every ours file."), ""]
        if any(r["source"] == "ours-task" for r in rs):
            L += ["Note: ours-task files were calibrated on the 419 BRACOL dev images (with the frozen prompt). Their dev",
                  "numbers and their tau (chosen on dev) are therefore in-sample; the test numbers below are not.", ""]
        pub = [r for r in rs if r["source"] not in ("bf16",) and not r["source"].startswith("ours")]
        # matched sizes
        L += ["### Matched sizes: ours minus best public at the same or larger size",
              "",
              "Comparator: the public file with the highest forced_macro_f1 among public files whose LM bytes are",
              "at least ours and at most 10% larger. If there is none, the smallest public file larger than ours.",
              "",
              "| ours | ours LM bytes | public comparator | public LM bytes | Δ forced_macro_f1 [paired 95% CI] | verdict | agree_bf16 ours / public (Δ) | letter_kld ours / public (Δ) |",
              "|---|---:|---|---:|---|---|---|---|"]
        for o in sorted(ours, key=lambda r: r["lm_bytes"]):
            cand = [p for p in pub if o["lm_bytes"] <= p["lm_bytes"] <= 1.10 * o["lm_bytes"]]
            note = ""
            if not cand:
                larger = sorted([p for p in pub if p["lm_bytes"] >= o["lm_bytes"]], key=lambda r: r["lm_bytes"])
                cand = larger[:1]
                note = " (nearest larger)"
            if not cand:
                L.append(f"| {o['source']} {o['quant_name']} | {o['lm_bytes']:,} | none | | | | | |")
                continue
            p = max(cand, key=lambda r: r["forced_macro_f1"])
            shown.add((model, p["source"], p["quant_name"]))
            d, lo, hi = paired(runs[(model, o["source"], o["quant_name"])], runs[(model, p["source"], p["quant_name"])])
            verdict = "no difference" if lo <= 0 <= hi else ("ours higher" if lo > 0 else "ours lower")
            L.append(f"| {o['source']} {o['quant_name']} | {o['lm_bytes']:,} | {p['source']} {p['quant_name']}{note} | "
                     f"{p['lm_bytes']:,} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {verdict} | "
                     f"{fmt(o['agree_bf16'])} / {fmt(p['agree_bf16'])} ({o['agree_bf16'] - p['agree_bf16']:+.4f}) | "
                     f"{fmt(o['letter_kld'], 5)} / {fmt(p['letter_kld'], 5)} ({o['letter_kld'] - p['letter_kld']:+.5f}) |")
        L.append("")
        # Unsloth only: same bytes, else the smallest larger Unsloth file
        uns = sorted([r for r in rs if r["source"] == "unsloth"], key=lambda r: r["lm_bytes"])
        if uns and ours:
            L += ["### Against Unsloth: ours minus the Unsloth file of the same bytes, or the smallest larger one", "",
                  "| ours | ours LM bytes | Unsloth file | Unsloth LM bytes | Δ forced_macro_f1 [paired 95% CI] | verdict | ours / Unsloth forced_macro_f1 |",
                  "|---|---:|---|---:|---|---|---|"]
            for o in sorted(ours, key=lambda r: (r["lm_bytes"], r["source"])):
                u = next((x for x in uns if x["lm_bytes"] >= o["lm_bytes"]), None)
                if u is None:
                    L.append(f"| {o['source']} {o['quant_name']} | {o['lm_bytes']:,} | none larger | | | | |")
                    continue
                d, lo, hi = paired(runs[(model, o["source"], o["quant_name"])], runs[(model, "unsloth", u["quant_name"])])
                verdict = "no difference" if lo <= 0 <= hi else ("ours higher" if lo > 0 else "ours lower")
                L.append(f"| {o['source']} {o['quant_name']} | {o['lm_bytes']:,} | {u['quant_name']} | {u['lm_bytes']:,} | "
                         f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {verdict} | {fmt(o['forced_macro_f1'])} / {fmt(u['forced_macro_f1'])} |")
            L.append("")
        # package view
        L += [f"### Under-1 GB package (LM + Q8_0 mmproj {mm:,} B < 1,000,000,000 B)", "",
              "| source | best file | LM bytes | total bytes | forced_macro_f1 [95% CI] | coarse_macro_f1 | agree_bf16 | letter_kld | tau | answered_acc / coverage | CPU s/img (proxy) | peak RSS MB |",
              "|---|---|---:|---:|---|---|---|---|---|---|---|---|"]
        fit = [r for r in rs if r["total_bytes"] < PACKAGE and r["source"] != "bf16"
               and (not r["source"].startswith("ours") or r in ours)]
        best_by = {}
        for r in fit:
            k = r["source"]
            if k not in best_by or r["forced_macro_f1"] > best_by[k]["forced_macro_f1"]:
                best_by[k] = r
        srcs = sorted(best_by, key=lambda s: (not s.startswith("ours"), s))
        for s in srcs:
            r = best_by[s]
            L.append(f"| {s} | {r['quant_name']} | {r['lm_bytes']:,} | {r['total_bytes']:,} | {fmt(r['forced_macro_f1'])} "
                     f"[{fmt(r['f1_ci_low'])}, {fmt(r['f1_ci_high'])}] | {fmt(r['coarse_macro_f1'])} | {fmt(r['agree_bf16'])} | "
                     f"{fmt(r['letter_kld'], 5)} | {fmt(r['tau'])} | {fmt(r['answered_acc'])} / {fmt(r['coverage'])} | "
                     f"{fmt(r['cpu4_sec_per_img'], 2)} | {fmt(r['cpu_peak_rss_mb'], 0)} |")
        uns_fit = [r for r in fit if r["source"] == "unsloth"]
        if not uns_fit:
            us = sorted([r for r in rs if r["source"] == "unsloth"], key=lambda r: r["lm_bytes"])
            if us:
                u = us[0]
                L.append(f"| unsloth (smallest; does not fit) | {u['quant_name']} | {u['lm_bytes']:,} | {u['total_bytes']:,} | "
                         f"{fmt(u['forced_macro_f1'])} [{fmt(u['f1_ci_low'])}, {fmt(u['f1_ci_high'])}] | {fmt(u['coarse_macro_f1'])} | "
                         f"{fmt(u['agree_bf16'])} | {fmt(u['letter_kld'], 5)} | {fmt(u['tau'])} | "
                         f"{fmt(u['answered_acc'])} / {fmt(u['coverage'])} | {fmt(u['cpu4_sec_per_img'], 2)} | {fmt(u['cpu_peak_rss_mb'], 0)} |")
        L.append("")
        # paired differences between the best ours package and each public package
        bo = [best_by[s] for s in srcs if s.startswith("ours")]
        if bo:
            o = max(bo, key=lambda r: r["forced_macro_f1"])
            L += [f"Best ours package: {o['source']} {o['quant_name']}. Paired difference against each public package:", ""]
            for s in srcs:
                if s.startswith("ours"):
                    continue
                p = best_by[s]
                d, lo, hi = paired(runs[(model, o["source"], o["quant_name"])], runs[(model, p["source"], p["quant_name"])])
                verdict = "no difference" if lo <= 0 <= hi else ("ours higher" if lo > 0 else "ours lower")
                L.append(f"- vs {s} {p['quant_name']}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}], {verdict}; "
                         f"total bytes {o['total_bytes']:,} vs {p['total_bytes']:,}.")
            if not uns_fit:
                us = sorted([r for r in rs if r["source"] == "unsloth"], key=lambda r: r["lm_bytes"])
                if us:
                    u = us[0]
                    d, lo, hi = paired(runs[(model, o["source"], o["quant_name"])], runs[(model, "unsloth", u["quant_name"])])
                    verdict = "no difference" if lo <= 0 <= hi else ("ours higher" if lo > 0 else "ours lower")
                    L.append(f"- vs unsloth {u['quant_name']} (smallest Unsloth file; its package is over 1 GB): "
                             f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}], {verdict}; total bytes {o['total_bytes']:,} vs {u['total_bytes']:,}.")
            L.append("")
        if fit:
            bp = max(fit, key=lambda r: r["forced_macro_f1"])
            L += [f"### Confusion matrix of the best under-1 GB package ({bp['source']} {bp['quant_name']}), BRACOL test",
                  "", "Rows: true label. Columns: forced prediction (A to E).", "",
                  "| true \\ pred | A | B | C | D | E |", "|---|---:|---:|---:|---:|---:|"]
            names = {"A": "A healthy", "B": "B rust", "C": "C cercospora", "D": "D phoma", "E": "E leaf miner"}
            for t in "ABCDE":
                L.append(f"| {names[t]} | " + " | ".join(str(bp["_confusion"][t][p]) for p in "ABCDE") + " |")
            L.append("")
            L.append("Per-class F1: " + ", ".join(f"{k} {v:.3f}" for k, v in bp["_per_class"].items()) + ".")
            L.append("")
    if jm:
        L += ["## Optional extra test set: JMuBEN + JMuBEN2 (field photos, Kenya)", "",
              "Distinct images only (the archives repeat files byte for byte), first 600 per class by SHA-256:",
              "A 62, B 600, C 322, D 600, E 600 (2,184; one damaged JPEG dropped). Same harness, prompt and mmproj; tau from each file's BRACOL dev run.", "",
              "| model | source | file | LM bytes | forced_macro_f1 [95% CI] | minus BF16 [paired 95% CI] | coarse_macro_f1 | agree_bf16 | letter_kld | answered_acc / coverage |",
              "|---|---|---|---:|---|---|---|---|---|---|"]
        final = odd.get("final")
        keep = []
        for r in jm:
            k = (r["model"], r["source"], r["quant_name"])
            if (r["source"] in ("bf16", "unsloth", "official") or k in shown
                    or (r["source"].startswith("ours") and (final is None or r["lm_file"] in final))):
                keep.append(r)
        for r in sorted(keep, key=lambda r: (r["model"], r["source"] != "bf16", r["lm_bytes"])):
            k = (r["model"], r["source"], r["quant_name"])
            vs = ""
            if r["source"] != "bf16" and k in runs_j and (r["model"], "bf16", "BF16") in runs_j:
                d, lo, hi = paired(runs_j[k], runs_j[(r["model"], "bf16", "BF16")])
                vs = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            L.append(f"| {r['model']} | {r['source']} | {r['quant_name']} | {r['lm_bytes']:,} | {fmt(r['forced_macro_f1'])} "
                     f"[{fmt(r['f1_ci_low'])}, {fmt(r['f1_ci_high'])}] | {vs} | {fmt(r['coarse_macro_f1'])} | {fmt(r['agree_bf16'])} | "
                     f"{fmt(r['letter_kld'], 5)} | {fmt(r['answered_acc'])} / {fmt(r['coverage'])} |")
        L.append("")
        L.append("Rows: BF16, every Unsloth and official file, the ours files used in the comparisons, and the public files that")
        L.append("appear as comparators in the BRACOL tables; every file is in results.csv (dataset JMuBEN).")
        L.append("")
    L += ["## Checks", ""]
    L.append(f"- Prompt SHA-256 over all runs: {', '.join(sorted(odd['prompt_sha']))}. llama.cpp commit: {', '.join(sorted(odd['commit']))}.")
    if odd["failed"]:
        for m_, s_, q_, tail in odd["failed"]:
            L.append(f"- Did not complete: {m_} {s_} {q_}. Last log lines: `{' '.join(tail.split())[-200:]}`")
    else:
        L.append("- Every listed file loaded and completed.")
    for m_, s_, q_, got, want in odd["stale"]:
        L.append(f"- Skipped a stale run: {m_} {s_} {q_} was measured on SHA-256 {got}…, files.csv now lists {want}….")
    if odd["letter_tokens"]:
        for m_, s_, q_, t in odd["letter_tokens"]:
            L.append(f"- Letter token ids differ from BF16: {m_} {s_} {q_}: {t}")
    else:
        L.append("- A to F are the same single tokens (ids 32 to 37) in every file.")
    if odd["low_letter_mass"]:
        with open(os.path.join(out, "low_letter_mass.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "source", "quant_name", "mean_letter_mass"])
            for m_, s_, q_, ms in sorted(odd["low_letter_mass"], key=lambda x: (x[0], x[3])):
                w.writerow([m_, s_, q_, f"{ms:.4f}"])
        for model in sorted(set(x[0] for x in odd["low_letter_mass"])):
            xs = [x for x in odd["low_letter_mass"] if x[0] == model]
            L.append(f"- {model}: {len(xs)} files put on average less than 90% of the first-token probability on A to F "
                     f"(the rest on other tokens, such as \"To\" or \"Here\"); their answers come from what the grammar leaves. "
                     f"Lowest: {min(xs, key=lambda x: x[3])[3]:.3f}. List: low_letter_mass.csv.")
    for model in sorted(set(r["model"] for r in test)):
        bf = [r for r in test if r["model"] == model and r["source"] == "bf16"]
        if not bf:
            continue
        hi = []
        for r in test:
            if r["model"] != model or r["source"] == "bf16":
                continue
            d, lo, hi_ = paired(runs[(model, r["source"], r["quant_name"])], runs[(model, "bf16", "BF16")])
            if lo > 0:
                hi.append(f"{r['source']} {r['quant_name']} {d:+.4f} [{lo:+.4f}, {hi_:+.4f}]")
        if hi:
            L.append(f"- {model}: quantized files above BF16 on test (paired interval above 0): " + "; ".join(hi) + ".")
        else:
            L.append(f"- {model}: no quantized file is above BF16 on test with a paired interval above 0.")
    L.append("")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True, help="prompt file stem, e.g. prompt_v1")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--files", default=os.environ.get("HN04B_FILES", os.path.join(H.ROOT, "selection", "files.csv")), help="files.csv (every built file)")
    ap.add_argument("--final", default=os.environ.get("HN04B_FINAL", os.path.join(H.ROOT, "selection", "final.csv")),
                    help="final.csv: when present, only these ours files enter the comparisons")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, runs, odd = load_rows(a)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["dataset"] != "BRACOL", r["split"] != "test", r["source"], r["lm_bytes"])):
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    pi = os.path.join(a.out, "per_image")
    os.makedirs(pi, exist_ok=True)
    import gzip
    for d in list(runs.values()) + list(runs_j.values()):
        with open(os.path.join(d, "images.csv"), "rb") as src, gzip.GzipFile(
                os.path.join(pi, f"{os.path.basename(d)}.csv.gz"), "wb", mtime=0) as dst:
            shutil.copyfileobj(src, dst)
        shutil.copy(os.path.join(d, "meta.json"), os.path.join(pi, f"{os.path.basename(d)}.meta.json"))
    test = [r for r in rows if r["split"] == "test" and r["dataset"] == "BRACOL"]
    jm = [r for r in rows if r["dataset"] == "JMuBEN"]
    charts(test, a.out, odd.get("final"))
    summary(test, runs, a.out, a.prompt, odd, jm)
    readme = os.path.join(a.out, "README.md")
    if not os.path.exists(readme):  # keep a hand-edited results/README.md
        open(readme, "w").write(RESULTS_README)
    print(f"{len(rows)} rows, {len(runs)} runs -> {a.out}")


if __name__ == "__main__":
    main()
