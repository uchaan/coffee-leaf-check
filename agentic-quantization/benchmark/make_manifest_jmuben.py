#!/usr/bin/env python3
"""Optional extra test set (protocol: benchmark/README.md): JMuBEN + JMuBEN2 (Mendeley t2r6rszp5c/1, tgv3zb82nd/1; CC BY 4.0).
Test only: per class, sort the images by SHA-256 of the file bytes and take the first 600.
The archives hold many byte-identical copies (Healthy: 18,984 files, 63 distinct images), so copies are
collapsed first (one row per distinct SHA-256 within a class, the first path in sort order); a class with
fewer than 600 distinct images contributes all of them. No SHA-256 occurs in two classes.
Files that PIL cannot decode (damaged JPEGs, e.g. Healthy/2 (691).jpg ends in 00 00, not FF D9) are dropped
before the selection and counted.
  python benchmark/make_manifest_jmuben.py --root $JMUBEN_ROOT --out manifest_jmuben.csv
"""
import argparse, collections, csv, hashlib, os

# folder name in the archives -> our letter
CLASS_TO_LETTER = {"healthy": "A", "leaf rust": "B", "cerscospora": "C", "phoma": "D", "miner": "E"}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--per-class", type=int, default=600)
a = ap.parse_args()
by = collections.defaultdict(list)
for dp, _, fns in os.walk(a.root):
    for fn in fns:
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = [p.lower() for p in os.path.relpath(dp, a.root).split(os.sep)]
        letter = next((CLASS_TO_LETTER[p] for p in parts if p in CLASS_TO_LETTER), None)
        if letter is None:
            continue
        rel = os.path.relpath(os.path.join(dp, fn), a.root)
        sha = hashlib.sha256(open(os.path.join(a.root, rel), "rb").read()).hexdigest()
        by[letter].append((sha, rel))
from PIL import Image


def decodes(path):
    try:
        with Image.open(path) as im:
            im.convert("RGB").load()
        return True
    except Exception:
        return False


rows = []
uniq = {}
bad = collections.Counter()
for L in sorted(by):
    first = {}
    for sha, rel in sorted(by[L]):
        first.setdefault(sha, rel)
    ok = []
    for sha, rel in sorted(first.items()):
        if len(ok) >= a.per_class:
            break
        if decodes(os.path.join(a.root, rel)):
            ok.append((sha, rel))
        else:
            bad[L] += 1
    uniq[L] = ok
    uniq[L + "_n"] = len(first)
    for sha, rel in uniq[L][: a.per_class]:
        rows.append((rel, sha, "test", L))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "sha256", "split", "label_letter"]); w.writerows(rows)
for L in sorted(by):
    print(L, "files", len(by[L]), "distinct", uniq[L + "_n"], "undecodable before the cut", bad[L], "used", len(uniq[L]))
