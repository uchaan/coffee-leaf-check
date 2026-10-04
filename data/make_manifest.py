#!/usr/bin/env python3
"""Build data/manifest.csv from the BRACOL leaf dataset (protocol: eval/README.md).

BRACOL root = the folder holding dataset.csv and images/ (leaf-level set).
Per class: sort images by SHA-256 of file bytes; first floor(25%) -> dev, rest -> test.
Rows with predominant_stress code 5 (no single predominant stress) are excluded,
as in the authors' own classification split (lara2018 classification/dataset/dataset.csv).
"""
import argparse, csv, hashlib, os, collections

# BRACOL predominant_stress code -> (name in lara2018 results.py, our letter)
CODE_TO_LETTER = {0: ("Healthy", "A"), 2: ("Rust", "B"), 4: ("Cercospora", "C"),
                  3: ("Phoma (brown leaf spot)", "D"), 1: ("Leaf miner", "E")}

ap = argparse.ArgumentParser()
ap.add_argument("--bracol-root", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.csv"))
a = ap.parse_args()

by_class = collections.defaultdict(list)
excluded = 0
with open(os.path.join(a.bracol_root, "dataset.csv")) as f:
    for r in csv.DictReader(f):
        code = int(r["predominant_stress"])
        if code not in CODE_TO_LETTER:
            excluded += 1
            continue
        rel = f"images/{r['id']}.jpg"
        sha = hashlib.sha256(open(os.path.join(a.bracol_root, rel), "rb").read()).hexdigest()
        by_class[CODE_TO_LETTER[code][1]].append((sha, rel))

rows = []
for letter in sorted(by_class):
    items = sorted(by_class[letter])
    n_dev = len(items) // 4
    for i, (sha, rel) in enumerate(items):
        rows.append((rel, sha, "dev" if i < n_dev else "test", letter))
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "sha256", "split", "label_letter"]); w.writerows(rows)

cnt = collections.Counter((r[3], r[2]) for r in rows)
print(f"excluded (code 5): {excluded}")
for L in sorted(by_class):
    print(L, "total", cnt[(L, 'dev')] + cnt[(L, 'test')], "dev", cnt[(L, 'dev')], "test", cnt[(L, 'test')])
print("all dev", sum(v for k, v in cnt.items() if k[1] == 'dev'), "test", sum(v for k, v in cnt.items() if k[1] == 'test'))
