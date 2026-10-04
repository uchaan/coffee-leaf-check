"""JMuBEN grouped split: how jmuben_grouped_split.csv and jmuben_train_originals.csv were made.

JMuBEN/JMuBEN2 ship each photo as several rotated/flipped copies. jmuben_distinct_manifest.csv lists the 3,756 distinct
images (byte-identical copies collapsed; `source_path` = file in the Mendeley archives, `in_eval` = part of the 2,184-image
benchmark set in ../../agentic-quantization/manifest_jmuben.csv). `images/<sha256>.jpg` is the benchmark preprocessing
(harness.preprocess: RGB, longest side 512 px, Lanczos, JPEG q90) of the archive file with that SHA-256.

Grouping: a 16x16 average hash is computed for all 8 rotations/flips of each image; two images of the same class are
linked when the best of the 8 variants of one is within 20 of 256 bits of the other (union-find). Groups (= original
photos) are split 70/30 per class, ordered by their smallest SHA-256, keeping one image per group (the smallest SHA-256).
Result: 799 originals, 559 train (split=dev) / 240 test.

usage: python make_jmuben_split.py JMUBEN_ROOT [MANIFEST]   (JMUBEN_ROOT holds images/<sha256>.jpg)"""
import csv, os, sys
from collections import defaultdict
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def group_images(root, manifest):
    """Returns (rows, group_id per row index): rows of the manifest and the original-photo group of each."""
    R = list(csv.DictReader(open(manifest)))
    def hashes(p):
        g = np.asarray(Image.open(p).convert("L").resize((16, 16), Image.BILINEAR), dtype=np.float32); out = []
        for k in range(4):
            for f in (False, True):
                x = np.rot90(g, k); x = np.fliplr(x) if f else x; out.append((x > x.mean()).flatten())
        return np.array(out)
    Hs = np.array([hashes(os.path.join(root, r["path"])) for r in R]); base = Hs[:, 0, :].astype(np.uint8)
    parent = list(range(len(R)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(R)):
        d = (Hs[i][:, None, :] != base[None, :, :]).sum(-1).min(0)
        for j in np.nonzero(d <= 20)[0]:
            if j > i and R[i]["label_letter"] == R[j]["label_letter"]: parent[find(j)] = find(i)
    return R, [find(i) for i in range(len(R))]


def split_groups(R, gid):
    """Returns ({group: 'train'|'test'}, {group: representative row})."""
    groups = defaultdict(list)
    for r, g in zip(R, gid): groups[g].append(r)
    split = {}
    for c in "ABCDE":
        gs = sorted([g for g, rs in groups.items() if rs[0]["label_letter"] == c], key=lambda g: min(r["sha256"] for r in groups[g]))
        k = round(0.7 * len(gs))
        for i, g in enumerate(gs): split[g] = "train" if i < k else "test"
    return split, {g: min(rs, key=lambda r: r["sha256"]) for g, rs in groups.items()}


if __name__ == "__main__":
    root = sys.argv[1]; manifest = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "jmuben_distinct_manifest.csv")
    R, gid = group_images(root, manifest); split, one = split_groups(R, gid)
    with open(os.path.join(HERE, "jmuben_train_originals.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "label_letter"]); w.writeheader()
        for g, r in one.items():
            if split[g] == "train": w.writerow({"path": r["path"], "sha256": r["sha256"], "label_letter": r["label_letter"]})
    with open(os.path.join(HERE, "jmuben_grouped_split.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sha256", "split", "label_letter"]); w.writeheader()
        for g, r in one.items(): w.writerow({"path": r["path"], "sha256": r["sha256"], "split": "dev" if split[g] == "train" else "test", "label_letter": r["label_letter"]})
    print(f"{len(R)} distinct images -> {len(one)} originals")
