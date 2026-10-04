#!/usr/bin/env python3
"""Rebuild public_gguf_log.csv from the downloaded files: revision and published SHA-256 (etag) from the
metadata hf_hub_download leaves next to each file, SHA-256 recomputed and checked against it.
local_path is written relative to DEST (<repo>/<file>).
  python eval/rebuild_public_log.py $HN04B_PUBLIC eval/public_gguf_log.csv
"""
import csv, datetime, hashlib, os, sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_public import REPOS, is_lm  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return p, h.hexdigest()


def main(dest, out):
    items = []
    for repo, model, source in REPOS:
        d = os.path.join(dest, repo)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not is_lm(fn):
                continue
            md = open(os.path.join(d, ".cache", "huggingface", "download", fn + ".metadata")).read().split("\n")
            items.append((repo, model, source, fn, os.path.join(d, fn), md[0], md[1],
                          datetime.datetime.utcfromtimestamp(float(md[2])).strftime("%F %T")))
    with Pool(6) as pool:
        hashes = dict(pool.map(sha, [i[4] for i in items]))
    bad = 0
    with open(out + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "revision", "file", "model", "source", "bytes", "sha256", "hf_published_sha256",
                    "downloaded_utc", "local_path"])
        for repo, model, source, fn, p, rev, etag, ts in items:
            s = hashes[p]
            bad += s != etag
            w.writerow([repo, rev, fn, model, source, os.path.getsize(p), s, etag, ts, os.path.join(repo, fn)])
    os.replace(out + ".tmp", out)
    print(len(items), "files;", bad, "with SHA-256 different from the HF-published hash")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
