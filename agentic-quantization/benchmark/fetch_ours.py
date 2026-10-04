#!/usr/bin/env python3
"""Poll selection/files.csv (a copy kept fresh from origin by the host), download every listed GGUF from
a Hugging Face repo (--repo or HN04B_HF_REPO), verify its SHA-256 against files.csv, then rebuild the jobs CSV.
  python benchmark/fetch_ours.py --repo OWNER/REPO --files selection/files.csv --dest $WORK/models/a100 --jobs $WORK/jobs.csv
"""
import argparse, csv, hashlib, os, subprocess, sys, time
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from queue_common import RUNS  # noqa: E402
ROOT = os.path.dirname(HERE)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def retire(runs, stem, old_sha):
    """Move every run, text-KLD and CPU result of a replaced file to runs/replaced/<stem>__<old sha>/."""
    import shutil
    dest = os.path.join(runs, "replaced", f"{stem}__{old_sha[:8]}")
    os.makedirs(dest, exist_ok=True)
    for sub in ["", "textkld", "cpu", "claims"]:
        d = os.path.join(runs, sub)
        if not os.path.isdir(d):
            continue
        for n in os.listdir(d):
            if f"__{stem}__" in n or n.endswith(f"__{stem}") or f"__{stem}.json" in n:
                shutil.move(os.path.join(d, n), os.path.join(dest, (sub + "__" if sub else "") + n))
    print(time.strftime("%T"), "retired results of", stem, "->", dest, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True); ap.add_argument("--dest", required=True)
    ap.add_argument("--jobs", required=True); ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--repo", default=os.environ.get("HN04B_HF_REPO"), help="Hugging Face repo holding the files of files.csv")
    a = ap.parse_args()
    if not a.repo: sys.exit("set --repo or HN04B_HF_REPO")
    repo = a.repo
    seen = {}
    while True:
        changed = False
        try:
            rows = list(csv.DictReader(open(a.files)))
        except Exception:
            rows = []
        last = {}
        for r in rows:  # a file listed twice: the last line (latest hand-off) wins
            last[r["file"]] = r
        rows = list(last.values())
        for r in rows:
            if not r["file"].endswith(".gguf"):
                continue
            p = os.path.join(a.dest, r["file"])
            key = (r["file"], r["sha256"])
            if seen.get(r["file"]) == r["sha256"]:
                continue
            if os.path.exists(p) and sha256(p) == r["sha256"]:
                seen[r["file"]] = r["sha256"]
                continue
            try:
                if os.path.exists(p):
                    os.remove(p)  # replaced upstream: drop the stale copy and every result made from it
                    retire(a.runs, os.path.basename(p).replace(".gguf", ""), seen.get(r["file"], "old"))
                hf_hub_download(repo, r["file"], local_dir=a.dest, force_download=True)
            except Exception as e:
                print(time.strftime("%T"), "not yet", r["file"], type(e).__name__, flush=True)
                continue
            got = sha256(p)
            if got != r["sha256"]:
                print(time.strftime("%T"), "SHA MISMATCH", r["file"], got, flush=True)
                os.remove(p)
                continue
            seen[r["file"]] = r["sha256"]
            changed = True
            print(time.strftime("%T"), "ok", r["file"], flush=True)
        if changed or not os.path.exists(a.jobs):
            subprocess.run([sys.executable, os.path.join(HERE, "make_jobs.py"), "--a100-dir", a.dest,
                            "--files", a.files, "--out", a.jobs])
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
