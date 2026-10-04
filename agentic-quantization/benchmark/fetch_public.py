#!/usr/bin/env python3
"""Download the public GGUFs (language-model files at or under 1 GB, plus Unsloth Q4_K_M) and log
repo, revision, file, bytes, SHA-256 and date to public_gguf_log.csv. Files are never renamed.
local_path is relative to DEST (<repo>/<file>); benchmark/make_jobs.py joins it to --public-dir.
  python benchmark/fetch_public.py $HN04B_PUBLIC data/public_gguf_log.csv
"""
import csv, datetime, hashlib, os, sys
from huggingface_hub import HfApi, hf_hub_download

LIMIT = 1_000_000_000
# (repo, model, source) in evaluation priority order
REPOS = [("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B", "unsloth"),
         ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B", "unsloth"),
         ("ggml-org/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B", "official"),
         ("bartowski/Qwen_Qwen3.5-2B-GGUF", "Qwen3.5-2B", "other:bartowski/Qwen_Qwen3.5-2B-GGUF"),
         ("bartowski/Qwen_Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B", "other:bartowski/Qwen_Qwen3.5-0.8B-GGUF"),
         ("lmstudio-community/Qwen3.5-2B-GGUF", "Qwen3.5-2B", "other:lmstudio-community/Qwen3.5-2B-GGUF"),
         ("lmstudio-community/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B", "other:lmstudio-community/Qwen3.5-0.8B-GGUF"),
         ("mradermacher/Qwen3.5-2B-GGUF", "Qwen3.5-2B", "other:mradermacher/Qwen3.5-2B-GGUF"),
         ("mradermacher/Qwen3.5-2B-i1-GGUF", "Qwen3.5-2B", "other:mradermacher/Qwen3.5-2B-i1-GGUF"),
         ("mradermacher/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B", "other:mradermacher/Qwen3.5-0.8B-GGUF"),
         ("mradermacher/Qwen3.5-0.8B-i1-GGUF", "Qwen3.5-0.8B", "other:mradermacher/Qwen3.5-0.8B-i1-GGUF")]

def is_lm(name):
    n = name.lower()
    return n.endswith(".gguf") and "mmproj" not in n and "imatrix" not in n

def main(dest, log):
    api = HfApi()
    done = set()
    if os.path.exists(log):
        done = {(r["repo"], r["file"]) for r in csv.DictReader(open(log))}
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["repo", "revision", "file", "model", "source", "bytes", "sha256", "hf_published_sha256",
                        "downloaded_utc", "local_path"])
        for repo, model, source in REPOS:
            info = api.model_info(repo, files_metadata=True)
            for s in info.siblings:
                if not is_lm(s.rfilename):
                    continue
                keep = s.size <= LIMIT or (source == "unsloth" and s.rfilename.endswith("-Q4_K_M.gguf"))
                if not keep or (repo, s.rfilename) in done:
                    continue
                p = hf_hub_download(repo, s.rfilename, revision=info.sha, local_dir=os.path.join(dest, repo))
                h = hashlib.sha256()
                with open(p, "rb") as g:
                    for b in iter(lambda: g.read(1 << 24), b""):
                        h.update(b)
                sha = h.hexdigest()
                lfs = getattr(s, "lfs", None)
                if lfs and getattr(lfs, "sha256", None) and lfs.sha256 != sha:
                    print("SHA MISMATCH", repo, s.rfilename, file=sys.stderr)
                w.writerow([repo, info.sha, s.rfilename, model, source, os.path.getsize(p), sha,
                            getattr(lfs, "sha256", "") if lfs else "",
                            datetime.datetime.utcnow().strftime("%F %T"), os.path.join(repo, s.rfilename)]); f.flush()
                print("ok", repo, s.rfilename, flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
