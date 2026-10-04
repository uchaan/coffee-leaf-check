#!/usr/bin/env python3
"""Write the jobs CSV for the queues: BF16 first, then ours (model/quantization/selection/files.csv), Unsloth, official, others.
Calibration is read from each GGUF's metadata (quantize.imatrix.*), never guessed.
  python eval/make_jobs.py --a100-dir $WORK/models/a100 --public-dir $HN04B_PUBLIC --out $WORK/jobs.csv
--a100-dir holds the files of model/quantization/selection/files.csv at their listed paths (<model>/<file>, plus <model>/<model>-BF16.gguf
and <model>/mmproj-<model>-Q8_0.gguf); public files are <public-dir>/<local_path of eval/public_gguf_log.csv>.
"""
import argparse, csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import struct

MODELS = ["Qwen3.5-2B", "Qwen3.5-0.8B"]
ORDER = {"bf16": 0, "ours-general": 1, "ours-task": 1, "unsloth": 2, "official": 3}

_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_kv(path, want_prefix="quantize."):
    """Read only the key/value header of a GGUF file; return the keys starting with want_prefix."""
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not GGUF")
        struct.unpack("<I", f.read(4)); struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))

        def rstr():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                (it,) = struct.unpack("<I", f.read(4)); (n,) = struct.unpack("<Q", f.read(8))
                if it == 8:
                    for _ in range(n):
                        (m,) = struct.unpack("<Q", f.read(8)); f.seek(m, 1)
                    return f"[{n} strings]"
                sz = struct.calcsize(_FMT[it]); f.seek(sz * n, 1)
                return f"[{n} values]"
            fmt = _FMT[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            k = rstr(); (t,) = struct.unpack("<I", f.read(4)); v = rval(t)
            if k.startswith(want_prefix):
                out[k] = v
    return out


def calib(path):
    try:
        kv = {k.split(".")[-1]: v for k, v in gguf_kv(path).items() if k.startswith("quantize.imatrix")}
    except Exception as e:
        return f"unreadable ({type(e).__name__})"
    if not kv:
        return "none (no imatrix metadata)"
    parts = [f"imatrix {os.path.basename(str(kv.get('file', '?')))}"]
    if "dataset" in kv:
        parts.append(f"dataset {os.path.basename(str(kv['dataset']))}")
    if "chunks_count" in kv:
        parts.append(f"{kv['chunks_count']} chunks")
    return "; ".join(parts)


def quant_name(model, fname):
    s = fname.replace(".gguf", "")
    for pre in [f"Qwen_{model}-", f"{model}-", f"{model}."]:
        if s.startswith(pre):
            return s[len(pre):]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a100-dir", required=True)
    ap.add_argument("--public-log", default=os.path.join(HERE, "public_gguf_log.csv"))
    ap.add_argument("--public-dir", default=os.environ.get("HN04B_PUBLIC", os.path.join(os.environ.get("WORK", "work"), "public")),
                    help="root that the relative local_path column of the public log is joined to")
    ap.add_argument("--files", default=os.environ.get("HN04B_FILES", os.path.join(ROOT, "model", "quantization", "selection", "files.csv")))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = []
    for r in csv.DictReader(open(a.files)):
        # LM files only: the ours-finetuned rows (LoRA adapter, fine-tuned mmproj) are not --lm inputs
        if r["variant"] in ("bf16", "ours-general", "ours-task"):
            src = r["variant"]
            p = os.path.join(a.a100_dir, r["file"])
            q = "BF16" if src == "bf16" else quant_name(r["model"], os.path.basename(r["file"]))
            if q.startswith("ours-"):
                q = q[len("ours-"):]  # e.g. general-1GBpkg, task-UD-IQ2_M, general2-1GBpkg-v2
            jobs.append({"model": r["model"], "source": src, "quant_name": q, "lm_path": p,
                         "calibration": r.get("calibration", "")})
    if os.path.exists(a.public_log):
        for r in csv.DictReader(open(a.public_log)):
            p = os.path.join(a.public_dir, r["local_path"])  # an absolute local_path is kept as is
            jobs.append({"model": r["model"], "source": r["source"], "quant_name": quant_name(r["model"], r["file"]),
                         "lm_path": p, "calibration": calib(p)})
    dedup = {}
    for j in jobs:  # one job per file; a file listed twice keeps its last line
        dedup[j["lm_path"]] = j
    jobs = list(dedup.values())
    jobs.sort(key=lambda j: (MODELS.index(j["model"]), ORDER.get(j["source"], 4)))
    with open(a.out + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "source", "quant_name", "lm_path", "calibration"])
        w.writeheader(); w.writerows(jobs)
    os.replace(a.out + ".tmp", a.out)
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
