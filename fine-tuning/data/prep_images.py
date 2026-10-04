"""Image inputs of the fine-tuning and baseline scripts, made from the raw datasets with the benchmark preprocessing
(harness.preprocess: RGB, longest side 512 px, Lanczos, JPEG q90). The SHA-256 of each source file is checked against the manifest.

  bracol : $BRACOL_ROOT/<path> -> $HN04B_CACHE/<sha256>.jpg for the 1,685 images of ../../agentic-quantization/manifest.csv,
           plus $WORK/bracol/dev512/<file name> for the 419 dev images (train_lora.py --image-dir)
  jmuben : JMUBEN_RAW/<source_path> -> $WORK/jmuben/images/<sha256>.jpg for the 3,756 distinct images of jmuben_distinct_manifest.csv
           (JMUBEN_RAW = the extracted JMuBEN + JMuBEN2 archives, same folder as make_manifest_jmuben.py --root)

usage: prep_images.py bracol
       prep_images.py jmuben JMUBEN_RAW"""
import csv, hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__)); AQ = os.path.join(HERE, "../../agentic-quantization")
sys.path.insert(0, os.path.join(AQ, "benchmark"))
from harness import preprocess   # noqa: E402

H = os.environ.get("WORK", "work"); CACHE = os.environ.get("HN04B_CACHE", f"{H}/bracol/cache512")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def run(rows, src_root, src_key, out_dir):
    os.makedirs(out_dir, exist_ok=True); bad = 0
    for r in rows:
        src = os.path.join(src_root, r[src_key])
        if sha256(src) != r["sha256"]: bad += 1; print(f"SHA-256 mismatch: {src}", flush=True); continue
        preprocess(src, r["sha256"], out_dir)
    print(f"{len(rows) - bad} images -> {out_dir}" + (f" ({bad} mismatches skipped)" if bad else ""), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["bracol"]:
        root = os.environ.get("BRACOL_ROOT", f"{H}/bracol/raw"); rows = list(csv.DictReader(open(f"{AQ}/manifest.csv")))
        run(rows, root, "path", CACHE)
        d = f"{H}/bracol/dev512"; os.makedirs(d, exist_ok=True)
        for r in rows:
            if r["split"] == "dev" and os.path.exists(f"{CACHE}/{r['sha256']}.jpg"):
                shutil.copyfile(f"{CACHE}/{r['sha256']}.jpg", os.path.join(d, os.path.basename(r["path"])))
        print(f"dev images by file name -> {d}")
    elif sys.argv[1:2] == ["jmuben"] and len(sys.argv) == 3:
        rows = list(csv.DictReader(open(os.path.join(HERE, "jmuben_distinct_manifest.csv"))))
        run(rows, sys.argv[2], "source_path", f"{H}/jmuben/images")
    else:
        sys.exit(__doc__)
