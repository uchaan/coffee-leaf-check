"""ImageFolder layout for yolo_cls.py: $WORK/yolo/<split>/{train,val,test}/<letter>/<sha256>.jpg (symlinks to the 512 px images).
hack   : train = BRACOL dev minus val, val = dev images with int(sha256, 16) % 5 == 0 (the 82 the adapters use), test = BRACOL test
hackjm : hack + the 559 JMuBEN training originals in train
usage: make_yolo_dirs.py hack|hackjm   (BRACOL 512 px images in $HN04B_CACHE, JMuBEN images under $WORK/jmuben)"""
import csv, os, sys
H = os.environ.get("WORK", "work"); CACHE = os.environ.get("HN04B_CACHE", f"{H}/bracol/cache512")
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, "../../.."); DATA = os.path.join(REPO, "data")
split = sys.argv[1]; rows = list(csv.DictReader(open(f"{REPO}/data/manifest.csv")))
dev = [r for r in rows if r["split"] == "dev"]
parts = {"train": [(f"{CACHE}/{r['sha256']}.jpg", r) for r in dev if int(r["sha256"], 16) % 5 != 0],
         "val": [(f"{CACHE}/{r['sha256']}.jpg", r) for r in dev if int(r["sha256"], 16) % 5 == 0],
         "test": [(f"{CACHE}/{r['sha256']}.jpg", r) for r in rows if r["split"] == "test"]}
if split == "hackjm":
    parts["train"] += [(f"{H}/jmuben/{r['path']}", r) for r in csv.DictReader(open(f"{DATA}/jmuben_train_originals.csv"))]
for part, items in parts.items():
    for src, r in items:
        d = f"{H}/yolo/{split}/{part}/{r['label_letter']}"; os.makedirs(d, exist_ok=True); dst = f"{d}/{r['sha256']}.jpg"
        if not os.path.lexists(dst): os.symlink(os.path.abspath(src), dst)
    print(part, len(items))
